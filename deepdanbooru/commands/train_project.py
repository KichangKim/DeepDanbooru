import os
import random
from sqlite3.dbapi2 import NotSupportedError
import time
import datetime

import tensorflow as tf

import deepdanbooru as dd


def export_model_as_float32(temporary_model, checkpoint_path, export_path):
    """
    Hotfix for exporting mixed precision model as float32.
    """
    checkpoint = tf.train.Checkpoint(model=temporary_model)

    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_path, max_to_keep=3
    )

    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    temporary_model.save(export_path, include_optimizer=False)


def train_project(project_path, source_model):
    project_context_path = os.path.join(project_path, "project.json")
    project_context = dd.io.deserialize_from_json(project_context_path)

    width = project_context["image_width"]
    height = project_context["image_height"]
    database_path = project_context["database_path"]
    minimum_tag_count = project_context["minimum_tag_count"]
    model_type = project_context["model"]
    optimizer_type = project_context["optimizer"]
    learning_rate = (
        project_context["learning_rate"]
        if "learning_rate" in project_context
        else 0.001
    )
    learning_rates = (
        project_context["learning_rates"]
        if "learning_rates" in project_context
        else None
    )
    minibatch_size = project_context["minibatch_size"]
    epoch_count = project_context["epoch_count"]
    export_model_per_epoch = (
        project_context["export_model_per_epoch"]
        if "export_model_per_epoch" in project_context
        else 10
    )
    checkpoint_frequency_mb = project_context["checkpoint_frequency_mb"]
    console_logging_frequency_mb = project_context["console_logging_frequency_mb"]
    rotation_range = project_context["rotation_range"]
    scale_range = project_context["scale_range"]
    shift_range = project_context["shift_range"]
    use_mixed_precision = (
        project_context["mixed_precision"]
        if "mixed_precision" in project_context
        else False
    )
    loss_type = (
        project_context["loss"] if "loss" in project_context else "binary_crossentropy"
    )
    checkpoint_path = os.path.join(project_path, "checkpoints")

    # disable PNG warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # tf.keras.backend.set_epsilon(1e-6)
    # tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
    # tf.config.gpu.set_per_process_memory_growth(True)

    if optimizer_type == "adam":
        optimizer = tf.optimizers.Adam(learning_rate)
        print("Using Adam optimizer ... ")
    elif optimizer_type == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
        print("Using SGD optimizer ... ")
    elif optimizer_type == "rmsprop":
        optimizer = tf.optimizers.RMSprop(learning_rate)
        print("Using RMSprop optimizer ... ")
    else:
        raise Exception(f"Not supported optimizer : {optimizer_type}")

    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("Optimizer is changed to LossScaleOptimizer.")

    if model_type == "resnet_152":
        model_delegate = dd.model.resnet.create_resnet_152
    elif model_type == "resnet_custom_v1":
        model_delegate = dd.model.resnet.create_resnet_custom_v1
    elif model_type == "resnet_custom_v2":
        model_delegate = dd.model.resnet.create_resnet_custom_v2
    elif model_type == "resnet_custom_v3":
        model_delegate = dd.model.resnet.create_resnet_custom_v3
    elif model_type == "resnet_custom_v4":
        model_delegate = dd.model.resnet.create_resnet_custom_v4
    else:
        raise Exception(f"Not supported model : {model_type}")

    print("Loading tags ... ")
    tags = dd.project.load_tags_from_project(project_path)
    output_dim = len(tags)

    print(f"Creating model ({model_type}) ... ")
    # tf.keras.backend.set_learning_phase(1)

    if source_model:
        model = tf.keras.models.load_model(source_model)
        print(
            f"Model : {model.input_shape} -> {model.output_shape} (loaded from {source_model})"
        )
    else:
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)

        inputs = tf.keras.Input(shape=(height, width, 3), dtype=tf.float32)  # HWC
        ouputs = model_delegate(inputs, output_dim)
        model = tf.keras.Model(inputs=inputs, outputs=ouputs, name=model_type)

        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("float32")
            tf.keras.mixed_precision.set_global_policy(policy)

            inputs_float32 = tf.keras.Input(
                shape=(height, width, 3), dtype=tf.float32
            )  # HWC
            ouputs_float32 = model_delegate(inputs_float32, output_dim)
            model_float32 = tf.keras.Model(
                inputs=inputs_float32, outputs=ouputs_float32, name=model_type
            )

            print("float32 model is created.")

        print(f"Model : {model.input_shape} -> {model.output_shape}")

    if loss_type == "binary_crossentropy":
        loss = loss = tf.keras.losses.BinaryCrossentropy()
    elif loss_type == "focal_loss":
        loss = dd.model.losses.focal_loss()
    else:
        raise NotSupportedError(f"Loss type '{loss_type}' is not supported.")
    print(f"Using loss : {loss_type}")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    print(f"Loading database ... ")
    image_records = dd.data.load_image_records(database_path, minimum_tag_count)

    # Checkpoint variables
    used_epoch = tf.Variable(0, dtype=tf.int64)
    used_minibatch = tf.Variable(0, dtype=tf.int64)
    used_sample = tf.Variable(0, dtype=tf.int64)
    offset = tf.Variable(0, dtype=tf.int64)
    random_seed = tf.Variable(0, dtype=tf.int64)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        used_epoch=used_epoch,
        used_minibatch=used_minibatch,
        used_sample=used_sample,
        offset=offset,
        random_seed=random_seed,
    )

    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_path, max_to_keep=3
    )

    if manager.latest_checkpoint:
        print(f"Checkpoint exists. Continuing training ... ({datetime.datetime.now()})")
        checkpoint.restore(manager.latest_checkpoint)
        print(
            f"used_epoch={int(used_epoch)}, used_minibatch={int(used_minibatch)}, used_sample={int(used_sample)}, offset={int(offset)}, random_seed={int(random_seed)}"
        )
    else:
        print(f"No checkpoint. Starting new training ... ({datetime.datetime.now()})")

    epoch_size = len(image_records)
    slice_size = minibatch_size * checkpoint_frequency_mb
    loss_sum = 0.0
    loss_count = 0
    used_sample_sum = 0
    last_time = time.time()

    while int(used_epoch) < epoch_count:
        print(f"Shuffling samples (epoch {int(used_epoch)}) ... ")
        epoch_random = random.Random(int(random_seed))
        epoch_random.shuffle(image_records)

        # Update learning rate
        if learning_rates:
            for learning_rate_per_epoch in learning_rates:
                if learning_rate_per_epoch["used_epoch"] <= int(used_epoch):
                    learning_rate = learning_rate_per_epoch["learning_rate"]
        print(f"Trying to change learning rate to {learning_rate} ...")
        optimizer.learning_rate.assign(learning_rate)
        print(f"Learning rate is changed to {optimizer.learning_rate} ...")

        while int(offset) < epoch_size:
            image_records_slice = image_records[
                int(offset) : min(int(offset) + slice_size, epoch_size)
            ]

            image_paths = [image_record[0] for image_record in image_records_slice]
            tag_strings = [image_record[1] for image_record in image_records_slice]

            dataset_wrapper = dd.data.DatasetWrapper(
                (image_paths, tag_strings),
                tags,
                width,
                height,
                scale_range=scale_range,
                rotation_range=rotation_range,
                shift_range=shift_range,
            )
            dataset = dataset_wrapper.get_dataset(minibatch_size)

            for (x_train, y_train) in dataset:
                sample_count = x_train.shape[0]

                step_result = model.train_on_batch(
                    x_train, y_train, reset_metrics=False
                )

                used_minibatch.assign_add(1)
                used_sample.assign_add(sample_count)
                used_sample_sum += sample_count
                loss_sum += step_result[0]
                loss_count += 1

                if int(used_minibatch) % console_logging_frequency_mb == 0:
                    # calculate logging informations
                    current_time = time.time()
                    delta_time = current_time - last_time
                    step_metric_precision = step_result[1]
                    step_metric_recall = step_result[2]
                    if step_metric_precision + step_metric_recall > 0.0:
                        step_metric_f1_score = (
                            2.0
                            * (step_metric_precision * step_metric_recall)
                            / (step_metric_precision + step_metric_recall)
                        )
                    else:
                        step_metric_f1_score = 0.0
                    average_loss = loss_sum / float(loss_count)
                    samples_per_seconds = float(used_sample_sum) / max(
                        delta_time, 0.001
                    )
                    progress = (
                        float(int(used_sample))
                        / float(epoch_size * epoch_count)
                        * 100.0
                    )
                    remain_seconds = float(
                        epoch_size * epoch_count - int(used_sample)
                    ) / max(samples_per_seconds, 0.001)
                    eta_datetime = datetime.datetime.now() + datetime.timedelta(
                        seconds=remain_seconds
                    )
                    eta_datetime_string = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Epoch[{int(used_epoch)}] Loss={average_loss:.6f}, P={step_metric_precision:.6f}, R={step_metric_recall:.6f}, F1={step_metric_f1_score:.6f}, Speed = {samples_per_seconds:.1f} samples/s, {progress:.2f} %, ETA = {eta_datetime_string}"
                    )

                    # reset for next logging
                    model.reset_metrics()
                    loss_sum = 0.0
                    loss_count = 0
                    used_sample_sum = 0
                    last_time = current_time

            offset.assign_add(slice_size)
            print(f"Saving checkpoint ... ({datetime.datetime.now()})")
            manager.save()

        used_epoch.assign_add(1)
        random_seed.assign_add(1)
        offset.assign(0)

        if export_model_per_epoch == 0 or int(used_epoch) % export_model_per_epoch == 0:
            print(f"Saving model ... (per epoch {export_model_per_epoch})")
            export_path = os.path.join(
                project_path, f"model-{model_type}.h5.e{int(used_epoch)}"
            )
            model.save(export_path, include_optimizer=False, save_format="h5")

            if use_mixed_precision:
                export_model_as_float32(
                    model_float32, checkpoint_path, export_path + ".float32.h5"
                )

    print("Saving model ...")
    model_path = os.path.join(project_path, f"model-{model_type}.h5")

    # tf.keras.experimental.export_saved_model throw exception now
    # see https://github.com/tensorflow/tensorflow/issues/27112
    model.save(model_path, include_optimizer=False)

    if use_mixed_precision:
        export_model_as_float32(
            model_float32, checkpoint_path, model_path + ".float32.h5"
        )

    print("Training is complete.")
    print(
        f"used_epoch={int(used_epoch)}, used_minibatch={int(used_minibatch)}, used_sample={int(used_sample)}"
    )
