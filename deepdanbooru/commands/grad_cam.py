import os

import tensorflow as tf
import numpy as np
from PIL import Image
import deepdanbooru as dd
from scipy import ndimage


@tf.function
def get_gradient(model, x, output_mask):
    with tf.GradientTape() as tape:
        output = model(x)
        gradcam_loss = tf.reduce_sum(tf.multiply(output_mask, output))

    return tape.gradient(gradcam_loss, x)


def norm_clip_grads(grads):
    upper_quantile = np.quantile(grads, 0.99)
    lower_quantile = np.quantile(grads, 0.01)
    clipped_grads = np.abs(np.clip(grads, lower_quantile, upper_quantile))

    return clipped_grads / np.max(clipped_grads)


def filter_grads(grads):
    return ndimage.median_filter(grads, 10)


def to_onehot(length, index):
    value = np.zeros(shape=(1, length), dtype=np.float32)
    value[0, index] = 1.0
    return value


def grad_cam(project_path, target_path, output_path, threshold):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(target_path):
        raise Exception(f"Target path {target_path} is not exists.")

    if os.path.isfile(target_path):
        taget_image_paths = [target_path]
    else:
        patterns = [
            "*.[Pp][Nn][Gg]",
            "*.[Jj][Pp][Gg]",
            "*.[Jj][Pp][Ee][Gg]",
            "*.[Gg][Ii][Ff]",
        ]

        taget_image_paths = dd.io.get_file_paths_in_directory(target_path, patterns)

        taget_image_paths = dd.extra.natural_sorted(taget_image_paths)

    model = dd.project.load_model_from_project(project_path)
    tags = dd.project.load_tags_from_project(project_path)
    width = model.input_shape[2]
    height = model.input_shape[1]

    dd.io.try_create_directory(output_path)

    for image_path in taget_image_paths:
        image = dd.data.load_image_for_evaluate(image_path, width=width, height=height)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        image_folder = os.path.join(output_path, image_name)
        dd.io.try_create_directory(image_folder)

        Image.fromarray(np.uint8(image * 255.0)).save(
            os.path.join(image_folder, f"input.png")
        )
        image_for_result = image
        image_shape = image.shape
        y = model.predict(
            image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
        )[0]

        result_dict = {}

        estimated_tags = []

        for i, tag in enumerate(tags):
            result_dict[tag] = y[i]

            if y[i] >= threshold:
                estimated_tags.append((i, tag))

        print(f"Tags of {image_path}:")

        for tag in tags:
            if result_dict[tag] >= threshold:
                print(f"({result_dict[tag]:05.3f}) {tag}")

        image = image.astype(np.float32)

        for estimated_tag in estimated_tags:
            print(f"Calculating grad-cam ... ({estimated_tag[1]})")
            grads = get_gradient(
                model, tf.Variable([image]), to_onehot(len(tags), estimated_tag[0])
            )[0]
            print("Normalizing gradients ...")
            grads = norm_clip_grads(grads)
            print("Filtering gradients ...")
            grads = filter_grads(grads)
            Image.fromarray(np.uint8(grads * 255.0)).save(
                os.path.join(
                    image_folder,
                    f"result-{estimated_tag[1]}.png".replace(":", "_").replace(
                        "/", "_"
                    ),
                )
            )
            mask_array = np.stack([np.max(grads, axis=-1)] * 3, axis=2)
            Image.fromarray(
                np.uint8(np.multiply(image_for_result, mask_array) * 255.0)
            ).save(
                os.path.join(
                    image_folder,
                    f"result-{estimated_tag[1]}-masked.png".replace(":", "_").replace(
                        "/", "_"
                    ),
                )
            )
