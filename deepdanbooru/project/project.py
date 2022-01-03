import os
import deepdanbooru as dd
import tensorflow as tf

DEFAULT_PROJECT_CONTEXT = {
    "image_width": 299,
    "image_height": 299,
    "database_path": None,
    "minimum_tag_count": 20,
    "model": "resnet_custom_v2",
    "minibatch_size": 32,
    "epoch_count": 10,
    "export_model_per_epoch": 10,
    "checkpoint_frequency_mb": 200,
    "console_logging_frequency_mb": 10,
    "loss": "binary_crossentropy",
    "optimizer": "adam",    
    "learning_rate": 0.001,
    "rotation_range": [0.0, 360.0],
    "scale_range": [0.9, 1.1],
    "shift_range": [-0.1, 0.1],
    "mixed_precision": False,
}


def load_project(project_path):
    project_context_path = os.path.join(project_path, "project.json")
    project_context = dd.io.deserialize_from_json(project_context_path)
    tags = dd.data.load_tags_from_project(project_path)

    model_type = project_context["model"]
    model_path = os.path.join(project_path, f"model-{model_type}.h5")
    model = tf.keras.models.load_model(model_path)

    return project_context, model, tags


def load_model_from_project(project_path, compile_model=True):
    project_context_path = os.path.join(project_path, "project.json")
    project_context = dd.io.deserialize_from_json(project_context_path)

    model_type = project_context["model"]
    model_path = os.path.join(project_path, f"model-{model_type}.h5")
    model = tf.keras.models.load_model(model_path, compile=compile_model)

    return model


def load_tags_from_project(project_path):
    tags_path = os.path.join(project_path, "tags.txt")

    return dd.data.load_tags(tags_path)
