from typing import List
import tensorflow as tf
import deepdanbooru as dd

# optimizations: see in tf.lite.Optimize
def convert_to_tflite_from_from_saved_model(
    project_path: str, model_path: str, save_path: str,
    optimizations: List[tf.lite.Optimize] = [tf.lite.Optimize.DEFAULT],
    verbose: bool = False
):
    if not model_path and not project_path:
        raise Exception("You must provide project path or model path.")

    if not save_path:
        raise Exception("You must provide a path to save tflite model.")

    if model_path:
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = tf.keras.models.load_model(model_path)
    else:
        if verbose:
            print(f"Loading model from project {project_path} ...")
        model = dd.project.load_model_from_project(project_path)

    if verbose:
        print("Converting ...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = optimizations
    tflite_model = converter.convert()

    if verbose:
        print("Saving ...")

    with open(save_path, "wb") as f:
        f.write(tflite_model)

    if verbose:
        print(f"Converted model has been saved to {save_path}")
