import tensorflow as tf
import deepdanbooru as dd

from .dataset_wrapper import DatasetWrapper
from .dataset import load_image_records
from .dataset import load_tags


def load_image_for_evaluate(path, width, height, normalize=True):
    image_raw = tf.io.read_file(path)
    image = tf.io.decode_png(image_raw, channels=3)

    image = tf.image.resize(
        image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)

    if normalize:
        image = image / 255.0

    return image
