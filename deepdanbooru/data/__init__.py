from typing import Any, Union

import six
import tensorflow as tf
import tensorflow_io as tfio

import deepdanbooru as dd

from .dataset import load_image_records, load_tags
from .dataset_wrapper import DatasetWrapper


def load_image_for_evaluate(
    input_: Union[str, six.BytesIO], width: int, height: int, normalize: bool = True
) -> Any:
    if isinstance(input_, six.BytesIO):
        image_raw = input_.getvalue()
    else:
        image_raw = tf.io.read_file(input_)
    try:
        image = tf.io.decode_png(image_raw, channels=3)
    except:
        image = tfio.image.decode_webp(image_raw)
        image = tfio.experimental.color.rgba_to_rgb(image)

    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    )
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)

    if normalize:
        image = image / 255.0

    return image
