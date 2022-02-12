import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import deepdanbooru as dd


class DatasetWrapper:
    """
    Wrapper class for data pipelining/augmentation.
    """

    def __init__(
        self, inputs, tags, width, height, scale_range, rotation_range, shift_range
    ):
        self.inputs = inputs
        self.width = width
        self.height = height
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.tag_all_array = np.array(tags)

    def get_dataset(self, minibatch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.inputs)
        dataset = dataset.map(
            self.map_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.map(
            self.map_transform_image_and_label,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(minibatch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.apply(
        #    tf.data.experimental.prefetch_to_device('/device:GPU:0'))

        return dataset

    def map_load_image(self, image_path, tag_string):
        image_raw = tf.io.read_file(image_path)
        try:
            image = tf.io.decode_png(image_raw, channels=3)
        except:
            image = tfio.image.decode_webp(image_raw)
            image = tfio.experimental.color.rgba_to_rgb(image)

        if self.scale_range:
            pre_scale = self.scale_range[1]
        else:
            pre_scale = 1.0

        size = (int(self.height * pre_scale), int(self.width * pre_scale))

        image = tf.image.resize(
            image,
            size=size,
            method=tf.image.ResizeMethod.AREA,
            preserve_aspect_ratio=True,
        )

        return (image, tag_string)

    def map_transform_image_and_label(self, image, tag_string):
        return tf.py_function(
            self.map_transform_image_and_label_py,
            (image, tag_string),
            (tf.float32, tf.float32),
        )

    def map_transform_image_and_label_py(self, image, tag_string):
        # transform image
        image = image.numpy()

        if self.scale_range:
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) * (
                1.0 / self.scale_range[1]
            )
        else:
            scale = None

        if self.rotation_range:
            rotation = random.uniform(self.rotation_range[0], self.rotation_range[1])
        else:
            rotation = None

        if self.shift_range:
            shift_x = random.uniform(self.shift_range[0], self.shift_range[1])
            shift_y = random.uniform(self.shift_range[0], self.shift_range[1])
            shift = (shift_x, shift_y)
        else:
            shift = None

        image = dd.image.transform_and_pad_image(
            image=image,
            target_width=self.width,
            target_height=self.height,
            rotation=rotation,
            scale=scale,
            shift=shift,
        )

        image = image / 255.0  # normalize to 0~1
        # image = image.astype(np.float32)

        # transform tag
        tag_string = tag_string.numpy().decode()
        tag_array = np.array(tag_string.split(" "))

        labels = np.where(np.isin(self.tag_all_array, tag_array), 1, 0).astype(
            np.float32
        )

        return (image, labels)
