import argparse
import os
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import skimage.transform
import tensorflow as tf

THRESHOLD = 0.5


def transform_and_pad_image(image, target_width, target_height):
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_array = image

    # centerize
    t = skimage.transform.AffineTransform(
        translation=(-image_width * 0.5, -image_height * 0.5))

    t += skimage.transform.AffineTransform(
        translation=(target_width * 0.5, target_height * 0.5))

    warp_shape = (target_height, target_width)

    image_array = skimage.transform.warp(
        image_array, (t).inverse, output_shape=warp_shape, order=1, mode='edge')

    return image_array


def prepare_image(path):
    image_raw = tf.io.read_file(path)
    image = tf.io.decode_png(image_raw, channels=3)

    image = tf.image.resize(image, size=(
        299, 299), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
    image = image.numpy()
    image = transform_and_pad_image(image, 299, 299)
    image = image / 255.0

    return image


def evaluate_image(
    image_path: str, model: Any, tags: List[str], threshold: float = THRESHOLD
) -> Iterator[Tuple[str, float]]:
    print('Loading image ...')
    image = prepare_image(image_path)
    image_shape = image.shape
    image = image.reshape(
        (1, image_shape[0], image_shape[1], image_shape[2]))

    print('Evaluating ...')
    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    for tag in tags:
        if result_dict[tag] >= threshold:
            yield tag, result_dict[tag]


def evaluate_images(
        image_paths: List[str], model_path: str, tags_path: str, cpu: bool,
        threshold: float = THRESHOLD, compile_: Optional[bool] = None):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print('Loading model ...')
    if compile_ is None:
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path, compile=compile_)

    print('Loading tags ...')
    with open(tags_path, 'r') as stream:
        tags = [tag for tag in (tag.strip() for tag in stream) if tag]

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f'Target path {image_path} is not exists.')
            continue
        print(f'Tags of {image_path}:')
        for tag, score in evaluate_image(image_path, model, tags, threshold):
            print(f'({score:05.3f}) {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Image file path.", type=str)
    parser.add_argument("--cpu", help="Use CPU.", action="store_true")
    parser.add_argument("--model", help="Path to model file.", default="model.h5")
    parser.add_argument("--tags", help="Path to tags file.", default="tags.txt")
    parser.add_argument(
        "--threshold", help="Score threshold.", type=float, default=0.5)

    args = parser.parse_args()

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(args.image_path):
        raise Exception(f'Target path {args.image_path} is not exists.')

    print('Loading model ...')
    model = tf.keras.models.load_model(args.model)

    print('Loading image ...')
    image = prepare_image(args.image_path)
    image_shape = image.shape
    image = image.reshape(
        (1, image_shape[0], image_shape[1], image_shape[2]))

    print('Loading tags ...')
    with open(args.tags, 'r') as stream:
        tags = [tag for tag in (tag.strip() for tag in stream) if tag]

    print('Evaluating ...')
    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    print(f'Tags of {args.image_path}:')
    for tag in tags:
        if result_dict[tag] >= args.threshold:
            print(f'({result_dict[tag]:05.3f}) {tag}')
