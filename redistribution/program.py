import argparse
import os

import numpy as np
import skimage.transform
import tensorflow as tf


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Image file path.", type=str)
    parser.add_argument("--cpu", help="Use CPU.", action="store_true")
    parser.add_argument(
        "--threshold", help="Score threshold.", type=float, default=0.5)

    args = parser.parse_args()

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(args.image_path):
        raise Exception(f'Target path {args.image_path} is not exists.')

    print('Loading model ...')
    model = tf.keras.models.load_model('model.h5')

    print('Loading image ...')
    image = prepare_image(args.image_path)
    image_shape = image.shape
    image = image.reshape(
        (1, image_shape[0], image_shape[1], image_shape[2]))

    print('Loading tags ...')
    with open('tags.txt', 'r') as stream:
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
