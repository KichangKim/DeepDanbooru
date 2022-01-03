import numpy as np
import tensorflow as tf
import deepdanbooru as dd


def resnet_bottleneck_block(
    x, output_filters, inter_filters, activation=True, se=False
):
    c1 = dd.model.layers.conv_bn_relu(x, inter_filters, (1, 1))
    c2 = dd.model.layers.conv_bn_relu(c1, inter_filters, (3, 3))
    c3 = dd.model.layers.conv_bn(
        c2, output_filters, (1, 1), bn_gamma_initializer="zeros"
    )

    if se:
        c3 = dd.model.layers.squeeze_excitation(c3)

    p = tf.keras.layers.Add()([c3, x])

    if activation:
        return tf.keras.layers.Activation("relu")(p)
    else:
        return p


def resnet_bottleneck_inc_block(
    x, output_filters, inter_filters, strides1x1=(1, 1), strides2x2=(2, 2), se=False
):
    c1 = dd.model.layers.conv_bn_relu(x, inter_filters, (1, 1), strides=strides1x1)
    c2 = dd.model.layers.conv_bn_relu(c1, inter_filters, (3, 3), strides=strides2x2)
    c3 = dd.model.layers.conv_bn(
        c2, output_filters, (1, 1), bn_gamma_initializer="zeros"
    )

    if se:
        c3 = dd.model.layers.squeeze_excitation(c3)

    strides = np.multiply(strides1x1, strides2x2)
    s = dd.model.layers.conv_bn(x, output_filters, (1, 1), strides=strides)  # shortcut

    p = tf.keras.layers.Add()([c3, s])

    return tf.keras.layers.Activation("relu")(p)


def resnet_original_bottleneck_model(
    x, filter_sizes, repeat_sizes, final_pool=True, se=False
):
    """
    https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/resnet_models.py
    """
    assert len(filter_sizes) == len(repeat_sizes)

    x = dd.model.layers.conv_bn_relu(x, filter_sizes[0] // 4, (7, 7), strides=(2, 2))
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    for i in range(len(repeat_sizes)):
        x = resnet_bottleneck_inc_block(
            x=x,
            output_filters=filter_sizes[i],
            inter_filters=filter_sizes[i] // 4,
            strides2x2=(2, 2) if i > 0 else (1, 1),
            se=se,
        )
        x = dd.model.layers.repeat_blocks(
            x=x,
            block_delegate=resnet_bottleneck_block,
            count=repeat_sizes[i],
            output_filters=filter_sizes[i],
            inter_filters=filter_sizes[i] // 4,
            se=se,
        )

    if final_pool:
        x = tf.keras.layers.AveragePooling2D((7, 7), name="ap_final")(x)

    return x


def resnet_longterm_bottleneck_model(
    x, filter_sizes, repeat_sizes, final_pool=True, se=False
):
    """
    Add long-term shortcut.
    """
    assert len(filter_sizes) == len(repeat_sizes)

    x = dd.model.layers.conv_bn_relu(x, filter_sizes[0] // 4, (7, 7), strides=(2, 2))
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    for i in range(len(repeat_sizes)):
        x = resnet_bottleneck_inc_block(
            x=x,
            output_filters=filter_sizes[i],
            inter_filters=filter_sizes[i] // 4,
            strides2x2=(2, 2) if i > 0 else (1, 1),
            se=se,
        )
        x_1 = dd.model.layers.repeat_blocks(
            x=x,
            block_delegate=resnet_bottleneck_block,
            count=repeat_sizes[i] - 1,
            output_filters=filter_sizes[i],
            inter_filters=filter_sizes[i] // 4,
            se=se,
        )
        x_1 = resnet_bottleneck_block(
            x_1,
            output_filters=filter_sizes[i],
            inter_filters=filter_sizes[i] // 4,
            activation=False,
            se=se,
        )

        x = tf.keras.layers.Add()([x_1, x])  # long-term shortcut
        x = tf.keras.layers.Activation("relu")(x)

    if final_pool:
        x = tf.keras.layers.AveragePooling2D((7, 7), name="ap_final")(x)

    return x


def create_resnet_152(x, output_dim):
    """
    Original ResNet-152 Model.
    """
    filter_sizes = [256, 512, 1024, 2048]
    repeat_sizes = [2, 7, 35, 2]

    x = resnet_original_bottleneck_model(
        x, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes
    )

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return x


def create_resnet_custom_v1(x, output_dim):
    """
    DeepDanbooru web (until 2019/04/20)
    Short, wide
    """
    filter_sizes = [256, 512, 1024, 2048, 4096]
    repeat_sizes = [2, 7, 35, 2, 2]

    x = resnet_original_bottleneck_model(
        x, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes, final_pool=False
    )

    x = dd.model.layers.conv_gap(x, output_dim)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return x


def create_resnet_custom_v2(x, output_dim):
    """
    Experimental (blazing-deep network)
    Deep, narrow
    """
    filter_sizes = [256, 512, 1024, 1024, 1024, 2048]
    repeat_sizes = [2, 7, 40, 16, 16, 6]

    x = resnet_original_bottleneck_model(
        x, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes, final_pool=False
    )

    x = dd.model.layers.conv_gap(x, output_dim)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return x


def create_resnet_custom_v3(x, output_dim):
    filter_sizes = [256, 512, 1024, 1024, 2048, 4096]
    repeat_sizes = [2, 7, 19, 19, 2, 2]

    x = resnet_original_bottleneck_model(
        x, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes, final_pool=False
    )

    x = dd.model.layers.conv_gap(x, output_dim)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return x


def create_resnet_custom_v4(x, output_dim):
    filter_sizes = [256, 512, 1024, 1024, 1024, 2048]
    repeat_sizes = [2, 7, 10, 10, 10, 2]

    x = resnet_original_bottleneck_model(
        x, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes, final_pool=False
    )

    x = dd.model.layers.conv_gap(x, output_dim)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return x
