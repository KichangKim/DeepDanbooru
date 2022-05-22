import tensorflow as tf


def conv(
    x, filters, kernel_size, strides=(1, 1), padding="same", initializer="he_normal"
):
    c = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=initializer,
        use_bias=False,
    )(x)

    return c


def conv_bn(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    initializer="he_normal",
    bn_gamma_initializer="ones",
):
    c = conv(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        initializer=initializer,
    )

    c_bn = tf.keras.layers.BatchNormalization(gamma_initializer=bn_gamma_initializer)(c)

    return c_bn


def conv_bn_relu(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    initializer="he_normal",
    bn_gamma_initializer="ones",
):
    c_bn = conv_bn(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        initializer=initializer,
        bn_gamma_initializer=bn_gamma_initializer,
    )

    return tf.keras.layers.Activation("relu")(c_bn)


def conv_gap(x, output_filters, kernel_size=(1, 1)):
    x = conv(x, filters=output_filters, kernel_size=kernel_size)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    return x


def repeat_blocks(x, block_delegate, count, **kwargs):
    assert count >= 0

    for _ in range(count):
        x = block_delegate(x, **kwargs)
    return x


def squeeze_excitation(x, reduction=16):
    """
    Squeeze-Excitation layer from https://arxiv.org/abs/1709.01507
    """
    output_filters = x.shape[-1]

    assert output_filters // reduction > 0

    s = x

    s = tf.keras.layers.GlobalAveragePooling2D()(s)
    s = tf.keras.layers.Dense(output_filters // reduction, activation="relu")(s)
    s = tf.keras.layers.Dense(output_filters, activation="sigmoid")(s)
    x = tf.keras.layers.Multiply()([x, s])

    return x
