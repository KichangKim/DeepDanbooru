import tensorflow as tf


def focal_loss(alpha=0.25, gamma=2.0, epsilon=1e-7):
    def loss(y_true, y_pred):
        value = -alpha * y_true * tf.math.pow(1.0 - y_pred, gamma) * tf.math.log(
            y_pred + epsilon
        ) - (1.0 - alpha) * (1.0 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(
            1.0 - y_pred + epsilon
        )

        return tf.math.reduce_mean(value)

    return loss


def binary_crossentropy(epsilon=1e-7):
    def loss(y_true, y_pred):
        clipped_y_pred = tf.clip_by_value(y_pred, epsilon, tf.float32.max)
        clipped_y_pred_nega = tf.clip_by_value(1.0 - y_pred, epsilon, tf.float32.max)
        value = (-y_true) * tf.math.log(clipped_y_pred) - (1.0 - y_true) * tf.math.log(
            clipped_y_pred_nega
        )

        return tf.math.reduce_mean(value)

    return loss
