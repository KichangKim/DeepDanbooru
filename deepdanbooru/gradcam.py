import tensorflow as tf
import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


def grad(y, x):
    V = tf.keras.layers.Lambda(lambda z: tf.gradients(z[0], z[1]), output_shape=[1])(
        [y, x]
    )
    return V


def grad_cam_test(model, x, some_variable):
    fixed_input = model.inputs
    fixed_output = tf.keras.layers.Lambda(
        lambda z: tf.keras.backend.gradients(z[0], z[1]), output_shape=[2]
    )([model.inputs[0], model.outputs[0]])

    grad_model = tf.keras.Model(inputs=fixed_input, outputs=fixed_output)

    return grad_model.predict(x)


def run_test():
    # Generate sample model
    x = tf.keras.Input(shape=(2))
    y = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    target = np.array([[1.0, 2.0]], dtype=np.float32)

    # Calculate gradient using numpy array
    input_numpy = np.array([[0.0, 0.0]])
    grad_output_numpy = grad_cam_test(model, input_numpy, target)
    print(f"numpy: {grad_output_numpy}")

    # Calculate gradient using tf.Variable
    input_variable = tf.constant([[0.0, 0.0]])
    grad_output_variable = grad_cam_test(model, input_variable, target)
    print(f"variable: {grad_output_variable}")


run_test()
