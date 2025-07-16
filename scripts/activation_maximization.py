import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf


def reverse_preprocess_image(x, dtype=int):
    """Convert from preprocessed BGR [0-centered] to RGB [0-255]"""
    mean = [103.939, 116.779, 123.68]
    x = np.array(x)
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    x = x[..., ::-1]  # BGR -> RGB
    return np.array(x, dtype=dtype)


def clip_x_np(x):
    """For post-processing / logging: Tensor -> clipped RGB"""
    tmp = reverse_preprocess_image(np.array(x), dtype=float)
    return preprocess_image_np(np.clip(tmp, 0, 255))


def preprocess_image_np(x):
    """Standard VGG preprocessing using NumPy"""
    mean = [103.939, 116.779, 123.68]
    x = x[..., ::-1]  # RGB -> BGR
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


def tf_clip_x(x):
    """Preprocess using TensorFlow ops to preserve gradients"""
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    x_rev = x + mean[None, None, None, :]           # add mean
    x_rev = tf.reverse(x_rev, axis=[-1])            # BGR -> RGB
    x_rev = tf.clip_by_value(x_rev, 0.0, 255.0)      # clip
    x_proc = tf.reverse(x_rev, axis=[-1])            # RGB -> BGR
    x_proc = x_proc - mean[None, None, None, :]      # subtract mean
    return x_proc


def generate(neuron_selection, iterations):
    os.makedirs('../plots/activation_maximization', exist_ok=True)

    model = VGG16(weights='imagenet')
    model_softmax = VGG16(weights='imagenet')

    # Disable softmax in the original model
    model.layers[-1].activation = None
    model = tf.keras.models.clone_model(model)
    model.set_weights(model_softmax.get_weights())

    # Initial image
    x = np.random.uniform(-64, 64, (224, 224, 3)) * 0.1
    x = tf.Variable(x[None, ...], dtype=tf.float32)  # Add batch dim

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.0)

    for i in range(iterations):
        # Apply Gaussian blur to the image after gradient update
        x_np = x.numpy()[0]  # Remove batch dimension: shape (224, 224, 3)
        x_np_blurred = cv2.GaussianBlur(x_np, ksize=(3, 3), sigmaX=0.38)

        # Assign blurred image back to the variable
        x.assign(tf.convert_to_tensor(x_np_blurred[None, ...], dtype=tf.float32))

        with tf.GradientTape() as tape:
            x_clipped = tf_clip_x(x)
            pred = model(x_clipped)
            loss = -pred[:, neuron_selection]

        G = tape.gradient(loss, x)
        if G is None:
            raise ValueError("Gradient is None. Graph might be broken.")

        optimizer.apply_gradients([(G, x)])

        # Logging
        x_np = x.numpy()[0]
        pred_value = model_softmax(np.array([clip_x_np(x_np)]))[-1][neuron_selection]
        print("i: {}, pred: {:.10f}".format(i, pred_value))

        # Visualize result
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(reverse_preprocess_image(np.array(x.numpy()[0])))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plot_path = f'../plots/activation_maximization/gen_rand_{neuron_selection}_{i}.png'
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    generate(130, iterations=150) # flamingo
    generate(31, iterations=150) # tree-frog
    generate(953, iterations=150) # pineapple
    generate(980, iterations=150) # volcano
    generate(251, iterations=150) # dalmatian
    generate(278, iterations=150) # kit fox
    generate(9, iterations=150) # ostrich
    generate(776, iterations=150) # sax
