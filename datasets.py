import os
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
import tensorflow as tf
# mean= [0.1307]
# std = [0.3081]
# x -= mean
# x /= std
def load_mnist():
    # the data, shuffled and split between train and test sets

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # x = x.reshape((x.shape[0], -1))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    # x -= mean
    # x /= std
    # print('MNIST samples', x.shape)
    return x, y
# 加载卷积数据集（数据个数，高，宽，通道）

def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    # elif dataset == 'fmnist':
    #     return load_fashion_mnist()
    # elif dataset == 'usps':
    #     return load_usps()
    else:
        raise ValueError('Not defined for loading %s' % dataset)

# 加载二维数据集（数据个数，特征）
def load_data(dataset):
    x, y = load_data_conv(dataset)
    print(x.shape)
    Image = custom_augment(x)
    print(type(Image))
    return x.reshape([x.shape[0], -1]), y

def random_apply(func, x, p):
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)

def custom_augment(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))

    # Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    # Random translations
    # image = random_apply(translate, image, p=0.5)
    # Randomly apply gausian blur
    image = random_apply(gaussian_blur, image, p=0.5)
    # Randomly apply transformation (color distortions)
    # image = random_apply(color_jitter, image, p=0.8)
    # Randomly apply grayscale
    # image = random_apply(color_drop, image, p=0.2)

    return image

def translate(image):
    (h, w) = tf.shape(image)[0], tf.shape(image)[1]
    image = tf.image.random_flip_left_right(image)

    f = tf.random.uniform([], minval=0, maxval=0.125, dtype=tf.float32)
    (dh, dw) = tf.cast(tf.cast(h, tf.float32) * f, tf.float32), \
               tf.cast(tf.cast(w, tf.float32) * f, tf.float32)

    return image

def gaussian_blur(image, kernel_size=23, padding='SAME'):
    sigma = tf.random.uniform((1,)) * 1.9 + 0.1

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    x = tf.clip_by_value(x, 0, 1)
    return x

def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x