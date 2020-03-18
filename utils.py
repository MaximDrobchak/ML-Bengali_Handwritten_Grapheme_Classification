
import random
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

from constants import IMAGE_SIZE, SHAPE, AUTO

def resize_and_crop_image(image, label):
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = IMAGE_SIZE[1]
    th = IMAGE_SIZE[0]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                    lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                   )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

def normalize(image, label):
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)
    return image, label

def crooping(image):
    crop_w = int(IMAGE_SIZE[0]*0.95)
    crop_h = int(IMAGE_SIZE[1]*0.95)
    crop_or_pad_s = int(IMAGE_SIZE[0]*0.1) + IMAGE_SIZE[0]
    crop_or_pad_p = int(IMAGE_SIZE[1]*0.1) + IMAGE_SIZE[1]
    image = tf.image.resize_with_crop_or_pad(image, crop_or_pad_s, crop_or_pad_p)
    image = tf.image.random_crop(image, [crop_w, crop_h, 3])
    return image

def augmentation(image, label):
    if random.randrange(2) % 2 == 0:
        image = crooping(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=random.randrange(4))
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_brightness(image, 0.5)
    return image, label

def force_image_sizes(dataset):
    reshape_images = lambda image, label: (tf.reshape(image, SHAPE), label)
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)
    return dataset

def display_9_images_from_dataset(dataset):
  plt.figure(figsize=(13,13))
  subplot=331
  for i, (image, _) in enumerate(dataset):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image.numpy().astype(np.uint8))
    subplot += 1
    if i==8:
      break
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()