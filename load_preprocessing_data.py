
import random
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

from constants import TRAINING_FILENAMES, VALIDATION_FILENAMES, BATCH_SIZE, VALIDATION_BATCH_SIZE, SHAPE, AUTO

from utils import resize_and_crop_image, augmentation, normalize, force_image_sizes

def read_tfrecord(example):
    features = {
      "image": tf.io.FixedLenFeature([], tf.string),
      "grapheme_root": tf.io.FixedLenFeature([], tf.int64),
      "vowel_diacritic": tf.io.FixedLenFeature([], tf.int64),
      "consonant_diacritic": tf.io.FixedLenFeature([], tf.int64),

      "label":         tf.io.FixedLenFeature([], tf.string),
      "size":          tf.io.FixedLenFeature([2], tf.int64),
      "head_root_hot": tf.io.VarLenFeature(tf.float32),
      "head_vowel_hot": tf.io.VarLenFeature(tf.float32),
      "head_consonant_hot": tf.io.VarLenFeature(tf.float32),
    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_image(example['image'], channels=3)

    head_root_hot = tf.sparse.to_dense(example['head_root_hot'])
    head_vowel_hot = tf.sparse.to_dense(example['head_vowel_hot'])
    head_consonant_hot = tf.sparse.to_dense(example['head_consonant_hot'])

    head_root_hot = tf.reshape(head_root_hot, [168])
    head_vowel_hot = tf.reshape(head_vowel_hot, [11])
    head_consonant_hot = tf.reshape(head_consonant_hot, [7])

    return image,  {"head_root": head_root_hot, "head_vowel": head_vowel_hot, "head_consonant": head_consonant_hot}


def read_test_tfrecord(example):
    TEST_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, TEST_TFREC_FORMAT)
    image = tf.image.decode_image(example['image'], channels=3)
    image_model = tf.cast(image, tf.float32)/255.0
    image_model = tf.reshape(image_model, SHAPE)
    head_root_hot_classes =  [x for x in range(168)]
    head_vowel_hot_classes =  [x  for x in range(11)]
    head_consonant_hot_classes = [x  for x in range(7)]
    label = example['label']
    return image_model, label

option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False



def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
#     dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)
    return dataset


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES)
    if random.randrange(3) % 3 != 0:
        dataset = dataset.map(augmentation, num_parallel_calls=AUTO)
        dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)
    dataset = dataset.map(normalize, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    dataset = dataset.shuffle(8036)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES)
    dataset = dataset.map(normalize, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset)
    dataset = dataset.batch(VALIDATION_BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_load_test_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_test_tfrecord, num_parallel_calls=AUTO)
    return dataset
