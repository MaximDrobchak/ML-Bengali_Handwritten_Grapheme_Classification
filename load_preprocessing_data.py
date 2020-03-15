
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

from constamts import TRAINING_FILENAMES, VALIDATION_FILENAMES, BATCH_SIZE, VALIDATION_BATCH_SIZE, SHAPE, AUTO

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

    grapheme_root = example['grapheme_root']
    vowel_diacritic = example['vowel_diacritic']
    consonant_diacritic = example['consonant_diacritic']

    head_root_hot = tf.sparse.to_dense(example['head_root_hot'])
    head_vowel_hot = tf.sparse.to_dense(example['head_vowel_hot'])
    head_consonant_hot = tf.sparse.to_dense(example['head_consonant_hot'])

    head_root_hot = tf.reshape(head_root_hot, [168])
    head_vowel_hot = tf.reshape(head_vowel_hot, [11])
    head_consonant_hot = tf.reshape(head_consonant_hot, [7])

    label  = example['label']
    height = example['size'][0]
    width  = example['size'][1]
    return image,  {"head_root": head_root_hot, "head_vowel": head_vowel_hot, "head_consonant": head_consonant_hot}

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
    dataset = dataset.map(augmentation, num_parallel_calls=AUTO)
    dataset = dataset.map(normalize, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(8036)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES)
    dataset = dataset.map(normalize, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset)
    dataset = dataset.batch(VALIDATION_BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset