
import tensorflow as tf
from tensorflow.keras import backend as K

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


TEST_DS_PATH =  '/data/test_tfrecords'
GCS_DS_PATH = '/data/train_tfrecords'

if strategy.num_replicas_in_sync == 1: # single GPU or CPU
    BATCH_SIZE = 256
    VALIDATION_BATCH_SIZE = 256
else: # TPU pod
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    VALIDATION_BATCH_SIZE = 8 * strategy.num_replicas_in_sync

FILENAMES = tf.io.gfile.glob(GCS_DS_PATH+'/*.tfrec')
TEST_FILENAMES  = tf.io.gfile.glob(TEST_DS_PATH+'/*.tfrec')

IMAGE_SIZE = [64, 64]

if K.image_data_format() == 'channels_first':
    SHAPE = (3,*IMAGE_SIZE)
else:
    SHAPE = (*IMAGE_SIZE, 3)

SIZE_TFRECORD = 128
split = int(len(FILENAMES)*0.81)
TRAINING_FILENAMES = FILENAMES[:split]
VALIDATION_FILENAMES = FILENAMES[split:]
STEP_PER_EPOCH = (len(TRAINING_FILENAMES)*SIZE_TFRECORD)//BATCH_SIZE
VALIDATION_STEP_PER_EPOCH = (len(VALIDATION_FILENAMES)*SIZE_TFRECORD)//VALIDATION_BATCH_SIZE
