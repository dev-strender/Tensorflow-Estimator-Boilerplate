import os
import tensorflow as tf

from .dataset_interface import IDataset


class CIFAR10(IDataset):
    def __init__(self, flags_obj):
        super(CIFAR10, self).__init__(flags_obj)
        self.HEIGHT = 32
        self.WIDTH = 32
        self.CHANNELS = 3

    def parse_fn(self, serialized_example):
        """overridden from base class"""
        features = tf.parse_example(
            [serialized_example],
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        )

        # decode byte image to [0, 255]
        image = tf.decode_raw(features['image'], tf.uint8)

        # byte string is just string. you should set the shape of image
        image.set_shape((1, self.HEIGHT * self.WIDTH * self.CHANNELS))

        # Reshape to original[c, h, w] format, and then convert to [h, w, c] format
        image = tf.reshape(image, [self.CHANNELS, self.HEIGHT, self.WIDTH])
        image = tf.cast(
            tf.transpose(image, [1, 2, 0]),
            tf.float32
        )

        # cast label to int
        label = tf.cast(features['label'], tf.int32)

        return image, label

    @staticmethod
    def augmentation(image, label):
        """augmentation function to image goes here.
        I did not applied any augmentation from now"""
        return image, label

    @staticmethod
    def normalization(image, label):
        """convert image from range [0, 255] -> [-0.5, 0.5] floats"""
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        return image, label

    def get_train_set(self):
        """overridden from base class"""
        train_filename = 'train.tfrecords'
        dataset = tf.data.TFRecordDataset(os.path.join(self.FLAGS.data_dir, train_filename))
        dataset = dataset.map(self.parse_fn)
        dataset = dataset.map(self.normalization)
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=50000).batch(self.FLAGS.batch_size)
        dataset = dataset.repeat(self.FLAGS.epochs_between_evals)

        return dataset

    def get_validation_set(self):
        """overridden from base class"""
        valid_filename = 'validation.tfrecords'
        dataset = tf.data.TFRecordDataset(os.path.join(self.FLAGS.data_dir, valid_filename))
        dataset = dataset.map(self.parse_fn)
        dataset = dataset.map(self.normalization)
        dataset = dataset.batch(self.FLAGS.batch_size)

        iterator = dataset.make_one_shot_iterator().get_next()

        return iterator

