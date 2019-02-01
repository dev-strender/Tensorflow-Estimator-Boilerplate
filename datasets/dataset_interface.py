from abc import *


class IDataset(metaclass=ABCMeta):
    def __init__(self, flags_obj):
        self.FLAGS = flags_obj

    @abstractmethod
    def parse_fn(self, serialized_example):
        """parse tfrecord files. you may should know how the data was generated to get exact feature"""
        pass

    @abstractmethod
    def get_train_set(self):
        """
        get train set using tf.data.dataset
        Be aware that when using this function in supervisor class, you should use lambda call
        to fit dataset into the same graph
        """
        pass

    @abstractmethod
    def get_validation_set(self):
        """get validation set using tf.data.dataset
        Be aware that when using this function in supervisor class, you should use lambda call
        to fit dataset into the same graph
        """
        pass
