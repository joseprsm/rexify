from typing import List

import abc

import tensorflow as tf


class BaseModel(tf.keras.Model, abc.ABC):

    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.layer_sizes = layer_sizes

    def get_config(self):
        return {
            'layer_sizes': self.layer_sizes
        }

