from abc import abstractmethod

import tensorflow as tf
import tensorflow_recommenders as tfrs


class BaseRecommender(tfrs.Model):

    query_model: tf.keras.Model
    candidate_model: tf.keras.Model
    task: tfrs.tasks.Task

    def __init__(
        self,
        user_id: str,
        user_dims: int,
        item_id: str,
        item_dims: int,
        embedding_dim: int,
        output_layers: list[int],
    ):
        super().__init__()
        self._user_id = user_id
        self._user_dims = user_dims

        self._item_id = item_id
        self._item_dims = item_dims

        self._embedding_dim = embedding_dim
        self._output_layers = output_layers

    @abstractmethod
    def call(self, inputs):
        pass

    @abstractmethod
    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        pass

    def get_config(self):
        return {
            "item_dims": self._item_dims,
            "user_dims": self._user_dims,
            "user_id": self._user_id,
            "item_id": self._item_id,
            "output_layers": self._output_layers,
        }
