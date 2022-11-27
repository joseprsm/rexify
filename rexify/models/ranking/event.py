import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.ranking.base import BaseRankingModel


class EventModel(BaseRankingModel):
    def __init__(self, layer_sizes: list[int] = None, n_dims: int = 1):
        super().__init__(layer_sizes=layer_sizes)
        self._n_dims = n_dims
        self.output_layer = tf.keras.layers.Dense(self._n_dims, activation="softmax")
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.CategoricalCrossentropy())

    def get_config(self):
        return {
            "layer_sizes": self._layer_sizes,
            "n_dims": self._n_dims,
        }
