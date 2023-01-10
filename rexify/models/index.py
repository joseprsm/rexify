import tensorflow as tf
import tensorflow_recommenders as tfrs


class BruteForceIndex(tfrs.layers.factorized_top_k.BruteForce):
    def __init__(
        self,
        query_model: tf.keras.Model,
        window_size,
        k: int = 10,
        name: str = None,
    ):
        super().__init__(query_model, k, name)
        self._window_size = window_size

    def call(self, queries: tf.Tensor, k: int = None):
        queries_shape = queries.shape[0] or 1
        inputs = {
            "user_id": queries,
            "history": tf.zeros(
                shape=(queries_shape, self._window_size), dtype=tf.int32
            ),
        }
        return tfrs.layers.factorized_top_k.BruteForce.call(self, inputs, k)
