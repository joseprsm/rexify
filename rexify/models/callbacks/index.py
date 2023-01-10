from abc import abstractmethod

import tensorflow as tf

from rexify.models.index import BruteForceIndex


class _IndexCallback(tf.keras.callbacks.Callback):

    index: tf.keras.Model

    def __init__(self, sample_query: dict[str, tf.Tensor], batch_size: int = 128):
        super().__init__()
        self._batch_size = batch_size
        self._sample_query = sample_query

    @abstractmethod
    def on_train_end(self):
        pass

    def _get_dataset(self):
        def zip_item_dataset(item):
            return (item["item_id"], self.model.candidate_model(item))

        candidates = self._get_candidates().batch(self._batch_size)
        return candidates.map(zip_item_dataset)

    def _get_candidates(self):
        def header_fn(item_id):
            return {"item_id": tf.cast(item_id, tf.int32)}

        return tf.data.Dataset.from_tensor_slices(
            self.model.candidate_model.identifiers
        ).map(header_fn)


class BruteForceCallback(_IndexCallback):
    def on_train_end(self, logs=None):
        brute_force = BruteForceIndex(
            self.model.query_model, window_size=self.model.window_size
        )
        brute_force.index_from_dataset(candidates=self._get_dataset())
        _ = brute_force(self._sample_query["user_id"])
        self.model.index = brute_force
