from abc import abstractmethod

import tensorflow as tf

from rexify.models.index import BruteForce, ScaNN


class _IndexCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, sample_query: dict[str, tf.Tensor], batch_size: int = 128, **index_args
    ):
        super().__init__()
        self._batch_size = batch_size
        self._sample_query = sample_query
        self._index_args = index_args

    @abstractmethod
    def set(self) -> tf.keras.Model:
        pass

    def on_train_end(self, logs=None):
        index = self.set()
        index.index_from_dataset(candidates=self._get_dataset())
        _ = index(self._sample_query["user_id"])
        self.model.index = index

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
    def set(self) -> tf.keras.Model:
        return BruteForce(
            self.model.query_model, self.model.window_size, **self._index_args
        )


class ScaNNCallback(_IndexCallback):
    def set(self) -> tf.keras.Model:
        return ScaNN(self.model.query_model, self.model.window_size, **self._index_args)
