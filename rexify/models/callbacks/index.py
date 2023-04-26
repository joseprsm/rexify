import tensorflow as tf

from rexify.models.index import BruteForce, ScaNN


class _IndexCallback(tf.keras.callbacks.Callback):

    INDEX: BruteForce | ScaNN

    def __init__(
        self,
        sample_query: dict[str, tf.Tensor],
        query_model: str = "query_model",
        batch_size: int = 128,
        **index_args,
    ):
        super().__init__()
        self._query_model = query_model
        self._batch_size = batch_size
        self._sample_query = sample_query
        self._index_args = index_args
        self._target = "user" if self._query_model == "query_model" else "item"

    def set(self) -> tf.keras.Model:
        query_model = getattr(self.model, self._query_model)
        return self.INDEX(query_model, self.model.window_size, **self._index_args)

    def on_train_end(self, logs=None):
        index = self.set()
        index.index_from_dataset(candidates=self._get_candidates_dataset())
        _ = index(self._sample_query[f"{self._target}_id"])
        setattr(self.model, f"{self._target}_index", index)

    def _get_candidates_dataset(self):
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

    INDEX = BruteForce


class ScaNNCallback(_IndexCallback):

    INDEX = ScaNN
