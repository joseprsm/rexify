import tensorflow as tf
import tensorflow_recommenders as tfrs


class _BaseIndex:
    def __init__(self, query_model: tf.keras.Model, window_size: int):
        self.query_model = query_model
        self._window_size = window_size

    def call(self, queries: tf.Tensor, k: int = None):
        queries_shape = queries.shape[0] or 1
        inputs = (
            {
                "user_id": queries,
                "history": tf.zeros(
                    shape=(queries_shape, self._window_size), dtype=tf.int32
                ),
            }
            if self.query_model.name.startswith("query")
            else {"item_id": queries}
        )
        return self.__class__.__bases__[1].call(self, inputs, k)


class BruteForce(_BaseIndex, tfrs.layers.factorized_top_k.BruteForce):
    def __init__(
        self,
        query_model: tf.keras.Model,
        window_size: int,
        k: int = 2,
        name: str = None,
    ):
        tfrs.layers.factorized_top_k.BruteForce.__init__(self, query_model, k, name)
        _BaseIndex.__init__(self, query_model, window_size)


class ScaNN(_BaseIndex, tfrs.layers.factorized_top_k.ScaNN):
    def __init__(
        self,
        query_model: tf.keras.Model,
        window_size: int,
        k: int = 10,
        distance_measure: str = "dot_product",
        num_leaves: int = 100,
        num_leaves_to_search: int = 10,
        training_iterations: int = 12,
        dimensions_per_block: int = 2,
        num_reordering_candidates: int = None,
        parallelize_batch_searches: bool = True,
        name: str = None,
    ):
        tfrs.layers.factorized_top_k.ScaNN.__init__(
            self,
            query_model,
            k,
            distance_measure,
            num_leaves,
            num_leaves_to_search,
            training_iterations,
            dimensions_per_block,
            num_reordering_candidates,
            parallelize_batch_searches,
            name,
        )
        _BaseIndex.__init__(self, query_model, window_size)
