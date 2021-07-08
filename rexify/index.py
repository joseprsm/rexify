import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.lookup import EmbeddingLookup


class ScaNN(tfrs.layers.factorized_top_k.ScaNN):

    def __init__(
            self,
            lookup_model: EmbeddingLookup,
            candidates: tf.data.Dataset,
            embeddings: tf.data.Dataset,
            sample_query,
            **kwargs
    ):
        super().__init__(lookup_model, **kwargs)
        self.candidates = candidates
        self.embeddings = embeddings
        self.index(candidates=embeddings, identifiers=candidates)
        _ = self([sample_query])

    def get_config(self):
        return {
            "lookup_model": self.query_model,
            "candidates": self.candidates,
            "embeddings": self.embeddings
        }
