from typing import Optional, List

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.query import QueryModel
from rexify.models.candidate import CandidateModel


class Recommender(tfrs.Model):
    def __init__(
        self,
        n_unique_items: int,
        n_unique_users: int,
        user_features: List[str],
        item_features: List[str],
        layer_sizes: Optional[List[int]] = None,
    ):
        super(Recommender, self).__init__()
        layer_sizes = layer_sizes if layer_sizes else [64, 32]
        self._user_features = user_features
        self._item_features = item_features

        self._layer_sizes = layer_sizes

        self.candidate_model = CandidateModel(
            n_unique_items, self._item_features[0], layer_sizes=layer_sizes
        )
        self.query_model = QueryModel(
            n_unique_users, self._user_features[0], layer_sizes=layer_sizes
        )

        self.task: tfrs.tasks.Task = tfrs.tasks.Retrieval()

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)
        return loss

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs["query"])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs["candidate"])
        return query_embeddings, candidate_embeddings

    def get_config(self):
        return {
            "query_params": self._query_params,
            "candidate_params": self._candidate_params,
            "layer_sizes": self._layer_sizes,
        }
