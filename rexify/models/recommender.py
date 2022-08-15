from typing import List, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.candidate import CandidateModel
from rexify.models.query import QueryModel


class Recommender(tfrs.Model):
    def __init__(
        self,
        n_unique_items: int,
        n_unique_users: int,
        user_id: str,
        item_id: str,
        layer_sizes: Optional[List[int]] = None,
    ):
        super(Recommender, self).__init__()
        layer_sizes = layer_sizes if layer_sizes else [64, 32]

        self._n_unique_items = n_unique_items
        self._n_unique_users = n_unique_users

        self._user_id = user_id
        self._item_id = item_id

        self._layer_sizes = layer_sizes

        self.candidate_model = CandidateModel(
            n_unique_items, self._item_id, layer_sizes=layer_sizes
        )
        self.query_model = QueryModel(
            n_unique_users, self._user_id, layer_sizes=layer_sizes
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
            "n_unique_items": self._n_unique_items,
            "n_unique_users": self._n_unique_users,
            "user_id": self._user_id,
            "item_id": self._item_id,
            "layer_sizes": self._layer_sizes,
        }
