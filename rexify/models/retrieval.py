from abc import ABC

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.candidate import CandidateModel
from rexify.models.query import QueryModel


class RetrievalMixin(tfrs.Model, ABC):
    def __init__(
        self,
        user_id: str,
        user_dims: int,
        item_id: str,
        item_dims: int,
        embedding_dim: int = 32,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
        **kwargs
    ):
        super().__init__()
        self._user_id = user_id
        self._user_dims = user_dims

        self._item_id = item_id
        self._item_dims = item_dims

        self._embedding_dim = embedding_dim
        self._output_layers = output_layers or [64, 32]
        self._feature_layers = feature_layers or [64, 32, 16]
        joint_args = {
            "embedding_dim": self._embedding_dim,
            "output_layers": self._output_layers,
            "feature_layers": self._feature_layers,
        }

        self.query_model = QueryModel(
            self._user_id, self._user_dims, self._item_dims, **joint_args
        )

        self.candidate_model = CandidateModel(
            self._item_id, self._item_dims, **joint_args
        )

        self.retrieval_task = tfrs.tasks.Retrieval()

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs["query"])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs["candidate"])
        return query_embeddings, candidate_embeddings
