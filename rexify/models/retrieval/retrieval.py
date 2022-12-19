from abc import ABC

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.retrieval.candidate import CandidateModel
from rexify.models.retrieval.query import QueryModel


class RetrievalMixin(tfrs.Model, ABC):
    def __init__(
        self,
        user_dims: int,
        item_dims: int,
        user_embeddings: pd.DataFrame,
        item_embeddings: pd.DataFrame,
        embedding_dim: int = 32,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
        **kwargs
    ):
        super().__init__()
        self._user_dims = user_dims
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
            self._user_dims,
            self._item_dims,
            identifiers=user_embeddings.index.values.astype(int),
            feature_embeddings=user_embeddings.values.astype(float),
            **joint_args
        )

        self.candidate_model = CandidateModel(
            self._item_dims,
            identifiers=item_embeddings.index.values.astype(int),
            feature_embeddings=item_embeddings.values.astype(float),
            **joint_args
        )

        self.retrieval_task = tfrs.tasks.Retrieval()

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs["query"])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs["candidate"])
        return query_embeddings, candidate_embeddings

    def get_loss(self, *embeddings):
        return self.retrieval_task(*embeddings)
