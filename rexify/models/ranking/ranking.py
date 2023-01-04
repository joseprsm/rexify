from abc import ABC

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import DenseSetterMixin
from rexify.models.ranking.event import EventModel


class RankingMixin(tfrs.Model, DenseSetterMixin, ABC):
    def __init__(
        self,
        ranking_features: list[str] = None,
        layer_sizes: list[int] = None,
        weights: dict[str, float] = None,
    ):
        super().__init__()
        self._ranking_features = ranking_features or []
        self._ranking_layers = layer_sizes or [64, 32]

        self.event_model = EventModel(self._rating_layers, n_dims=self._ranking_dims)

    def get_loss(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        events: tf.Tensor,
    ):
        inputs = tf.concat([query_embeddings, candidate_embeddings], axis=1)
        loss = self.event_model(inputs, events)
        return loss
