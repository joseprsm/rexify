from abc import ABC
from typing import Any

import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingMixin(tfrs.Model, ABC):
    def __init__(
        self,
        ranking_features: list[str] = None,
        ranking_layers: list[int] = None,
        ranking_weights: list[float] = None,
        **kwargs
    ):
        super().__init__()
        self._ranking_features = ranking_features
        self._ranking_layers = ranking_layers or [64, 32]
        self._ranking_weights = ranking_weights

        if self._ranking_features:
            self._ranking_weights = self._ranking_weights or [1] * len(
                self._ranking_features
            )

            self.rating_models = {
                ranking_feature: self._get_rating_model(self._ranking_layers)
                for ranking_feature in ranking_features
            }

            self.ranking_tasks = {
                ranking_feature: tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )
                for ranking_feature in ranking_features
            }

    def get_ranking_loss(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        event_types: tf.Tensor,
        ratings: tf.Tensor,
    ):
        loss = 0
        inputs = tf.concat(
            [query_embeddings, candidate_embeddings, event_types], axis=1
        )

        # this method is never called when self._ranking_features is None
        for event_type in self._ranking_features:
            features, labels = self._filter(inputs, ratings, event_types, event_type)
            predictions = self.rating_models[event_type](features)
            loss += self._ranking_weights[event_type] * self.ranking_tasks[event_type](
                labels=labels, predictions=predictions
            )
        return loss

    @staticmethod
    def _get_rating_model(layer_sizes) -> tf.keras.Model:
        layers = [tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes]
        if layer_sizes[-1] != 1:
            layers.append(tf.keras.layers.Dense(1))
        return tf.keras.Sequential(layers)

    @staticmethod
    def _filter(
        features: tf.Tensor,
        labels: tf.Tensor,
        condition_tensor: tf.Tensor,
        value: str | int | Any,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        condition = tf.reduce_all(tf.equal(condition_tensor, value), axis=1)
        indices = tf.where(condition)
        features = tf.gather_nd(features, indices)
        labels = tf.gather_nd(labels, indices)
        return features, labels
