from abc import ABC
from typing import Any

import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingMixin(tfrs.Model, ABC):
    def __init__(
        self,
        ranking_features: list[str] = None,
        ranking_layers: list[int] = None,
        ranking_weights: dict[str, float] = None,
    ):
        super().__init__()
        self._ranking_features = ranking_features
        self._ranking_layers = ranking_layers or [64, 32]
        self._ranking_weights = ranking_weights

        self.rating_models = self._get_ranking_models()
        self.ranking_tasks = self._get_ranking_tasks()

    def get_ranking_loss(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        event_types: tf.Tensor,
        ratings: tf.Tensor,
    ):
        loss = 0
        event_types = tf.reshape(event_types, (-1, 1))
        inputs = tf.concat([query_embeddings, candidate_embeddings], axis=1)

        for event_type in self._ranking_features:
            features, labels = self._filter(inputs, ratings, event_types, event_type)
            predictions = self.rating_models[event_type](features)
            loss += 1 * self.ranking_tasks[event_type](
                labels=labels, predictions=predictions
            )

        return loss

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

    def _get_ranking_models(self) -> dict[str, tf.keras.Model] | None:
        ranking_models = None
        if self._ranking_features:
            ranking_models = {
                ranking_feature: self._get_rating_model(self._ranking_layers)
                for ranking_feature in self._ranking_features
            }
        return ranking_models

    def _get_ranking_tasks(self):
        ranking_tasks = None
        if self._ranking_features:
            ranking_tasks = {
                ranking_feature: tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )
                for ranking_feature in self._ranking_features
            }
        return ranking_tasks

    @staticmethod
    def _get_rating_model(layer_sizes) -> tf.keras.Model:
        layers = [tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes]
        if layer_sizes[-1] != 1:
            layers.append(tf.keras.layers.Dense(1))
        return tf.keras.Sequential(layers)
