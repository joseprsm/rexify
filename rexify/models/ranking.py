from abc import ABC

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import DenseSetterMixin


class RankingMixin(tfrs.Model, DenseSetterMixin, ABC):
    def __init__(
        self,
        n_dims: int = 1,
        rating_features: list[str] = None,
        layer_sizes: list[int] = None,
        weights: dict[str, float] = None,
    ):
        super().__init__()
        self._n_dims = n_dims
        self._rating_features = rating_features or []
        self._layer_sizes = layer_sizes or [64, 32]
        self._weights = weights

        self.event_model = self._set_dense_layers(self._layer_sizes)
        self.event_model.add(tf.keras.layers.Dense(self._n_dims, activation="softmax"))
        self.event_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.CategoricalCrossentropy()
        )

        if len(self._rating_features) > 0:
            self.rating_models = self._get_rating_models()
            self.rating_tasks = self._get_ranking_tasks()

    def get_loss(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        events: tf.Tensor,
    ):
        inputs = tf.concat([query_embeddings, candidate_embeddings], axis=1)
        x = self.event_model(inputs)
        loss = self.event_task(labels=events, predictions=x)
        return loss

    def _get_rating_models(self) -> dict[str, tf.keras.Model] | None:
        rating_models = None
        if self._rating_features:
            rating_models = {
                feature: self._get_rating_model(self._layer_sizes)
                for feature in self._rating_features
            }
        return rating_models

    def _get_ranking_tasks(self):
        ranking_tasks = None
        if self._rating_features:
            ranking_tasks = {
                ranking_feature: tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )
                for ranking_feature in self._rating_features
            }
        return ranking_tasks

    def _get_rating_model(self, layer_sizes) -> tf.keras.Model:
        model = self._set_dense_layers(layer_sizes)
        if layer_sizes[-1] != 1:
            model.add(tf.keras.layers.Dense(1))
        return model
