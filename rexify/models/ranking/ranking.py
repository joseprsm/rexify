from abc import ABC

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import DenseSetterMixin


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

        # todo: validate ranking weights
        self._ranking_weights = weights or {
            feature: 1.0 for feature in self._ranking_features
        }
        self._ranking_models = {
            feature: self._get_ranking_model() for feature in self._ranking_features
        }
        self._ranking_tasks = {
            feature: tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())
            for feature in self._ranking_features
        }

    def get_loss(
        self,
        query_embeddings: tf.Tensor,
        candidate_embeddings: tf.Tensor,
        ranks: dict[str, tf.Tensor],
    ):
        loss = 0
        inputs = tf.concat([query_embeddings, candidate_embeddings], axis=1)
        for feature, model in self._ranking_models.items():
            rating_preds = self._call_layers(model, inputs)
            loss += (
                self._ranking_tasks[feature](
                    labels=ranks[feature], predictions=rating_preds
                )
                * self._ranking_weights[feature]
            )
        return loss

    def _get_ranking_model(self) -> list[tf.keras.layers.Layer]:
        model = self._set_dense_layers(self._ranking_layers)
        model.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        return model
