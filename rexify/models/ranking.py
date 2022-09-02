from abc import ABC

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
        inputs: tf.Tensor,
    ):
        loss = 0

        # this method is never called when self._ranking_features is None
        for i, feature in enumerate(self._ranking_features):
            # assumes data follows the same indexing as in the schema
            labels = inputs[i]
            rating_model = self.rating_models[feature]

            predictions = rating_model(
                tf.concat([query_embeddings, candidate_embeddings], axis=1)
            )

            loss += self._ranking_weights[i] * self.ranking_tasks[feature](
                labels=labels, predictions=predictions
            )
        return loss

    @staticmethod
    def _get_rating_model(layer_sizes) -> tf.keras.Model:
        layers = [tf.keras.layers.Dense(num_neurons) for num_neurons in layer_sizes]
        if layer_sizes[-1] != 1:
            layers.append(tf.keras.layers.Dense(1))
        return tf.keras.Sequential(layers)
