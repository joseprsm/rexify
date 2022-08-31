import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import BaseRecommender


class RankingModel(BaseRecommender):
    """

    Args:
        user_id (str): the user ID feature name
        user_dims (int): number possible values for the user ID feature
        item_id (str): the item ID feature name
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        output_layers (list): number of neurons in each layer for the output models
    """

    def __init__(
        self,
        user_id: str,
        user_dims: int,
        item_id: str,
        item_dims: int,
        embedding_dim: int = 32,
        output_layers: list[int] = None,
    ):
        super().__init__(
            user_id=user_id,
            user_dims=user_dims,
            item_id=item_id,
            item_dims=item_dims,
            embedding_dim=embedding_dim,
            output_layers=output_layers or [64, 32],
        )

        self.query_model = tf.keras.layers.Embedding(
            self._user_dims, self._embedding_dim
        )
        self.candidate_model = tf.keras.layers.Embedding(
            self._item_dims, self._embedding_dim
        )

        self.rating_model = self._get_rating_model(self._layer_sizes)
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, inputs, *_):
        user_embedding = self.query_model(inputs[self._user_id])
        item_embedding = self.candidate_model(inputs[self._item_id])
        x = tf.concat([user_embedding, item_embedding], axis=1)
        rating = self.rating_model(x)
        return rating

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        labels = inputs.pop("rating")
        rating_predictions = self(inputs)
        return self.task(labels=labels, predictions=rating_predictions)

    @staticmethod
    def _get_rating_model(layer_sizes: list[int]) -> tf.keras.Model:
        layers = [
            tf.keras.layers.Dense(num_neurons, activation="relu")
            for num_neurons in layer_sizes[:-1]
        ]
        layers.append(tf.keras.layers.Dense(1))
        return tf.keras.Sequential(layers)
