import pandas as pd
import tensorflow as tf

from rexify.models.callbacks import BruteForceCallback
from rexify.models.ranking import RankingMixin
from rexify.models.retrieval import RetrievalMixin


class Recommender(RetrievalMixin, RankingMixin):
    """The main Recommender model.

    It expects a `tf.data.Dataset`, composed of two keys: "query" and "candidate";
    the query part of the dataset has three keys:

    * the user ID feature name, a scalar;
    * `user_features`, an array representing the user features
    * `context_features`, an array representing the context features

    The candidate part of the data set has two keys:

    * the item ID feature name, a scalar;
    * `item_features`, an array representing the item features

    The query tower model takes the user ID feature and passes it by an embedding layer. The
    user and context features are concatenated and passed by a number of dense layers. The
    item ID feature is similarly passed to an Embedding layer. Its outputs are then concatenated
    to the outputs of the features model whose inputs are the item features, and are then
    passed by a number of Dense layers.

    An optional Ranking model is also included, granted there are `ranking_features`.

    Args:
        user_dims (int): number possible values for the user ID feature
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        feature_layers (list): number of neurons in each layer for the feature models
        output_layers (list): number of neurons in each layer for the output models

    Examples:
        >>> from rexify.models import Recommender
        >>> model = Recommender()
        >>> model.compile()

        >>> import numpy as np
        >>> inputs = tf.data.Dataset.from_tensor_slices(np.concatenate([np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 1, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 5, size=100).reshape(-1, 1),], axis=1)).map(lambda x: {'query': {'user_id': x[0], 'user_features': x[1:3], 'context_features': x[3:4]}, 'candidate': {'item_id': x[4], 'item_features': x[5:]}}).batch(128)

        >>> _ = model.fit(inputs, verbose=0)

    """

    def __init__(
        self,
        user_dims: int,
        item_dims: int,
        user_embeddings: pd.DataFrame,
        item_embeddings: pd.DataFrame,
        embedding_dim: int = 32,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
        ranking_features: list[str] = None,
        ranking_layers: list[int] = None,
        ranking_weights: dict[str, float] = None,
    ):
        RetrievalMixin.__init__(
            self,
            user_dims=user_dims + 1,
            item_dims=item_dims + 1,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            embedding_dim=embedding_dim,
            feature_layers=feature_layers,
            output_layers=output_layers,
        )

        RankingMixin.__init__(
            self,
            ranking_features=ranking_features,
            layer_sizes=ranking_layers,
            weights=ranking_weights,
        )

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        embeddings = self(inputs)  # Recommender inherits RetrievalMixin's call method
        loss = RetrievalMixin.get_loss(self, *embeddings)
        loss += RankingMixin.get_loss(self, *embeddings, inputs["rank"])
        return loss

    def fit(
        self,
        x: tf.data.Dataset,
        batch_size: int = None,
        epochs: int = 1,
        callbacks: list[tf.keras.callbacks.Callback] = None,
        validation_data=None,
    ):
        # required to set index shapes
        sample_query = list(x.batch(1).take(1))[0]["query"]
        base_callbacks = [BruteForceCallback(sample_query)]
        callbacks = base_callbacks if callbacks is None else callbacks
        x = x.batch(batch_size) if batch_size else x

        return super().fit(
            x, epochs=epochs, validation_data=validation_data, callbacks=callbacks
        )

    def get_config(self):
        return {
            "item_dims": self._item_dims,
            "user_dims": self._user_dims,
            "output_layers": self._output_layers,
            "feature_layers": self._feature_layers,
            "ranking_layers": self._ranking_layers,
            "ranking_features": self._ranking_features,
            "ranking_weights": self._ranking_weights,
        }
