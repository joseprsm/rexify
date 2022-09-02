import tensorflow as tf

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
        user_id (str): the user ID feature name
        user_dims (int): number possible values for the user ID feature
        item_id (str): the item ID feature name
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        feature_layers (list): number of neurons in each layer for the feature models
        output_layers (list): number of neurons in each layer for the output models

    Examples:
        >>> from rexify.models import Recommender
        >>> model = Recommender('user_id', 15, 'item_id', 15)
        >>> model.compile()

        >>> import numpy as np
        >>> inputs = tf.data.Dataset.from_tensor_slices(np.concatenate([np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 1, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 5, size=100).reshape(-1, 1),], axis=1)).map(lambda x: {'query': {'user_id': x[0], 'user_features': x[1:3], 'context_features': x[3:4]}, 'candidate': {'item_id': x[4], 'item_features': x[5:]}}).batch(128)

        >>> _ = model.fit(inputs, verbose=0)

    """

    def __init__(
        self,
        user_id: str,
        user_dims: int,
        item_id: str,
        item_dims: int,
        embedding_dim: int = 32,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
        ranking_features: list[str] = None,
        ranking_layers: list[int] = None,
        ranking_weights: list[float] = None,
    ):
        RetrievalMixin.__init__(
            self,
            user_id=user_id,
            user_dims=user_dims,
            item_id=item_id,
            item_dims=item_dims,
            embedding_dim=embedding_dim,
            feature_layers=feature_layers,
            output_layers=output_layers,
        )

        RankingMixin.__init__(
            self,
            ranking_features=ranking_features,
            ranking_layers=ranking_layers,
            ranking_weights=ranking_weights,
        )

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        embeddings = self(inputs)  # Recommender inherits RetrievalMixin's call method
        loss = self.get_retrieval_loss(*embeddings)
        if self._ranking_features:
            loss += self.get_ranking_loss(*embeddings, inputs["rank"])
        return loss

    def get_config(self):
        return {
            "item_dims": self._item_dims,
            "user_dims": self._user_dims,
            "user_id": self._user_id,
            "item_id": self._item_id,
            "output_layers": self._output_layers,
            "feature_layers": self._feature_layers,
            "ranking_layers": self._ranking_layers,
            "ranking_features": self._ranking_features,
            "ranking_weights": self._ranking_weights,
        }
