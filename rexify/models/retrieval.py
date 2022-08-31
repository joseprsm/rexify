import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.base import BaseRecommender
from rexify.models.candidate import CandidateModel
from rexify.models.query import QueryModel


class RetrievalModel(BaseRecommender):
    """The main Recommendation model responsible for generating query and candidate embeddings.

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

    Args:
        user_id (str): the user ID feature name
        user_dims (int): number possible values for the user ID feature
        item_id (str): the item ID feature name
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        feature_layers (list): number of neurons in each layer for the feature models
        output_layers (list): number of neurons in each layer for the output models

    Attributes:
         query_model (rexify.models.tower.TowerModel): the Query Tower model
         candidate_model (rexify.models.tower.TowerModel): the Candidate Tower model

    Examples:

        >>> from rexify.models import RetrievalModel
        >>> model = RetrievalModel('user_id', 15, 'item_id', 15)
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
    ):
        super().__init__(
            user_id=user_id,
            user_dims=user_dims,
            item_id=item_id,
            item_dims=item_dims,
            embedding_dim=embedding_dim,
            output_layers=output_layers or [64, 32],
        )

        self._feature_layers = feature_layers or [64, 32, 16]

        self.query_model = QueryModel(
            self._user_id,
            self._user_dims,
            embedding_dim=self._embedding_dim,
            output_layers=self._output_layers,
            feature_layers=self._feature_layers,
        )

        self.candidate_model = CandidateModel(
            self._item_id,
            self._item_dims,
            embedding_dim=self._embedding_dim,
            output_layers=self._output_layers,
            feature_layers=self._feature_layers,
        )

        self.task = tfrs.tasks.Retrieval()

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs["query"])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs["candidate"])
        return query_embeddings, candidate_embeddings

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)
        return loss

    def get_config(self):
        config = super().get_config()
        config["feature_layers"] = self._feature_layers
        return config
