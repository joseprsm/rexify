import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.candidate import CandidateModel
from rexify.models.query import QueryModel


class Recommender(tfrs.Model):
    """

    Args:
        user_id (str): the user ID feature
        user_dims (int): number possible values for the user ID feature
        item_id (str): the item ID feature
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        feature_layers (list): number of neurons in each layer for the feature models
        output_layers (list): number of neurons in each layer for the output models

    Attributes:
         query_model (rexify.models.tower.TowerModel): the Query Tower model
         candidate_model (rexify.models.tower.TowerModel): the Candidate Tower model
         task (tfrs.tasks.Task): the task of

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
    ):
        super().__init__()

        self._user_id = user_id
        self._user_dims = user_dims

        self._item_id = item_id
        self._item_dims = item_dims

        self._embedding_dim = embedding_dim
        self._feature_layers = feature_layers or [64, 32, 16]
        self._output_layers = output_layers or [64, 32]

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
        self.task: tfrs.tasks.Task = tfrs.tasks.Retrieval()

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)
        return loss

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs["query"])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs["candidate"])
        return query_embeddings, candidate_embeddings

    def get_config(self):
        return {
            "item_dims": self._item_dims,
            "user_dims": self._user_dims,
            "user_id": self._user_id,
            "item_id": self._item_id,
            "output_layers": self._output_layers,
            "feature_layers": self._feature_layers,
        }
