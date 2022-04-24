from typing import Optional, List, Dict, Any

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.query import QueryModel
from rexify.models.candidate import CandidateModel

USER_FEATURES = ["userId"]
ITEM_FEATURES = ["itemId"]


class Recommender(tfrs.Model):
    def __init__(
        self,
        query_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
        layer_sizes: Optional[List[int]] = None,
        activation: Optional[str] = "relu",
    ):
        super(Recommender, self).__init__()
        layer_sizes = layer_sizes if layer_sizes else [64, 32]
        self._query_params = query_params
        self._candidate_params = candidate_params
        self._layer_sizes = layer_sizes
        self._activation = activation

        self.candidate_model = CandidateModel(**self._candidate_params)
        self.query_model = QueryModel(**self._query_params)

        self.task: tfrs.tasks.Task = tfrs.tasks.Retrieval()

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)
        return loss

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(
            {feature: inputs[feature] for feature in USER_FEATURES}
        )
        candidate_embeddings: tf.Tensor = self.candidate_model(
            {feature: inputs[feature] for feature in ITEM_FEATURES}
        )
        return query_embeddings, candidate_embeddings

    def get_config(self):
        return {
            "query_params": self._query_params,
            "candidate_params": self._candidate_params,
            "layer_sizes": self._layer_sizes,
            "activation": self._activation,
        }
