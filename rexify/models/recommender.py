from typing import Optional, List, Dict, Union, Any

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.embedding import CategoricalModel
from rexify.models.query import QueryModel
from rexify.models.candidate import CandidateModel


class Recommender(tfrs.Model):

    def __init__(self,
                 schema: Dict[str, Union[str, Dict[str, str]]],
                 params: Dict[str, Dict[str, Any]],
                 layer_sizes: Optional[List[int]] = None,
                 activation: Optional[str] = 'relu'):
        super(Recommender, self).__init__()
        layer_sizes = layer_sizes if layer_sizes else [64, 32]
        self._schema = schema
        self._params = params
        self._layer_sizes = layer_sizes
        self._activation = activation

        candidate_params = {'layer_sizes': layer_sizes, 'activation': activation}
        self.candidate_model = CandidateModel(
            schema=schema['item'],
            params=params['item'],
            **candidate_params)

        query_params = candidate_params.copy()
        query_params.update({'recurrent_layers': [50, 50], 'layer_type': 'LSTM'})
        self.query_model = QueryModel(
            candidate_model=CategoricalModel(
                **params['item']['itemId']),
            schema=schema['user'],
            params=params['user'],
            **query_params)

        self.task: tfrs.tasks.Task = tfrs.tasks.Retrieval()

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)
        return loss

    def call(self, inputs, *_):
        query_embeddings: tf.Tensor = self.query_model(inputs)
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs)
        return query_embeddings, candidate_embeddings

    def get_config(self):
        return {
            'schema': self._schema,
            'params': self._params,
            'layer_sizes': self._layer_sizes,
            'activation': self._activation
        }
