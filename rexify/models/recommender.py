from typing import Optional, List

import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.models.query import QueryModel
from rexify.models.candidate import CandidateModel


class Recommender(tfrs.Model):

    def __init__(self,
                 nb_users: int,
                 nb_items: int,
                 layer_sizes: Optional[List[int]] = None,
                 activation: Optional[str] = 'relu'):
        super(Recommender, self).__init__()
        layer_sizes = layer_sizes if layer_sizes else [64, 32]
        self._nb_users = nb_users
        self._nb_items = nb_items
        self._layer_sizes = layer_sizes
        self._activation = activation

        candidate_params = {
            'schema': {'itemId': 'categorical'},
            'layer_sizes': layer_sizes,
            'activation': activation,
            'params': {
                'itemId': {
                    'input_dim': nb_users,
                    'embedding_dim': 32
                }
            }
        }
        self.candidate_model = CandidateModel(schema=self._schema, **candidate_params)

        query_params = {
            'schema': {'userId': 'categorical'},
            'layer_sizes': layer_sizes,
            'activation': activation,
            'params': {
                'userId': {
                    'input_dim': nb_items,
                    'embedding_dim': 16
                }
            }
        }
        self.query_model = QueryModel(**query_params)

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
            'nb_users': self._nb_users,
            'nb_items': self._nb_items,
            'layer_sizes': self._layer_sizes,
            'activation': self._activation
        }
