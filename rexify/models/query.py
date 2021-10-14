from typing import Dict, List, Any

import tensorflow as tf

from rexify.models.tower import Tower
from rexify.models.sequence import SequenceModel


class QueryModel(Tower):

    def __init__(self,
                 candidate_model: tf.keras.Model,
                 schema: Dict[str, str],
                 params: Dict[str, Dict[str, Any]],
                 recurrent_layers: List[int],
                 layer_type: str,
                 ff_layers: List[int],
                 activation: str = 'relu'):
        super(QueryModel, self).__init__(
            schema=schema,
            params=params,
            layer_sizes=ff_layers,
            activation=activation)

        self.sequence_model = SequenceModel(
            candidate_model=candidate_model,
            layer_type=layer_type,
            layer_sizes=recurrent_layers)

    def call_feature_models(self, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        feature_embeddings: List[tf.Tensor] = [
            model(inputs[feature_name])
            for feature_name, model in self.feature_models.items()]
        sequence_embeddings: tf.Tensor = self.sequence_model(inputs['sequence'])
        return feature_embeddings + [sequence_embeddings]
