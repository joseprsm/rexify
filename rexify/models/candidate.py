from typing import Dict, Any, List

import tensorflow as tf

from rexify.models.embedding import CategoricalModel
from rexify.models.tower import Tower


class CandidateModel(Tower):

    def __init__(self,
                 schema: Dict[str, str],
                 params: Dict[str, Dict[str, Any]],
                 layer_sizes: List[int],
                 activation: str = 'relu'):

        self.item_model = CategoricalModel(**params['itemId'])
        _ = params.pop('itemId')

        super(CandidateModel, self).__init__(
            schema=schema,
            params=params,
            layer_sizes=layer_sizes,
            activation=activation)

    def call_feature_models(self, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        item_embeddings: tf.Tensor = self.item_model(inputs['itemId'])
        feature_embeddings: List[tf.Tensor] = [
            model(inputs[feature_name])
            for feature_name, model in self.feature_models.items()]
        return [item_embeddings] + feature_embeddings
