from typing import List, Dict, Any
from abc import abstractmethod

import tensorflow as tf

from rexify.models.categorical import CategoricalModel


class Tower(tf.keras.Model):

    def __init__(self,
                 schema: Dict[str, str],
                 params: Dict[str, Dict[str, Any]],
                 layer_sizes: List[int],
                 activation: str):
        super(Tower, self).__init__()
        self._schema = schema
        self._params = params
        self._activation = activation
        self._layer_sizes = layer_sizes

        self.dense_layers = self._get_dense_layers(layer_sizes, activation=activation)
        self.feature_models: Dict[str, tf.keras.Model] = self._feature_factory(schema, params)

    def call(self, inputs: Dict[str, tf.Tensor], *_):
        # retrieves the respective embedding for each feature present
        x: List[tf.Tensor] = self.call_feature_models(inputs)
        # if there is more than one feature, concatenate embeddings, else retrieve the single tensor
        x = tf.concat(x, axis=1) if len(x) > 1 else x[0]
        return self.dense_layers(x)

    @staticmethod
    def _get_dense_layers(layer_sizes: List[int], activation: str) -> tf.keras.Model:
        model = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            model.add(tf.keras.layers.Dense(layer_size, activation=activation))
        model.add(tf.keras.layers.Dense(layer_sizes[-1]))
        return model

    @staticmethod
    def _feature_factory(schema: Dict[str, str],
                         params: Dict[str, Dict[str, Any]]) -> Dict[str, tf.keras.Model]:

        def get_model(feature_name: str, dtype: str) -> tf.keras.Model:
            if dtype == 'categorical':
                return CategoricalModel(**params[feature_name])

        return {
            feature_name: get_model(feature_name, dtype)
            for feature_name, dtype in schema.items()
        }

    def get_config(self):
        return {
            'layer_sizes': self._layer_sizes,
            'schema': self._schema,
            'params': self._params,
            'activation': self._activation
        }

    @abstractmethod
    def call_feature_models(self, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        pass
