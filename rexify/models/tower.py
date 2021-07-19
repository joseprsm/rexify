from typing import List, Text, Dict, Any, Optional

import tensorflow as tf

from rexify.models.features import CategoricalModel


class Tower(tf.keras.Model):

    def __init__(self,
                 schema: Dict[Text, Text],
                 layer_sizes: List[int],
                 params: Optional[Dict[Text, Dict]] = None,
                 suffix: Optional[Text] = '',
                 **kwargs):
        super().__init__(**kwargs)
        self.schema: Dict[Text, Text] = schema
        self.params: Dict[Text, Dict] = params
        self.layer_sizes: List[int] = layer_sizes
        self.suffix: Text = suffix
        self.dense_layers: tf.keras.Model = self._get_dense_layers(layer_sizes)
        self.models: Dict[Text, tf.keras.Model] = self._get_feature_models(schema, params)

    def call(self, inputs: Dict[Text, tf.Tensor], *_):
        x = [model(inputs[self._get_feature_name(feature_name)]) for feature_name, model in self.models.items()]
        x = tf.concat(x, axis=1) if len(x) > 1 else x[0]
        return self.dense_layers(x)

    def get_config(self):
        return {'layer_sizes': self._layer_sizes}

    @staticmethod
    def _get_dense_layers(layer_sizes) -> tf.keras.Model:
        model = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        model.add(tf.keras.layers.Dense(layer_sizes[-1]))
        return model

    @staticmethod
    def _get_feature_models(schema: Dict[Text, Text], params: Dict[Text, Any]) -> Dict[Text, tf.keras.Model]:

        def get_model(feature_name, dtype) -> tf.keras.Model:
            if dtype == 'categorical':
                return CategoricalModel(**params[feature_name])

        return {
            feature_name: get_model(feature_name, dtype)
            for feature_name, dtype in schema.items()
        }

    def _get_feature_name(self, feature_name):
        if self.suffix:
            feature_name += self.suffix
            return feature_name
        return feature_name
