import tempfile
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

from rexify.models.tower import Tower


class TowerTest(tf.test.TestCase):

    def setUp(self):
        raw_data = [{'userId': 42}]
        raw_data_metadata = dataset_metadata.DatasetMetadata(
            schema_utils.schema_from_feature_spec({'userId': tf.io.FixedLenFeature([], tf.int64)}))

        def preprocessing_fn(inputs):
            x = tft.compute_and_apply_vocabulary(inputs['userId'])
            return {'userId': x}

        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            transformed_dataset, transform_fn = (
                    (raw_data, raw_data_metadata) |
                    tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

        transformed_data, transformed_metadata = transformed_dataset
        self._inputs = {'userId': tf.constant([transformed_data[0]['userId']])}

        self._schema = {'userId': 'categorical'}
        self._params = {'userId': {'input_dim': 3, 'embedding_dim': 32}}
        self._layer_sizes = [64, 32]
        self._tower = Tower(schema=self._schema, params=self._params, layer_sizes=self._layer_sizes, activation='relu')

    def testCall(self):
        x = self._tower(self._inputs)
        self.assertEqual(x.shape, tf.TensorShape([1, 32]))


if __name__ == '__main__':
    tf.test.main()
