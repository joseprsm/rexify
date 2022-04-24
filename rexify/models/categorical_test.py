import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

from rexify.models.categorical import CategoricalModel


class CategoricalModelTest(tf.test.TestCase):
    def setUp(self):
        raw_data = [{"userId": 42}]
        raw_data_metadata = dataset_metadata.DatasetMetadata(
            schema_utils.schema_from_feature_spec(
                {"userId": tf.io.FixedLenFeature([], tf.int64)}
            )
        )

        def preprocessing_fn(inputs):
            x = tft.compute_and_apply_vocabulary(inputs["userId"])
            return {"userId_transformed": x}

        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            transformed_dataset, _ = (
                raw_data,
                raw_data_metadata,
            ) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)

        transformed_data = transformed_dataset[0]
        self._inputs = transformed_data[0]["userId_transformed"]
        self._model = CategoricalModel(1, 32)

    def testCall(self):
        self.assertEqual(self._inputs, 0)
        res = self._model(tf.constant(self._inputs))
        self.assertEqual(res.shape, tf.TensorShape([32]))

    def testConfig(self):
        config = self._model.get_config()
        self.assertIn("input_dim", list(config.keys()))
        self.assertIn("embedding_dim", list(config.keys()))


if __name__ == "__main__":
    tf.test.main()
