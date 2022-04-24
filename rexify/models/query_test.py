import tensorflow as tf

from rexify.models.query import QueryModel


class QueryModelTest(tf.test.TestCase):
    def testCallFeatureModels(self):
        model = QueryModel(
            candidate_model=tf.keras.Sequential(
                [
                    tf.keras.layers.Hashing(num_bins=10),
                    tf.keras.layers.Embedding(input_dim=11, output_dim=16),
                    tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1, 1, 16))),
                ]
            ),
            schema={"userId": "categorical"},
            params={"userId": {"input_dim": 10, "embedding_dim": 16}},
            recurrent_layers=[64, 32],
            layer_type="LSTM",
            layer_sizes=[45, 12],
        )

        inputs = {"userId": tf.constant([1]), "sequence": tf.constant("1")}
        output = model(inputs)

        self.assertEqual(output.shape, tf.TensorShape([1, 12]))
