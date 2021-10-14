import tensorflow as tf

from rexify.models.sequence import SequenceModel


class SequenceModelTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.model = SequenceModel(
            candidate_model=tf.keras.Sequential([
                tf.keras.layers.Hashing(num_bins=10),
                tf.keras.layers.Embedding(input_dim=11, output_dim=16),
                tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1, 1, 16)))]),
            layer_sizes=[64, 32])

    def testCall(self):
        inputs = tf.constant('?')
        output = self.model(inputs)
        self.assertEqual(output.shape, tf.TensorShape([1, 32]))
