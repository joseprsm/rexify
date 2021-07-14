import tensorflow as tf

from rexify.models import EmbeddingLookup

from tests.utils import get_sample_data


class EmbeddingLookupTest(tf.test.TestCase):

    def setUp(self):
        super(EmbeddingLookupTest, self).setUp()
        self._vocabulary, self._embeddings = get_sample_data()

    def testCall(self):
        model = EmbeddingLookup(vocabulary=self._vocabulary, embeddings=self._embeddings)
        inputs = tf.constant([self._vocabulary[0]])
        x = model.call(inputs)
        self.assertIsInstance(x, tf.Tensor)
        self.assertEqual(x.shape, tf.TensorShape([32]))

        tokens = tf.strings.split(inputs, sep=None)
        self.assertIsInstance(tokens, tf.RaggedTensor)

        ids = model.token_to_id.lookup(tokens)
        self.assertIsInstance(ids, tf.RaggedTensor)
        self.assertEqual(ids[0, 0].numpy(), 0)

        embeddings = tf.nn.embedding_lookup(
            params=model.embeddings, ids=ids)
        self.assertIsInstance(embeddings, tf.RaggedTensor)
        self.assertEqual(embeddings[0, 0].shape, tf.TensorShape([32]))
        self.assertEqual(sum(tf.cast(self._embeddings[0] == embeddings[0, 0], tf.int32)).numpy(), 32)

    def testConfig(self):
        model = EmbeddingLookup(vocabulary=self._vocabulary, embeddings=self._embeddings)
        config = model.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn('sample_query', config.keys())


if __name__ == '__main__':
    tf.test.main()
