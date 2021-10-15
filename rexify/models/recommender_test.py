import tensorflow as tf

from rexify.models import recommender


class RecommenderTest(tf.test.TestCase):

    def testBuild(self):
        num_bins = 100
        model = recommender.build(num_bins, num_bins)
        self.assertIsInstance(model, recommender.Recommender)
        for attr in ['query_model', 'candidate_model']:
            query_model = getattr(model, attr)
            self.assertEqual(
                query_model.layers[0].get_config()['num_bins'],
                num_bins)

    def testCall(self):
        rex = recommender.build(50, 50)
        sample_query = {'userId': tf.constant([1]), 'itemId': tf.constant([1])}
        query_embeddings, candidate_embeddings = rex(sample_query)
        self.assertIsInstance(query_embeddings, tf.Tensor)
        self.assertIsInstance(candidate_embeddings, tf.Tensor)

        self.assertEqual(query_embeddings.shape, tf.TensorShape([1, 32]))
        self.assertEqual(candidate_embeddings.shape, tf.TensorShape([1, 32]))

        query_embedding_1 = rex.query_model(tf.constant([[1]]))
        self.assertTrue((tf.reduce_sum(tf.cast(
            query_embeddings == query_embedding_1,
            tf.int32)) == 32).numpy())

    def testComputeLoss(self):
        rex = recommender.build(50, 50)
        sample_query = {'userId': tf.constant([1]), 'itemId': tf.constant([1])}
        loss = rex.compute_loss(sample_query)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, tf.TensorShape([]))

    def testConfig(self):
        rex = recommender.build(50, 50)
        self.assertIsInstance(rex.get_config(), dict)


if __name__ == '__main__':
    tf.test.main()
