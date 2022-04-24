import tensorflow as tf

from rexify.models.recommender import Recommender
from rexify.features.sequence import slide_transform

from tests.utils import get_sample_params, get_sample_schema, load_mock_events


class RecommenderTest(tf.test.TestCase):
    def setUp(self):
        super(RecommenderTest, self).setUp()
        self._recommender_args = {
            "schema": get_sample_schema(),
            "params": get_sample_params(),
            "layer_sizes": [64, 32],
            "activation": "relu",
        }
        self.model = Recommender(**self._recommender_args)

    def testCall(self):
        events = load_mock_events()
        inputs = slide_transform(events, self._recommender_args["schema"])
        inputs = list(inputs.batch(1).take(1))[0]

        query_embeddings, candidate_embeddings = self.model(inputs)
        self.assertIsInstance(query_embeddings, tf.Tensor)
        self.assertIsInstance(candidate_embeddings, tf.Tensor)

        self.assertEqual(query_embeddings.shape, tf.TensorShape([1, 32]))
        self.assertEqual(candidate_embeddings.shape, tf.TensorShape([1, 32]))

        # query_embedding_1 = self.model.query_model(tf.constant([[1]]))
        # self.assertTrue((tf.reduce_sum(tf.cast(
        #     query_embeddings == query_embedding_1,
        #     tf.int32)) == 32).numpy())

    def testComputeLoss(self):
        events = load_mock_events()
        mock_data = slide_transform(events, self._recommender_args["schema"])
        loss = self.model.compute_loss(list(mock_data.batch(1).take(1))[0])
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, tf.TensorShape([]))

    def testConfig(self):
        self.assertIsInstance(self.model.get_config(), dict)


if __name__ == "__main__":
    tf.test.main()
