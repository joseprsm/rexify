import os
import json

import pandas as pd
import tensorflow as tf

from rexify import Recommender, FeatureExtractor

EVENTS_PATH = os.path.join("tests", "data", "events.csv")
SCHEMA_PATH = os.path.join("tests", "data", "schema.json")


def mock():
    events = pd.read_csv(EVENTS_PATH)
    with open(SCHEMA_PATH, "r") as f:
        feat = FeatureExtractor(json.load(f))
    feat.fit(events)
    return events, feat


class RecommenderTest(tf.test.TestCase):
    def setUp(self):
        super(RecommenderTest, self).setUp()
        self.inputs, self.feat = mock()
        self.inputs = self.feat.make_dataset(self.inputs)
        self.inputs = list(self.inputs.batch(1).take(1))[0]
        self.model = Recommender(**self.feat.model_params)

    def testCall(self):
        query_embeddings, candidate_embeddings = self.model(self.inputs)

        self.assertIsInstance(query_embeddings, tf.Tensor)
        self.assertIsInstance(candidate_embeddings, tf.Tensor)

        self.assertEqual(query_embeddings.shape, tf.TensorShape([1, 32]))
        self.assertEqual(candidate_embeddings.shape, tf.TensorShape([1, 32]))

    def testComputeLoss(self):
        loss = self.model.compute_loss(self.inputs)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, tf.TensorShape([]))

    def testConfig(self):
        config = self.feat.model_params
        config["layer_sizes"] = [64, 32]
        self.assertEqual(self.model.get_config(), config)
