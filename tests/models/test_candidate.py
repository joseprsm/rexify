import os
import json

import pandas as pd
import tensorflow as tf

from rexify import FeatureExtractor
from rexify.models.candidate import CandidateModel

EVENTS_PATH = os.path.join("tests", "data", "events.csv")
SCHEMA_PATH = os.path.join("tests", "data", "schema.json")


def mock():
    events = pd.read_csv(EVENTS_PATH)
    with open(SCHEMA_PATH, "r") as f:
        feat = FeatureExtractor(json.load(f))
    feat.fit(events)
    return events, feat


class QueryModelTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.inputs, self.feat = mock()
        self.inputs = self.feat.make_dataset(self.inputs)
        self.inputs = list(self.inputs.batch(1).take(1))[0]["candidate"]
        self.model = CandidateModel(
            n_items=self.feat.model_params["n_unique_items"],
            item_id=self.feat.model_params["item_id"],
            layer_sizes=[64, 32],
        )

    def testCall(self):
        query_embeddings = self.model(self.inputs)
        self.assertIsInstance(query_embeddings, tf.Tensor)
        self.assertEqual(query_embeddings.shape, tf.TensorShape([1, 32]))

    def testConfig(self):
        config = {
            "n_items": self.feat.model_params["n_unique_items"],
            "item_id": self.feat.model_params["item_id"],
            "embedding_dim": 32,
            "layer_sizes": [64, 32],
        }
        self.assertEqual(self.model.get_config(), config)
