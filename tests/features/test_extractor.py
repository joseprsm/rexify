import os

import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.features import FeatureExtractor

EVENTS_PATH = os.path.join("tests", "data", "events.csv")

events = pd.read_csv(EVENTS_PATH)
schema = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}
feat = FeatureExtractor(schema=schema)
feat.fit(events)


def test_model_params():
    assert feat.model_params == {
        "n_unique_items": 535,
        "item_id": "item_id",
        "n_unique_users": 881,
        "user_id": "user_id",
    }


def test_output_features():
    assert np.all(feat.output_features == ["user_id", "item_id"])


def test_fit():
    assert len(feat._ppl.steps) != 0


def test_transform_output_ids():
    features = feat.transform(events)
    e = next(features.as_numpy_iterator())
    assert isinstance(e["query"]["user_id"], np.int64)
    assert isinstance(e["candidate"]["item_id"], np.int64)


def test_transform_output_type():
    features = feat.transform(events)
    assert isinstance(features, tf.data.Dataset)


def test_transform_nunique():
    features = feat.transform(events)
    assert np.all(
        pd.DataFrame(
            list(
                features.map(
                    lambda x: [x["query"]["user_id"], x["candidate"]["item_id"]]
                ).as_numpy_iterator()
            )
        )
        .nunique()
        .values
        == events.nunique().values
    )
