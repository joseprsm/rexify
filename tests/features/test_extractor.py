import os

import numpy as np
import pandas as pd

from rexify.features import FeatureExtractor

EVENTS_PATH = os.path.join("tests", "data", "events.csv")

events = pd.read_csv(EVENTS_PATH)
schema = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}
feat = FeatureExtractor(schema=schema)
feat.fit(events)


def test_model_params():
    assert feat.model_params == {
        "n_unique_items": 3047,
        "item_id": "item_id",
        "n_unique_users": 89088,
        "user_id": "user_id",
    }


def test_output_features():
    assert np.all(feat.output_features == ["user_id", "item_id"])


def test_fit():
    assert len(feat._ppl.steps) != 0


def test_transform_output_shape():
    features = feat.transform(events)
    assert features.shape == events.shape


def test_transform_nunique():
    features = feat.transform(events)
    assert np.all(pd.DataFrame(features).nunique().values == events.nunique().values)
