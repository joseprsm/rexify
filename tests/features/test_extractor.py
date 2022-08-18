import os
from numbers import Number
from typing import Callable

import numpy as np
import pandas as pd

from rexify.features import FeatureExtractor


EVENTS_PATH = os.path.join("tests", "data", "events.csv")
TOWERS = ["query", "candidate"]

events = pd.read_csv(EVENTS_PATH)
schema = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}
feat = FeatureExtractor(schema=schema)
out = feat.fit_transform(events)
out = feat.make_dataset(out)
out = list(out.take(1))[0]

query_header = list(schema["user"].keys())
query_header += list(schema["context"].keys()) if "context" in schema.keys() else list()
candidate_header = list(schema["item"].keys())

headers = {"query": query_header, "candidate": candidate_header}


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
    assert len(feat._ppl.transformers_) != 0


def test_transform_output_ids():
    features = feat.transform(events)
    features = feat.make_dataset(features)
    e = next(features.as_numpy_iterator())
    assert isinstance(e["query"]["user_id"], Number)
    assert isinstance(e["candidate"]["item_id"], Number)


def test_transform_output_type():
    features = feat.transform(events)
    assert isinstance(features, np.ndarray)


def test_transform_nunique():
    features = feat.transform(events)
    features = feat.make_dataset(features)
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


def test_dataset_main_keys():
    assert np.all(np.in1d(list(out.keys()), TOWERS))


def test_dataset_header_output():
    assert isinstance(feat._get_header_fn(), Callable)
    assert feat._get_header_fn()(np.array([1, 2])) == {
        "query": {"user_id": 1},
        "candidate": {"item_id": 2},
    }


def test_dataset_candidate_keys():
    assert np.all(np.in1d(list(out["candidate"].keys()), headers["candidate"]))


def test_dataset_query_keys():
    assert np.all(np.in1d(list(out["query"].keys()), headers["query"]))
