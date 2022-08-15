import os
from typing import Callable

import numpy as np
import pandas as pd

from rexify.features.dataset import TfDatasetGenerator

EVENTS_PATH = os.path.join("tests", "data", "events.csv")

events = pd.read_csv(EVENTS_PATH)
schema = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}
data_gen = TfDatasetGenerator(schema)
out = data_gen.make_dataset(events)
out = list(out.take(1))[0]

towers = ["query", "candidate"]
query_header = list(schema["user"].keys())
query_header += list(schema["context"].keys()) if "context" in schema.keys() else list()
candidate_header = list(schema["item"].keys())

headers = {"query": query_header, "candidate": candidate_header}


def test_dataset_main_keys():
    assert np.all(np.in1d(list(out.keys()), towers))


def test_dataset_header_output():
    assert isinstance(data_gen._get_header_fn(schema), Callable)
    assert data_gen._get_header_fn(schema)(np.array([1, 2])) == {
        "query": {"user_id": 1},
        "candidate": {"item_id": 2},
    }


def test_dataset_candidate_keys():
    assert np.all(np.in1d(list(out["candidate"].keys()), headers["candidate"]))


def test_dataset_query_keys():
    assert np.all(np.in1d(list(out["query"].keys()), headers["query"]))
