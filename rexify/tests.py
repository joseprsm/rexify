import itertools
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd


_EVENT_TYPES = ["Page View", "Add to Cart", "Purchase"]

np.random.seed(42)


def get_sample_data():
    event_types = np.array(_EVENT_TYPES)

    today = datetime.today().date()
    hours = np.random.randint(0, 23, size=100)
    minutes = np.random.randint(0, 59, size=100)
    timestamps = [
        datetime(today.year, today.month, today.day, hour=h, minute=m)
        for h, m in zip(hours, minutes)
    ]

    return pd.DataFrame(
        {
            "user_id": np.random.randint(0, 15, size=100),  # user_id
            "gender": np.random.randint(0, 2, size=100),  # gender
            "age": np.random.randint(15, 65, size=100),  # age
            "item_id": np.random.randint(0, 30, size=100),  # item_id
            "type": np.random.randint(0, 5, size=100),  # type
            "price": np.random.randint(0, 1_000, size=100),  # price
            "event_type": event_types[np.random.randint(0, 3, size=100)],  # event_type
            "timestamp": timestamps,
        }
    )


def get_mock_schema(
    use_categorical: bool = False,
    use_numerical: bool = False,
    use_context: bool = False,
    use_rank: bool = False,
):
    schema = base = {
        "user": {"user_id": "id"},
        "item": {"item_id": "id"},
        "rank": [
            {"name": "Purchase"},
        ],
    }

    if use_categorical:
        with_categorical = deepcopy(base)
        with_categorical["user"]["gender"] = "categorical"
        with_categorical["item"]["type"] = "categorical"
        schema = with_categorical

    if use_numerical:
        with_numerical = deepcopy(schema)
        with_numerical["user"]["age"] = "numerical"
        with_numerical["item"]["price"] = "numerical"
        schema = with_numerical

    if use_context:
        with_context = deepcopy(schema)
        with_context["context"] = {}
        with_context["context"]["timestamp"] = "timestamp"
        schema = with_context

    if use_rank:
        with_rank = deepcopy(schema)
        ranking_features = with_rank["rank"]
        ranking_features += [
            {"name": "Page View", "weight": 0.1},
            {"name": "Add to Cart", "weight": 0.9},
        ]

    return schema


def get_mock_schemas():
    return [
        get_mock_schema(*args) for args in itertools.product([True, False], repeat=4)
    ]
