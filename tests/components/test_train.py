import itertools
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest

from rexify.components.load.task import load
from rexify.components.train.task import train


TEMP_DIR = Path(mkdtemp())


def generate_mock_schemas():

    base = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}

    with_categorical = deepcopy(base)
    with_categorical["user"]["is_client"] = "categorical"
    with_categorical["item"]["type"] = "categorical"

    with_numerical = deepcopy(with_categorical)
    with_numerical["user"]["age"] = "numerical"
    with_numerical["item"]["price"] = "numerical"

    with_context = deepcopy(with_numerical)
    with_context["context"] = {}
    with_context["context"]["event_type"] = "categorical"
    with_context["context"]["days_without_purchases"] = "numerical"

    with_rank = deepcopy(with_context)
    with_rank["rank"] = [
        {"name": "rating", "weight": 0.5},
        {"name": "minutes_watched"},
    ]

    schemas = {
        "base": base,
        "with_categorical": with_categorical,
        "with_numerical": with_numerical,
        "with_context": with_context,
        "with_rank": with_rank,
    }

    os.makedirs(TEMP_DIR / "schemas")

    for k, v in schemas.items():
        with open(TEMP_DIR / "schemas" / f"{k}.json", "w") as f:
            json.dump(v, f)


def generate_events():
    pd.DataFrame(
        np.concatenate(
            [
                np.random.randint(0, 15, size=100).reshape(-1, 1),
                np.random.randint(0, 2, size=100).reshape(-1, 1),
                np.random.randint(15, 65, size=100).reshape(-1, 1),
                np.random.randint(0, 15, size=100).reshape(-1, 1),
                np.random.randint(0, 5, size=100).reshape(-1, 1),
                np.random.randint(0, 1_000, size=100).reshape(-1, 1),
                np.random.randint(0, 5, size=100).reshape(-1, 1),
                np.random.randint(0, 365, size=100).reshape(-1, 1),
                np.random.randint(0, 5, size=100).reshape(-1, 1),
                np.random.randint(0, 40, size=100).reshape(-1, 1),
            ],
            axis=1,
        ),
        columns=[
            "user_id",
            "is_client",
            "age",
            "item_id",
            "type",
            "price",
            "event_type",
            "days_without_purchases",
            "rating",
            "minutes_watched",
        ],
    ).to_csv(TEMP_DIR / "events.csv", index=False)


generate_events()
generate_mock_schemas()

event_path = [TEMP_DIR / "events.csv"]
schema_paths = list(TEMP_DIR.glob("schemas/*"))
extractor_dirs = [TEMP_DIR / "extractor"]
train_data_dirs = [extractor_dirs[0], TEMP_DIR / "train_dir"]
test_data_dirs = [extractor_dirs[0], TEMP_DIR / "test_dir"]
items_dirs = [extractor_dirs[0], TEMP_DIR / "items_dir"]
users_dirs = [extractor_dirs[0], TEMP_DIR / "users_dir"]
model_dirs = [TEMP_DIR / "model"]
batch_sizes = [64, 128]

args = list(
    itertools.product(
        event_path,
        schema_paths,
        extractor_dirs,
        train_data_dirs,
        test_data_dirs,
        items_dirs,
        users_dirs,
        model_dirs,
        batch_sizes,
    )
)


@pytest.mark.parametrize(
    "events_path,schema_path,extractor_dir,train_data_dir,test_data_dir,items_dir,users_dir,model_dir,batch_size",
    args,
)
def test_train(
    events_path: str,
    schema_path: str,
    extractor_dir: str,
    train_data_dir: str,
    test_data_dir: str,
    items_dir: str,
    users_dir: str,
    model_dir: str,
    batch_size: int,
):
    load(
        events_path,
        schema_path,
        extractor_dir,
        train_data_dir,
        test_data_dir,
        items_dir,
        users_dir,
    )
    train(train_data_dir, extractor_dir, model_dir, 1, batch_size)
