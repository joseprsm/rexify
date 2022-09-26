import itertools
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import pytest

from rexify.components.load.task import load
from rexify.components.train.task import train
from rexify.tests import get_mock_schemas, get_sample_data


TEMP_DIR = Path(mkdtemp())


def generate_mock_schemas():
    schemas = get_mock_schemas()
    os.makedirs(TEMP_DIR / "schemas")

    for i, schema in enumerate(schemas):
        with open(TEMP_DIR / "schemas" / f"{i}.json", "w") as f:
            json.dump(schema, f)


def generate_events():
    get_sample_data().to_csv(TEMP_DIR / "events.csv", index=False)


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
