import itertools
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import pytest

from rexify.components.load.task import load
from rexify.tests import get_mock_schemas, get_sample_data


TEMP_DIR = Path(mkdtemp())


def generate_mock_schemas():
    schemas = get_mock_schemas()
    os.makedirs(TEMP_DIR / "schemas")

    for k, v in schemas.items():
        with open(TEMP_DIR / "schemas" / f"{k}.json", "w") as f:
            json.dump(v, f)


def generate_events():
    get_sample_data().to_csv(TEMP_DIR / "events.csv", index=False)


generate_events()
generate_mock_schemas()

event_path = [TEMP_DIR / "events.csv"]
schema_paths = list(TEMP_DIR.glob("schemas/*"))
extractor_dirs = [TEMP_DIR / "outputs"]
train_data_dirs = [extractor_dirs[0], TEMP_DIR / "train_dir"]
test_data_dirs = [extractor_dirs[0], TEMP_DIR / "test_dir"]
items_dirs = [extractor_dirs[0], TEMP_DIR / "items_dir"]
users_dirs = [extractor_dirs[0], TEMP_DIR / "users_dir"]

args = list(
    itertools.product(
        event_path,
        schema_paths,
        extractor_dirs,
        train_data_dirs,
        test_data_dirs,
        items_dirs,
        users_dirs,
    )
)


@pytest.mark.parametrize(
    "events_path,schema_path,extractor_dir,train_data_dir,test_data_dir,items_dir,users_dir",
    args,
)
def test_load(
    events_path: str,
    schema_path: str,
    extractor_dir: str,
    train_data_dir: str,
    test_data_dir: str,
    items_dir: str,
    users_dir: str,
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
