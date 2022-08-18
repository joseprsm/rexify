import os
from tempfile import mkdtemp, tempdir

import pandas as pd

from rexify.components.load.task import _make_dirs, _read, load


EVENTS_PATH = os.path.join("tests", "data", "events.csv")
SCHEMA_PATH = os.path.join("tests", "data", "schema.json")


def test_data_read():
    events, schema = _read(EVENTS_PATH, SCHEMA_PATH)
    assert isinstance(events, pd.DataFrame)
    assert isinstance(schema, dict)
    assert events.shape[0] != 0
    assert events.shape[1] != 0


def test_make_dirs():
    _make_dirs(tempdir)


def test_task():
    output_dir = mkdtemp()
    load(
        events_path=EVENTS_PATH,
        schema_path=SCHEMA_PATH,
        train_data_dir=output_dir,
        test_data_dir=output_dir,
        extractor_dir=output_dir,
        users_dir=output_dir,
        items_dir=output_dir,
    )
