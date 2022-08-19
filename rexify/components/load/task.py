import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rexify.features import FeatureExtractor
from rexify.utils import flatten, get_target_id, make_dirs


def _read(events_path, schema_path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    events = pd.read_csv(events_path)

    with open(schema_path, "r") as f:
        schema = json.loads(f.read().replace("'", '"'))

    features = [list(schema[target].keys()) for target in ["user", "item"]]
    events = events.loc[~np.any(pd.isnull(events), axis=1), flatten(features)]
    return events, schema


# noinspection PyTypeChecker,PydanticTypeChecker
def load(
    events_path: str,
    schema_path: str,
    extractor_dir: str,
    train_data_dir: str,
    test_data_dir: str,
    items_dir: str,
    users_dir: str,
    test_size: float = 0.3,
):

    events, schema = _read(events_path, schema_path)
    train, test = train_test_split(events, test_size=test_size)

    feat = FeatureExtractor(schema)
    train = feat.fit_transform(train)
    test = feat.transform(test)
    feat.save(extractor_dir)

    make_dirs(train_data_dir, test_data_dir, items_dir, users_dir)

    train_path = Path(train_data_dir) / "train.csv"
    test_path = Path(test_data_dir) / "test.csv"

    np.savetxt(train_path, train, delimiter=",")
    np.savetxt(test_path, test, delimiter=",")

    transformed_events = feat.transform(events)

    def get_unique_target_ids(target: str) -> np.ndarray:
        id_feature = get_target_id(schema, target)[0]
        return np.unique(
            transformed_events[:, np.argwhere(events.columns == id_feature)[0, 0]]
        ).astype(int)

    items = get_unique_target_ids("item")
    items_path = Path(items_dir) / "items.csv"
    np.savetxt(items_path, items)

    users = get_unique_target_ids("user")
    users_path = Path(users_dir) / "users.csv"
    np.savetxt(users_path, users)


@click.command()
@click.option("--events-path", type=str)
@click.option("--schema-path", type=str)
@click.option("--items-dir", type=str)
@click.option("--users-dir", type=str)
@click.option("--extractor-dir", type=str)
@click.option("--train-data-dir", type=str)
@click.option("--test-data-dir", type=str)
@click.option("--test-size", type=float, default=0.3)
def load_cmd(**kwargs):
    load(**kwargs)


if __name__ == "__main__":
    load_cmd()
