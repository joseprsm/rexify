import json
import click
import pickle

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from rexify.utils import flatten, get_target_id
from rexify.features import FeatureExtractor


@click.command()
@click.option("--events-path", type=str)
@click.option("--schema-path", type=str)
@click.option("--items-dir", type=str)
@click.option("--users-dir", type=str)
@click.option("--extractor-dir", type=str)
@click.option("--train-data-dir", type=str)
@click.option("--test-data-dir", type=str)
@click.option("--test-size", type=float, default=0.3)
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

    events = pd.read_csv(events_path)

    with open(schema_path, "r") as f:
        schema = json.loads(f.read().replace("'", '"'))

    features = [list(schema[target].keys()) for target in ["user", "item"]]
    events = events.loc[~np.any(pd.isnull(events), axis=1), flatten(features)]

    train, test = train_test_split(events, test_size=test_size)

    feat = FeatureExtractor(schema)

    train = feat.fit_transform(train)
    test = feat.transform(test)

    Path(extractor_dir).mkdir(parents=True, exist_ok=True)
    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    Path(test_data_dir).mkdir(parents=True, exist_ok=True)

    extractor_path = Path(extractor_dir) / "feat.pkl"
    train_path = Path(train_data_dir) / "train.csv"
    test_path = Path(test_data_dir) / "test.csv"

    with open(extractor_path, "wb") as f:
        pickle.dump(feat, f)

    np.savetxt(train_path, train, delimiter=",")
    np.savetxt(test_path, test, delimiter=",")

    transformed_events = feat.transform(events)

    item_id = get_target_id(schema, "item")
    items = np.unique(
        transformed_events[:, np.argwhere(events.columns == item_id)[0, 0]]
    ).astype(int)

    Path(items_dir).mkdir(parents=True, exist_ok=True)
    items_path = Path(items_dir) / "items.csv"
    np.savetxt(items_path, items)

    user_id = get_target_id(schema, "user")
    users = np.unique(
        transformed_events[:, np.argwhere(events.columns == user_id)[0, 0]]
    ).astype(int)

    Path(users_dir).mkdir(parents=True, exist_ok=True)
    users_path = Path(users_dir) / "users.csv"
    np.savetxt(users_path, users)


if __name__ == "__main__":
    load()
