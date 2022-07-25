import json
import click

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from rexify.utils import flatten, get_target_id


@click.command()
@click.option("--events-path", type=str)
@click.option("--schema-path", type=str)
@click.option("--items-dir", type=str)
@click.option("--users-dir", type=str)
@click.option("--train-data-dir", type=str)
@click.option("--test-data-dir", type=str)
@click.option("--test-size", type=float, default=0.3)
def load(
    events_path: str,
    schema_path: str,
    train_data_dir: str,
    test_data_dir: str,
    items_dir: str,
    users_dir: str,
    test_size: float = 0.3,
):

    events = pd.read_csv(events_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    features = [list(schema[target].keys()) for target in ["user", "item"]]
    events = events.loc[~np.any(pd.isnull(events), axis=1), flatten(features)]

    train, test = train_test_split(events, test_size=test_size)

    ppl = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )

    train = ppl.fit_transform(train)
    test = ppl.transform(test)

    Path(train_data_dir).mkdir(parents=True, exist_ok=True)
    Path(test_data_dir).mkdir(parents=True, exist_ok=True)

    train_path = Path(train_data_dir) / "train.csv"
    test_path = Path(test_data_dir) / "test.csv"

    np.savetxt(train_path, train, delimiter=",")
    np.savetxt(test_path, test, delimiter=",")

    transformed_events = ppl.transform(events)

    item_id = get_target_id(schema, "item")
    items = transformed_events[:, np.argwhere(events.columns == item_id)[0, 0]]

    Path(items_dir).mkdir(parents=True, exist_ok=True)
    items_path = Path(items_dir) / "items.csv"
    np.savetxt(items_path, items)

    user_id = get_target_id(schema, "user")
    users = transformed_events[:, np.argwhere(events.columns == user_id)[0, 0]]

    Path(users_dir).mkdir(parents=True, exist_ok=True)
    users_path = Path(users_dir) / "users.csv"
    np.savetxt(users_path, users)


if __name__ == "__main__":
    load()
