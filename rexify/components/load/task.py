import json
import click

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from rexify.utils import flatten, get_target_id


@click.command()
@click.option("--events-path", type=str)
@click.option("--schema-path", type=str)
@click.option("--items-path", type=str)
@click.option("--train-data-path", type=str)
@click.option("--test-data-path", type=str)
@click.option("--test-size", type=float, default=0.3)
def load(
    events_path: str,
    schema_path: str,
    train_data_path: str,
    test_data_path: str,
    items_path: str,
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

    np.savetxt(train_data_path, train, delimiter=",")
    np.savetxt(test_data_path, test, delimiter=",")

    item_id = get_target_id(schema, 'item')
    items = ppl.transform(events)[:, np.argwhere(events.columns == item_id)[0, 0]]
    np.savetxt(items_path, items)


if __name__ == "__main__":
    load()
