from typing import Dict, List

import json
import click

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from rexify.utils import flatten
from rexify.features.transformers import PreprocessingPipeline


def _create_pipeline(schema):

    categorical_args = _get_features(schema, value="categorical")
    numerical_args = _get_features(schema, value="numerical")
    embedding_args = {
        "features": _get_features(schema, value="userId")["features"]
        + _get_features(schema, value="itemId")["features"]
    }
    date_args = _get_features(schema, value="date")

    return PreprocessingPipeline(
        embedding_args, numerical_args, date_args, categorical_args
    )


def _get_features(schema, value) -> Dict[str, List[str]]:

    return {
        "features": list(
            map(
                lambda x: x[0],
                flatten(
                    [
                        list(
                            filter(
                                lambda x: x[1] in [value], list(schema[target].items())
                            )
                        )
                        for target in ["user", "item", "context"]
                    ]
                ),
            )
        )
    }


# noinspection PyPep8Naming
@click.command()
@click.option("--events-path", type=str)
@click.option("--users-path", type=str)
@click.option("--items-path", type=str)
@click.option("--schema-path", type=str)
@click.option("--train-data-path", type=str)
@click.option("--test-data-path", type=str)
@click.option("--test-size", type=float, default=0.3)
def load(
    events_path: str,
    users_path: str,
    items_path: str,
    schema_path: str,
    train_data_path: str,
    test_data_path: str,
    test_size: float = 0.3,
):

    events = pd.read_csv(events_path)
    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    users = users.loc[:, list(schema["user"].keys())]
    items = items.loc[:, list(schema["item"].keys())]

    user_feature = _get_features(schema, value="userId")["features"][0]
    item_feature = _get_features(schema, value="itemId")["features"][0]
    features = list(...)

    events = events.merge(users, on=user_feature).merge(items, on=item_feature)
    events = events[[features]]

    ppl = _create_pipeline(schema)

    events = events[~np.any(pd.isnull(events), axis=1), :]
    train, test = train_test_split(events, test_size=test_size)

    train = ppl.fit_transform(train)
    test = ppl.transform(test)

    np.savetxt(train_data_path, train)
    np.savetxt(test_data_path, test)


if __name__ == "__main__":
    load()
