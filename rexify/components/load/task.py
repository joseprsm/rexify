from typing import Union, Dict, List, Optional

import os
import json
import click
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from rexify.features.transformers import PreprocessingPipeline, ColumnTransformer
from rexify.utils import flatten

INPUT_DIR = "/mnt/data"
SCHEMA_PATH = os.path.join(INPUT_DIR, "schema.json")


# noinspection PyTypeChecker
def _create_pipeline(
    schema: Dict[str, Union[Dict[str, str]], str]
) -> ColumnTransformer:

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


def _get_features(
    schema: Optional[Dict[str, Dict[str, str]]] = None,
    value: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[str]]:

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
@click.option("--output-dir", type=str)
@click.option("--schema-path", type=str)
@click.option("--test-size", type=float, default=0.3)
def load(
    output_dir: Union[str, bytes, os.PathLike],
    schema_path: Union[str, bytes, os.PathLike] = SCHEMA_PATH,
    test_size: float = 0.3,
):

    event_path = os.path.join(INPUT_DIR, "events.csv")
    events = pd.read_csv(event_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    ppl = _create_pipeline(schema)

    events = events[~np.any(pd.isnull(events), axis=1), :]
    X = ppl.fit_transform(events)

    X_train, X_test = train_test_split(X, test_size=test_size)
    X_train.to_csv(os.path.join(output_dir, "train.csv"))
    X_test.to_csv(os.path.join(output_dir, "test.csv"))

    pipeline_output_path = os.path.join(output_dir, "pipelines.pkl")

    with open(pipeline_output_path, "wb") as f:
        pickle.dump(ppl, f)


if __name__ == "__main__":
    load()
