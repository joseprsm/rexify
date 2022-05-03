from typing import Union

import os
import click
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

INPUT_DIR = "/mnt/data/raw"

ppl = make_pipeline(...)


# noinspection PyPep8Naming
@click.command()
@click.option("--output-dir", type=str)
@click.option("--test-size", type=float, default=0.3)
def preprocess(
    output_dir: Union[str, bytes, os.PathLike],
    test_size: float = 0.3,
):

    event_path = os.path.join(INPUT_DIR, "events.csv")
    events = pd.read_csv(event_path)

    X = ppl.fit_transform(events)

    X_train, X_test = train_test_split(X, test_size=test_size)
    X_train.to_csv(os.path.join(output_dir, "train.csv"))
    X_test.to_csv(os.path.join(output_dir, "test.csv"))
