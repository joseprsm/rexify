import json

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from rexify import DataFrame, FeatureExtractor


def load(
    events_path: str = typer.Option(...),
    users_path: str = typer.Option(...),
    items_path: str = typer.Option(...),
    schema_path: str = typer.Option(...),
    test_size: float = typer.Option(0.3),
):

    events = pd.read_csv(events_path)
    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    fe = FeatureExtractor(schema, users, items, return_dataset=False)

    train, val = train_test_split(events, test_size=test_size)
    train: DataFrame = fe.fit(train).transform(train)
    val: DataFrame = fe.transform(val)


if __name__ == "__main__":
    typer.run(load)
