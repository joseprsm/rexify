from pathlib import Path

import json
import click

import pandas as pd
import tensorflow as tf

from rexify.models import Recommender


@click.command()
@click.option("--training-data-dir", type=str)
@click.option("--schema-path", type=str)
@click.option("--model-dir", type=str)
@click.option("--learning-rate", type=float, default=0.2)
@click.option("--epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=512)
def train(
    training_data_dir: str,
    schema_path: str,
    model_dir: str,
    learning_rate: float = 0.1,
    epochs: int = 100,
    batch_size: int = 512,
):

    train_path = Path(training_data_dir) / "train.csv"
    train_df: pd.DataFrame = pd.read_csv(train_path, header=None)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    user_feature = [k for k, v in schema["user"].items() if v == "id"]
    item_feature = [k for k, v in schema["item"].items() if v == "id"]

    train_df.columns = user_feature + item_feature

    nb_users = train_df.loc[:, user_feature].nunique()[0]
    nb_items = train_df.loc[:, item_feature].nunique()[0]

    training_data: tf.data.Dataset = _make_dataset(train_df)

    model = Recommender(nb_items, nb_users, user_feature, item_feature)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model.fit(
        training_data.batch(batch_size),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / "model"
    model.save(model_path)


def _make_dataset(data):

    return tf.data.Dataset.from_tensor_slices(data.values).map(
        lambda x: {
            "query": {data.columns[0]: x[0]},
            "candidate": {data.columns[1]: x[1]},
        }
    )


if __name__ == "__main__":
    train()
