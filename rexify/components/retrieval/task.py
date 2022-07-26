from pathlib import Path

import json
import click
import numpy as np
import tensorflow as tf

import scann

from rexify.utils import get_target_id


@click.command()
@click.option("--users-dir")
@click.option("--schema-path")
@click.option("--model-dir")
@click.option("--index-dir")
@click.option("--predictions-dir")
@click.option("--k")
def retrieval(
    users_dir: str,
    schema_path: str,
    model_dir: str,
    index_dir: str,
    predictions_dir: str,
    k: int = 20,
):
    users_path = Path(users_dir) / "users.csv"
    users = np.loadtxt(str(users_path), delimiter=",")

    model_path = Path(model_dir) / "model"
    model = tf.keras.models.load_model(model_path)

    index_path = Path(index_dir) / "model"
    index = tf.keras.models.load_model(index_path)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    user_id = get_target_id(schema, "user")

    def add_header(x):
        return {user_id: x}

    users = tf.data.Dataset.from_tensor_slices(users).map(add_header)
    user_embeddings = model.query_model(users)
    _, predictions = index(user_embeddings, k)

    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "preds.csv"
    np.savetxt(predictions_path, predictions)


if __name__ == "__main__":
    retrieval()
