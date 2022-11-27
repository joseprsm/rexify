import json
from pathlib import Path
from typing import Dict

import click
import numpy as np
import scann  # noqa: F401
import tensorflow as tf

from rexify.utils import get_target_id


@click.command()
@click.option("--users-dir")
@click.option("--schema-path")
@click.option("--model-dir")
@click.option("--index-dir")
@click.option("--predictions-dir")
@click.option("--batch-size", default=512)
def retrieval(
    users_dir: str,
    schema_path: str,
    model_dir: str,
    index_dir: str,
    predictions_dir: str,
    batch_size: int = 512,
):
    users_path = Path(users_dir) / "users.csv"
    users = np.loadtxt(str(users_path), delimiter=",")
    users = users[users != -1]

    model_path = Path(model_dir) / "model"
    model = tf.keras.models.load_model(model_path)

    index_path = Path(index_dir) / "model"
    index = tf.keras.models.load_model(index_path)

    with open(schema_path, "r") as f:
        schema = json.loads(f.read().replace("'", '"'))

    user_id = get_target_id(schema, "user")

    def add_header(x):
        return {user_id: tf.cast(x, tf.float32)}

    users_tf = (
        tf.data.Dataset.from_tensor_slices(users).map(add_header).batch(batch_size)
    )

    predictions = np.concatenate(
        [
            users.reshape(-1, 1),
            np.concatenate(
                [
                    get_recommendations(
                        query_model=model.query_model,
                        index=index,
                        user_batch=user_batch,
                    )
                    for user_batch in list(users_tf)
                ],
                axis=0,
            ),
        ],
        axis=1,
    )

    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "preds.csv"
    np.savetxt(predictions_path, predictions)


def get_recommendations(
    query_model: tf.keras.Model,
    index: tf.keras.Model,
    user_batch: Dict[str, tf.Tensor],
):
    user_embeddings = query_model(user_batch)
    _, predictions = index(user_embeddings)
    return predictions


if __name__ == "__main__":
    retrieval()
