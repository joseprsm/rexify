import json
from pathlib import Path

import click
import numpy as np
import scann as _  # noqa: F401
import tensorflow as tf
import tensorflow_recommenders as tfrs

from rexify.utils import get_target_id


@click.command()
@click.option("--items-dir", type=str)
@click.option("--schema-path", type=str)
@click.option("--model-dir", type=str)
@click.option("--index-dir", type=str)
def index(
    items_dir: str,
    schema_path: str,
    model_dir: str,
    index_dir: str,
):
    with open(schema_path, "r") as f:
        schema = json.loads(f.read().replace("'", '"'))

    user_id = get_target_id(schema, "user")
    item_id = get_target_id(schema, "item")

    items_path = Path(items_dir) / "items.csv"
    items = np.loadtxt(items_path)
    items = items[items != -1]  # remove unknown indices
    items = tf.data.Dataset.from_tensor_slices(items.reshape(-1, 1))

    model_path = Path(model_dir) / "model"
    model = tf.keras.models.load_model(model_path)

    item_embeddings = items.map(lambda x: {item_id: x}).map(model.candidate_model)

    scann = tfrs.layers.factorized_top_k.ScaNN(k=50, num_reordering_candidates=1_000)
    scann.index_from_dataset(tf.data.Dataset.zip((items, item_embeddings)))
    scann(model.query_model({user_id: [42]}), k=1)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "model"
    tf.keras.models.save_model(
        scann,
        index_path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]),
    )


if __name__ == "__main__":
    index()
