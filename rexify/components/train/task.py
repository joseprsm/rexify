from typing import List, Union

import os
import click
import tensorflow as tf

from rexify.models import Recommender


@click.command()
@click.option("--input-dir", type=str)
@click.option("--output-dir", type=str)
@click.option("--layer-sizes", type=int)
@click.option("--activation", type=int, default="leaky_relu")
@click.option("--batch-size", type=int, default=512)
@click.option("--epochs", type=int, default=100)
def train(
    input_dir: Union[str, bytes, os.PathLike],
    output_dir: Union[str, bytes, os.PathLike],
    layer_sizes: List[int] = None,
    activation: str = "leaky_relu",
    batch_size: int = 512,
    epochs: int = 100,
):

    layer_sizes = layer_sizes or [64, 32]
    training_data: tf.data.Dataset = ...

    query_params = {
        "schema": {"userId": "categorical"},
        "layer_sizes": layer_sizes,
        "activation": activation,
        "params": {"userId": {"input_dim": nb_items, "embedding_dim": 16}},
    }

    candidate_params = {
        "schema": {"itemId": "categorical"},
        "layer_sizes": layer_sizes,
        "activation": activation,
        "params": {"itemId": {"input_dim": nb_users, "embedding_dim": 32}},
    }

    model: Recommender = Recommender(
        query_params=query_params,
        candidate_params=candidate_params,
        layer_sizes=layer_sizes,
        activation=activation,
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(0.2))
    model.fit(
        training_data.batch(batch_size),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
    )

    model_dir = ...
    model.save(model_dir)


if __name__ == "__main__":
    train()
