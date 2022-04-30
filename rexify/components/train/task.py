from typing import List, Union, Dict

import os
import click
import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.models import Recommender


@click.command()
@click.option("--input-dir", type=str)
@click.option("--output-dir", type=str)
@click.option("--layer-sizes", type=int)
@click.option("--activation", type=int, default="leaky_relu")
@click.option("--batch-size", type=int, default=512)
@click.option("--epochs", type=int, default=100)
@click.option("--learning-rate", "-lr", type=float, default=0.2)
@click.option("--user-id", type=str, default="userId")
@click.option("--item-id", type=str, default="itemId")
def train(
    input_dir: Union[str, bytes, os.PathLike],
    output_dir: Union[str, bytes, os.PathLike],
    schema: Dict[str, Dict[str, str]],
    layer_sizes: List[int] = None,
    activation: str = "leaky_relu",
    batch_size: int = 512,
    epochs: int = 100,
    learning_rate: float = 0.2,
    user_id: str = "userId",
    item_id: str = "itemId",
):

    layer_sizes = layer_sizes or [64, 32]

    train_df: pd.DataFrame = _load_data(input_dir)
    nb_users: np.ndarray = np.unique(train_df[[user_id]])
    nb_items: np.ndarray = np.unique(train_df[[item_id]])

    training_data: tf.data.Dataset = _make_dataset(
        users=train_df.loc[schema["user"].keys()],
        items=train_df.loc[schema["item"].keys()],
        context=train_df.loc[schema["context"].keys()],
    )

    # todo: add missing query schema param
    query_params = {
        "layer_sizes": layer_sizes,
        "activation": activation,
        "params": {"userId": {"input_dim": nb_items, "embedding_dim": 16}},
    }

    # todo: add missing candidate schema param
    candidate_params = {
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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model.fit(
        training_data.batch(batch_size),
        epochs=epochs,
        callbacks=[
            # todo: add tensorboard, mlflow callbacks
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
    )

    model_dir = ...
    model.save(model_dir)


def _load_data(data_dir: Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    pass


def _make_dataset(users, items, context):
    """

    Args:
        users: pandas.DataFrame with user features
        items: pandas.DataFrame with item features
        context: pandas.DataFrame with event features

    Returns: a zipped tf.data.Dataset

    """

    @tf.autograph.experimental.do_not_convert
    def add_features_header(features):
        return lambda x: {features.columns[i]: x[i] for i in range(features.shape[1])}

    @tf.autograph.experimental.do_not_convert
    def add_main_header(query: pd.DataFrame, candidate: pd.DataFrame):
        return {"query": query, "candidate": candidate}

    @tf.autograph.experimental.do_not_convert
    def add_query_header(user_features, context_features):
        return {"user": user_features, "context": context_features}

    def create_dataset(data):
        return tf.data.Dataset.from_tensor_slices(data).map(add_features_header(data))

    return tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip((create_dataset(users), create_dataset(context))).map(
                add_query_header
            ),
            create_dataset(items),
        )
    ).map(add_main_header)


if __name__ == "__main__":
    train()
