from pathlib import Path

import click
import numpy as np
import tensorflow as tf

from rexify import FeatureExtractor
from rexify.models import Recommender
from rexify.utils import make_dirs


def train(
    training_data_dir: str,
    extractor_dir: str,
    model_dir: str,
    epochs: int = 100,
    batch_size: int = 512,
):
    feat_path = Path(extractor_dir) / "feat.pkl"
    feat = FeatureExtractor.load(feat_path)

    train_path = Path(training_data_dir) / "train.csv"
    train_df = np.loadtxt(train_path, delimiter=",")
    train_df = feat.make_dataset(train_df).batch(batch_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

    model = Recommender(**feat.model_params)
    model.compile()
    model.fit(train_df, epochs=epochs, callbacks=callbacks)

    make_dirs(model_dir)
    model_path = Path(model_dir) / "model"
    model.save(model_path)


@click.command()
@click.option("--training-data-dir", type=str)
@click.option("--extractor-dir", type=str)
@click.option("--model-dir", type=str)
@click.option("--learning-rate", type=float, default=0.2)
@click.option("--epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=512)
def train_cmd(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    train_cmd()
