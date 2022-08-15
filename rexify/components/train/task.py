import os
import pickle
from pathlib import Path

import click
import pandas as pd
import tensorflow as tf

from rexify import FeatureExtractor
from rexify.models import Recommender


@click.command()
@click.option("--training-data-dir", type=str)
@click.option("--extractor-dir", type=str)
@click.option("--model-dir", type=str)
@click.option("--learning-rate", type=float, default=0.2)
@click.option("--epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=512)
def train(
    training_data_dir: str,
    extractor_dir: str,
    model_dir: str,
    learning_rate: float = 0.1,
    epochs: int = 100,
    batch_size: int = 512,
):

    feat_path = os.path.join(extractor_dir, "feat.pkl")
    with open(feat_path, "rb") as f:
        feat: FeatureExtractor = pickle.load(f)

    train_path = Path(training_data_dir) / "train.csv"
    train_df = pd.read_csv(train_path, header=None)
    train_df.columns = feat.output_features
    training_data = feat.make_dataset(train_df)

    model = Recommender(**feat.model_params)

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


if __name__ == "__main__":
    train()
