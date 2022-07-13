from typing import Union, Optional

import os

import click
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import scann

from rexify.models import EmbeddingLookup


def generate_ann(
    lookup_model: EmbeddingLookup,
    embeddings: Union[tf.data.Dataset, np.ndarray],
    candidates: tf.data.Dataset,
    sample_query: str,
) -> tfrs.layers.factorized_top_k.ScaNN:
    index = tfrs.layers.factorized_top_k.ScaNN(
        lookup_model, k=50, num_reordering_candidates=1_000
    )
    # todo: fix data types
    index.index(embeddings, candidates)
    _ = index(tf.constant([sample_query]))
    return index


@click.command()
@click.option("--output-dir", type=str)
def index(
    output_dir: Union[str, bytes, os.PathLike],
):
    lookup_model = EmbeddingLookup()
    scann = generate_ann(...)
    output_path: Union[str, bytes, os.PathLike] = os.path.join(output_dir, "scann")
    scann.save(
        output_path, options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
    )


if __name__ == "__main__":
    index()
