import numpy as np
import pytest
import tensorflow as tf

from rexify.models.candidate import CandidateModel


SAMPLE_MODEL_PARAMS = ("item_id", 15)


def get_base_inputs():
    return np.concatenate(
        [
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 1, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
        ],
        axis=1,
    )


def get_base_dataset():
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {"item_id": x[0], "item_features": x[1:]},
        )
        .batch(16)
    )


def get_missing_features_dataset(strip_features: list[str]):
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {
                "item_id": x[0]
                if "item_id" not in strip_features
                else tf.constant(dtype=tf.float32),
                "item_features": x[1:]
                if "item_features" not in strip_features
                else tf.constant([], dtype=tf.float32),
            },
        )
        .batch(16)
    )


def get_model_params():
    return [
        ("item_id", 9, 16, [64, 32], [64, 32, 16]),
        ("string", 9, 8, [32], [64]),
        # (12, 9, 'string', 9, 8, [32], [64])
    ]


datasets = [get_base_dataset(), get_missing_features_dataset(["item_features"])]


@pytest.mark.parametrize("model_params", get_model_params())
def test_init(model_params):
    CandidateModel(*model_params)


@pytest.mark.parametrize("inputs", datasets)
def test_call(inputs):
    model = CandidateModel(*SAMPLE_MODEL_PARAMS)
    inputs = list(inputs.take(1))[0]
    candidate_embeddings = model(inputs)
    assert candidate_embeddings.shape == tf.TensorShape([16, 32])


def test_config():
    model = CandidateModel(*SAMPLE_MODEL_PARAMS)
    assert model.get_config() == {
        "id_features": "item_id",
        "n_dims": 15,
        "embedding_dim": 32,
        "layer_sizes": [64, 32],
        "feature_layers": [64, 32, 16],
        "item_id": "item_id",
        "n_items": 15,
    }
