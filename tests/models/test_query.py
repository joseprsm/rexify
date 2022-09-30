import numpy as np
import pytest
import tensorflow as tf

from rexify.models.query import QueryModel


SAMPLE_MODEL_PARAMS = ("user_id", 15, 9)


def get_base_inputs():
    return np.concatenate(
        [
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 1, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
            np.random.randint(0, 15, size=100).reshape(-1, 1),
        ],
        axis=1,
    )


def get_base_dataset():
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {
                "user_id": x[0],
                "user_features": x[1:3],
                "context_features": x[3:],
            },
        )
        .batch(16)
    )


def get_missing_features_dataset(strip_features: list[str]):
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {
                "user_id": x[0]
                if "user_id" not in strip_features
                else tf.constant(dtype=tf.float32),
                "user_features": x[1:3]
                if "user_features" not in strip_features
                else tf.constant([], dtype=tf.float32),
                "context_features": x[3:]
                if "context_features" not in strip_features
                else tf.constant([], dtype=tf.float32),
            },
        )
        .batch(16)
    )


def get_model_params():
    return [
        ("user_id", 9, 9, 16, [64, 32], [64, 32, 16]),
        ("user_id", 9, 9, 8, [32], [64]),
        # (12, 9, 'string', 9, 8, [32], [64])
    ]


datasets = [get_base_dataset(), get_missing_features_dataset(["item_features"])]


@pytest.mark.parametrize("model_params", get_model_params())
def test_init(model_params):
    QueryModel(*model_params)


@pytest.mark.parametrize("inputs", datasets)
def test_call(inputs):
    model = QueryModel(*SAMPLE_MODEL_PARAMS)
    inputs = list(inputs.take(1))[0]
    embeddings = model(inputs)
    assert embeddings.shape == tf.TensorShape([16, 32])


def test_config():
    model = QueryModel(*SAMPLE_MODEL_PARAMS)
    assert model.get_config() == {
        "id_features": "user_id",
        "n_dims": 15,
        "embedding_dim": 32,
        "layer_sizes": [64, 32],
        "feature_layers": [64, 32, 16],
        "user_id": "user_id",
        "n_users": 15,
        "n_items": 9,
    }
