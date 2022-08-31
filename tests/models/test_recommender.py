import itertools
from tempfile import mkdtemp

import numpy as np
import pytest
import tensorflow as tf

from rexify import RetrievalModel
from rexify.utils import flatten


SAMPLE_MODEL_PARAMS = ("user_id", 15, "item_id", 15)


def get_base_inputs():
    return np.concatenate(
        [
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 1, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 5, size=100).reshape(-1, 1),
        ],
        axis=1,
    )


def get_base_dataset():
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {
                "query": {
                    "user_id": x[0],
                    "user_features": x[1:3],
                    "context_features": x[3:4],
                },
                "candidate": {"item_id": x[4], "item_features": x[5:]},
            }
        )
        .batch(16)
    )


def get_dataset_missing_features(missing_feature: list[str] = None):
    return (
        tf.data.Dataset.from_tensor_slices(get_base_inputs())
        .map(
            lambda x: {
                "query": {
                    "user_id": x[0],
                    "user_features": x[1:3]
                    if "user_features" not in missing_feature
                    else tf.constant([], dtype=tf.float32),
                    "context_features": x[3:4]
                    if "context_features" not in missing_feature
                    else tf.constant([], dtype=tf.float32),
                },
                "candidate": {
                    "item_id": x[4],
                    "item_features": x[5:]
                    if "item_features" not in missing_feature
                    else tf.constant([], dtype=tf.float32),
                },
            }
        )
        .batch(16)
    )


datasets = [get_base_dataset()] + [
    get_dataset_missing_features(feature)
    for feature in flatten(
        [
            list(
                itertools.combinations(
                    ["user_features", "context_features", "item_features"], i
                )
            )
            for i in range(1, 4)
        ]
    )
]


def get_model_params():
    return [
        ("string", 9, "string", 9, 16, [64, 32], [64, 32, 16]),
        ("string", 9, "string", 9, 8, [32], [64]),
        # (12, 9, 'string', 9, 8, [32], [64])
    ]


@pytest.mark.parametrize("inputs", datasets)
def test_call(inputs):
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    query_embeddings, candidate_embeddings = model(list(inputs.take(1))[0])

    assert query_embeddings.shape == tf.TensorShape([16, 32])
    assert candidate_embeddings.shape == tf.TensorShape([16, 32])


@pytest.mark.parametrize("inputs", datasets)
def test_fit(inputs):
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    model.compile()
    model.fit(inputs, epochs=1)


@pytest.mark.parametrize("inputs", datasets)
def test_compile(inputs):
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(inputs, epochs=1)


@pytest.mark.parametrize("inputs", datasets)
def test_save(inputs):
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(inputs, epochs=1)
    model.save(mkdtemp())


@pytest.mark.parametrize("model_params", get_model_params())
def test_init(model_params):
    RetrievalModel(*model_params)


@pytest.mark.parametrize("inputs", datasets)
def test_compute_loss(inputs):
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    inputs = list(inputs.take(1))[0]
    loss = model.compute_loss(inputs)

    assert loss.dtype == tf.float32


def test_config():
    model = RetrievalModel(*SAMPLE_MODEL_PARAMS)
    assert model.get_config() == {
        "item_dims": 15,
        "user_dims": 15,
        "user_id": "user_id",
        "item_id": "item_id",
        "output_layers": [64, 32],
        "feature_layers": [64, 32, 16],
    }
