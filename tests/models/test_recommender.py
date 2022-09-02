import itertools

import numpy as np
import pytest
import tensorflow as tf

from rexify import Recommender
from rexify.utils import flatten


BASE_MODEL_PARAMS = ["user_id", 15, "item_id", 15]


def get_base_inputs():
    return np.concatenate(
        [
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 1, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
            np.random.randint(0, 1_000, size=100).reshape(-1, 1),
            np.random.randint(0, 15, size=100).reshape(-1, 1),
            np.random.randint(0, 5, size=100).reshape(-1, 1),
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
                "candidate": {"item_id": x[4], "item_features": x[5:6]},
                "rank": x[6:],
            }
        )
        .batch(32)
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
                    "item_features": x[5:6]
                    if "item_features" not in missing_feature
                    else tf.constant([], dtype=tf.float32),
                },
                "rank": x[6:]
                if "rank" not in missing_feature
                else tf.constant([], dtype=tf.float32),
            }
        )
        .batch(32)
    )


datasets = [get_base_dataset()] + [
    get_dataset_missing_features(feature)
    for feature in flatten(
        [
            list(
                itertools.combinations(
                    ["user_features", "context_features", "item_features", "rank"], i
                )
            )
            for i in range(1, 5)
        ]
    )
]


def get_model_params():
    embedding_dims = [16, 8]
    feature_layers = [None]
    output_layers = [None]

    ranking_features = [["rating"], None]
    ranking_layers = [[32, 16], None]
    ranking_weights = [None]

    model_params = [
        BASE_MODEL_PARAMS + list(params)
        for params in itertools.product(
            embedding_dims,
            feature_layers,
            output_layers,
            ranking_features,
            ranking_layers,
            ranking_weights,
        )
    ]

    return model_params


args = list(itertools.product(datasets, get_model_params()))
args = np.array(args, dtype=object)[
    ~(
        np.array([list(a[0].take(1))[0]["rank"].shape[-1] == 0 for a in args])
        & np.array([a[1][-3] is not None for a in args])
    )
].tolist()


@pytest.mark.parametrize("inputs,params", args)
def test_compile(inputs, params):
    model = Recommender(*params)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1))


@pytest.mark.parametrize("model_params", get_model_params())
def test_init(model_params):
    Recommender(*model_params)


@pytest.mark.parametrize("inputs,params", args)
def test_call(inputs, params):
    model = Recommender(*params)
    inputs = list(inputs.take(1))[0]
    res = model(inputs)

    assert type(res) == tuple
    assert len(res) == 2
    assert res[0].shape == res[1].shape

    query_embeddings, candidate_embeddings = res
    assert query_embeddings.shape == tf.TensorShape([32, 32])
    assert candidate_embeddings.shape == tf.TensorShape([32, 32])


@pytest.mark.parametrize("inputs,params", args)
def test_compute_loss(inputs, params):
    model = Recommender(*params)
    inputs = list(inputs.take(1))[0]
    embeddings = model(inputs)
    compare_loss = model.get_retrieval_loss(*embeddings)

    total_loss = model.compute_loss(inputs)
    if params[-3]:
        compare_loss += model.get_ranking_loss(*embeddings, inputs["rank"])

    assert total_loss.dtype == tf.float32
    assert total_loss == compare_loss


def test_config():
    model = Recommender(*BASE_MODEL_PARAMS)
    assert model.get_config() == {
        "item_dims": 15,
        "user_dims": 15,
        "user_id": "user_id",
        "item_id": "item_id",
        "output_layers": [64, 32],
        "feature_layers": [64, 32, 16],
        "ranking_features": None,
        "ranking_layers": [64, 32],
        "ranking_weights": None,
    }
