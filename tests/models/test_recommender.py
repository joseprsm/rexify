import pytest
import tensorflow as tf

from rexify import Recommender
from rexify.tests import get_dataset, get_mock_schemas


_BASE_MODEL_PARAMS = ["user_id", 15, "item_id", 30]


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_init(schema):
    _, params = get_dataset(schema)
    Recommender(**params)


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_compile(schema):
    _, params = get_dataset(schema)
    model = Recommender(**params)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1))


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_call(schema):
    ds, params = get_dataset(schema)
    model = Recommender(**params)
    inputs = list(ds.batch(32).take(1))[0]
    res = model(inputs)

    assert type(res) == tuple
    assert len(res) == 2
    assert res[0].shape == res[1].shape

    query_embeddings, candidate_embeddings = res
    assert query_embeddings.shape == tf.TensorShape([32, 32])
    assert candidate_embeddings.shape == tf.TensorShape([32, 32])


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_compute_loss(schema):
    ds, params = get_dataset(schema)
    model = Recommender(**params)
    inputs = list(ds.batch(32).take(1))[0]
    embeddings = model(inputs)

    total_loss = model.compute_loss(inputs)
    assert total_loss.dtype == tf.float32

    compare_loss = model.retrieval_task(*embeddings)
    compare_loss += model.get_ranking_loss(
        *embeddings, inputs["event_type"], inputs["rating"]
    )
    assert total_loss == compare_loss


def test_config():
    model = Recommender(*_BASE_MODEL_PARAMS)
    assert model.get_config() == {
        "item_dims": 30,
        "user_dims": 15,
        "user_id": "user_id",
        "item_id": "item_id",
        "output_layers": [64, 32],
        "feature_layers": [64, 32, 16],
        "ranking_features": None,
        "ranking_layers": [64, 32],
        "ranking_weights": None,
    }
