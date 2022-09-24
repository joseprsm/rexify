import tempfile
from pathlib import Path
from tempfile import mkdtemp

import pytest
import tensorflow as tf

from rexify.features import FeatureExtractor
from rexify.tests import get_mock_schemas, get_sample_data


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_init(schema):
    FeatureExtractor(schema)


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_fit(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    feat.fit(events)


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_transform(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    feat.fit_transform(events)


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_save(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    feat.fit(events)
    feat.save(mkdtemp())


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_load(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    feat.fit(events)

    with tempfile.TemporaryDirectory() as tempdir:
        feat.save(tempdir)
        FeatureExtractor.load(Path(tempdir) / "feat.pkl")


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_model_params(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    feat.fit(events)
    assert feat.model_params["item_id"] == "item_id"
    assert feat.model_params["user_id"] == "user_id"
    assert feat.model_params["item_dims"] > 0
    assert feat.model_params["user_dims"] > 0
    assert len(feat.model_params["ranking_features"]) != 0


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_make_dataset(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema)
    x = feat.fit_transform(events)
    feat.make_dataset(x)


@pytest.mark.parametrize("schema", get_mock_schemas())
def test_dataset_header_output_with_features(schema):
    events = get_sample_data()
    feat = FeatureExtractor(schema=schema)
    feat.fit(events)

    targets = ["user", "item"]
    targets += ["context"] if "context" in schema.keys() else []

    count = {
        target: sum(
            [
                1 if v != "categorical" else events.loc[:, k].nunique()
                for k, v in schema[target].items()
                if v != "id"
            ]
        )
        for target in targets
    }

    if "rank" in schema.keys():
        count["rank"] = len(schema["rank"])

    header = feat._get_header_fn()(tf.ones(sum(list(count.values())) + 2))

    expected_result = {
        "query": {
            "user_id": tf.constant(1),
            "user_features": tf.ones(count["user"]),
            "context_features": tf.ones(count["context"])
            if "context" in schema.keys()
            else tf.constant([], dtype=tf.float32),
        },
        "candidate": {
            "item_id": tf.constant(1),
            "item_features": tf.ones(count["item"]),
        },
    }

    assert tf.reduce_all(
        header["query"]["user_features"] == expected_result["query"]["user_features"]
    )
    assert tf.reduce_all(
        header["query"]["context_features"]
        == expected_result["query"]["context_features"]
    )
    assert tf.reduce_all(
        header["query"]["user_id"]
        == tf.cast(expected_result["query"]["user_id"], tf.float32)
    )
    assert tf.reduce_all(
        header["candidate"]["item_features"]
        == expected_result["candidate"]["item_features"]
    )
    assert tf.reduce_all(
        header["candidate"]["item_id"]
        == tf.cast(expected_result["candidate"]["item_id"], tf.float32)
    )
