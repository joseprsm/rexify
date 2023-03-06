import pandas as pd
import pytest
import tensorflow as tf

from rexify import FeatureExtractor, Schema


class TestFeatureExtractor:
    @pytest.fixture(scope="class")
    def schema(self):
        user_id = "user_id"
        item_id = "item_id"
        timestamp = "timestamp"
        event_type = "event_type"
        user_features = {"age": "number", "gender": "category"}
        item_features = {"price": "number", "category": "category"}
        return Schema(
            user_id, item_id, timestamp, event_type, user_features, item_features
        )

    @pytest.fixture(scope="class")
    def data(self):
        return pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [10, 20, 10, 20, 30, 40],
                "timestamp": [1, 2, 3, 4, 5, 6],
                "event_type": ["p", "p", "p", "p", "p", "p"],
            }
        )

    @pytest.fixture(scope="class")
    def users(self):
        return pd.DataFrame(
            {"user_id": [1, 2, 3], "age": [25, 30, 35], "gender": ["M", "F", "M"]}
        )

    @pytest.fixture(scope="class")
    def items(self):
        return pd.DataFrame(
            {"item_id": [10, 20, 30], "price": [1, 2, 3], "category": ["1", "2", "3"]}
        )

    @pytest.fixture(scope="class")
    def feat(self, schema, users, items):
        return FeatureExtractor(schema, users, items)

    def test_fit(self, data, feat):
        _ = feat.fit(data)

    def test_transform(self, data, feat):
        transformed = feat.fit(data).transform(data)
        assert isinstance(transformed, tf.data.Dataset)
        example = list(transformed.take(1))[0]
        assert isinstance(example, dict)
