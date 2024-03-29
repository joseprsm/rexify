import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from rexify import FeatureExtractor, Output, Schema
from rexify.features.transform import CustomTransformer


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
        users, items = self._save_users_items(users, items)
        return FeatureExtractor(schema, users, items)

    def test_fit(self, data, feat):
        _ = feat.fit(data)

    def test_transform(self, data, feat):
        transformed = feat.fit(data).transform(data)
        assert isinstance(transformed, Output)

    @pytest.fixture(scope="class")
    def custom_feat(self, schema, users, items):
        users["custom_feature"] = np.random.randint(100, 200, size=users.shape[0])
        users, items = self._save_users_items(users, items)
        return FeatureExtractor(
            schema,
            users,
            items,
            custom_transformers=[
                CustomTransformer("user", StandardScaler(), ["custom_feature"])
            ],
        )

    def test_fit_custom(self, data, feat, custom_feat):
        _ = feat.fit(data)
        _ = custom_feat.fit(data)
        assert feat.model_params["user_embeddings"].shape[1] == 3
        assert custom_feat.model_params["user_embeddings"].shape[1] == 4

    def test_save_load(self, data, feat):
        _ = feat.fit(data).transform(data)
        tmp_dir = tempfile.mkdtemp()
        feat.save(tmp_dir)
        feat_path = Path(tmp_dir) / "feature_extractor.pickle"
        assert feat_path.exists()

        fe = FeatureExtractor.load(feat_path)
        assert fe

    @pytest.fixture(scope="class")
    def fe_no_data(self, schema, users, items):
        users, items = self._save_users_items(users, items)
        return FeatureExtractor(schema, users, items, return_dataset=False)

    def test_make_dataset(self, data, fe_no_data):
        transformed = fe_no_data.fit(data).transform(data)

        tmp_dir = tempfile.mkdtemp()
        transformed_path = Path(tmp_dir)
        transformed.save(transformed_path)

        df = Output.load(transformed_path)
        df.to_dataset()

    def _save_users_items(self, users, items) -> tuple[str, str]:
        tmp_dir = tempfile.mkdtemp()

        users_path = os.path.join(tmp_dir, "users.csv")
        users.to_csv(users_path)

        items_path = os.path.join(tmp_dir, "items.csv")
        items.to_csv(items_path)

        return users_path, items_path
