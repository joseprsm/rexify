import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.features.base import HasSchemaMixin
from rexify.schema import Schema
from rexify.utils import get_target_id, make_dirs


class DataFrame(pd.DataFrame, HasSchemaMixin):
    def __init__(
        self,
        data: pd.DataFrame,
        schema: Schema,
        ranking_features: list[str] | None = None,
    ):
        super().__init__(data)
        HasSchemaMixin.__init__(self, schema)
        self._ranking_features = ranking_features

    def save(self, path: str | Path, name: str = None):
        path = Path(path)
        path = path / name if name else path

        history = pd.DataFrame(np.stack(self.loc[:, "history"].values))

        make_dirs(path)
        history.to_csv(path / "history.csv", index=None)
        self.drop("history", axis=1).to_csv(path / "features.csv", index=None)

        with open(path / "ranks.json", "w") as f:
            json.dump(self._ranking_features, f)

        self.schema.save(path / "schema.json")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        history = pd.read_csv(path / "history.csv")
        features = pd.read_csv(path / "features.csv")
        features["history"] = history.values.tolist()
        del history

        schema = Schema.from_json(path / "schema.json")
        with open(path / "ranks.json", "r") as f:
            ranking_features = json.load(f)

        return DataFrame(features, schema=schema, ranking_features=ranking_features)

    def to_dataset(self) -> tf.data.Dataset:
        return self._make_dataset().map(self._get_header_fn())

    def _make_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.zip(
            (
                self._get_target_vector_dataset(self, self._schema, "user"),
                self._get_target_vector_dataset(self, self._schema, "item"),
                tf.data.Dataset.from_tensor_slices(
                    np.stack(self["history"].values).astype(np.int32)
                ),
                self._get_ranking_dataset(self),
            )
        )

    @staticmethod
    def _get_target_vector_dataset(
        data, schema: Schema, target: str
    ) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, get_target_id(schema, target)]
            .values.reshape(-1)
            .astype(np.int32)
        )

    @staticmethod
    def _get_header_fn():
        @tf.autograph.experimental.do_not_convert
        def header_fn(user_id, item_id, history, ranks):
            return {
                "query": {"user_id": user_id, "history": history},
                "candidate": {"item_id": item_id},
                "rank": ranks,
            }

        return header_fn

    def _get_ranking_dataset(self, data) -> tf.data.Dataset:
        @tf.autograph.experimental.do_not_convert
        def add_header(x):
            return {
                self._ranking_features[i]: x[i]
                for i in range(len(self._ranking_features))
            }

        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, self._ranking_features].values.astype(np.int32)
        ).map(add_header)
