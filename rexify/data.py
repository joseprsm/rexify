import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.features.base import HasSchemaMixin
from rexify.schema import Schema
from rexify.utils import get_target_id


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

    def save(self):
        raise NotImplementedError

    def to_dataset(self) -> tf.data.Dataset:
        ds = self._make_dataset()
        ds = ds.map(self._get_header_fn())
        return ds

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
