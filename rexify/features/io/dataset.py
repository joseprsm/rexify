import numpy as np
import pandas as pd
import tensorflow as tf

from rexify.features.io.schema import HasSchemaInput
from rexify.types import Schema
from rexify.utils import get_target_id


class TFDatasetGenerator(HasSchemaInput):
    def make_dataset(self, X: pd.DataFrame) -> tf.data.Dataset:
        ds = self._get_dataset(X)
        ds = ds.map(self._get_header_fn())
        return ds

    def _get_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        return tf.data.Dataset.zip(
            (
                self._get_target_vector_dataset(data, self._schema, "user"),
                self._get_target_vector_dataset(data, self._schema, "item"),
                tf.data.Dataset.from_tensor_slices(
                    np.stack(data["history"].values).astype(int)
                ),
                tf.data.Dataset.from_tensor_slices(
                    np.stack(data[self._schema["event"]].values).astype(float)
                ),
            )
        )

    @staticmethod
    def _get_target_vector_dataset(
        data, schema: Schema, target: str
    ) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(
            data.loc[:, get_target_id(schema, target)].values.reshape(-1)
        )

    @staticmethod
    def _get_header_fn():
        def header_fn(user_id, item_id, history, event):
            return {
                "query": {"user_id": user_id, "history": history},
                "candidate": {"item_id": item_id},
                "event": event,
            }

        return header_fn
