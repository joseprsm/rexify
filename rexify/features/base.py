import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline

from rexify.schema import Schema
from rexify.utils import get_target_feature, get_target_id, make_dirs


class HasSchemaMixin:
    def __init__(self, schema: Schema):
        self._schema = schema

    @property
    def schema(self):
        return self._schema


class HasTargetMixin:

    _SUPPORTED_TARGETS = ["user", "item"]

    def __init__(self, target: str):
        self._target = target

    @property
    def target(self):
        return self._target

    @classmethod
    def _validate_target(cls, target: str):
        if target not in cls._SUPPORTED_TARGETS:
            raise ValueError(f"Target {target} not supported")


class Serializable:
    def save(self, output_dir: str, filename: str):
        make_dirs(output_dir)
        output_path = Path(output_dir) / filename
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        return feat


class TFDatasetGenerator(HasSchemaMixin):
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
                    np.stack(data["history"].values).astype(np.int32)
                ),
                tf.data.Dataset.from_tensor_slices(
                    np.stack(data[self._schema.event_type].values).astype(np.float32)
                ),
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
        def header_fn(user_id, item_id, history, event):
            return {
                "query": {"user_id": user_id, "history": history},
                "candidate": {"item_id": item_id},
                "event": event,
            }

        return header_fn


class BaseEncoder(HasSchemaMixin):

    ppl: Pipeline
    _targets: list[str]

    def __init__(self, dtype: str, target: str, schema: Schema):
        super().__init__(schema)
        self._type = dtype
        self._name = target.lower() + "_" + self._type + "Pipeline"
        self._targets = self._get_features(self._schema, target, self._type)

    @staticmethod
    def _get_features(schema, target, dtype) -> list[str]:
        return get_target_feature(schema, target, dtype)

    def __iter__(self):
        for x in [self._name, self.ppl, self._targets]:
            yield x

    def as_tuple(self):
        return tuple(self)

    @property
    def name(self):
        return self._name
