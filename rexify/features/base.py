import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline

from rexify.schema import Schema
from rexify.utils import get_target_feature, make_dirs


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
