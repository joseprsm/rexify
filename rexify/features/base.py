import pickle
import re
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
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
    def save(self, output_dir: str, filename: str = None):
        make_dirs(output_dir)
        filename = (
            filename or self._camel_to_snake_case(self.__class__.__name__) + ".pickle"
        )
        output_path = Path(output_dir) / filename
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        return feat

    @staticmethod
    def _camel_to_snake_case(name: str):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


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


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer: TransformerMixin, target_features: list[str]):
        super().__init__()
        self.transformer = transformer
        self.target_features = target_features

        self._column_transformer = make_column_transformer(
            (self.transformer, self.target_features),
        )

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        pass
