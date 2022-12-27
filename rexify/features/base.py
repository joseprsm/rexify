from sklearn.pipeline import Pipeline

from rexify.dataclasses import Schema
from rexify.utils import get_target_feature


class HasSchemaInput:
    def __init__(self, schema: Schema):
        self._schema = schema

    @property
    def schema(self):
        return self._schema


class HasTargetInput:

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


class BaseEncoder(HasSchemaInput):

    ppl: Pipeline
    _targets: list[str]

    def __init__(self, dtype: str, schema: Schema):
        super().__init__(schema)
        self._type = dtype
        self._name = self._type + "Pipeline"

    def __iter__(self):
        for x in [self._name, self.ppl, self._targets]:
            yield x

    @property
    def name(self):
        return self._name

    def as_tuple(self):
        return tuple(self)


class BaseFeatureEncoder(BaseEncoder):
    def __init__(self, dtype: str, target: str, schema: Schema):
        super().__init__(dtype, schema)
        self._name = target.lower() + "_" + self._name
        self._targets = self._get_features(self._schema, target, self._type)

    @staticmethod
    def _get_features(schema, target, dtype) -> list[str]:
        return get_target_feature(schema, target, dtype)
