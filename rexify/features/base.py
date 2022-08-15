from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from rexify.constants import SUPPORTED_DATA_TYPES
from rexify.exceptions.schema import (
    ContextIdSchemaException,
    DataTypeNotSupportedSchemaException,
    EmptySchemaException,
    MissingIdSchemaException,
    MissingKeysSchemaException,
    TooManyIdFeaturesSchemaException,
)


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError


class HasSchemaInput:
    def __init__(self, schema):
        self._validate(schema)
        self.schema = schema

    @staticmethod
    def _validate(schema):
        if schema == {}:
            raise EmptySchemaException()

        targets = ["user", "item"]
        if np.any(~np.in1d(targets, list(schema.keys()))):
            keys = np.array(targets)[~np.in1d(targets, list(schema.keys()))].tolist()
            raise MissingKeysSchemaException(keys)

        else:
            for target in targets:
                if "id" not in schema[target].values():
                    raise MissingIdSchemaException(target)
                elif np.sum(np.in1d(list(schema[target].values()), ["id"])) > 1:
                    raise TooManyIdFeaturesSchemaException(target)
                elif np.any(
                    ~np.in1d(list(schema[target].values()), SUPPORTED_DATA_TYPES)
                ):
                    raise DataTypeNotSupportedSchemaException()

        if "context" in schema.keys():
            if "id" in schema["context"].values():
                raise ContextIdSchemaException()
            elif np.any(
                ~np.in1d(list(schema["context"].values()), SUPPORTED_DATA_TYPES)
            ):
                raise DataTypeNotSupportedSchemaException()
