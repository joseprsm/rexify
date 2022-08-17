import pytest

from rexify.exceptions.schema import (
    ContextIdSchemaException,
    DataTypeNotSupportedSchemaException,
    EmptySchemaException,
    MissingIdSchemaException,
    MissingKeysSchemaException,
    TooManyIdFeaturesSchemaException,
)
from rexify.features.base import BaseTransformer, HasSchemaInput


def test_fit():
    transformer = BaseTransformer()
    assert transformer.fit(None) == transformer


def test_transform():
    with pytest.raises(NotImplementedError):
        BaseTransformer().transform(None)


def test_empty_schema():
    schema = dict()

    with pytest.raises(EmptySchemaException):
        HasSchemaInput(schema=schema)


def test_empty_user_attributes_schema():
    schema = {"item": {"item_id": "id"}}

    with pytest.raises(MissingKeysSchemaException):
        HasSchemaInput(schema=schema)


def test_empty_item_attributes_schema():
    schema = {"user": {"user_id": "id"}}

    with pytest.raises(MissingKeysSchemaException):
        HasSchemaInput(schema=schema)


def test_too_many_user_ids_schema():
    schema = {"user": {"a": "id", "b": "id"}, "item": {"c": "id"}}

    with pytest.raises(TooManyIdFeaturesSchemaException):
        HasSchemaInput(schema=schema)


def test_too_many_item_ids_schema():
    schema = {"user": {"a": "id"}, "item": {"b": "id", "c": "id"}}

    with pytest.raises(TooManyIdFeaturesSchemaException):
        HasSchemaInput(schema=schema)


def test_context_id_schema():
    schema = {"user": {"a": "id"}, "item": {"b": "id"}, "context": {"c": "id"}}

    with pytest.raises(ContextIdSchemaException):
        HasSchemaInput(schema=schema)


def test_unsupported_user_attribute_dtype_schema():
    schema = {
        "user": {"a": "id", "d": "error"},
        "item": {"b": "id"},
        "context": {"c": "id"},
    }

    with pytest.raises(DataTypeNotSupportedSchemaException):
        HasSchemaInput(schema=schema)


def test_unsupported_item_attribute_dtype_schema():
    schema = {
        "user": {"a": "id"},
        "item": {"b": "id", "d": "error"},
    }

    with pytest.raises(DataTypeNotSupportedSchemaException):
        HasSchemaInput(schema=schema)


def test_unsupported_context_attribute_dtype_schema():
    schema = {
        "user": {"a": "id"},
        "item": {"b": "id"},
        "context": {"d": "error"},
    }

    with pytest.raises(DataTypeNotSupportedSchemaException):
        HasSchemaInput(schema=schema)


def test_missing_user_id_schema():
    schema = {
        "user": {"a": "categorical"},
        "item": {"b": "id"},
        "context": {"d": "categorical"},
    }

    with pytest.raises(MissingIdSchemaException):
        HasSchemaInput(schema=schema)


def test_missing_item_id_schema():
    schema = {
        "user": {"a": "id"},
        "item": {"b": "categorical"},
        "context": {"d": "categorical"},
    }

    with pytest.raises(MissingIdSchemaException):
        HasSchemaInput(schema=schema)
