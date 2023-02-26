import json
import tempfile

import pytest

from rexify.schema import Schema, _TargetSchema


def test_init():
    user_id = "user_id"
    item_id = "item_id"
    timestamp = "timestamp"
    event_type = "event_type"
    user_features = {"age": "number", "gender": "category"}
    item_features = {"price": "number", "category": "category"}
    schema = Schema(
        user_id=user_id,
        item_id=item_id,
        timestamp=timestamp,
        event_type=event_type,
        user_features=user_features,
        item_features=item_features,
    )

    assert schema.user.id == "user_id"
    assert schema.user.age == "number"
    assert schema.user.gender == "category"
    assert schema.item.id == "item_id"
    assert schema.item.price == "number"
    assert schema.item.category == "category"
    assert schema.timestamp == timestamp
    assert schema.event_type == event_type


def test_from_dict():
    schema_dict = {
        "user": {"user_id": "id", "age": "number", "gender": "category"},
        "item": {"item_id": "id", "price": "number", "category": "category"},
        "timestamp": "timestamp",
        "event_type": "event_type",
    }

    schema = Schema.from_dict(schema_dict)

    assert schema.user.id == "user_id"
    assert schema.user.age == "number"
    assert schema.user.gender == "category"
    assert schema.item.id == "item_id"
    assert schema.item.price == "number"
    assert schema.item.category == "category"
    assert schema.timestamp == "timestamp"
    assert schema.event_type == "event_type"


def test_load():
    schema_dict = {
        "user": {"user_id": "id", "age": "number", "gender": "category"},
        "item": {"item_id": "id", "price": "number", "category": "category"},
        "timestamp": "timestamp",
        "event_type": "event_type",
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(schema_dict, f)
        f.seek(0)
        schema = Schema.from_json(f.name)

    assert schema.user.id == "user_id"
    assert schema.user.age == "number"
    assert schema.user.gender == "category"
    assert schema.item.id == "item_id"
    assert schema.item.price == "number"
    assert schema.item.category == "category"
    assert schema.timestamp == "timestamp"
    assert schema.event_type == "event_type"


def test_target_schema():
    # Test data types are valid
    target = _TargetSchema("id", feature1="category", feature2="number")
    assert hasattr(target, "id")
    assert hasattr(target, "feature1")
    assert target.feature1 == "category"
    assert hasattr(target, "feature2")
    assert target.feature2 == "number"

    # Test unsupported data type throws error
    with pytest.raises(ValueError, match=r"Data type not supported"):
        _ = _TargetSchema("id", feature1="string")


def test_schema_io():
    # Test Schema to_dict method
    user_id = "user_id"
    item_id = "item_id"
    timestamp = "timestamp"
    event_type = "event_type"
    user_features = {"age": "number", "gender": "category"}
    item_features = {"price": "number", "category": "category"}
    schema = Schema(
        user_id, item_id, timestamp, event_type, user_features, item_features
    )
    assert schema.to_dict() == {
        "user": {"user_id": "id", "age": "number", "gender": "category"},
        "item": {"item_id": "id", "price": "number", "category": "category"},
        "timestamp": "timestamp",
        "event_type": "event_type",
    }

    # Test Schema from_dict method
    schema_dict = schema.to_dict()
    schema_loaded = Schema.from_dict(schema_dict)
    assert schema_loaded.to_dict() == schema.to_dict()

    # Test Schema load method
    with open("test_schema.json", "w") as f:
        json.dump(schema_dict, f, indent=4)
    schema_loaded = Schema.from_json("test_schema.json")
    assert schema_loaded.to_dict() == schema.to_dict()

    # Test Schema save method
    schema.save("test_schema.json")
    with open("test_schema.json", "r") as f:
        schema_loaded = json.load(f)
    assert schema_loaded == schema_dict
