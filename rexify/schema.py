import json

from rexify.utils import get_target_id


class _JSONSerializable:
    def to_dict(self):
        return self.__dict__


class _TargetSchema(_JSONSerializable):

    _SUPPORTED_DATA_TYPES = ["category", "number"]

    def __init__(self, id_: str, **features):
        setattr(self, id_, "id")
        for feature_name, dtype in features.items():
            self._validate_features(feature_name, dtype)
            setattr(self, feature_name, dtype)

    @classmethod
    def _validate_features(cls, feature_name: str, dtype: str):
        if dtype not in cls._SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"""
                Data type not supported for feature `{feature_name}`.
                Supported data types are: {cls._SUPPORTED_DATA_TYPES}
                """
            )


class Schema(_JSONSerializable):
    def __init__(
        self,
        user_id: str,
        item_id: str,
        timestamp: str,
        event_type: str,
        user_features: dict[str, str] = None,
        item_features: dict[str, str] = None,
    ):
        user_features = user_features or {}
        item_features = item_features or {}
        self.user = _TargetSchema(user_id, **user_features)
        self.item = _TargetSchema(item_id, **item_features)
        self.timestamp = timestamp
        self.event_type = event_type

    @classmethod
    def load(cls, schema_path: str):
        with open(schema_path, "r") as f:
            schema = json.load(f)
        return Schema.from_dict(schema)

    @classmethod
    def from_dict(cls, schema: dict[str, str | dict[str, str]]):
        schema_ = schema.copy()
        user_id = get_target_id(schema_, "user")[0]
        _ = schema_["user"].pop(user_id)

        item_id = get_target_id(schema_, "item")[0]
        _ = schema_["item"].pop(item_id)

        return Schema(
            user_id=user_id,
            item_id=item_id,
            timestamp=schema_["timestamp"],
            event_type=schema_["event_type"],
            user_features=schema_["user"],
            item_features=schema_["item"],
        )

    def to_dict(self) -> dict[str, str | dict[str, str]]:
        schema = super().to_dict()
        schema["user"] = self.user.to_dict()
        schema["item"] = self.item.to_dict()
        return schema

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
