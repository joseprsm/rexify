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
    def from_dict(cls, schema: dict[str, str | dict[str, str]]):
        schema_ = schema.copy()
        user_id = schema_["user"].pop("id")
        item_id = schema_["item"].pop("id")
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
