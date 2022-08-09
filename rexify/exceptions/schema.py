from typing import List

from rexify.constants import SUPPORTED_DATA_TYPES


class EmptySchemaException(Exception):
    def __init__(self):
        self.message = "Schema can't be empty"
        super().__init__(self.message)


class MissingKeysSchemaException(Exception):
    def __init__(self, missing_keys: List[str]):
        self.missing_keys = missing_keys
        self.message = f"{self.missing_keys} not in Schema Keys"
        super().__init__(self.message)


class MissingIdSchemaException(Exception):
    def __init__(self, missing_attr_key: str):
        self.missing_attr_key = missing_attr_key
        self.message = f'No `id` feature for "{missing_attr_key}" key'
        super().__init__(self.message)


class TooManyIdFeaturesSchemaException(Exception):
    def __init__(self, target_key: str):
        self.message = f'Too many `id` features for "{target_key}" key'
        super().__init__(self.message)


class ContextIdSchemaException(Exception):
    def __init__(self):
        self.message = "Context can't have `id` attributes"


class DataTypeNotSupportedSchemaException(Exception):
    def __init__(self):
        self.message = (
            f"Rexify only supports {', '.join(SUPPORTED_DATA_TYPES)} data types"
        )
