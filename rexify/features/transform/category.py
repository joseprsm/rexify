from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from rexify.features.base import BaseFeatureEncoder
from rexify.types import Schema


class CategoricalEncoder(BaseFeatureEncoder):
    def __init__(self, schema: Schema, target: str):
        super().__init__(dtype="categorical", target=target, schema=schema)

        self.ppl = make_pipeline(OneHotEncoder(sparse=False))
