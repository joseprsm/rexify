from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from rexify.dataclasses import Schema
from rexify.features.base import BaseEncoder


class CategoricalEncoder(BaseEncoder):
    def __init__(self, schema: Schema, target: str):
        super().__init__(dtype="categorical", target=target, schema=schema)
        self.ppl = make_pipeline(OneHotEncoder(sparse=False))
