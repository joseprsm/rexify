from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from rexify.features.base import BaseEncoder
from rexify.schema import Schema


class CategoricalEncoder(BaseEncoder):
    def __init__(self, schema: Schema, target: str):
        super().__init__(dtype="category", target=target, schema=schema)
        self.ppl = make_pipeline(OneHotEncoder(sparse=False))
