from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from rexify.features.base import BaseEncoder
from rexify.schema import Schema


class NumericalEncoder(BaseEncoder):
    def __init__(self, schema: Schema, target: str):
        super().__init__(dtype="number", target=target, schema=schema)
        self.ppl = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))
