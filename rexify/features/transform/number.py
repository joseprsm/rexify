from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from rexify.features.base import BaseFeatureEncoder
from rexify.types import Schema


class NumericalEncoder(BaseFeatureEncoder):
    def __init__(self, schema: Schema, target: str):
        super().__init__(dtype="numerical", target=target, schema=schema)

        self.ppl = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))
