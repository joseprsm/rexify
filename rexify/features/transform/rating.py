from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from rexify.features.transform.base import BaseEncoder
from rexify.types import Schema
from rexify.utils import get_ranking_features


class RatingEncoder(BaseEncoder):
    def __init__(self, schema: Schema):
        super().__init__(dtype="rating", schema=schema)
        self.ppl = make_pipeline(MinMaxScaler())
        self.targets = get_ranking_features(schema)
