import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

from rexify.features.base import HasSchemaInput
from rexify.types import Schema
from rexify.utils import get_target_id


class IDEncoder(BaseEstimator, TransformerMixin, HasSchemaInput):
    def __init__(self, schema: Schema):
        super().__init__(schema)
        target_features = get_target_id(schema, "user") + get_target_id(schema, "item")
        encoder_args = self._get_encoder_args()
        self._transformer = make_column_transformer(
            (OrdinalEncoder(**encoder_args), target_features),
            remainder="passthrough",
        )

    def fit(self, X: pd.DataFrame, y=None):
        self._transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self._transformer.transform(X)
        columns = self._get_features_names_out()
        return pd.DataFrame(x, columns=columns)

    def _get_features_names_out(self) -> list[str]:
        features = self._transformer.get_feature_names_out()
        return [name.split("__")[-1] for name in features]

    @staticmethod
    def _get_encoder_args():
        return {
            "dtype": np.int64,
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
        }
