import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

from rexify.features.base import HasSchemaMixin, HasTargetMixin
from rexify.utils import get_target_id


class IDEncoder(BaseEstimator, TransformerMixin, HasSchemaMixin, HasTargetMixin):

    _transformer: ColumnTransformer

    def __init__(self, schema, target):
        HasSchemaMixin.__init__(self, schema)
        HasTargetMixin.__init__(self, target)

    def fit(self, X: pd.DataFrame, y=None):
        target_features = get_target_id(self._schema, self._target)
        encoder_args = self._get_encoder_args(X, target_features)
        self._transformer = make_column_transformer(
            (OrdinalEncoder(**encoder_args), target_features),
            remainder="passthrough",
        )
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
    def _get_encoder_args(df: pd.DataFrame, target_features: list[str]):
        value = df[target_features].nunique().sum()
        return {
            "dtype": np.int64,
            "handle_unknown": "use_encoded_value",
            "unknown_value": value,
        }

    @property
    def transformer(self):
        return self._transformer

    @property
    def target_feature(self):
        return self._transformer.transformers[0][-1][0]
