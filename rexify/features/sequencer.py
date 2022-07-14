from typing import Optional, List

import pandas as pd

from rexify.features.base import BaseTransformer


class Sequencer(BaseTransformer):
    def __init__(
        self,
        user_col: Optional[str] = "account_id",
        item_col: Optional[str] = "program_id",
        context: Optional[List[str]] = None,
        min_len: Optional[int] = 4,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.context = context or ["date"]
        self.min_len = min_len

    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:

        feature_cols = [self.item_col] + self.context

        def mask(x: pd.DataFrame):
            return [list(x[col]) for col in feature_cols]

        sequences: pd.DataFrame = (
            X.set_index(self.user_col).groupby(level=-1).apply(mask).apply(pd.Series)
        )
        sequences.columns = feature_cols

        sequences = sequences.loc[
            sequences.loc[:, self.item_col].apply(len) >= self.min_len
        ]

        return sequences
