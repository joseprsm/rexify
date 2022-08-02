from typing import Optional

import numpy as np

from rexify.features.base import BaseTransformer


class CyclicalTransformer(BaseTransformer):
    def __init__(
        self, num_bins: int = 24, fn: Optional[str] = "sin", drop_columns: bool = True
    ):
        super().__init__()
        self.num_bins = num_bins
        self.drop_columns = drop_columns
        self.fn = fn
        self.fn_ = getattr(np, fn)

    def transform(self, X):
        transformed = self.fn_((2.0 * X * np.pi / self.num_bins))
        transformed = (
            np.concatenate([X, transformed], axis=1)
            if not self.drop_columns
            else transformed
        )
        return transformed
