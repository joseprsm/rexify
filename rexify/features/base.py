from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    @abstractmethod
    def transform(self, X):
        pass
