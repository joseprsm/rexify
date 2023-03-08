from sklearn.base import TransformerMixin


class CustomTransformer(tuple):
    def __new__(
        cls, target: str, transformer: TransformerMixin, features: list[str]
    ) -> tuple:
        name = f"{target}_{''.join([f[0] for f in features])}_customTransformer"
        return tuple.__new__(CustomTransformer, (name, transformer, features))
