import tensorflow as tf

from rexify.models.tower import TowerModel


class CandidateModel(TowerModel):
    """Tower model responsible for computing the candidate representations

    Args:
        item_id (str): the item ID feature
        n_items (str): number possible values for the ID feature
        embedding_dim (int): output dimension of the embedding layer
        output_layers (list): number of neurons in each layer for the output model
        feature_layers (list): number of neurons in each layer for the feature model

    Examples:

    >>> from rexify.models.candidate import CandidateModel
    >>> model = CandidateModel('item_id', 15)
    >>> model({'item_id': tf.constant([1]), 'item_features': tf.constant([[1, 1, 1]])})
    <tf.Tensor: shape=(1, 32), dtype=float32, numpy=
    array([[...]], dtype=float32)>
    """

    def __init__(
        self,
        item_id: str,
        n_items: int,
        embedding_dim: int = 32,
        output_layers: list[int] = None,
        feature_layers: list[int] = None,
    ):
        super().__init__(item_id, n_items, embedding_dim, output_layers, feature_layers)

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        x = self.embedding_layer(inputs[self._id_feature])
        if inputs["item_features"].shape[1] != 0:
            feature_embedding = self.feature_model(inputs["item_features"])
            x = tf.concat([x, feature_embedding], axis=1)
        else:
            self.feature_model.build(input_shape=tf.TensorShape([]))
        x = self.output_model(x)
        return x

    def get_config(self):
        config = super().get_config()
        config["item_id"] = self._id_feature
        config["n_items"] = self._n_dims
        return config
