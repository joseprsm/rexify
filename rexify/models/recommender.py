import tensorflow as tf
import tensorflow_recommenders as tfrs


class Recommender(tfrs.Model):

    """
    Main Recommender Model.

    Args:
        query_bins: number of hashing bins to use for the query model
        candidate_bins: number of hashing bins to use for the candidate model
        output_dim: the output embedding dimension of both query and candidate models
    """

    def __init__(self, query_bins, candidate_bins, output_dim: int = 32):
        super().__init__()
        self.query_model: tf.keras.Model = self.set_embedding_model(query_bins, output_dim)
        self.candidate_model: tf.keras.Model = self.set_embedding_model(candidate_bins, output_dim)
        self.task: tfrs.tasks.Task = tfrs.tasks.Retrieval()  # TODO: add candidates to compute accuracy metrics

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embeddings, candidate_embeddings = self(inputs)
        loss = self.task(query_embeddings, candidate_embeddings)  # TODO: add support for multiple tasks
        # TODO: add event weights
        return loss

    # TODO: specify input signature
    def call(self, inputs, training=None, mask=None):
        # TODO: move inputs filters to tower models
        query_embeddings: tf.Tensor = self.query_model(inputs['userId'])
        candidate_embeddings: tf.Tensor = self.candidate_model(inputs['itemId'])
        return query_embeddings, candidate_embeddings

    def get_config(self):
        return {}  # TODO: add config

    @staticmethod
    def set_embedding_model(num_bins: int, output_dim: int = 32) -> tf.keras.Model:
        """Placeholder model for query and candidate model"""
        # TODO: create tower models
        return tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Hashing(num_bins=num_bins),
            tf.keras.layers.Embedding(input_dim=num_bins, output_dim=output_dim)
        ])


def build(query_bins: int,
          candidate_bins: int,
          output_dims: int = 32,
          learning_rate: float = 0.2) -> Recommender:
    model = Recommender(query_bins, candidate_bins, output_dims)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))
    return model
