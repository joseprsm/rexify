from typing import Dict, Text, List, Any, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2

from rexify import utils
from rexify.models import Recommender, EmbeddingLookup


def _get_models(input_dict) -> Tuple[Recommender, EmbeddingLookup]:
    """Retrieves the two models used for the ScaNN building component"""
    model: Recommender = utils.load_model(input_dict["model"])
    lookup_model: EmbeddingLookup = utils.load_model(
        input_dict["lookup_model"], model_name="lookup_model"
    )
    return model, lookup_model


class Executor(base_executor.BaseExecutor):
    def Do(
        self,
        input_dict: Dict[Text, List[types.Artifact]],
        output_dict: Dict[Text, List[types.Artifact]],
        exec_properties: Dict[Text, Any],
    ) -> Optional[execution_result_pb2.ExecutorOutput]:

        model, lookup_model = _get_models(input_dict)
        candidates = utils.read_split_examples(
            input_dict["candidates"], schema=exec_properties["schema"]
        )
        vocabulary: np.array = (
            tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x["itemId"])])
            .predict(candidates)
            .reshape(-1)
        )

        # todo: fix training input data
        candidate_embeddings = np.concatenate(
            list(
                candidates.batch(1)
                .batch(512)
                .map(model.candidate_model)
                .as_numpy_iterator()
            )
        )
        candidate_embeddings = candidate_embeddings.reshape(
            (-1, candidate_embeddings.shape[-1])
        )
        sample_query = tf.constant([lookup_model.get_config()["sample_query"]])

        scann = self._generate_ann(
            lookup_model=lookup_model,
            embeddings=candidate_embeddings,
            candidates=vocabulary,
            sample_query=sample_query,
        )

        utils.export_model(
            output_dict["index"],
            scann,
            "index",
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]),
        )

    @staticmethod
    def _generate_ann(
        lookup_model: EmbeddingLookup,
        embeddings: Union[tf.Tensor, np.ndarray],
        candidates: Union[tf.Tensor, np.ndarray],
        sample_query: Dict[str, tf.Tensor],
        **kwargs
    ) -> tf.keras.Model:
        """Generates a ScaNN TensorFlow model"""
        scann = tfrs.layers.factorized_top_k.ScaNN(lookup_model, **kwargs)
        scann.index(embeddings, candidates)
        _ = scann(sample_query)
        return scann
