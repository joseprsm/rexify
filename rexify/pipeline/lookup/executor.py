from typing import Dict, Text, List, Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import Artifact

from rexify import utils
from rexify.models import Recommender, EmbeddingLookup


def _get_examples(input_dict, exec_properties):
    return utils.read_split_examples(
        input_dict["examples"], schema=exec_properties["schema"]
    )


def _get_recommender(input_dict):
    return utils.load_model(input_dict["model"])


class Executor(base_executor.BaseExecutor):
    def Do(
        self,
        input_dict: Dict[Text, List[Artifact]],
        output_dict: Dict[Text, List[Artifact]],
        exec_properties: Dict[Text, Any],
    ) -> Optional[execution_result_pb2.ExecutorOutput]:
        self._log_startup(input_dict, output_dict, exec_properties)

        examples: tf.data.Dataset = _get_examples(input_dict, exec_properties)
        model: Recommender = _get_recommender(input_dict)

        lookup_model = EmbeddingLookup(
            *self.get_lookup_params(
                examples=examples,
                model=model,
                query_model=exec_properties["query_model"],
                feature_key=exec_properties["feature_key"],
            )
        )

        # call to build the model
        lookup_model([lookup_model.vocabulary[0]])

        utils.export_model(output_dict["lookup_model"], lookup_model, "lookup_model")

    @staticmethod
    def get_lookup_params(
        examples, model, query_model, feature_key
    ) -> Tuple[np.array, np.array]:
        """Retrieves and computes the parameters for the EmbeddingLookup Model."""

        # todo: fix training input data
        batched_examples = examples.batch(1).batch(512)
        embedding_model: tf.keras.Model = getattr(model, query_model)

        # using the .predict method of the models to return numpy arrays
        embeddings: np.array = np.concatenate(
            list(batched_examples.map(embedding_model).as_numpy_iterator())
        )

        vocabulary: np.array = (
            tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[feature_key])])
            .predict(batched_examples)
            .reshape(-1)
        )

        return vocabulary, embeddings
