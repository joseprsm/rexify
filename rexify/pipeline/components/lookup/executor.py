from typing import Dict, Text, List, Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import Artifact

from rexify import utils
from rexify.models import Recommender, EmbeddingLookup


class Executor(base_executor.BaseExecutor):

    def Do(
            self,
            input_dict: Dict[Text, List[Artifact]],
            output_dict: Dict[Text, List[Artifact]],
            exec_properties: Dict[Text, Any]
    ) -> Optional[execution_result_pb2.ExecutorOutput]:
        self._log_startup(input_dict, output_dict, exec_properties)

        examples: tf.data.Dataset = utils.read_split_examples(input_dict['examples'], schema=exec_properties['schema'])
        model: Recommender = utils.load_model(input_dict['model'])

        lookup_model = EmbeddingLookup(
            *self.get_lookup_params(
                examples=examples,
                model=model,
                query_model=exec_properties['query_model'],
                feature_key=exec_properties['feature_key']))
        lookup_model([lookup_model.vocabulary[0]])

        utils.export_model(output_dict['lookup_model'], lookup_model, 'lookup_model')

    @staticmethod
    def get_lookup_params(examples, model, query_model, feature_key) -> Tuple[np.array, np.array]:

        batched_examples = examples.batch(512)
        embedding_model: tf.keras.Model = getattr(model, query_model)
        embeddings: np.array = embedding_model.predict(batched_examples.map(lambda x: x[feature_key]))

        vocabulary: np.array = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[feature_key])
        ]).predict(batched_examples).reshape(-1)

        return vocabulary, embeddings
