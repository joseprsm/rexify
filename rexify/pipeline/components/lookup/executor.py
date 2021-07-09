from typing import Dict, Text, List, Any, Optional, Tuple

import os
import numpy as np
import tensorflow as tf

from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact_utils, Artifact

from rexify import utils
from rexify.models import Recommender, EmbeddingLookupModel


class Executor(base_executor.BaseExecutor):

    def Do(
            self,
            input_dict: Dict[Text, List[Artifact]],
            output_dict: Dict[Text, List[Artifact]],
            exec_properties: Dict[Text, Any]
    ) -> Optional[execution_result_pb2.ExecutorOutput]:
        self._log_startup(input_dict, output_dict, exec_properties)

        examples_artifact: Artifact = artifact_utils.get_single_uri(input_dict['examples'])
        examples: tf.data.Dataset = utils.read_split_examples(examples_artifact)

        model_uri: Text = artifact_utils.get_single_uri(input_dict['model'])
        model: Recommender = tf.saved_model.load(model_uri)

        lookup_model = EmbeddingLookupModel(
            *self.get_lookup_params(
                examples=examples,
                model=model,
                query_model=exec_properties['query_model'],
                feature_key=exec_properties['feature_key']))

        output_dir = artifact_utils.get_single_uri(output_dict['lookup_model'])
        output_uri = os.path.join(output_dir, 'lookup_model')

        fileio.makedirs(os.path.dirname(output_uri))
        lookup_model.save(output_uri, save_format='tf')

    @staticmethod
    def get_lookup_params(examples, model, query_model, feature_key) -> Tuple[np.array, np.array]:

        batched_examples = examples.batch(512)
        embedding_model: tf.keras.Model = getattr(model, query_model)
        embeddings: np.array = embedding_model.predict(batched_examples.map(lambda x: x[feature_key]))

        vocabulary: np.array = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[feature_key])
        ]).predict(batched_examples)

        return vocabulary, embeddings
