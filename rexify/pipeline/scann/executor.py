from typing import Dict, Text, List, Any, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2

from rexify import utils
from rexify.models import Recommender

SAMPLE_QUERY = '42'


def generate_ann(
        lookup_model: tf.keras.Model,
        embeddings: tf.data.Dataset,
        candidates: tf.data.Dataset,
        sample_query: str,
        feature_key: str, **kwargs) -> tf.keras.Model:
    """Generates a ScaNN TensorFlow model"""
    scann = tfrs.layers.factorized_top_k.ScaNN(lookup_model, **kwargs)
    # noinspection PyTypeChecker
    scann.index(embeddings, candidates.map(lambda x: x[feature_key]))
    _ = scann(tf.constant([sample_query]))
    return scann


class Executor(base_executor.BaseExecutor):

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> Optional[execution_result_pb2.ExecutorOutput]:

        model: Recommender = utils.load_model(input_dict['model'])
        candidates = utils.read_split_examples(input_dict['candidates'], schema=exec_properties['schema'])

        candidate_embeddings: tf.data.Dataset = candidates.batch(512).map(model.candidate_model)
        if len(list(candidate_embeddings.take(1))[0].shape) > 2:
            candidate_embeddings = candidate_embeddings.unbatch()

        scann = generate_ann(
            lookup_model=model.query_model,
            candidates=candidates,
            embeddings=candidate_embeddings,
            sample_query=SAMPLE_QUERY,
            feature_key=exec_properties['feature_key'],
            **exec_properties.get('custom_config', {}))

        utils.export_model(
            output_dict['index'], scann, 'index',
            options=tf.saved_model.SaveOptions(namespace_whitelist=['Scann']))
