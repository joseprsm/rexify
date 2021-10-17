from typing import Dict, Text, List, Any, Optional, Tuple

import tensorflow as tf
import tensorflow_recommenders as tfrs

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2

from rexify import utils
from rexify.models import Recommender, EmbeddingLookup


def generate_ann(
        lookup_model: EmbeddingLookup,
        embeddings: tf.data.Dataset,
        candidates: tf.data.Dataset,
        sample_query: Text, **kwargs) -> tf.keras.Model:
    """Generates a ScaNN TensorFlow model"""
    scann = tfrs.layers.factorized_top_k.ScaNN(lookup_model, **kwargs)
    scann.index(embeddings, candidates.map(lambda x: x['itemId']))
    _ = scann(tf.constant([sample_query]))
    return scann


def _get_models(input_dict) -> Tuple[Recommender, EmbeddingLookup]:
    """Retrieves the two models used for the ScaNN building component"""
    model: Recommender = utils.load_model(input_dict['model'])
    lookup_model: EmbeddingLookup = utils.load_model(
        input_dict['lookup_model'], model_name='lookup_model')
    return model, lookup_model


def _get_candidates(input_dict, exec_properties):
    candidates: tf.data.Dataset = utils.read_split_examples(
        input_dict['candidates'], schema=exec_properties['schema'])
    return candidates


class Executor(base_executor.BaseExecutor):

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> Optional[execution_result_pb2.ExecutorOutput]:

        model, lookup_model = _get_models(input_dict)
        # candidates must be of shape ()
        candidates = _get_candidates(input_dict, exec_properties)

        candidate_embeddings: tf.data.Dataset = candidates.batch(512).map(model.candidate_model)
        if len(list(candidate_embeddings.take(1))[0].shape) > 2:
            candidate_embeddings = candidate_embeddings.unbatch()

        # retrieving the `sample_query` from the EmbeddingLookup
        # model config file
        sample_query = lookup_model.get_config()['sample_query']

        scann = generate_ann(
            lookup_model=lookup_model,
            candidates=candidates,
            embeddings=candidate_embeddings,
            sample_query=sample_query,
            **exec_properties.get('custom_config', {}))

        utils.export_model(
            output_dict['index'], scann, 'index',
            options=tf.saved_model.SaveOptions(namespace_whitelist=['Scann']))
