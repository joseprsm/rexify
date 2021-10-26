from typing import List

import os
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.types import Artifact, artifact_utils


def load(artifact_list: List[Artifact], model_name: str = 'Format-Serving'):
    model_uri: str = artifact_utils.get_single_uri(artifact_list)
    model: tf.keras.Model = tf.keras.models.load_model(os.path.join(model_uri, model_name))
    return model


def export(artifact_list: List[Artifact], model: tf.keras.Model, model_name: str, **kwargs):
    output_dir = artifact_utils.get_single_uri(artifact_list)
    output_uri = os.path.join(output_dir, model_name)
    fileio.makedirs(os.path.dirname(output_uri))
    model.save(output_uri, save_format='tf', **kwargs)
