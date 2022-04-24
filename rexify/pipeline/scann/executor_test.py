import os

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tfx.dsl.io import fileio

from tfx.types import standard_artifacts

from unittest import mock

from . import executor

from tests.utils import mock_candidates, mock_embeddings, mock_recommender, mock_lookup


def get_mock_models(*_):
    return mock_recommender, mock_lookup


def get_mock_candidates(*_):
    return mock_candidates


class ScannExecutorTest(tf.test.TestCase):
    def testGenerateANN(self):
        sample_query = list(mock_candidates.take(1))[0]["itemId"].numpy()
        scann = executor.generate_ann(
            mock_lookup,
            mock_embeddings,
            mock_candidates,
            str(sample_query),
            num_leaves=10,
        )
        self.assertIsInstance(scann, tfrs.layers.factorized_top_k.ScaNN)

    @mock.patch.multiple(
        executor, _get_models=get_mock_models, _get_candidates=get_mock_candidates
    )
    def testDo(self):
        index = standard_artifacts.Model()
        index.uri = self.get_temp_dir()
        output_dict = {"index": [index]}

        exec_properties = {"feature_key": "itemId", "custom_config": {"num_leaves": 10}}
        scann_gen = executor.Executor()
        scann_gen.Do({}, output_dict, exec_properties)

        output_file = os.path.join(index.uri, "index")
        self.assertTrue(fileio.exists(output_file))


if __name__ == "__main__":
    tf.test.main()
