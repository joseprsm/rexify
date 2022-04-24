from unittest import mock

import os
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.types import standard_artifacts

from . import executor
from tests.utils import mock_recommender, mock_candidates


def get_mock_model(*_):
    return mock_recommender


def get_mock_candidates(*_):
    return mock_candidates


class LookupGenExecutorTest(tf.test.TestCase):
    def testGetLookupParams(self):
        params = executor.Executor.get_lookup_params(
            mock_candidates, mock_recommender, "candidate_model", "itemId"
        )
        self.assertEqual(len(params[0]), len(mock_candidates))
        self.assertEqual(params[1].shape[-1], 32)

    @mock.patch.multiple(
        executor, _get_examples=get_mock_candidates, _get_recommender=get_mock_model
    )
    def testDo(self):
        lookup_model = standard_artifacts.Model()
        lookup_model.uri = self.get_temp_dir()
        output_dict = {"lookup_model": [lookup_model]}

        exec_properties = {"query_model": "candidate_model", "feature_key": "itemId"}
        lookup_gen = executor.Executor()
        lookup_gen.Do({}, output_dict, exec_properties)

        output_file = os.path.join(lookup_model.uri, "lookup_model")
        self.assertTrue(fileio.exists(output_file))


if __name__ == "__main__":
    tf.test.main()
