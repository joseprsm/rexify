import json

import tensorflow as tf
from tfx.types import standard_artifacts, channel_utils

from . import component


class LookupGenComponentTest(tf.test.TestCase):

    def setUp(self):
        super(LookupGenComponentTest, self).setUp()
        examples = standard_artifacts.Examples()
        examples.split_names = json.dumps(['train'])
        self._examples = channel_utils.as_channel([examples])
        self._model = channel_utils.as_channel([standard_artifacts.Model()])
        self._lookup_model = channel_utils.as_channel([standard_artifacts.Model()])

    def testConstruct(self):
        lookup_component = component.LookupGen(
            examples=self._examples,
            model=self._model,
            lookup_model=self._lookup_model,
            query_model='candidate_model',
            feature_key='itemId',
            schema=json.dumps({'itemId': 'x'}))

        self.assertEqual(standard_artifacts.Examples.TYPE_NAME, lookup_component.inputs['examples'].type_name)
        self.assertIn('train', json.loads(lookup_component.inputs['examples'].get()[0].split_names))
        self.assertEqual(standard_artifacts.Model.TYPE_NAME, lookup_component.inputs['model'].type_name)
        self.assertEqual(standard_artifacts.Model.TYPE_NAME, lookup_component.outputs['lookup_model'].type_name)


if __name__ == '__main__':
    tf.test.main()
