import json

import tensorflow as tf
from tfx.types import standard_artifacts, channel_utils

from . import component


class ScaNNGenComponentTest(tf.test.TestCase):
    def setUp(self):
        super(ScaNNGenComponentTest, self).setUp()
        candidates = standard_artifacts.Examples()
        candidates.split_names = json.dumps(["train"])
        self._candidates = channel_utils.as_channel([candidates])
        self._model = channel_utils.as_channel([standard_artifacts.Model()])
        self._lookup_model = channel_utils.as_channel([standard_artifacts.Model()])
        self._index = channel_utils.as_channel([standard_artifacts.Model()])

    def testConstruct(self):
        scann_component = component.ScaNNGen(
            candidates=self._candidates,
            model=self._model,
            lookup_model=self._lookup_model,
            index=self._index,
            schema=json.dumps({"userId": 1, "itemId": 42}),
            feature_key="userId",
        )

        self.assertEqual(
            standard_artifacts.Examples.TYPE_NAME,
            scann_component.inputs["candidates"].type_name,
        )
        self.assertIn(
            "train",
            json.loads(scann_component.inputs["candidates"].get()[0].split_names),
        )
        self.assertEqual(
            standard_artifacts.Model.TYPE_NAME,
            scann_component.inputs["model"].type_name,
        )
        self.assertEqual(
            standard_artifacts.Model.TYPE_NAME,
            scann_component.inputs["lookup_model"].type_name,
        )
        self.assertEqual(
            standard_artifacts.Model.TYPE_NAME,
            scann_component.outputs["index"].type_name,
        )


if __name__ == "__main__":
    tf.test.main()
