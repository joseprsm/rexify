import tensorflow as tf

from rexify.models.tower import Tower

from tests.utils import get_sample_schema, get_sample_params


class TowerTest(tf.test.TestCase):

    def setUp(self):
        super(TowerTest, self).setUp()
        self._tower_args = {
            'schema': get_sample_schema(),
            'params': get_sample_params(),
            'layer_sizes': [64, 32],
            'activation': 'relu'}
        self.model = Tower(**self._tower_args)

    def testGetConfig(self):
        config = self.model.get_config()
        self.assertIsInstance(config, dict)
        self.assertIsNotNone(config)
        self.assertNotEqual(config, dict())
        self.assertAllInSet(['layer_sizes'], list(config.keys()))


if __name__ == '__main__':
    tf.test.main()
