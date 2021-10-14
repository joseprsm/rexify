import tensorflow as tf

from rexify.models.base import BaseModel


class BaseModelTest(tf.test.TestCase):

    def testGetConfig(self):
        model = BaseModel([64, 32])
        config = model.get_config()
        self.assertIsInstance(config, dict)
        self.assertIsNotNone(config)
        self.assertNotEqual(config, dict())
        self.assertAllInSet(['layer_sizes'], list(config.keys()))

    def testConfigWrongSizes(self):
        pass

    def testEmptyLayerSizes(self):
        pass


if __name__ == '__main__':
    tf.test.main()
