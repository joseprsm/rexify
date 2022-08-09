import tensorflow as tf

from rexify.utils import get_target_ids
from rexify.features.base import HasSchemaInput


class TfDatasetGenerator(HasSchemaInput):
    def make_dataset(self, X) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.map(self._get_header_fn(self.schema))
        return ds

    @staticmethod
    def _get_header_fn(schema):

        header = get_target_ids(schema)

        def add_header(x):
            return {
                "query": {header[0]: x[0]},
                "candidate": {header[1]: x[1]},
            }

        return add_header
