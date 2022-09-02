import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer

from rexify.features.base import HasSchemaInput
from rexify.utils import get_first, get_target_id


class TfDatasetGenerator(HasSchemaInput):

    _ppl: ColumnTransformer

    def make_dataset(self, X) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.map(self._get_header_fn())
        return ds

    def _get_header_fn(self):
        user_id = get_target_id(self.schema, "user")[0]
        item_id = get_target_id(self.schema, "item")[0]

        (
            user_id_idx,
            user_features_idx,
            item_id_idx,
            item_features_idx,
            context_features_idx,
            rank_features_idx,
        ) = self._get_indices()

        def add_header(x):
            header = {
                "query": {
                    user_id: tf.gather(x, user_id_idx)[0],
                },
                "candidate": {
                    item_id: tf.gather(x, item_id_idx)[0],
                },
            }

            header["query"]["user_features"] = (
                tf.gather(x, user_features_idx)
                if len(user_features_idx) != 0
                else tf.constant([])
            )

            header["query"]["context_features"] = (
                tf.gather(x, context_features_idx)
                if len(context_features_idx) != 0
                else tf.constant([])
            )

            header["candidate"]["item_features"] = (
                tf.gather(x, item_features_idx)
                if len(item_features_idx) != 0
                else tf.constant([])
            )

            header["rank"] = (
                tf.gather(x, rank_features_idx)
                if len(rank_features_idx) != 0
                else tf.constant([])
            )

            return header

        return add_header

    def _get_indices(self):
        feature_names = pd.Series(self._ppl.get_feature_names_out())
        target_feature_names: pd.Series = feature_names.str.split("_").map(get_first)
        pipeline_names: pd.Series = feature_names.str.split("_").map(lambda x: x[1])
        id_pipeline_mask: pd.Series = pipeline_names == "idPipeline"

        def get_target_indices(target):
            feature_mask: pd.Series = target_feature_names == target

            def apply_mask(mask) -> list:
                return (
                    np.argwhere(np.logical_and(mask, feature_mask.values))
                    .reshape(-1)
                    .tolist()
                )

            id_list = apply_mask(id_pipeline_mask.values) if target != "context" else []
            features_list = apply_mask(~id_pipeline_mask.values)

            return id_list, features_list

        user_ids, user_features = get_target_indices("user")
        item_ids, item_features = get_target_indices("item")
        _, context_features = get_target_indices("context")
        _, rank_features = get_target_indices("rank")

        return (
            user_ids,
            user_features,
            item_ids,
            item_features,
            context_features,
            rank_features,
        )
