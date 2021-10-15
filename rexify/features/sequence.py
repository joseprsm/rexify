from typing import Dict, Union

import tensorflow as tf


def slide_transform(data: tf.data.Dataset,
                    schema: Dict[str, Union[str, dict]],
                    window_size: int = 3):

    def key_fn(elem: Dict[str, tf.Tensor]):
        return elem['userId']

    def reduce_fn(_, window):
        return window.batch(batch_size=window_size)

    sliding_window = data.apply(
        tf.data.experimental.group_by_window(
            key_func=key_fn,
            reduce_func=reduce_fn,
            window_size=window_size))

    return sliding_window.map(_filter_by_keys(schema))


def _filter_by_keys(schema: Dict[str, Union[dict, str]]):

    def get_custom_keys(target_key: str):
        return [
            k for k in schema[target_key].keys()
            if k != f'{target_key}Id']

    keys = [
        key for custom_key_list in [
            get_custom_keys(target)
            for target in ['user', 'item']
        ] for key in custom_key_list]

    def filter_fn(x: Dict[str, tf.Tensor]):
        header = {
            'userId': x['userId'][0],
            'sequence': x['itemId'][:-1],
            'date': x['date'][-1],
            'target': x['itemId'][-1]}

        header.update({key: x[key][-1] for key in keys})
        return header

    return filter_fn
