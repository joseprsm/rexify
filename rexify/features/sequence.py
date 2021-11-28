from typing import Dict, Union

import tensorflow as tf


def slide_transform(data: tf.data.Dataset,
                    schema: Dict[str, Union[str, dict]],
                    window_size: int = 3):

    def key_fn(elem: Dict[str, tf.Tensor]):
        return elem['userId']

    def reduce_fn(_, window):
        return get_sliding_batches(window, window_size)

    sliding_window = data.group_by_window(
            key_func=key_fn,
            reduce_func=reduce_fn,
            window_size=window_size)

    return sliding_window.\
        map(_filter_by_keys(schema)).\
        filter(lambda x: len(x['sequence']) == window_size - 1)


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
            'itemId': x['itemId'][-1]}

        header.update({key: x[key][-1] for key in keys if key not in ['userId', 'itemId', 'sequence']})
        return header

    return filter_fn


# this fn doesn't work properly in `group_by_window`
# for some reason. fix later.
def get_sliding_batches(window: tf.data.Dataset,
                        window_size: int = 3):

    ds = window.batch(window_size)
    for i in range(1, window_size):
        ds = ds.concatenate(
            window.skip(i).batch(window_size))

    return ds
