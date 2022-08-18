from typing import Any, List


def flatten(xss: List[List[Any]]):
    return [x for xs in xss for x in xs]


def get_target_id(schema: dict, target: str):
    return get_target_feature(schema, target, "id")


def get_target_feature(schema: dict, target: str, type_: str):
    def mask(x: tuple):
        return x[1] == type_

    def get_first(x: tuple):
        return x[0]

    return list(map(get_first, filter(mask, schema[target].items())))
