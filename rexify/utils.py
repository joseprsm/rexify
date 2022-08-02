from typing import Any, List


def flatten(xss: List[List[Any]]):
    return [x for xs in xss for x in xs]


def get_target_id(schema: dict, target: str):
    def mask(x: tuple):
        return x[1] == "id"

    return list(filter(mask, schema[target].items()))[0][0]


def get_target_ids(schema: dict):
    return [get_target_id(schema, target) for target in ["user", "item"]]
