from pathlib import Path
from typing import Any, List


def flatten(xss: List[List[Any]]):
    return [x for xs in xss for x in xs]


def get_target_id(schema: dict, target: str):
    return get_target_feature(schema, target, "id")


def get_target_feature(schema: dict, target: str, type_: str):
    def mask(x: tuple):
        return x[1] == type_

    return list(map(get_first, filter(mask, schema[target].items())))


def make_dirs(*args):
    for dir_ in args:
        Path(dir_).mkdir(parents=True, exist_ok=True)


def get_first(x: tuple):
    return x[0]


def get_schema_features(
    schema: dict[str, dict[str, Any] | list[dict[str, str | float]]]
):
    return flatten([list(v.keys()) for k, v in schema.items() if k != "rank"])


def get_ranking_features(
    schema: dict[str, dict[str, Any] | list[dict[str, str | float]]]
):
    return [feat["name"] for feat in schema["rank"]]
