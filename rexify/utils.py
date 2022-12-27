from pathlib import Path

from rexify.dataclasses import Schema


def get_target_id(schema: Schema, target: str):
    return getattr(schema, target).id


def get_target_feature(schema: Schema, target: str, type_: str):
    def mask(x: tuple):
        return x[1] == type_

    return list(
        map(lambda x: x[0], filter(mask, getattr(schema, target).to_dict().items()))
    )


def make_dirs(*args):
    for dir_ in args:
        Path(dir_).mkdir(parents=True, exist_ok=True)
