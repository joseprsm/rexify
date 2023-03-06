from pathlib import Path


def _get_target(schema, target: str):
    return getattr(schema, target).to_dict() if type(schema) != dict else schema[target]


def get_target_id(schema, target: str) -> list[str]:
    if type(schema) != dict:
        return [getattr(schema, target).id]
    return [k for k, v in schema[target].items() if v == "id"]


def get_target_feature(schema, target: str, type_: str):
    def mask(x: tuple):
        return x[1] == type_

    schema_dict = _get_target(schema, target)
    return list(map(lambda x: x[0], filter(mask, schema_dict.items())))


def make_dirs(*args):
    for dir_ in args:
        Path(dir_).mkdir(parents=True, exist_ok=True)
