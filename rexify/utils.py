from pathlib import Path


def get_target_id(schema, target: str):
    schema_dict = (
        getattr(schema, target).to_dict() if type(schema) != dict else schema[target]
    )

    return [k for k, v in schema_dict.items() if v == "id"]


def get_target_feature(schema, target: str, type_: str):
    def mask(x: tuple):
        return x[1] == type_

    return list(
        map(lambda x: x[0], filter(mask, getattr(schema, target).to_dict().items()))
    )


def make_dirs(*args):
    for dir_ in args:
        Path(dir_).mkdir(parents=True, exist_ok=True)
