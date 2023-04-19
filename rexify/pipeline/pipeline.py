import os
from pathlib import Path

from kfp.components import load_component_from_file, load_component_from_url
from kfp.v2.compiler import Compiler
from kfp.v2.dsl import pipeline


BASE_PATH = Path("rexify") / "components"
PIPELINE_NAME = os.environ.get("PIPELINE_NAME")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")


def _load_component(task: str):
    try:
        return load_component_from_file(BASE_PATH / task / "component.yaml")
    except FileNotFoundError:
        return load_component_from_url(
            f"https://raw.githubusercontent.com/joseprsm/rexify/main/rexify/components/{task}/component.yaml"
        )


download_op = _load_component("download")
copy_op = _load_component("copy")
load_op = _load_component("load")
train_op = _load_component("train")
index_op = _load_component("index")
retrieval_op = _load_component("retrieval")


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline_fn(
    event_uri: str = None,
    schema: dict = None,
    epochs: int = 100,
):
    events_task = download_op(input_uri=event_uri or os.environ.get("INPUT_DATA_URL"))

    schema_task = copy_op(content=schema)

    load_task = load_op(
        events=events_task.outputs["data"],
        schema=schema_task.outputs["data"],
    )

    train_task = train_op(
        train_data=load_task.outputs["train"],
        feat=load_task.outputs["feat"],
        epochs=epochs,
    )

    index_task = index_op(
        items=load_task.outputs["items"],
        model=train_task.outputs["model"],
        schema=schema_task.outputs["data"],
    )

    retrieval_task = retrieval_op(  # noqa: F841
        users=load_task.outputs["users"],
        schema=schema_task.outputs["data"],
        index=index_task.outputs["index"],
        model=train_task.outputs["model"],
    )


def compile_(package_path: str = "pipeline.json", **ppl_params):
    Compiler().compile(
        pipeline_func=pipeline_fn,
        package_path=package_path,
        pipeline_parameters=ppl_params,
    )


if __name__ == "__main__":
    compile_()
