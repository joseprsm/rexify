import os

from kfp.v2.dsl import pipeline
from kfp.v2.compiler import Compiler
from kfp.components import load_component_from_file

BASE_PATH = os.path.join("rexify", "components")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")


def _load_component(task: str):
    return load_component_from_file(os.path.join(BASE_PATH, task, "component.yaml"))


download_op = _load_component("download")
load_op = _load_component("load")
train_op = _load_component("train")


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline_fn():
    events_downloader_task = download_op(input_uri=os.environ.get("INPUT_DATA_URL"))
    schema_downloader_task = download_op(input_uri=os.environ.get("SCHEMA_URL"))

    load_task = load_op(
        events=events_downloader_task.outputs["data"],
        schema=schema_downloader_task.outputs["data"],
    )

    train_task = train_op(
        train_data=load_task.outputs["train"],
        schema_path=schema_downloader_task.outputs["data"],
    )


if __name__ == "__main__":
    Compiler().compile(pipeline_func=pipeline_fn, package_path="pipeline.json")
