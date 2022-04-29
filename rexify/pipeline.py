import os
import kfp

from kfp.components import load_component_from_file

BASE_PATH = os.path.join("rexify", "components")
KFP_HOST = os.environ.get("KFP_HOST", "http://localhost:3000")


def _load_component(task: str):
    return load_component_from_file(os.path.join(BASE_PATH, task, "component.yaml"))


preprocess_op = _load_component("preprocess")
train_op = _load_component("train")


# noinspection PyUnusedLocal
@kfp.dsl.pipeline()
def pipeline_fn():
    preprocess_task = preprocess_op()
    train_task = train_op()


client = kfp.Client(host=KFP_HOST)
client.create_run_from_pipeline_func(pipeline_fn, arguments={})
