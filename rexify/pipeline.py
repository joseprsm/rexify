import os
import kfp

from kfp.components import load_component_from_file
from kfp.onprem import mount_pvc

BASE_PATH = os.path.join("rexify", "components")
KFP_HOST = os.environ.get("KFP_HOST", "http://localhost:3000")


def _load_component(task: str):
    return load_component_from_file(os.path.join(BASE_PATH, task, "component.yaml"))


preprocess_op = _load_component("preprocess")
train_op = _load_component("train")
index_op = _load_component("index")
deploy_op = _load_component("deploy")


# noinspection PyUnusedLocal
@kfp.dsl.pipeline()
def pipeline_fn():
    preprocess_task = preprocess_op()
    preprocess_task.apply(mount_pvc('rexify-pvc', 'data-vol', '/mnt/data'))

    train_task = train_op(input_dir=preprocess_task.outputs["output_dir"])
    train_task.apply(mount_pvc('rexify-pvc', 'data-vol', '/mnt/data'))

    index_task = index_op(model_dir=train_task.outputs["model_dir"])
    index_task.apply(mount_pvc('rexify-pvc', 'data-vol', '/mnt/data'))

    deploy_task = deploy_op(index_dir=index_task.outputs["index_dir"])


if __name__ == '__main__':
    client = kfp.Client(host=KFP_HOST)
    client.create_run_from_pipeline_func(pipeline_fn, arguments={})
