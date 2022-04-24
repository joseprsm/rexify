import os

# from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.pipeline import Pipeline

from rexify import pipeline

PIPELINE_NAME = "rexify_test"
PIPELINE_ROOT = os.path.join("pipelines", PIPELINE_NAME)
METADATA_PATH = os.path.join("metadata", PIPELINE_NAME, "metadata.db")
SERVING_MODEL_DIR = os.path.join("serving_model", PIPELINE_NAME)


ppl = pipeline.build(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    data_root="data/events",
    items_root="data/items",
    run_fn="rexify.train.run_fn",
    schema={"userId": "", "itemId": ""},
    serving_model_dir=SERVING_MODEL_DIR,
    metadata_path=METADATA_PATH,
)


def test_pipeline_components():
    assert isinstance(ppl, Pipeline)
    assert len(ppl.components) > 0
