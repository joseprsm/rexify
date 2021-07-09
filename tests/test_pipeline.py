import os

# from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from rexify import pipeline

PIPELINE_NAME = "rexify_test"
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)


ppl = pipeline.build(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    data_root='data',
    run_fn='rexify.train.run_fn',
    serving_model_dir=SERVING_MODEL_DIR,
    metadata_path=METADATA_PATH)


def test_pipeline_components():
    components = ['CsvExampleGen', 'Trainer', 'Pusher']
    assert len(ppl.components) > 0

    for i in range(len(components)):
        component = components[i]
        assert ppl.components[i].component_type.split('.')[-1] == component


# def test_pipeline_run():
#     current_model_list = os.listdir(SERVING_MODEL_DIR)
#     LocalDagRunner().run(ppl)
#     new_model_list = os.listdir(SERVING_MODEL_DIR)
#     assert len(new_model_list) > len(current_model_list)
