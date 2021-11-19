import json
import os
import datetime as dt

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner, AirflowPipelineConfig

from rexify import pipeline

_airflow_config = {
    'schedule_interval': None,
    'start_date': dt.datetime.today()
}


RUN_FN = 'rexify.pipeline.train.run_fn'
PIPELINE_NAME = 'rexify'
PIPELINE_OUTPUT = 'output'
PIPELINE_OUTPUT = os.path.join('.', PIPELINE_OUTPUT)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

events = os.path.join('data', 'events')
items = os.path.join('data', 'items')

with open(os.path.join('data', 'schema.json'), 'r') as f:
    schema = json.load(f)

ppl = pipeline.build(
    run_fn=RUN_FN,
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_OUTPUT,
    metadata_path=METADATA_PATH,
    data_root=events,
    items_root=items,
    schema=schema,
    serving_model_dir=None,
    enable_cache=True)


DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(ppl)
