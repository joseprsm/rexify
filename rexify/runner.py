from typing import List, Optional

import os
import configparser

import tfx
import tfx.proto

from tfx.components.base import executor_spec
from tfx.components.base.base_component import BaseComponent
from tfx.components.trainer import executor as trainer_executor

from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.pipeline import Pipeline

from kfp import onprem

from rexify.pipeline.lookup import LookupGen
from rexify.pipeline.scann import ScaNNGen

config = configparser.ConfigParser()
config.read(os.environ.get('CONFIG_FILE', 'config.ini'))

RUN_FN = config.get('PIPELINE', 'run_fn')
PIPELINE_NAME = config.get('PIPELINE', 'pipeline_name')
PIPELINE_ROOT = config.get('PIPELINE', 'pipeline_root')
DATA_ROOT = config.get('PIPELINE', 'data_root')
OUTPUT_BUCKET = config.get('PIPELINE', 'output_bucket')

# todo: make this a parameter
schema = {"itemId": "categorical", 'userId': 'categorical'}

mount_volume_op = onprem.mount_pvc('refixy-pvc', 'shared-volume', OUTPUT_BUCKET)


def create_pipeline(events_root: str,
                    items_root: str,
                    users_root: str,
                    serving_model_dir: str,
                    run_fn: Optional[str] = RUN_FN,
                    pipeline_name: Optional[str] = PIPELINE_NAME,
                    pipeline_root: Optional[str] = PIPELINE_ROOT,
                    enable_cache: Optional[bool] = True):

    components: List[BaseComponent] = list()

    event_gen = tfx.components.CsvExampleGen(input_base=events_root).with_id('EventGen')
    components.append(event_gen)

    trainer_args = dict(
        run_fn=run_fn,
        examples=event_gen.outputs['examples'],
        train_args=tfx.proto.trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=tfx.proto.trainer_pb2.EvalArgs(num_steps=1000),
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor))
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    user_gen = tfx.components.CsvExampleGen(input_base=users_root).with_id('UserGen')
    components.append(user_gen)

    lookup_gen = LookupGen(
        examples=user_gen.outputs['examples'],
        model=trainer.outputs['model'],
        query_model='query_model',
        feature_key='userId',
        schema=str({'userId': 'categorical'}))
    components.append(lookup_gen)

    item_gen = tfx.components.CsvExampleGen(input_base=items_root).with_id('ItemGen')
    components.append(item_gen)

    scann_gen = ScaNNGen(
        candidates=item_gen.outputs['examples'],
        model=trainer.outputs['model'],
        lookup_model=lookup_gen.outputs['lookup_model'],
        schema=str({'itemId': 'categorical'}),
        feature_key='itemId')
    components.append(scann_gen)

    pusher_args = dict(
        model=scann_gen.outputs['index'],
        push_destination=tfx.proto.pusher_pb2.PushDestination(
            filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
    pusher = tfx.components.Pusher(**pusher_args)
    components.append(pusher)

    return Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache)


def get_kubeflow_metadata_config() -> kubeflow_pb2.KubeflowMetadataConfig:
    metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    metadata_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    metadata_config.mysql_db_service_port.environment_variable = 'MYSQL_SERVICE_PORT'
    metadata_config.mysql_db_name.value = 'metadb'
    metadata_config.mysql_db_user.value = 'root'
    metadata_config.mysql_db_password.value = ''
    return metadata_config


if __name__ == '__main__':
    data_root = os.path.join(OUTPUT_BUCKET, 'data')
    pipeline_args = dict(
        events_root=os.environ.get('EVENTS_ROOT', os.path.join(data_root, 'events')),
        items_root=os.environ.get('ITEMS_ROOT', os.path.join(data_root, 'items')),
        users_root=os.environ.get('USERS_ROOT', os.path.join(data_root, 'users')),
        serving_model_dir=os.path.join(PIPELINE_ROOT, 'serving_model'))
    pipeline = create_pipeline(**pipeline_args)
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=get_kubeflow_metadata_config(),
        pipeline_operator_funcs=[mount_volume_op])
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)


