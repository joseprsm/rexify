from typing import List, Dict, Optional

import json

import tfx
import tfx.proto

from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.base import executor_spec

from rexify.pipeline import steps as rexify_components


def build(pipeline_name: str,
          pipeline_root: str,
          schema: Dict[str, Dict[str, str]],
          run_fn: str,
          serving_model_dir: str,
          metadata_path: str,
          enable_cache: Optional[bool] = True) -> Pipeline:

    components: List[BaseComponent] = list()

    event_downloader = rexify_components.Downloader(
        table='events',
        features_table='events_features'
    ).with_id('EventDownloader')
    components.append(event_downloader)

    event_gen = tfx.components.CsvExampleGen(
        input_base=event_downloader.outputs['output_path']
    ).with_id('EventGen')
    components.append(event_gen)

    user_downloader = rexify_components.Downloader(
        table='users',
        features_table='users_features'
    ).with_id('UserDownloader')
    components.append(user_downloader)

    user_gen = tfx.components.CsvExampleGen(
        input_base=user_downloader.outputs['output_path']
    ).with_id('UserGen')
    components.append(user_gen)

    item_downloader = rexify_components.Downloader(
        table='items',
        features_tables='items_features'
    ).with_id('ItemDownloader')
    components.append(item_downloader)

    item_gen = tfx.components.CsvExampleGen(
        input_base=item_downloader.outputs['output_path']
    ).with_id('ItemGen')
    components.append(item_gen)

    transform_args = dict()
    transform = tfx.components.Transform(**transform_args)
    components.append(transform)

    trainer_args = dict(
        run_fn=run_fn,
        examples=event_gen.outputs['examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=1000),
        eval_args=tfx.proto.EvalArgs(num_steps=1000),
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor))
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    lookup_gen = rexify_components.LookupGen(
        examples=item_gen.outputs['examples'],
        model=trainer.outputs['model'],
        query_model='candidate_model',
        feature_key='itemId',
        schema=json.dumps(schema))
    components.append(lookup_gen)

    scann_gen = rexify_components.ScaNNGen(
        candidates=item_gen.outputs['examples'],
        model=trainer.outputs['model'],
        lookup_model=lookup_gen.outputs['lookup_model'],
        schema=json.dumps(schema),
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
        enable_cache=enable_cache,
        metadata_connection_config=sqlite_metadata_connection_config(metadata_path))
