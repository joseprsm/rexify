from typing import List, Optional

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
          data_root: str,
          items_root: str,
          schema: str,
          run_fn: str,
          serving_model_dir: str,
          metadata_path: str,
          enable_cache: Optional[bool] = True) -> Pipeline:

    components: List[BaseComponent] = list()

    event_gen = tfx.components.CsvExampleGen(input_base=data_root).with_id('EventGen')
    components.append(event_gen)

    trainer_args = dict(
        run_fn=run_fn,
        examples=event_gen.outputs['examples'],
        train_args=tfx.proto.trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=tfx.proto.trainer_pb2.EvalArgs(num_steps=1000),
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor))
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    item_gen = tfx.components.CsvExampleGen(input_base=items_root).with_id('ItemGen')
    components.append(item_gen)

    lookup_gen = rexify_components.LookupGen(
        examples=item_gen.outputs['examples'],
        model=trainer.outputs['model'],
        query_model='candidate_model',
        feature_key='itemId',
        schema=json.loads(schema))
    components.append(lookup_gen)

    scann_gen = rexify_components.ScaNNGen(
        candidates=item_gen.outputs['examples'],
        model=trainer.outputs['model'],
        lookup_model=lookup_gen.outputs['lookup_model'],
        schema=json.loads(schema),
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
