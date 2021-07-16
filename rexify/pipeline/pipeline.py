from typing import List, Text, Dict

import json

from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.components import CsvExampleGen, Trainer, Pusher
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.base import executor_spec
from tfx.proto import trainer_pb2
from tfx.proto.pusher_pb2 import PushDestination

from rexify.pipeline.components import LookupGen, ScaNNGen


def build(
        pipeline_name: Text,
        pipeline_root: Text,
        data_root: Text,
        items_root: Text,
        schema: Dict[Text, Text],
        run_fn: Text,
        serving_model_dir: Text,
        metadata_path: Text
) -> Pipeline:

    components: List[BaseComponent] = list()

    example_gen = CsvExampleGen(input_base=data_root).with_id('event_gen')
    components.append(example_gen)

    trainer_args = dict(
        run_fn=run_fn,
        examples=example_gen.outputs.examples,
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=1000),
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor))
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    item_gen = CsvExampleGen(input_base=items_root).with_id('item_gen')
    components.append(item_gen)

    lookup_gen = LookupGen(
        examples=item_gen.outputs.examples,
        model=trainer.outputs.model,
        query_model='candidate_model',
        feature_key='itemId',
        schema=json.dumps(schema))
    components.append(lookup_gen)

    scann_gen = ScaNNGen(
        candidates=item_gen.outputs.examples,
        model=trainer.outputs.model,
        lookup_model=lookup_gen.outputs.lookup_model,
        schema=json.dumps(schema),
        feature_key='itemId')
    components.append(scann_gen)

    pusher_args = dict(
        model=scann_gen.outputs.index,
        push_destination=PushDestination(
            filesystem=PushDestination.Filesystem(
                base_directory=serving_model_dir)))
    pusher = Pusher(**pusher_args)
    components.append(pusher)

    return Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=sqlite_metadata_connection_config(metadata_path))
