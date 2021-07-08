from typing import List

from tfx.orchestration.pipeline import Pipeline
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, Trainer, Pusher
from tfx.proto.pusher_pb2 import PushDestination


def build(
        data_path,
        run_fn,
        serving_model_dir,
        pipeline_name,
        pipeline_root,
        metadata_connection_config,
        beam_pipeline_args
) -> Pipeline:

    components: List[BaseComponent] = list()

    example_gen = CsvExampleGen(input_base=data_path)
    components.append(example_gen)

    statistics_gen = StatisticsGen(examples=example_gen.outputs.examples)
    components.append(statistics_gen)

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs.statistics,
        infer_feature_shape=True)
    components.append(schema_gen)

    trainer_args = dict(
        run_fn=run_fn,
        schema=schema_gen.outputs.schema,
        examples=example_gen.outputs.examples)
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    pusher_args = dict(
        model=trainer.outputs.model,
        push_destination=PushDestination(
            filesystem=PushDestination.filesystem(
                base_directory=serving_model_dir)))
    pusher = Pusher(**pusher_args)
    components.append(pusher)

    return Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args)
