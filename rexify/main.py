from typing import Union, Optional

import os
import json
import click
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

RUN_FN = config.get('PIPELINE', 'run_fn')
PIPELINE_NAME = config.get('PIPELINE', 'pipeline_name')
PIPELINE_OUTPUT = os.environ.get('REXIFY_PIPELINE_OUTPUT', config.get('PIPELINE', 'pipeline_output'))
PIPELINE_OUTPUT = os.path.join('.', PIPELINE_OUTPUT)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
BACKEND_CONFIG = os.environ.get('REXIFY_BACKEND', config.get('PIPELINE', 'backend'))
ENABLE_CACHE = False


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.option('--events', required=True, help='Path to events CSV')
@click.option('--users', help='Path to users CSV')
@click.option('--items', help='Path to items CSV')
@click.option('--schema', help='Path to schema JSON')
@click.option('--output', help='Output artifacts locations')
def run(events: Union[str, bytes, os.PathLike],
        users: Optional[Union[str, bytes, os.PathLike]] = None,
        items: Optional[Union[str, bytes, os.PathLike]] = None,
        schema: Optional[Union[str, bytes, os.PathLike]] = None,
        output: Optional[Union[str, bytes, os.PathLike]] = None):
    """
    Run a Rexify pipeline.
    """

    runner = runner_factory()
    with open(schema, 'r') as f:
        schema = json.load(f)

    from rexify.pipeline import pipeline
    ppl = pipeline.build(
        run_fn=RUN_FN,
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_OUTPUT,
        metadata_path=METADATA_PATH,
        data_root=events,
        items_root=items,
        schema=schema,
        serving_model_dir=output,
        enable_cache=ENABLE_CACHE)

    runner.run(ppl)


def runner_factory():
    if BACKEND_CONFIG == 'kubeflow':
        from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner
        return KubeflowDagRunner()
    elif BACKEND_CONFIG == 'local':
        from tfx.orchestration.local.local_dag_runner import LocalDagRunner
        return LocalDagRunner()
    raise ValueError('Invalid backend config')


if __name__ == '__main__':
    cli()
