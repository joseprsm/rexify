import warnings

import typer
from kfp.v2.compiler import Compiler
from kfp.v2.dsl import pipeline

from rexify.pipeline import PIPELINE_ROOT
from rexify.pipeline.components import load, train


@pipeline(name="pipeline", pipeline_root=PIPELINE_ROOT)
def pipeline(
    events: str,
    users: str,
    items: str,
    schema: str,
    epochs: int = 100,
    batch_size: int = 512,
):

    load_task = load(
        events=events,
        users=users,
        items=items,
        schema=schema,
    )

    train_task = train(  # noqa:F841
        feature_extractor=load_task.outputs["feature_extractor"],
        train_data=load_task.outputs["train_data"],
        validation_data=load_task.outputs["validation_data"],
        batch_size=batch_size,
        epochs=epochs,
    )


def compile(
    output_path: str = typer.Option(
        None, help="Output path for the pipeline definition JSON file"
    ),
    parameter: list[str] = typer.Option(
        None, "--parameter", "-p", help="Pipeline parameter, KEY=VALUE"
    ),
):
    output_path = output_path if output_path else "pipeline.json"

    pipeline_parameters = (
        {k: v for k, v in [param.split("=") for param in parameter]}
        if parameter
        else None
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Compiler().compile(
            pipeline_func=pipeline,
            package_path=output_path,
            pipeline_parameters=pipeline_parameters,
        )


if __name__ == "__main__":
    typer.run(compile)
