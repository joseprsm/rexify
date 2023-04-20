import typer
from kfp.v2.compiler import Compiler
from kfp.v2.dsl import pipeline

from rexify.pipeline.components import load, train


@pipeline(name="pipeline")
def pipeline(
    events_uri: str = None,
    users_uri: str = None,
    items_uri: str = None,
    schema: str = None,
    epochs: int = 100,
    batch_size: int = 512,
):

    load_task = load(
        events=events_uri,
        users=users_uri,
        items=items_uri,
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
    output_path: str = typer.Option(None, help="Pipeline definition output path"),
    parameter: list[str] = typer.Option(None, help="Pipeline parameter, KEY=VALUE"),
):
    output_path = output_path if output_path else "pipeline.json"

    pipeline_parameters = (
        {k: v for k, v in [param.split("=") for param in parameter]}
        if parameter
        else None
    )

    Compiler().compile(
        pipeline_func=pipeline,
        package_path=output_path,
        pipeline_parameters=pipeline_parameters,
    )


if __name__ == "__main__":
    typer.run(compile)
