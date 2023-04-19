from kfp.v2.compiler import Compiler
from kfp.v2.dsl import pipeline

from rexify.pipeline.components import preprocess, train


@pipeline(name="pipeline")
def pipeline():
    preprocess_op = preprocess()
    train_op = train(  # noqa:F841
        feature_extractor=preprocess_op.outputs["feature_extractor"],
        train_data=preprocess_op.outputs["train_data"],
        validation_data=preprocess_op.outputs["validation_data"],
    )


if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
    )
