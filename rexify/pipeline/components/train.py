from kfp.v2.dsl import Artifact, Dataset, Input, Model, Output, component

from rexify import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def train(
    feature_extractor: Input[Artifact],
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    model: Output[Model],
    batch_size: int = 512,
    epochs: int = 10,
):
    from rexify import DataFrame, FeatureExtractor, Recommender

    fe = FeatureExtractor.load(feature_extractor.path)
    train_data = DataFrame.load(train_data.path)
    validation_data = DataFrame.load(validation_data.path)

    fit_params = {"batch_size": batch_size, "epochs": epochs}
    recommender = Recommender(**fe.model_params)
    recommender.compile()
    recommender.fit(train_data, validation_data=validation_data, **fit_params)
    recommender.save(model.path)
