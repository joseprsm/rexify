from kfp.v2.dsl import Artifact, Dataset, Output, component

from rexify import BASE_IMAGE


@component(base_image=BASE_IMAGE)
def load(
    events: str,
    users: str,
    items: str,
    schema: str,
    feature_extractor: Output[Artifact],
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_size: float = 0.3,
):
    import json

    from rexify import Events, FeatureExtractor, Output, Schema

    schema = Schema.from_dict(json.loads(schema))
    train, val = Events.load(events, schema=schema).split(test_size=test_size)

    fe = FeatureExtractor(schema, users, items, return_dataset=False)
    train: Output = fe.fit(train).transform(train)
    val: Output = fe.transform(val)

    fe.save(feature_extractor.path)
    train.save(train_data.path, "train.csv")
    val.save(validation_data.path, "val.csv")
