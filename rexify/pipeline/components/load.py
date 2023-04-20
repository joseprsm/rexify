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

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from rexify import DataFrame, FeatureExtractor

    events = pd.read_csv(events)
    users = pd.read_csv(users)
    items = pd.read_csv(items)
    schema = json.loads(schema)

    fe = FeatureExtractor(schema, users, items, return_dataset=False)

    train, val = train_test_split(events, test_size=test_size)
    train: DataFrame = fe.fit(train).transform(train)
    val: DataFrame = fe.transform(val)

    fe.save(feature_extractor.path)
    train.save(train_data.path, "train.csv")
    val.save(validation_data.path, "val.csv")
