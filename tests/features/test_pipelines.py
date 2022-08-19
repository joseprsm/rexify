import pytest
from sklearn.pipeline import Pipeline

from rexify.features.pipelines import (
    CategoricalPipeline,
    IdentifierPipeline,
    NumericalPipeline,
    _BaseFeaturePipeline,
)


schema = {
    "user": {
        "user_id": "id",
        "cat_1": "categorical",
        "num_1": "numerical",
        "num_2": "numerical",
    },
    "item": {"item_id": "id"},
}


def test_base_feature_pipeline_get_features():
    with pytest.raises(NotImplementedError):
        _BaseFeaturePipeline._get_features(None, None)


def test_base_feature_pipeline__new__():
    with pytest.raises(AttributeError):
        _BaseFeaturePipeline(None, None)


def test_id_pipeline_get_features():
    assert IdentifierPipeline._get_features(schema, "user") == ["user_id"]


def test_id_pipeline_new():
    ppl = IdentifierPipeline(schema, "user")
    assert isinstance(ppl, tuple)
    assert ppl[0] == "user_idPipeline"
    assert isinstance(ppl[1], Pipeline)
    assert len(ppl[1].steps) > 0


def test_categorical_pipeline_get_features():
    assert CategoricalPipeline._get_features(schema, "user") == ["cat_1"]


def test_categorical_pipeline_new():
    ppl = CategoricalPipeline(schema, "user")
    assert isinstance(ppl, tuple)
    assert ppl[0] == "user_categoricalPipeline"
    assert isinstance(ppl[1], Pipeline)
    assert len(ppl[1].steps) > 0


def test_numerical_pipeline_get_features():
    assert NumericalPipeline._get_features(schema, "user") == ["num_1", "num_2"]


def test_numerical_pipeline_new():
    id_ppl = NumericalPipeline(schema, "user")
    assert isinstance(id_ppl, tuple)
    assert id_ppl[0] == "user_numericalPipeline"
    assert isinstance(id_ppl[1], Pipeline)
    assert len(id_ppl[1].steps) > 0
