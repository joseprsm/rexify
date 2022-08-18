import json
import os
import tempfile
from typing import Callable

import numpy as np
import pytest

from rexify.pipeline import _load_component, compile_, pipeline_fn


def create_pipeline_spec():
    schema = {"user": {"user_id": "id"}, "item": {"item_id": "id"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = os.path.join(tmpdir, "pipeline.json")
        compile_(event_uri="", schema=schema, package_path=package_path)
        with open(package_path, "r") as f:
            pipeline_spec = json.load(f)["pipelineSpec"]
    return pipeline_spec


def test_load_non_existent_component():
    with pytest.raises(FileNotFoundError):
        _load_component("error")


def test_load_download_component():
    assert isinstance(_load_component("download"), Callable)


def test_load_load_component():
    assert isinstance(_load_component("copy"), Callable)


def test_load_copy_component():
    assert isinstance(_load_component("load"), Callable)


def test_load_train_component():
    assert isinstance(_load_component("train"), Callable)


def test_load_index_component():
    assert isinstance(_load_component("index"), Callable)


def test_load_retrieval_component():
    assert isinstance(_load_component("retrieval"), Callable)


def test_pipeline_fn_schema():
    pipeline_fn(event_uri="sample", schema=dict())


def test_compile_schema():
    create_pipeline_spec()


def test_pipeline_components():
    ppl_spec = create_pipeline_spec()
    component_names = [
        "comp-copy",
        "comp-download",
        "comp-index",
        "comp-load",
        "comp-retrieval",
        "comp-train",
    ]
    assert np.all(np.in1d(component_names, list(ppl_spec["components"].keys())))
    for comp_name in component_names:
        assert sum([comp_name in k for k in ppl_spec["components"].keys()]) == 1
