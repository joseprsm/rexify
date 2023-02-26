from pathlib import Path

import pytest

from rexify.utils import get_target_feature, get_target_id, make_dirs


@pytest.fixture
def schema():
    return {
        "target1": {"key1": "id", "key2": "value1"},
        "target2": {"key3": "value2", "key4": "id"},
        "target3": {"key5": "value3", "key6": "value4"},
    }


def test_get_target_id(schema):
    assert get_target_id(schema, "target1") == ["key1"]
    assert get_target_id(schema, "target2") == ["key4"]
    assert get_target_id(schema, "target3") == []


def test_get_target_feature(schema):
    assert get_target_feature(schema, "target1", "id") == ["key1"]
    assert get_target_feature(schema, "target1", "value1") == ["key2"]
    assert get_target_feature(schema, "target2", "id") == ["key4"]
    assert get_target_feature(schema, "target2", "value2") == ["key3"]
    assert get_target_feature(schema, "target3", "value3") == ["key5"]
    assert get_target_feature(schema, "target3", "value4") == ["key6"]
    assert get_target_feature(schema, "target3", "value5") == []


def test_make_dirs(tmpdir):
    dir1 = tmpdir.mkdir("dir1")
    dir2 = tmpdir.mkdir("dir2")
    make_dirs(dir1, dir2)

    assert Path(dir1).exists()
    assert Path(dir2).exists()
