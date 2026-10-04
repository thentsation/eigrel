import importlib.util
from pathlib import Path

import pytest

from eigrel.cli import _has_java

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip Spark tests where pyspark or a Java runtime is missing."""
    spark_items = [item for item in items if item.get_closest_marker('spark')]
    if not spark_items:
        return
    reason = None
    if importlib.util.find_spec('pyspark') is None:
        reason = 'pyspark is not installed'
    elif not _has_java():
        reason = 'no Java runtime'
    if reason:
        for item in spark_items:
            item.add_marker(pytest.mark.skip(reason=reason))
