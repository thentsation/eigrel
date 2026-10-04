import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from eigrel.runner import has_java

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture(autouse=True)
def isolate_mlflow_environment() -> Iterator[None]:
    """MLflow writes MLFLOW_* variables (e.g. the active experiment) into os.environ; undo them."""
    before = {k: v for k, v in os.environ.items() if k.startswith('MLFLOW_')}
    yield
    for key in [k for k in os.environ if k.startswith('MLFLOW_')]:
        del os.environ[key]
    os.environ.update(before)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip Spark tests where pyspark or a Java runtime is missing."""
    spark_items = [item for item in items if item.get_closest_marker('spark')]
    if not spark_items:
        return
    reason = None
    if importlib.util.find_spec('pyspark') is None:
        reason = 'pyspark is not installed'
    elif not has_java():
        reason = 'no Java runtime'
    if reason:
        for item in spark_items:
            item.add_marker(pytest.mark.skip(reason=reason))
