"""What the language knows about: data sources, algorithms and metrics.

This is backend-neutral. Each backend maps these names to its own libraries.
"""

from dataclasses import dataclass
from typing import Literal

from eigrel.compiler.ir import Task

ParamKind = Literal['int', 'number']
# path: a file path; url: a database URL, literal or env("VAR"); table: a database table, optionally
# schema-qualified; bigquery_table: a fully qualified project.dataset.table.
SourceArgKind = Literal['path', 'url', 'table', 'bigquery_table']


@dataclass(frozen=True)
class Source:
    args: tuple[tuple[str, SourceArgKind], ...]
    example: str


SOURCES: dict[str, Source] = {
    'csv': Source((('path', 'path'),), 'csv("data/customers.csv")'),
    'parquet': Source((('path', 'path'),), 'parquet("data/customers.parquet")'),
    'json': Source((('path', 'path'),), 'json("data/customers.jsonl")'),
    'sql': Source(
        (('connection URL', 'url'), ('table', 'table')),
        'sql(env("DATABASE_URL"), "customers")',
    ),
    'bigquery': Source((('table', 'bigquery_table'),), 'bigquery("project.dataset.table")'),
}


@dataclass(frozen=True)
class Algorithm:
    tasks: tuple[Task, ...]
    # Parameter name -> kind. Every algorithm parameter must be positive.
    params: dict[str, ParamKind]


ALGORITHMS: dict[str, Algorithm] = {
    'random_forest': Algorithm(
        ('classification', 'regression'),
        {'trees': 'int', 'max_depth': 'int', 'min_samples_leaf': 'int'},
    ),
    'decision_tree': Algorithm(
        ('classification', 'regression'),
        {'max_depth': 'int', 'min_samples_leaf': 'int'},
    ),
    'gradient_boosting': Algorithm(
        ('classification', 'regression'),
        {'trees': 'int', 'learning_rate': 'number', 'max_depth': 'int'},
    ),
    'xgboost': Algorithm(
        ('classification', 'regression'),
        {'trees': 'int', 'learning_rate': 'number', 'max_depth': 'int'},
    ),
    'logistic_regression': Algorithm(('classification',), {'max_iter': 'int', 'c': 'number'}),
    'linear_regression': Algorithm(('regression',), {}),
}

METRICS: dict[str, Task] = {
    'accuracy': 'classification',
    'precision': 'classification',
    'recall': 'classification',
    'f1': 'classification',
    'auc': 'classification',
    'mae': 'regression',
    'mse': 'regression',
    'rmse': 'regression',
    'r2': 'regression',
}

DEFAULT_METRICS: dict[Task, tuple[str, ...]] = {
    'classification': ('accuracy', 'precision', 'recall', 'f1'),
    'regression': ('mae', 'rmse', 'r2'),
}

DEFAULT_VALIDATION = 0.2
DEFAULT_SEED = 42


# What each backend can do. `eigrel plan` checks programs against it, and docs/LANGUAGE.md shows it
# as a table that a test keeps in sync. Values: 'yes', 'no', or a short note on the limit.
BACKENDS = ('python', 'spark', 'sql')
CAPABILITIES: dict[str, dict[str, str]] = {
    'csv, parquet, json sources': {'python': 'yes', 'spark': 'yes', 'sql': 'yes (DuckDB)'},
    'sql() sources': {'python': 'yes', 'spark': 'yes (JDBC)', 'sql': 'yes'},
    'bigquery() sources': {'python': 'yes', 'spark': 'yes (connector)', 'sql': 'yes'},
    'filter, select, fill, drop_missing': {'python': 'yes', 'spark': 'yes', 'sql': 'yes'},
    'training and evaluation': {'python': 'yes', 'spark': 'yes', 'sql': 'no'},
    'missing values in numeric features': {
        'python': 'random_forest, decision_tree, xgboost',
        'spark': 'no',
        'sql': 'no',
    },
    'multiclass gradient_boosting': {'python': 'yes', 'spark': 'no', 'sql': 'no'},
    'auc on multiclass targets': {'python': 'yes', 'spark': 'no', 'sql': 'no'},
    'stratified validation split': {'python': 'yes', 'spark': 'no', 'sql': 'no'},
    'time-ordered split (assumptions)': {'python': 'yes', 'spark': 'yes', 'sql': 'no'},
    'register (MLflow)': {'python': 'yes', 'spark': 'yes', 'sql': 'no'},
    'predict': {'python': 'yes', 'spark': 'yes', 'sql': 'no'},
}


def handles_missing_values(backend: str, algorithm: str) -> bool:
    """Whether a backend trains this algorithm on features with missing numeric values."""
    supported = CAPABILITIES['missing values in numeric features'][backend]
    return algorithm in supported.split(', ')
