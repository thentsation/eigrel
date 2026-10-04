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
