"""Read schemas and statistics from a program's data sources, for `eigrel plan`.

Files are queried with DuckDB and databases through SQLAlchemy, using the SQL backend's queries,
so filters and projections run inside the engine and only counts come back. BigQuery is not
probed: every query there costs money.
"""

import importlib.util
import os
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eigrel.backends import sql
from eigrel.compiler import ir
from eigrel.compiler.semantic import ExprType

NUMBER_TYPES = (
    'TINYINT',
    'SMALLINT',
    'INTEGER',
    'BIGINT',
    'HUGEINT',
    'UTINYINT',
    'USMALLINT',
    'UINTEGER',
    'UBIGINT',
    'UHUGEINT',
    'FLOAT',
    'DOUBLE',
    'DECIMAL',
    'REAL',
)


class ProbeError(Exception):
    """A source could not be read; the message says why."""


@dataclass(frozen=True)
class Column:
    name: str
    # 'number', 'string', 'bool' or 'unknown', as the semantic analysis sees it.
    type: ExprType
    # The engine's own type name, recorded in .eigstate to detect drift precisely.
    native: str


class Engine(Protocol):
    def columns(self, load: ir.Load) -> tuple[Column, ...]: ...

    def rows(self, query: str) -> list[tuple[Any, ...]]: ...


def duckdb_type(native: str) -> ExprType:
    upper = native.upper()
    if upper.startswith(NUMBER_TYPES):
        return 'number'
    if upper == 'VARCHAR':
        return 'string'
    if upper == 'BOOLEAN':
        return 'bool'
    return 'unknown'


def python_type(python: type | None) -> ExprType:
    if python is bool:
        return 'bool'
    if python is str:
        return 'string'
    if python is not None and (issubclass(python, int | float) or python.__name__ == 'Decimal'):
        return 'number'
    return 'unknown'


class DuckDBEngine:
    """Files (CSV, Parquet, JSON), queried in place relative to the program's directory."""

    def __init__(self, base: Path) -> None:
        if importlib.util.find_spec('duckdb') is None:
            raise ProbeError('eigrel plan needs duckdb to read files: pip install "eigrel[python]"')
        import duckdb

        self.connection = duckdb.connect()
        self.base = base

    def columns(self, load: ir.Load) -> tuple[Column, ...]:
        path = self.base / str(load.args[0])
        if not path.exists():
            raise ProbeError(f'cannot read {load.args[0]}: no such file')
        described = self._run(f'DESCRIBE {sql.select_all(load)}')
        return tuple(Column(name, duckdb_type(native), native) for name, native, *_ in described)

    def rows(self, query: str) -> list[tuple[Any, ...]]:
        return self._run(query)

    def _run(self, query: str) -> list[tuple[Any, ...]]:
        # Relative paths in the generated SQL are relative to the program, like `eigrel run`.
        current = Path.cwd()
        os.chdir(self.base)
        try:
            return self.connection.sql(query).fetchall()
        except Exception as exc:  # duckdb raises many specific error types
            raise ProbeError(str(exc).splitlines()[0]) from exc
        finally:
            os.chdir(current)


class DatabaseEngine:
    """A database behind a SQLAlchemy URL."""

    def __init__(self, url: str, base: Path) -> None:
        if importlib.util.find_spec('sqlalchemy') is None:
            raise ProbeError(
                'eigrel plan needs sqlalchemy to read databases: pip install "eigrel[sql]"'
            )
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool

        current = Path.cwd()
        os.chdir(base)  # sqlite:///relative.db is relative to the program
        try:
            self.engine = create_engine(url, poolclass=NullPool)
            self.engine.connect().close()
        except Exception as exc:  # driver errors vary by database
            raise ProbeError(f'cannot connect to the database: {str(exc).splitlines()[0]}') from exc
        finally:
            os.chdir(current)

    def columns(self, load: ir.Load) -> tuple[Column, ...]:
        from sqlalchemy import inspect

        schema, _, table = str(load.args[1]).rpartition('.')
        try:
            described = inspect(self.engine).get_columns(table, schema=schema or None)
        except Exception as exc:  # NoSuchTableError, permission errors, ...
            raise ProbeError(
                f'cannot read table {load.args[1]}: {str(exc).splitlines()[0]}'
            ) from exc
        if not described:
            raise ProbeError(f'table {load.args[1]} does not exist')
        return tuple(
            Column(c['name'], python_type(_python_type(c['type'])), str(c['type']))
            for c in described
        )

    def rows(self, query: str) -> list[tuple[Any, ...]]:
        from sqlalchemy import text

        try:
            with closing(self.engine.connect()) as connection:
                return [tuple(row) for row in connection.execute(text(query))]
        except Exception as exc:  # driver errors vary by database
            raise ProbeError(str(exc).splitlines()[0]) from exc


def _python_type(sql_type: Any) -> type | None:
    try:
        python: type = sql_type.python_type
    except NotImplementedError:
        return None
    return python


def engine_for(load: ir.Load, base: Path, cache: dict[str, Engine]) -> Engine:
    """The engine that can query a source, shared between datasets that use the same one."""
    if load.format in sql.FILE_READERS:
        key = 'files'
        if key not in cache:
            cache[key] = DuckDBEngine(base)
        return cache[key]
    if load.format == 'bigquery':
        raise ProbeError('BigQuery sources are not probed, because every query costs money')
    url = load.args[0]
    if isinstance(url, ir.EnvVar):
        if url.name not in os.environ:
            raise ProbeError(f'environment variable {url.name} is not set')
        url = os.environ[url.name]
    if url not in cache:
        cache[url] = DatabaseEngine(url, base)
    return cache[url]


def count(engine: Engine, query: str) -> int:
    (value,) = engine.rows(f'SELECT COUNT(*) FROM ({query}) AS probe')[0]
    return int(value)


def missing(engine: Engine, query: str, columns: Sequence[str], quote: str) -> dict[str, int]:
    """Missing values per column."""
    parts = ', '.join(
        f'COUNT(*) - COUNT({quote}{c.replace(quote, quote * 2)}{quote})' for c in columns
    )
    values = engine.rows(f'SELECT {parts} FROM ({query}) AS probe')[0]
    return {column: int(value) for column, value in zip(columns, values, strict=True)}


def value_counts(
    engine: Engine, query: str, column: str, quote: str, limit: int = 20
) -> tuple[list[tuple[Any, int]], int]:
    """The most frequent values of a column, and how many distinct values it has."""
    name = f'{quote}{column.replace(quote, quote * 2)}{quote}'
    top = engine.rows(
        f'SELECT {name}, COUNT(*) AS n FROM ({query}) AS probe'
        f' WHERE {name} IS NOT NULL GROUP BY {name} ORDER BY n DESC, {name} LIMIT {limit}'
    )
    (distinct,) = engine.rows(f'SELECT COUNT(DISTINCT {name}) FROM ({query}) AS probe')[0]
    return [(value, int(n)) for value, n in top], int(distinct)
