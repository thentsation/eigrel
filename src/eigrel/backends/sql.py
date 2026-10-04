"""SQL backend: turn the data part of the IR into one SELECT per dataset.

Each dataset is compiled for the engine its source lives in: the database of a `sql()` source,
BigQuery for `bigquery()`, and DuckDB for files, which can query CSV, Parquet and JSON directly.
Models are not part of the SQL output.
"""

from dataclasses import dataclass

from eigrel import __version__
from eigrel.backends import UnsupportedError
from eigrel.compiler import ast, ir

FILE_READERS = {'csv': 'read_csv_auto', 'parquet': 'read_parquet', 'json': 'read_json_auto'}
OPERATORS = {'and': 'AND', 'or': 'OR', '==': '=', '!=': '<>'}


@dataclass(frozen=True)
class Dialect:
    name: str
    quote: str
    # BigQuery has no % operator, only MOD().
    mod_function: bool = False
    # Supports SELECT * REPLACE (expr AS column).
    star_replace: bool = False


ANSI = Dialect('ansi', '"')
DUCKDB = Dialect('duckdb', '"', star_replace=True)
BIGQUERY = Dialect('bigquery', '`', mod_function=True, star_replace=True)
MYSQL = Dialect('mysql', '`')
# Database URL scheme (before any +driver) -> dialect; anything else uses ANSI quoting.
URL_DIALECTS = {
    'mysql': MYSQL,
    'mariadb': MYSQL,
    'postgresql': Dialect('postgresql', '"'),
    'postgres': Dialect('postgresql', '"'),
    'sqlite': Dialect('sqlite', '"'),
}


DatasetOp = ir.Filter | ir.Select | ir.Fill | ir.DropMissing


@dataclass(frozen=True)
class Query:
    dataset: str
    source: ir.Load
    # The operations applied to the source, in order.
    ops: tuple[DatasetOp, ...]
    # The source's columns when known (eigrel plan reads them), so fills and drop_missing can be
    # expressed without a select.
    source_columns: tuple[str, ...] | None = None


def dialect_for(load: ir.Load) -> Dialect:
    if load.format in FILE_READERS:
        return DUCKDB
    if load.format == 'bigquery':
        return BIGQUERY
    url = load.args[0]
    if isinstance(url, ir.EnvVar):
        return ANSI
    scheme = url.split(':', 1)[0].split('+', 1)[0].lower()
    return URL_DIALECTS.get(scheme, ANSI)


def dataset_queries(graph: ir.Graph) -> list[Query]:
    """The final state of every dataset: its source and the operations applied to it."""
    latest: dict[str, int] = {}
    for op in graph.ops:
        if isinstance(op, ir.Load | ir.Filter | ir.Select | ir.Fill | ir.DropMissing):
            latest[op.name] = op.id
    return [_query(graph, name, op_id) for name, op_id in latest.items()]


def _query(graph: ir.Graph, name: str, op_id: int) -> Query:
    ops: list[DatasetOp] = []
    op = graph.ops[op_id]
    while not isinstance(op, ir.Load):
        assert isinstance(op, ir.Filter | ir.Select | ir.Fill | ir.DropMissing)
        ops.append(op)
        op = graph.ops[op.input]
    return Query(name, op, tuple(reversed(ops)))


def render(query: Query) -> str:
    """One SELECT for the whole chain.

    Semantic analysis guarantees every column used exists where it is used, so filters and the
    final projection merge into a single statement. Filled columns are tracked as COALESCE
    expressions, so later filters and the projection see the filled values.
    """
    dialect = dialect_for(query.source)
    filled: dict[str, str] = {}
    conditions: list[str] = []
    columns: tuple[str, ...] | None = query.source_columns
    for op in query.ops:
        match op:
            case ir.Filter():
                conditions.append(_expr(op.condition, dialect, filled))
            case ir.Select():
                columns = op.columns
            case ir.Fill():
                for column, value in op.values:
                    current = filled.get(column, _identifier(column, dialect))
                    filled[column] = f'COALESCE({current}, {_value(value)})'
            case ir.DropMissing():
                dropped = op.columns if op.columns is not None else columns
                if dropped is None:
                    raise UnsupportedError(
                        f"SQL needs the column names for drop_missing in '{query.dataset}';"
                        ' list them (drop_missing a, b) or select the columns first'
                    )
                for column in dropped:
                    reference = filled.get(column, _identifier(column, dialect))
                    conditions.append(f'({reference} IS NOT NULL)')
    sql = f'SELECT {_projection(query.dataset, columns, filled, dialect)}'
    sql += f' FROM {_from(query.source, dialect)}'
    if conditions:
        sql += ' WHERE ' + ' AND '.join(conditions)
    return sql


def _projection(
    dataset: str, columns: tuple[str, ...] | None, filled: dict[str, str], dialect: Dialect
) -> str:
    if columns is not None:
        return ', '.join(
            f'{filled[c]} AS {_identifier(c, dialect)}' if c in filled else _identifier(c, dialect)
            for c in columns
        )
    if not filled:
        return '*'
    if not dialect.star_replace:
        raise UnsupportedError(
            f"{dialect.name} SQL cannot fill columns of '{dataset}' without knowing all of them;"
            ' select the columns first'
        )
    replaced = ', '.join(f'{expr} AS {_identifier(c, dialect)}' for c, expr in filled.items())
    return f'* REPLACE ({replaced})'


def select_all(load: ir.Load) -> str:
    """SELECT * from a single source, as used by the Python backend to read databases."""
    return render(Query(load.name, load, ()))


def generate(graph: ir.Graph, source_name: str = '<eigrel>') -> str:
    """Generate SQL for every dataset in the program."""
    blocks = [f'-- Generated by eigrel {__version__} from {source_name}. Do not edit.']
    for query in dataset_queries(graph):
        dialect = dialect_for(query.source)
        blocks.append(
            f'-- dataset {query.dataset} ({query.source.format}, {dialect.name} dialect)\n'
            f'{render(query)};'
        )
    models = sorted({op.name for op in graph.ops if isinstance(op, ir.Train)})
    if models:
        blocks.append(f'-- Models are not compiled to SQL: {", ".join(models)}')
    return '\n\n'.join(blocks) + '\n'


def _from(load: ir.Load, dialect: Dialect) -> str:
    if load.format in FILE_READERS:
        return f'{FILE_READERS[load.format]}({_string(str(load.args[0]))})'
    if load.format == 'bigquery':
        return f'`{load.args[0]}`'
    table = str(load.args[1])
    return '.'.join(_identifier(part, dialect) for part in table.split('.'))


def _identifier(name: str, dialect: Dialect) -> str:
    return dialect.quote + name.replace(dialect.quote, dialect.quote * 2) + dialect.quote


def _string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _value(value: ir.Value) -> str:
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, str):
        return _string(value)
    return repr(value)


def _expr(expr: ast.Expr, dialect: Dialect, filled: dict[str, str]) -> str:
    match expr:
        case ast.Name(value=column):
            return filled.get(column, _identifier(column, dialect))
        case ast.IntLiteral(value=number) | ast.FloatLiteral(value=number):
            return repr(number)
        case ast.StringLiteral(value=text):
            return _string(text)
        case ast.BoolLiteral(value=flag):
            return 'TRUE' if flag else 'FALSE'
        case ast.Unary(op='not', operand=operand):
            return f'(NOT {_expr(operand, dialect, filled)})'
        case ast.Unary(op=op, operand=operand):
            return f'({op}{_expr(operand, dialect, filled)})'
        case ast.Binary(op='%', left=left, right=right) if dialect.mod_function:
            return f'MOD({_expr(left, dialect, filled)}, {_expr(right, dialect, filled)})'
        case ast.Binary(op=op, left=left, right=right):
            sql_op = OPERATORS.get(op, op)
            return f'({_expr(left, dialect, filled)} {sql_op} {_expr(right, dialect, filled)})'
    raise AssertionError(f'semantic analysis rejects {expr!r} in filters')  # pragma: no cover
