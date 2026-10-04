"""Eigrel IR: the program as a graph of operations.

Every operation produces one value and refers to its inputs by id, so the program forms a DAG.
Backends generate code from this graph, and the optimizer (v0.5) will rewrite it.
"""

from dataclasses import dataclass, field
from typing import Literal

from eigrel.compiler import ast
from eigrel.compiler.tokens import Location

Task = Literal['classification', 'regression']
_NOWHERE = Location(0, 0)
Value = int | float | str | bool


@dataclass(frozen=True)
class Op:
    id: int
    # The Eigrel name this value is bound to (dataset or model name).
    name: str


@dataclass(frozen=True)
class EnvVar:
    """A value read from an environment variable when the program runs, e.g. a database URL."""

    name: str


SourceArg = str | EnvVar


@dataclass(frozen=True)
class Load(Op):
    format: str
    args: tuple[SourceArg, ...]


@dataclass(frozen=True)
class Filter(Op):
    input: int
    condition: ast.Expr


@dataclass(frozen=True)
class Select(Op):
    input: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Fill(Op):
    input: int
    values: tuple[tuple[str, Value], ...]


@dataclass(frozen=True)
class DropMissing(Op):
    input: int
    # None means rows with a missing value in any column.
    columns: tuple[str, ...] | None


@dataclass(frozen=True)
class Train(Op):
    input: int
    algorithm: str
    task: Task
    params: tuple[tuple[str, Value], ...]
    # None means every column except the target.
    features: tuple[str, ...] | None
    target: str
    validation: float
    seed: int


@dataclass(frozen=True)
class Evaluate(Op):
    model: int
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class Register(Op):
    model: int
    registered_name: str
    experiment: str | None
    # The latest evaluation of the model before it is registered, whose metrics are logged.
    evaluation: int | None


@dataclass
class Graph:
    ops: list[Op] = field(default_factory=list)

    def add(self, op: Op) -> int:
        self.ops.append(op)
        return op.id

    def next_id(self) -> int:
        return len(self.ops)


def format_expr(expr: ast.Expr) -> str:
    """Render an expression back to Eigrel syntax, fully parenthesized."""
    match expr:
        case ast.Name(value=name):
            return name
        case ast.StringLiteral(value=text):
            return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
        case ast.BoolLiteral(value=flag):
            return 'true' if flag else 'false'
        case ast.IntLiteral(value=number) | ast.FloatLiteral(value=number):
            return repr(number)
        case ast.ListExpr(items=items):
            return '[' + ', '.join(format_expr(item) for item in items) + ']'
        case ast.Call(func=func, args=args):
            return f'{func}(' + ', '.join(format_expr(arg) for arg in args) + ')'
        case ast.Unary(op='not', operand=operand):
            return f'(not {format_expr(operand)})'
        case ast.Unary(op=op, operand=operand):
            return f'({op}{format_expr(operand)})'
        case ast.Binary(op=op, left=left, right=right):
            return f'({format_expr(left)} {op} {format_expr(right)})'
    raise AssertionError(f'unknown expression {expr!r}')  # pragma: no cover


def format_value(value: Value) -> str:
    """Render a literal value in Eigrel syntax."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return format_expr(ast.StringLiteral(value, loc=_NOWHERE))
    return repr(value)


def _format_source_arg(arg: SourceArg) -> str:
    if isinstance(arg, EnvVar):
        return f'env("{arg.name}")'
    return format_expr(ast.StringLiteral(arg, loc=_NOWHERE))


def format_graph(graph: Graph) -> str:
    """Text form of the IR, one operation per line."""
    lines = []
    for op in graph.ops:
        match op:
            case Load():
                args = ', '.join(_format_source_arg(arg) for arg in op.args)
                body = f'load {op.format}({args})'
            case Filter():
                body = f'filter %{op.input} {format_expr(op.condition)}'
            case Select():
                body = f'select %{op.input} [{", ".join(op.columns)}]'
            case Fill():
                values = ', '.join(
                    f'{column} = {format_value(value)}' for column, value in op.values
                )
                body = f'fill %{op.input} [{values}]'
            case DropMissing():
                columns = 'any column' if op.columns is None else ', '.join(op.columns)
                body = f'drop_missing %{op.input} [{columns}]'
            case Train():
                params = ', '.join(f'{key}={value!r}' for key, value in op.params)
                features = 'all but target' if op.features is None else ', '.join(op.features)
                body = (
                    f'train %{op.input} {op.algorithm}({params}) {op.task}'
                    f' features=[{features}] target={op.target}'
                    f' validation={op.validation} seed={op.seed}'
                )
            case Evaluate():
                body = f'evaluate %{op.model} [{", ".join(op.metrics)}]'
            case Register():
                body = f'register %{op.model} as {format_value(op.registered_name)}'
                if op.experiment is not None:
                    body += f' experiment={format_value(op.experiment)}'
                if op.evaluation is not None:
                    body += f' metrics=%{op.evaluation}'
            case _:  # pragma: no cover
                raise AssertionError(f'unknown op {op!r}')
        lines.append(f'%{op.id} = {body}  # {op.name}')
    return '\n'.join(lines)
