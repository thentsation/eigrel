from dataclasses import dataclass, field, fields
from typing import Any

from eigrel.compiler.tokens import Location


@dataclass(frozen=True)
class Node:
    loc: Location = field(compare=False, kw_only=True)


# Expressions


@dataclass(frozen=True)
class Name(Node):
    value: str


@dataclass(frozen=True)
class IntLiteral(Node):
    value: int


@dataclass(frozen=True)
class FloatLiteral(Node):
    value: float


@dataclass(frozen=True)
class StringLiteral(Node):
    value: str


@dataclass(frozen=True)
class BoolLiteral(Node):
    value: bool


@dataclass(frozen=True)
class ListExpr(Node):
    items: tuple['Expr', ...]


@dataclass(frozen=True)
class Call(Node):
    func: str
    args: tuple['Expr', ...]


@dataclass(frozen=True)
class Unary(Node):
    op: str
    operand: 'Expr'


@dataclass(frozen=True)
class Binary(Node):
    op: str
    left: 'Expr'
    right: 'Expr'


Expr = (
    Name
    | IntLiteral
    | FloatLiteral
    | StringLiteral
    | BoolLiteral
    | ListExpr
    | Call
    | Unary
    | Binary
)


# Block contents


@dataclass(frozen=True)
class Param(Node):
    name: str
    value: Expr


@dataclass(frozen=True)
class FilterOp(Node):
    condition: Expr


@dataclass(frozen=True)
class SelectOp(Node):
    columns: tuple[Name, ...]


@dataclass(frozen=True)
class FillOp(Node):
    values: tuple[Param, ...]


@dataclass(frozen=True)
class DropMissingOp(Node):
    # Empty means any column.
    columns: tuple[Name, ...]


TransformOp = FilterOp | SelectOp | FillOp | DropMissingOp


# Statements


@dataclass(frozen=True)
class DatasetDecl(Node):
    name: str
    source: Call


@dataclass(frozen=True)
class TransformDecl(Node):
    dataset: str
    ops: tuple[TransformOp, ...]


@dataclass(frozen=True)
class FeaturesDecl(Node):
    dataset: str
    columns: tuple[Name, ...]


@dataclass(frozen=True)
class ModelDecl(Node):
    name: str
    algorithm: str
    params: tuple[Param, ...]


@dataclass(frozen=True)
class TrainStmt(Node):
    model: str
    params: tuple[Param, ...]


@dataclass(frozen=True)
class EvaluateStmt(Node):
    model: str
    params: tuple[Param, ...]


@dataclass(frozen=True)
class RegisterStmt(Node):
    model: str
    params: tuple[Param, ...]


Statement = (
    DatasetDecl | TransformDecl | FeaturesDecl | ModelDecl | TrainStmt | EvaluateStmt | RegisterStmt
)


@dataclass(frozen=True)
class Program(Node):
    statements: tuple[Statement, ...]


def to_dict(value: Any) -> Any:
    """Convert an AST into plain JSON-serializable data."""
    if isinstance(value, Node):
        data: dict[str, Any] = {'node': type(value).__name__}
        for f in fields(value):
            if f.name != 'loc':
                data[f.name] = to_dict(getattr(value, f.name))
        data['loc'] = {'line': value.loc.line, 'column': value.loc.column}
        return data
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    return value
