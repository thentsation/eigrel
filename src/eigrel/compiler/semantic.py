"""Semantic analysis: validate a parsed program and lower it into the IR."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from eigrel.compiler import ast, ir
from eigrel.compiler.catalog import (
    ALGORITHMS,
    DEFAULT_METRICS,
    DEFAULT_SEED,
    DEFAULT_VALIDATION,
    METRICS,
    SOURCES,
    SourceArgKind,
)
from eigrel.compiler.errors import SemanticError
from eigrel.compiler.tokens import Location

ExprType = Literal['bool', 'number', 'string', 'unknown']
ENV_VAR = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
SQL_TABLE = re.compile(r'[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?')
BIGQUERY_TABLE = re.compile(r'[a-z][a-z0-9-]{4,29}\.[A-Za-z0-9_]+\.[A-Za-z0-9_$-]+')
TASKS: tuple[ir.Task, ...] = ('classification', 'regression')


@dataclass
class Dataset:
    op: int
    loc: Location
    # None until a `select` makes the column set known at compile time.
    columns: tuple[str, ...] | None = None
    features: tuple[str, ...] | None = None


@dataclass
class Model:
    decl: ast.ModelDecl
    task: ir.Task
    params: tuple[tuple[str, ir.Value], ...]
    trained: int | None = None
    train_loc: Location | None = None


@dataclass
class Analyzer:
    graph: ir.Graph = field(default_factory=ir.Graph)
    datasets: dict[str, Dataset] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    # Metrics each model is evaluated with anywhere in the program, used to infer its task.
    planned_metrics: dict[str, list[ast.Name]] = field(default_factory=dict)

    def analyze(self, program: ast.Program) -> ir.Graph:
        self._collect_planned_metrics(program)
        for statement in program.statements:
            match statement:
                case ast.DatasetDecl():
                    self._dataset(statement)
                case ast.TransformDecl():
                    self._transform(statement)
                case ast.FeaturesDecl():
                    self._features(statement)
                case ast.ModelDecl():
                    self._model(statement)
                case ast.TrainStmt():
                    self._train(statement)
                case ast.EvaluateStmt():
                    self._evaluate(statement)
        return self.graph

    # Statements

    def _dataset(self, decl: ast.DatasetDecl) -> None:
        self._check_new_name(decl.name, decl.loc)
        call = decl.source
        source = SOURCES.get(call.func)
        if source is None:
            raise SemanticError(
                f"unknown data source '{call.func}'; expected one of {_names(SOURCES)}",
                call.loc,
            )
        if len(call.args) != len(source.args):
            expected = ', '.join(description for description, _ in source.args)
            raise SemanticError(
                f'{call.func}() takes {len(source.args)} argument'
                f'{"" if len(source.args) == 1 else "s"} ({expected}), e.g. {source.example}',
                call.loc,
            )
        args = tuple(
            self._source_arg(arg, description, kind)
            for arg, (description, kind) in zip(call.args, source.args, strict=True)
        )
        op = self.graph.add(ir.Load(self.graph.next_id(), decl.name, call.func, args))
        self.datasets[decl.name] = Dataset(op, decl.loc)

    def _source_arg(self, arg: ast.Expr, description: str, kind: SourceArgKind) -> ir.SourceArg:
        if kind == 'url' and isinstance(arg, ast.Call) and arg.func == 'env':
            if len(arg.args) != 1 or not isinstance(arg.args[0], ast.StringLiteral):
                raise SemanticError('env() takes the name of an environment variable', arg.loc)
            name = arg.args[0].value
            if not ENV_VAR.fullmatch(name):
                raise SemanticError(
                    f"'{name}' is not a valid environment variable name", arg.args[0].loc
                )
            return ir.EnvVar(name)
        if not isinstance(arg, ast.StringLiteral):
            hint = ' or env("VAR")' if kind == 'url' else ''
            raise SemanticError(f'the {description} must be a string{hint}', arg.loc)
        value = arg.value
        if not value:
            raise SemanticError(f'the {description} must not be empty', arg.loc)
        if kind == 'table' and not SQL_TABLE.fullmatch(value):
            raise SemanticError(
                f"'{value}' is not a table name; use table or schema.table", arg.loc
            )
        if kind == 'bigquery_table' and not BIGQUERY_TABLE.fullmatch(value):
            raise SemanticError(
                f"'{value}' is not a BigQuery table; use project.dataset.table", arg.loc
            )
        return value

    def _transform(self, decl: ast.TransformDecl) -> None:
        dataset = self._dataset_named(decl.dataset, decl.loc)
        for op in decl.ops:
            match op:
                case ast.FilterOp():
                    kind = self._expr_type(op.condition, dataset)
                    if kind not in ('bool', 'unknown'):
                        raise SemanticError(
                            f'filter condition must be a boolean expression, not a {kind}',
                            op.condition.loc,
                        )
                    dataset.op = self.graph.add(
                        ir.Filter(self.graph.next_id(), decl.dataset, dataset.op, op.condition)
                    )
                case ast.SelectOp():
                    columns = self._columns(op.columns, dataset)
                    dataset.op = self.graph.add(
                        ir.Select(self.graph.next_id(), decl.dataset, dataset.op, columns)
                    )
                    dataset.columns = columns

    def _features(self, decl: ast.FeaturesDecl) -> None:
        dataset = self._dataset_named(decl.dataset, decl.loc)
        if dataset.features is not None:
            raise SemanticError(f"features for '{decl.dataset}' are already declared", decl.loc)
        if not decl.columns:
            raise SemanticError('features must list at least one column', decl.loc)
        dataset.features = self._columns(decl.columns, dataset)

    def _model(self, decl: ast.ModelDecl) -> None:
        self._check_new_name(decl.name, decl.loc)
        algorithm = ALGORITHMS.get(decl.algorithm)
        if algorithm is None:
            raise SemanticError(
                f"unknown algorithm '{decl.algorithm}'; available: {_names(ALGORITHMS)}",
                decl.loc,
            )
        task: ir.Task | None = None
        params: list[tuple[str, ir.Value]] = []
        for param in decl.params:
            if param.name == 'task':
                task = self._task_param(param, decl.algorithm, algorithm.tasks)
                continue
            kind = algorithm.params.get(param.name)
            if kind is None:
                allowed = _names(['task', *algorithm.params])
                raise SemanticError(
                    f"'{decl.algorithm}' has no parameter '{param.name}'; expected {allowed}",
                    param.loc,
                )
            params.append((param.name, self._positive_number(param, integer=kind == 'int')))
        if task is None:
            task = self._infer_task(decl, algorithm.tasks)
        self.models[decl.name] = Model(decl, task, tuple(params))

    def _train(self, stmt: ast.TrainStmt) -> None:
        model = self._model_named(stmt.model, stmt.loc)
        if model.train_loc is not None:
            raise SemanticError(
                f"model '{stmt.model}' is already trained at line {model.train_loc.line}",
                stmt.loc,
            )
        params = self._params(stmt.params, ('data', 'target', 'validation', 'seed'), 'train')

        data_name = self._resolve_training_data(stmt, params.get('data'))
        dataset = self.datasets[data_name]

        if 'target' not in params:
            raise SemanticError(
                f'train needs a target column, e.g. train {stmt.model} {{ target = label }}',
                stmt.loc,
            )
        target_param = params['target']
        target = self._name_value(target_param, 'target')
        self._check_column(target, target_param.value.loc, dataset)

        if dataset.features is not None:
            if target in dataset.features:
                raise SemanticError(
                    f"target '{target}' is also declared as a feature of '{data_name}'",
                    target_param.value.loc,
                )
            for column in dataset.features:
                self._check_column(column, stmt.loc, dataset, context='feature ')

        validation = DEFAULT_VALIDATION
        if 'validation' in params:
            validation = float(self._number(params['validation']))
            if not 0 < validation < 1:
                raise SemanticError(
                    'validation must be a fraction between 0 and 1, e.g. 0.2',
                    params['validation'].value.loc,
                )
        seed = DEFAULT_SEED
        if 'seed' in params:
            value = self._number(params['seed'], integer=True)
            if value < 0:
                raise SemanticError('seed must not be negative', params['seed'].value.loc)
            seed = int(value)

        model.trained = self.graph.add(
            ir.Train(
                self.graph.next_id(),
                stmt.model,
                dataset.op,
                model.decl.algorithm,
                model.task,
                model.params,
                dataset.features,
                target,
                validation,
                seed,
            )
        )
        model.train_loc = stmt.loc

    def _evaluate(self, stmt: ast.EvaluateStmt) -> None:
        model = self._model_named(stmt.model, stmt.loc)
        if model.trained is None:
            raise SemanticError(
                f"model '{stmt.model}' must be trained before it is evaluated", stmt.loc
            )
        params = self._params(stmt.params, ('metrics',), 'evaluate')
        metrics = DEFAULT_METRICS[model.task]
        if 'metrics' in params:
            names = self._metric_names(params['metrics'])
            for name in names:
                if name.value not in METRICS:
                    raise SemanticError(
                        f"unknown metric '{name.value}'; available: {_names(METRICS)}", name.loc
                    )
                if METRICS[name.value] != model.task:
                    raise SemanticError(
                        f"metric '{name.value}' is for {METRICS[name.value]}, but"
                        f" '{stmt.model}' is a {model.task} model",
                        name.loc,
                    )
            metrics = _unique([name.value for name in names], names, 'metric')
        self.graph.add(ir.Evaluate(self.graph.next_id(), stmt.model, model.trained, metrics))

    # Tasks

    def _collect_planned_metrics(self, program: ast.Program) -> None:
        for statement in program.statements:
            if not isinstance(statement, ast.EvaluateStmt):
                continue
            for param in statement.params:
                if param.name == 'metrics' and isinstance(param.value, ast.ListExpr):
                    names = [item for item in param.value.items if isinstance(item, ast.Name)]
                    self.planned_metrics.setdefault(statement.model, []).extend(names)

    def _infer_task(self, decl: ast.ModelDecl, supported: tuple[ir.Task, ...]) -> ir.Task:
        if len(supported) == 1:
            return supported[0]
        metrics = [m for m in self.planned_metrics.get(decl.name, []) if m.value in METRICS]
        tasks = {METRICS[m.value] for m in metrics}
        if len(tasks) > 1:
            conflicting = next(m for m in metrics if METRICS[m.value] != METRICS[metrics[0].value])
            raise SemanticError(
                f"'{decl.name}' is evaluated with both classification and regression metrics;"
                ' use one kind, or set task = classification|regression',
                conflicting.loc,
            )
        return tasks.pop() if tasks else 'classification'

    def _task_param(
        self, param: ast.Param, algorithm: str, supported: tuple[ir.Task, ...]
    ) -> ir.Task:
        value = self._name_value(param, 'task')
        for task in TASKS:
            if value == task:
                if task not in supported:
                    raise SemanticError(f"'{algorithm}' does not support {task}", param.value.loc)
                return task
        raise SemanticError(
            f"task must be classification or regression, not '{value}'", param.value.loc
        )

    def _resolve_training_data(self, stmt: ast.TrainStmt, data: ast.Param | None) -> str:
        if data is not None:
            name = self._name_value(data, 'data')
            self._dataset_named(name, data.value.loc)
            return name
        with_features = [name for name, d in self.datasets.items() if d.features is not None]
        if len(with_features) == 1:
            return with_features[0]
        if not with_features and len(self.datasets) == 1:
            return next(iter(self.datasets))
        if not self.datasets:
            raise SemanticError('train needs a dataset, but none is declared yet', stmt.loc)
        raise SemanticError(
            f"cannot tell which dataset trains '{stmt.model}'; add data = <dataset>", stmt.loc
        )

    # Expressions

    def _expr_type(self, expr: ast.Expr, dataset: Dataset) -> ExprType:
        match expr:
            case ast.Name():
                self._check_column(expr.value, expr.loc, dataset)
                return 'unknown'
            case ast.BoolLiteral():
                return 'bool'
            case ast.IntLiteral() | ast.FloatLiteral():
                return 'number'
            case ast.StringLiteral():
                return 'string'
            case ast.Unary(op='not'):
                self._expect_type(expr.operand, dataset, 'bool', "'not'")
                return 'bool'
            case ast.Unary():
                self._expect_type(expr.operand, dataset, 'number', f"'{expr.op}'")
                return 'number'
            case ast.Binary(op='and' | 'or'):
                self._expect_type(expr.left, dataset, 'bool', f"'{expr.op}'")
                self._expect_type(expr.right, dataset, 'bool', f"'{expr.op}'")
                return 'bool'
            case ast.Binary(op='==' | '!='):
                left = self._expr_type(expr.left, dataset)
                right = self._expr_type(expr.right, dataset)
                if 'unknown' not in (left, right) and left != right:
                    raise SemanticError(f'cannot compare a {left} with a {right}', expr.loc)
                return 'bool'
            case ast.Binary(op='<' | '<=' | '>' | '>='):
                left = self._expr_type(expr.left, dataset)
                right = self._expr_type(expr.right, dataset)
                if 'bool' in (left, right):
                    raise SemanticError(f"'{expr.op}' cannot order booleans", expr.loc)
                if 'unknown' not in (left, right) and left != right:
                    raise SemanticError(f'cannot compare a {left} with a {right}', expr.loc)
                return 'bool'
            case ast.Binary():
                self._expect_type(expr.left, dataset, 'number', f"'{expr.op}'")
                self._expect_type(expr.right, dataset, 'number', f"'{expr.op}'")
                return 'number'
        raise SemanticError('function calls and lists are not supported in filters', expr.loc)

    def _expect_type(
        self, expr: ast.Expr, dataset: Dataset, expected: ExprType, operator: str
    ) -> None:
        actual = self._expr_type(expr, dataset)
        if actual not in (expected, 'unknown'):
            raise SemanticError(
                f'{operator} expects a {expected}, but this is a {actual}', expr.loc
            )

    # Helpers

    def _check_new_name(self, name: str, loc: Location) -> None:
        previous = self.datasets.get(name) or self.models.get(name)
        if previous is not None:
            where = previous.loc if isinstance(previous, Dataset) else previous.decl.loc
            raise SemanticError(f"'{name}' is already defined at line {where.line}", loc)

    def _dataset_named(self, name: str, loc: Location) -> Dataset:
        if name in self.datasets:
            return self.datasets[name]
        hint = ' (it is a model)' if name in self.models else ''
        raise SemanticError(f"unknown dataset '{name}'{hint}", loc)

    def _model_named(self, name: str, loc: Location) -> Model:
        if name in self.models:
            return self.models[name]
        hint = ' (it is a dataset)' if name in self.datasets else ''
        raise SemanticError(f"unknown model '{name}'{hint}", loc)

    def _columns(self, names: tuple[ast.Name, ...], dataset: Dataset) -> tuple[str, ...]:
        for name in names:
            self._check_column(name.value, name.loc, dataset)
        return _unique([name.value for name in names], list(names), 'column')

    def _check_column(
        self, column: str, loc: Location, dataset: Dataset, context: str = ''
    ) -> None:
        if dataset.columns is not None and column not in dataset.columns:
            raise SemanticError(
                f"{context}column '{column}' does not exist here; available columns:"
                f' {", ".join(dataset.columns)}',
                loc,
            )

    def _params(
        self, params: tuple[ast.Param, ...], allowed: tuple[str, ...], statement: str
    ) -> dict[str, ast.Param]:
        for param in params:
            if param.name not in allowed:
                raise SemanticError(
                    f"{statement} has no parameter '{param.name}'; expected {_names(allowed)}",
                    param.loc,
                )
        return {param.name: param for param in params}

    def _name_value(self, param: ast.Param, what: str) -> str:
        if not isinstance(param.value, ast.Name):
            raise SemanticError(f'{what} must be a name, not a value', param.value.loc)
        return param.value.value

    def _number(self, param: ast.Param, integer: bool = False) -> int | float:
        value, sign = param.value, 1
        if isinstance(value, ast.Unary) and value.op == '-':
            value, sign = value.operand, -1
        if isinstance(value, ast.IntLiteral):
            return sign * value.value
        if isinstance(value, ast.FloatLiteral) and not integer:
            return sign * value.value
        kind = 'an integer' if integer else 'a number'
        raise SemanticError(f"'{param.name}' must be {kind}", param.value.loc)

    def _positive_number(self, param: ast.Param, integer: bool) -> int | float:
        value = self._number(param, integer)
        if value <= 0:
            raise SemanticError(f"'{param.name}' must be greater than 0", param.value.loc)
        return value

    def _metric_names(self, param: ast.Param) -> list[ast.Name]:
        if not isinstance(param.value, ast.ListExpr) or not param.value.items:
            raise SemanticError('metrics must be a non-empty list, e.g. [accuracy, f1]', param.loc)
        names = []
        for item in param.value.items:
            if not isinstance(item, ast.Name):
                raise SemanticError('metrics must be names, e.g. accuracy', item.loc)
            names.append(item)
        return names


def _names(names: Iterable[str]) -> str:
    return ', '.join(sorted(names))


def _unique(values: list[str], nodes: list[ast.Name], what: str) -> tuple[str, ...]:
    seen: set[str] = set()
    for value, node in zip(values, nodes, strict=True):
        if value in seen:
            raise SemanticError(f"{what} '{value}' is listed twice", node.loc)
        seen.add(value)
    return tuple(values)


def analyze(program: ast.Program) -> ir.Graph:
    """Check that a program is meaningful and lower it into the IR graph."""
    return Analyzer().analyze(program)
