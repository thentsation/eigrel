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
    # None until a `select` (or the source's schema) makes the column set known at compile time.
    columns: tuple[str, ...] | None = None
    features: tuple[str, ...] | None = None
    # Column types, known when the source's schema is.
    types: dict[str, ExprType] = field(default_factory=dict)
    # Declared with `assumptions`: the column that orders rows in time.
    time: str | None = None
    assumptions_loc: Location | None = None
    # Fills applied so far, in order, so `predict` can replay them on serving data.
    fills: list[tuple[str, ir.Value]] = field(default_factory=list)


@dataclass
class Model:
    decl: ast.ModelDecl
    task: ir.Task
    params: tuple[tuple[str, ir.Value], ...]
    trained: int | None = None
    train_loc: Location | None = None
    evaluated: int | None = None
    register_loc: Location | None = None
    # Known at training time: the feature columns (and their types) the model expects.
    features: tuple[str, ...] | None = None
    feature_types: dict[str, ExprType] = field(default_factory=dict)
    fills: tuple[tuple[str, ir.Value], ...] = ()


@dataclass
class Analyzer:
    graph: ir.Graph = field(default_factory=ir.Graph)
    datasets: dict[str, Dataset] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    # Metrics each model is evaluated with anywhere in the program, used to infer its task.
    planned_metrics: dict[str, list[ast.Name]] = field(default_factory=dict)
    # Schemas read from the data by `eigrel plan`: dataset name -> (column, type) pairs.
    schemas: dict[str, tuple[tuple[str, ExprType], ...]] = field(default_factory=dict)

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
                case ast.RegisterStmt():
                    self._register(statement)
                case ast.AssumptionsDecl():
                    self._assumptions(statement)
                case ast.PredictStmt():
                    self._predict(statement)
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
        op = self.graph.add(ir.Load(self.graph.next_id(), decl.name, call.func, args, loc=decl.loc))
        dataset = Dataset(op, decl.loc)
        if decl.name in self.schemas:
            dataset.columns = tuple(column for column, _ in self.schemas[decl.name])
            dataset.types = dict(self.schemas[decl.name])
        self.datasets[decl.name] = dataset

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
                        ir.Filter(
                            self.graph.next_id(),
                            decl.dataset,
                            dataset.op,
                            op.condition,
                            loc=op.loc,
                        )
                    )
                case ast.SelectOp():
                    columns = self._columns(op.columns, dataset)
                    dataset.op = self.graph.add(
                        ir.Select(
                            self.graph.next_id(), decl.dataset, dataset.op, columns, loc=op.loc
                        )
                    )
                    dataset.columns = columns
                    dataset.types = {c: t for c, t in dataset.types.items() if c in columns}
                case ast.FillOp():
                    values = []
                    for param in op.values:
                        self._check_column(param.name, param.loc, dataset)
                        value = self._literal(param.value)
                        self._check_fill_type(param, value, dataset)
                        values.append((param.name, value))
                    dataset.fills.extend(values)
                    _unique(
                        [column for column, _ in values],
                        [ast.Name(p.name, loc=p.loc) for p in op.values],
                        'column',
                    )
                    dataset.op = self.graph.add(
                        ir.Fill(
                            self.graph.next_id(),
                            decl.dataset,
                            dataset.op,
                            tuple(values),
                            loc=op.loc,
                        )
                    )
                case ast.DropMissingOp():
                    dropped = self._columns(op.columns, dataset) if op.columns else None
                    dataset.op = self.graph.add(
                        ir.DropMissing(
                            self.graph.next_id(), decl.dataset, dataset.op, dropped, loc=op.loc
                        )
                    )

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
                dataset.time,
                loc=stmt.loc,
            )
        )
        model.train_loc = stmt.loc
        if dataset.features is not None:
            model.features = dataset.features
        elif dataset.columns is not None:
            excluded = {target, dataset.time}
            model.features = tuple(c for c in dataset.columns if c not in excluded)
        if model.features is not None:
            model.feature_types = {
                c: dataset.types[c] for c in model.features if c in dataset.types
            }
        model.fills = tuple(dataset.fills)

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
        model.evaluated = self.graph.add(
            ir.Evaluate(self.graph.next_id(), stmt.model, model.trained, metrics, loc=stmt.loc)
        )

    def _assumptions(self, decl: ast.AssumptionsDecl) -> None:
        dataset = self._dataset_named(decl.dataset, decl.loc)
        if dataset.assumptions_loc is not None:
            raise SemanticError(
                f"assumptions for '{decl.dataset}' are already declared at line"
                f' {dataset.assumptions_loc.line}',
                decl.loc,
            )
        params = self._params(decl.params, ('time',), 'assumptions')
        if 'time' not in params:
            raise SemanticError(
                f'assumptions needs a time column, e.g. assumptions {decl.dataset} {{ time = day }}',
                decl.loc,
            )
        column = self._name_value(params['time'], 'time')
        self._check_column(column, params['time'].value.loc, dataset)
        if dataset.types.get(column) == 'string':
            raise SemanticError(
                f"time column '{column}' is text; use a date, timestamp or number column",
                params['time'].value.loc,
            )
        dataset.time = column
        dataset.assumptions_loc = decl.loc

    def _predict(self, stmt: ast.PredictStmt) -> None:
        model = self._model_named(stmt.model, stmt.loc)
        if model.trained is None:
            raise SemanticError(
                f"model '{stmt.model}' must be trained before it predicts", stmt.loc
            )
        params = self._params(stmt.params, ('data', 'output'), 'predict')
        if 'data' not in params or 'output' not in params:
            raise SemanticError(
                f'predict needs data and output, e.g. predict {stmt.model}'
                ' { data = new_rows, output = csv("scored.csv") }',
                stmt.loc,
            )
        data_name = self._name_value(params['data'], 'data')
        dataset = self._dataset_named(data_name, params['data'].value.loc)
        output = params['output'].value
        if (
            not isinstance(output, ast.Call)
            or output.func not in ('csv', 'parquet', 'json')
            or len(output.args) != 1
            or not isinstance(output.args[0], ast.StringLiteral)
            or not output.args[0].value
        ):
            raise SemanticError(
                'output must be a file: csv("path"), parquet("path") or json("path")', output.loc
            )
        # Train/serve skew: the serving data must provide every feature the model was trained
        # on, with the same types, and receives the same fills as the training data.
        if model.features is not None and dataset.columns is not None:
            lacking = [c for c in model.features if c not in dataset.columns]
            if lacking:
                raise SemanticError(
                    f"'{data_name}' lacks features the model was trained on: {', '.join(lacking)};"
                    ' serving data must provide every training feature',
                    params['data'].value.loc,
                )
        for column, expected in model.feature_types.items():
            actual = dataset.types.get(column, 'unknown')
            if 'unknown' not in (expected, actual) and expected != actual:
                raise SemanticError(
                    f"feature '{column}' is a {expected} in the training data but a {actual}"
                    f" in '{data_name}'",
                    params['data'].value.loc,
                )
        fills = tuple(
            (column, value)
            for column, value in model.fills
            if model.features is None or column in model.features
        )
        self.graph.add(
            ir.Predict(
                self.graph.next_id(),
                stmt.model,
                model.trained,
                dataset.op,
                fills,
                output.func,
                output.args[0].value,
                loc=stmt.loc,
            )
        )

    def _register(self, stmt: ast.RegisterStmt) -> None:
        model = self._model_named(stmt.model, stmt.loc)
        if model.trained is None:
            raise SemanticError(
                f"model '{stmt.model}' must be trained before it is registered", stmt.loc
            )
        if model.register_loc is not None:
            raise SemanticError(
                f"model '{stmt.model}' is already registered at line {model.register_loc.line}",
                stmt.loc,
            )
        params = self._params(stmt.params, ('name', 'experiment'), 'register')
        name = self._text(params['name']) if 'name' in params else stmt.model
        experiment = self._text(params['experiment']) if 'experiment' in params else None
        self.graph.add(
            ir.Register(
                self.graph.next_id(),
                stmt.model,
                model.trained,
                name,
                experiment,
                model.evaluated,
                loc=stmt.loc,
            )
        )
        model.register_loc = stmt.loc

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
                return dataset.types.get(expr.value, 'unknown')
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

    def _literal(self, expr: ast.Expr) -> ir.Value:
        """A constant: a number (possibly negative), string or boolean."""
        match expr:
            case ast.IntLiteral(value=v) | ast.FloatLiteral(value=v):
                return v
            case ast.StringLiteral(value=v) | ast.BoolLiteral(value=v):
                return v
            case ast.Unary(op='-', operand=ast.IntLiteral(value=v) | ast.FloatLiteral(value=v)):
                return -v
        raise SemanticError('fill values must be constants, e.g. 0, "unknown" or false', expr.loc)

    def _check_fill_type(self, param: ast.Param, value: ir.Value, dataset: Dataset) -> None:
        expected = dataset.types.get(param.name, 'unknown')
        actual: ExprType = (
            'bool' if isinstance(value, bool) else 'string' if isinstance(value, str) else 'number'
        )
        if expected != 'unknown' and expected != actual:
            raise SemanticError(
                f"cannot fill the {expected} column '{param.name}' with a {actual}",
                param.value.loc,
            )

    def _text(self, param: ast.Param) -> str:
        if not isinstance(param.value, ast.StringLiteral) or not param.value.value:
            raise SemanticError(f"'{param.name}' must be a non-empty string", param.value.loc)
        return param.value.value

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


def analyze(
    program: ast.Program, schemas: dict[str, tuple[tuple[str, ExprType], ...]] | None = None
) -> ir.Graph:
    """Check that a program is meaningful and lower it into the IR graph.

    `schemas` holds the columns of each dataset's source when they are known (from `eigrel plan`),
    which lets the analysis check column names and types against the real data.
    """
    return Analyzer(schemas=dict(schemas or {})).analyze(program)
