"""`eigrel plan`: the consequences of a program, computed from its data before anything runs.

Like `terraform plan`, the plan reads the current state of the world (the data sources), shows
what every step will do to it, and flags what would go wrong. Nothing is trained.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from eigrel import __version__
from eigrel.backends import sql
from eigrel.compiler import EigrelError, analyze, ir, parse
from eigrel.compiler.ir import format_expr
from eigrel.compiler.tokens import Location
from eigrel.probe import Column, Engine, ProbeError, count, engine_for, missing, value_counts

Severity = Literal['error', 'warning', 'info']
Backend = Literal['python', 'spark']

# Algorithms whose Python backend cannot train with missing numeric values. scikit-learn's trees
# and XGBoost handle them; on Spark every algorithm fails, because VectorAssembler rejects nulls.
NO_MISSING_VALUES_IN_PYTHON = {'logistic_regression', 'linear_regression', 'gradient_boosting'}
IMBALANCE = 0.10
MIN_VALIDATION_ROWS_PER_CLASS = 5
CONTINUOUS_DISTINCT = 20
TINY_TRAINING_SET = 30
ROW_DRIFT = 0.5


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    loc: Location | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            'severity': self.severity,
            'code': self.code,
            'message': self.message,
        }
        if self.loc is not None:
            data |= {'line': self.loc.line, 'column': self.loc.column}
        return data


@dataclass
class Step:
    op: ir.Op
    description: str
    rows: int | None
    columns: tuple[Column, ...] | None


@dataclass
class DatasetPlan:
    name: str
    source: str
    columns: tuple[Column, ...] | None
    probed: bool
    steps: list[Step] = field(default_factory=list)


@dataclass
class TrainPlan:
    op: ir.Train
    dataset: str
    rows: int | None = None
    train_rows: int | None = None
    validation_rows: int | None = None
    stratified: bool | None = None
    features: tuple[str, ...] | None = None
    missing: dict[str, int] = field(default_factory=dict)
    classes: list[tuple[Any, int]] | None = None
    distinct: int | None = None


@dataclass
class Plan:
    program: str
    backend: Backend
    datasets: list[DatasetPlan] = field(default_factory=list)
    trains: list[TrainPlan] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    graph: ir.Graph | None = None
    state_path: Path | None = None
    state_exists: bool = False

    @property
    def ok(self) -> bool:
        return not any(f.severity == 'error' for f in self.findings)

    def add(self, severity: Severity, code: str, message: str, loc: Location | None) -> None:
        self.findings.append(Finding(severity, code, message, loc))


def build_plan(path: Path, backend: Backend = 'python') -> Plan:
    """Analyze a program against its data. Errors become findings; nothing is raised."""
    plan = Plan(str(path), backend, state_path=path.with_suffix('.eigstate'))
    try:
        source = path.read_text(encoding='utf-8')
    except OSError as exc:
        plan.add('error', 'io', f'cannot read {path}: {exc.strerror}', None)
        return plan
    try:
        program = parse(source)
        graph = analyze(program)
    except EigrelError as exc:
        plan.add('error', type(exc).__name__.removesuffix('Error').lower(), exc.message, exc.loc)
        return plan

    engines: dict[str, Engine] = {}
    probes: dict[str, Engine] = {}
    schemas: dict[str, tuple[Column, ...]] = {}
    for load in (op for op in graph.ops if isinstance(op, ir.Load)):
        source_text = ir.format_source(load)
        try:
            engine = engine_for(load, path.parent, engines)
            schemas[load.name] = engine.columns(load)
            probes[load.name] = engine
        except ProbeError as exc:
            not_probed = load.format == 'bigquery'
            severity: Severity = 'info' if not_probed else 'error'
            plan.add(severity, 'not-probed' if not_probed else 'source', str(exc), load.loc)
        plan.datasets.append(
            DatasetPlan(load.name, source_text, schemas.get(load.name), load.name in probes)
        )
    if not plan.ok:
        return plan

    # Check the program again, now against the columns and types the data really has.
    try:
        graph = analyze(
            program, {name: tuple((c.name, c.type) for c in cols) for name, cols in schemas.items()}
        )
    except EigrelError as exc:
        plan.add('error', 'schema', exc.message, exc.loc)
        return plan
    plan.graph = graph

    datasets = {d.name: d for d in plan.datasets}
    columns: dict[str, tuple[Column, ...] | None] = dict(schemas)
    for op in graph.ops:
        match op:
            case ir.Load() | ir.Filter() | ir.Select() | ir.Fill() | ir.DropMissing():
                _step(plan, datasets[op.name], op, graph, probes.get(op.name), columns, schemas)
            case ir.Train():
                trainer = probes.get(_dataset_of(graph, op.input))
                _train(plan, op, graph, trainer, columns, schemas)
            case ir.Evaluate() | ir.Register():
                pass
    _compare_state(plan, schemas, datasets)
    return plan


def _step(
    plan: Plan,
    dataset: DatasetPlan,
    op: ir.Op,
    graph: ir.Graph,
    engine: Engine | None,
    columns: dict[str, tuple[Column, ...] | None],
    schemas: dict[str, tuple[Column, ...]],
) -> None:
    current = columns.get(op.name)
    if isinstance(op, ir.Select) and current is not None:
        by_name = {c.name: c for c in current}
        current = tuple(by_name[c] for c in op.columns)
    columns[op.name] = current
    rows = None
    if engine is not None:
        try:
            rows = count(engine, _query(graph, op.id, schemas))
        except ProbeError as exc:
            plan.add('error', 'probe', f'{dataset.name}: {exc}', op.loc)
    dataset.steps.append(Step(op, _describe(op), rows, current))
    if rows == 0 and not isinstance(op, ir.Load | ir.Select):
        plan.add('error', 'empty', f"'{op.name}' has no rows left after this step", op.loc)
    elif rows == 0:
        plan.add('error', 'empty', f"'{op.name}' has no rows", op.loc)


def _train(
    plan: Plan,
    op: ir.Train,
    graph: ir.Graph,
    engine: Engine | None,
    columns: dict[str, tuple[Column, ...] | None],
    schemas: dict[str, tuple[Column, ...]],
) -> None:
    dataset = _dataset_of(graph, op.input)
    known = columns.get(dataset)
    train = TrainPlan(op, dataset)
    plan.trains.append(train)
    if op.features is not None:
        train.features = op.features
    elif known is not None:
        train.features = tuple(c.name for c in known if c.name != op.target)
    if engine is None:
        return
    quote = sql.dialect_for(_load_of(graph, op.input)).quote
    try:
        query = _query(graph, op.input, schemas)
        train.rows = count(engine, query)
        involved = [*(train.features or ()), op.target]
        train.missing = missing(engine, query, involved, quote)
        if op.task == 'classification':
            train.classes, train.distinct = value_counts(engine, query, op.target, quote)
    except ProbeError as exc:
        plan.add('error', 'probe', f'{op.name}: {exc}', op.loc)
        return

    rows = train.rows
    train.validation_rows = math.ceil(rows * op.validation)
    train.train_rows = rows - train.validation_rows
    types = {c.name: c.type for c in known or ()}
    _check_training(plan, train, types)


def _check_training(plan: Plan, train: TrainPlan, types: Mapping[str, str]) -> None:
    op = train.op
    rows = train.rows or 0
    if rows < TINY_TRAINING_SET:
        plan.add('warning', 'tiny', f"'{op.name}' trains on only {rows} rows", op.loc)
    target_missing = train.missing.get(op.target, 0)
    if target_missing:
        plan.add(
            'error',
            'target-missing',
            f"target '{op.target}' has {target_missing} missing values and training will fail;"
            f' add `drop_missing {op.target}` before training',
            op.loc,
        )
    numeric_missing = {
        c: n
        for c, n in train.missing.items()
        if n and c != op.target and types.get(c, 'number') != 'string'
    }
    cannot_handle = plan.backend == 'spark' or op.algorithm in NO_MISSING_VALUES_IN_PYTHON
    if numeric_missing and cannot_handle:
        listed = ', '.join(f'{c} ({n})' for c, n in numeric_missing.items())
        where = 'on Spark' if plan.backend == 'spark' else f'with {op.algorithm}'
        plan.add(
            'error',
            'missing-values',
            f'features with missing values cannot be trained {where}: {listed};'
            ' use fill or drop_missing first',
            op.loc,
        )
    if op.task == 'regression' and types.get(op.target) == 'string':
        plan.add(
            'error',
            'text-target',
            f"regression needs a numeric target, but '{op.target}' is text",
            op.loc,
        )
    if op.task != 'classification' or not train.classes:
        return
    distinct = train.distinct or 0
    if distinct == 1:
        value = train.classes[0][0]
        plan.add(
            'error',
            'one-class',
            f"target '{op.target}' has a single class ({value!r}); there is nothing to learn",
            op.loc,
        )
        return
    if types.get(op.target) == 'number' and distinct > CONTINUOUS_DISTINCT:
        plan.add(
            'warning',
            'continuous-target',
            f"target '{op.target}' has {distinct} distinct numeric values; did you mean"
            ' task = regression?',
            op.loc,
        )
        return
    if plan.backend == 'spark' and op.algorithm == 'gradient_boosting' and distinct > 2:
        plan.add(
            'error',
            'capability',
            f"gradient_boosting on Spark supports two classes, but '{op.target}' has {distinct}",
            op.loc,
        )
    smallest_value, smallest = min(train.classes, key=lambda pair: pair[1])
    largest_value, largest = train.classes[0]
    total = sum(n for _, n in train.classes)
    train.stratified = plan.backend == 'python' and smallest >= 2
    if plan.backend == 'python' and smallest < 2:
        plan.add(
            'warning',
            'no-stratify',
            f'class {smallest_value!r} has {smallest} row, so the split cannot be stratified',
            op.loc,
        )
    if total and smallest / total < IMBALANCE:
        plan.add(
            'warning',
            'imbalance',
            f'class imbalance: {smallest_value!r} has {smallest} rows'
            f' ({smallest / total:.0%}) against {largest} for {largest_value!r}',
            op.loc,
        )
    in_validation = math.floor(smallest * op.validation)
    if in_validation < MIN_VALIDATION_ROWS_PER_CLASS:
        expected = f'about {in_validation}' if in_validation else 'less than one'
        plan.add(
            'warning',
            'small-validation',
            f'{expected} row(s) of class {smallest_value!r} expected in the validation split'
            f' ({op.validation:.0%}); its metrics will be unstable',
            op.loc,
        )


# State and drift


def fingerprint(columns: tuple[Column, ...]) -> str:
    data = json.dumps([[c.name, c.native] for c in columns])
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def state_document(plan: Plan) -> dict[str, Any]:
    sources = {}
    for dataset in plan.datasets:
        if dataset.columns is None:
            continue
        rows = dataset.steps[0].rows if dataset.steps else None
        sources[dataset.name] = {
            'source': dataset.source,
            'columns': [[c.name, c.native] for c in dataset.columns],
            'rows': rows,
            'fingerprint': fingerprint(dataset.columns),
        }
    return {'eigrel': __version__, 'program': Path(plan.program).name, 'datasets': sources}


def save_state(plan: Plan) -> Path:
    assert plan.state_path is not None
    plan.state_path.write_text(json.dumps(state_document(plan), indent=2) + '\n', encoding='utf-8')
    return plan.state_path


def _compare_state(
    plan: Plan, schemas: dict[str, tuple[Column, ...]], datasets: dict[str, DatasetPlan]
) -> None:
    assert plan.state_path is not None
    if not plan.state_path.exists():
        return
    plan.state_exists = True
    try:
        recorded = json.loads(plan.state_path.read_text(encoding='utf-8'))['datasets']
    except (OSError, ValueError, KeyError, TypeError):
        plan.add('warning', 'state', f'{plan.state_path.name} is not a valid state file', None)
        return
    for name, columns in schemas.items():
        load = datasets[name].steps[0].op if datasets[name].steps else None
        loc = load.loc if load is not None else None
        before = recorded.get(name)
        if before is None:
            plan.add('info', 'drift-new', f"'{name}' is not in the recorded state yet", loc)
            continue
        old = {c: t for c, t in before.get('columns', [])}
        new = {c.name: c.native for c in columns}
        removed = [c for c in old if c not in new]
        added = [c for c in new if c not in old]
        changed = [f'{c} {old[c]} → {new[c]}' for c in new if c in old and old[c] != new[c]]
        if removed:
            plan.add(
                'warning', 'drift-removed', f"'{name}' lost columns: {', '.join(removed)}", loc
            )
        if changed:
            plan.add(
                'warning', 'drift-type', f"'{name}' column types changed: {', '.join(changed)}", loc
            )
        if added:
            plan.add('info', 'drift-added', f"'{name}' has new columns: {', '.join(added)}", loc)
        old_rows, new_rows = before.get('rows'), datasets[name].steps[0].rows
        if isinstance(old_rows, int) and isinstance(new_rows, int) and old_rows != new_rows:
            change = (new_rows - old_rows) / old_rows if old_rows else math.inf
            severity: Severity = 'warning' if abs(change) > ROW_DRIFT else 'info'
            plan.add(severity, 'drift-rows', f"'{name}' rows changed: {old_rows} → {new_rows}", loc)


# Graph helpers


def _chain(graph: ir.Graph, op_id: int) -> tuple[ir.Load, tuple[sql.DatasetOp, ...]]:
    ops: list[sql.DatasetOp] = []
    op = graph.ops[op_id]
    while not isinstance(op, ir.Load):
        assert isinstance(op, ir.Filter | ir.Select | ir.Fill | ir.DropMissing)
        ops.append(op)
        op = graph.ops[op.input]
    return op, tuple(reversed(ops))


def _query(graph: ir.Graph, op_id: int, schemas: dict[str, tuple[Column, ...]]) -> str:
    load, ops = _chain(graph, op_id)
    known = schemas.get(load.name)
    columns = tuple(c.name for c in known) if known is not None else None
    return sql.render(sql.Query(load.name, load, ops, columns))


def _load_of(graph: ir.Graph, op_id: int) -> ir.Load:
    return _chain(graph, op_id)[0]


def _dataset_of(graph: ir.Graph, op_id: int) -> str:
    return graph.ops[op_id].name


def _describe(op: ir.Op) -> str:
    match op:
        case ir.Load():
            return f'load {ir.format_source(op)}'
        case ir.Filter():
            return f'filter {format_expr(op.condition)}'
        case ir.Select():
            return f'select {", ".join(op.columns)}'
        case ir.Fill():
            return 'fill ' + ', '.join(f'{c} = {ir.format_value(v)}' for c, v in op.values)
        case ir.DropMissing():
            return 'drop_missing ' + (', '.join(op.columns) if op.columns else '(any column)')
    raise AssertionError(f'not a dataset step: {op!r}')  # pragma: no cover


# Output


def to_json(plan: Plan) -> dict[str, Any]:
    def columns_json(columns: tuple[Column, ...] | None) -> list[dict[str, str]] | None:
        if columns is None:
            return None
        return [{'name': c.name, 'type': c.type, 'native': c.native} for c in columns]

    def step_json(step: Step) -> dict[str, Any]:
        loc = step.op.loc
        data: dict[str, Any] = {'op': step.description, 'rows': step.rows}
        data['columns'] = [c.name for c in step.columns] if step.columns is not None else None
        if loc is not None:
            data['line'] = loc.line
        return data

    return {
        'eigrel': __version__,
        'program': plan.program,
        'backend': plan.backend,
        'ok': plan.ok,
        'datasets': [
            {
                'name': d.name,
                'source': d.source,
                'probed': d.probed,
                'columns': columns_json(d.columns),
                'steps': [step_json(s) for s in d.steps],
            }
            for d in plan.datasets
        ],
        'trainings': [
            {
                'model': t.op.name,
                'algorithm': t.op.algorithm,
                'task': t.op.task,
                'dataset': t.dataset,
                'target': t.op.target,
                'features': list(t.features) if t.features is not None else None,
                'rows': t.rows,
                'train_rows': t.train_rows,
                'validation_rows': t.validation_rows,
                'stratified': t.stratified,
                'missing': t.missing,
                'classes': (
                    [{'value': v, 'rows': n} for v, n in t.classes]
                    if t.classes is not None
                    else None
                ),
                'distinct': t.distinct,
            }
            for t in plan.trains
        ],
        'findings': [f.to_json() for f in plan.findings],
        'state': {
            'path': str(plan.state_path) if plan.state_path else None,
            'exists': plan.state_exists,
        },
    }


SYMBOLS = {'error': '✗', 'warning': '!', 'info': '·'}
# For terminals and pipes that cannot encode the symbols (e.g. cp1252 on Windows).
ASCII = str.maketrans({'✗': 'x', '·': '-', '✓': 'ok', '→': '->', '←': '<-', '…': '...'})


def render_for(plan: Plan, encoding: str | None) -> str:
    """The rendered plan, with ASCII symbols if `encoding` cannot represent the others."""
    text = render(plan)
    try:
        text.encode(encoding or 'ascii')
    except (UnicodeEncodeError, LookupError):
        return text.translate(ASCII)
    return text


def render(plan: Plan) -> str:
    lines = [f'Plan for {plan.program} ({plan.backend} backend)', '']
    for dataset in plan.datasets:
        lines.append(f'dataset {dataset.name} ← {dataset.source}')
        if dataset.columns is not None:
            described = ', '.join(f'{c.name} {c.native.lower()}' for c in dataset.columns)
            lines.append(f'  columns: {described}')
        previous: int | None = None
        for step in dataset.steps:
            if isinstance(step.op, ir.Load):
                if step.rows is not None:
                    lines.append(f'  rows: {step.rows}')
                previous = step.rows
                continue
            lines.append(f'  {step.description:<44} {_rows(previous, step.rows)}')
            previous = step.rows
        lines.append('')
    for train in plan.trains:
        op = train.op
        lines.append(f'train {op.name}: {op.algorithm} {op.task} on {train.dataset}')
        if train.rows is not None:
            approx = '' if plan.backend == 'python' else '~'
            how = ', stratified' if train.stratified else ''
            lines.append(
                f'  split: {approx}{train.train_rows} train / {approx}{train.validation_rows}'
                f' validation ({op.validation:.0%}{how})'
            )
        if train.features is not None:
            lines.append(f'  features: {", ".join(train.features)}')
        if train.classes:
            total = sum(n for _, n in train.classes)
            shown = ', '.join(f'{v!r} {n} ({n / total:.0%})' for v, n in train.classes[:6])
            more = f', … {train.distinct} classes' if (train.distinct or 0) > 6 else ''
            lines.append(f'  target {op.target}: {shown}{more}')
        gaps = {c: n for c, n in train.missing.items() if n}
        if train.rows is not None:
            if gaps:
                lines.append('  missing values: ' + ', '.join(f'{c} {n}' for c, n in gaps.items()))
            else:
                lines.append('  missing values: none')
        lines.append('')
    for finding in plan.findings:
        where = f'line {finding.loc.line}: ' if finding.loc is not None else ''
        lines.append(f'{SYMBOLS[finding.severity]} {finding.severity}: {where}{finding.message}')
    errors = sum(f.severity == 'error' for f in plan.findings)
    warnings = sum(f.severity == 'warning' for f in plan.findings)
    if errors or warnings:
        lines.append(f'{errors} error(s), {warnings} warning(s)')
    else:
        lines.append('✓ no problems found')
    if plan.state_path is not None and plan.ok and not plan.state_exists:
        lines.append(
            f'state: no {plan.state_path.name} yet; `eigrel plan --save` records the data schema'
            ' so later plans can detect drift'
        )
    return '\n'.join(lines).rstrip('\n') + '\n'


def _rows(before: int | None, after: int | None) -> str:
    if after is None:
        return ''
    if before is None or before == after:
        return f'{after} rows'
    return f'{before} → {after} rows ({after - before:+d})'
