"""Python backend: turn the IR into a readable pandas + scikit-learn script."""

from dataclasses import dataclass, field

from eigrel import __version__
from eigrel.backends import Requirement, sql
from eigrel.backends.naming import assign_names
from eigrel.compiler import ast, ir


def runtime_requirements(graph: ir.Graph) -> list[Requirement]:
    """The modules a program needs at run time, depending on its sources and models."""
    formats = {op.format for op in graph.ops if isinstance(op, ir.Load)}
    requirements = [Requirement('pandas', 'python')]
    if any(isinstance(op, ir.Train) for op in graph.ops):
        requirements.append(Requirement('sklearn', 'python'))
    if 'parquet' in formats:
        requirements.append(Requirement('pyarrow', 'python'))
    if 'sql' in formats:
        requirements.append(Requirement('sqlalchemy', 'sql'))
    if 'bigquery' in formats:
        requirements.append(Requirement('google.cloud.bigquery', 'bigquery'))
        requirements.append(Requirement('db_dtypes', 'bigquery'))
    if any(isinstance(op, ir.Train) and op.algorithm == 'xgboost' for op in graph.ops):
        requirements.append(Requirement('xgboost', 'xgboost'))
    if any(isinstance(op, ir.Register) for op in graph.ops):
        requirements.append(Requirement('mlflow', 'mlflow'))
    return requirements


# Eigrel algorithm -> (module, classifier class, regressor class, accepts random_state)
ESTIMATORS: dict[str, tuple[str, str | None, str | None, bool]] = {
    'random_forest': (
        'sklearn.ensemble',
        'RandomForestClassifier',
        'RandomForestRegressor',
        True,
    ),
    'decision_tree': ('sklearn.tree', 'DecisionTreeClassifier', 'DecisionTreeRegressor', True),
    'gradient_boosting': (
        'sklearn.ensemble',
        'GradientBoostingClassifier',
        'GradientBoostingRegressor',
        True,
    ),
    'xgboost': ('xgboost', 'XGBClassifier', 'XGBRegressor', True),
    'logistic_regression': ('sklearn.linear_model', 'LogisticRegression', None, True),
    'linear_regression': ('sklearn.linear_model', None, 'LinearRegression', False),
}

# Eigrel parameter name -> scikit-learn keyword, where they differ.
PARAM_NAMES = {'trees': 'n_estimators', 'c': 'C'}

# Metric -> (scikit-learn function, uses the classification `average` argument)
METRIC_FUNCTIONS = {
    'accuracy': ('accuracy_score', False),
    'precision': ('precision_score', True),
    'recall': ('recall_score', True),
    'f1': ('f1_score', True),
    'auc': ('roc_auc_score', False),
    'mae': ('mean_absolute_error', False),
    'mse': ('mean_squared_error', False),
    'rmse': ('root_mean_squared_error', False),
    'r2': ('r2_score', False),
}

BINARY_OPERATORS = {'and': '&', 'or': '|'}

# Variables the generated code creates for each model, as suffixes of the model's name.
MODEL_HELPERS = (
    '_X',
    '_y',
    '_X_train',
    '_X_test',
    '_y_train',
    '_y_test',
    '_pred',
    '_average',
    '_proba',
    '_scores',
    '_metrics',
    '_encoder',
    '_registered',
    '_text',
    '_rows',
    '_input',
    '_scored',
)

# Names the generated code imports or calls; Eigrel names must not shadow them.
RESERVED = {
    'pd',
    'mlflow',
    'skops',
    'logging',
    'warnings',
    'LabelEncoder',
    'Pipeline',
    'ColumnTransformer',
    'OneHotEncoder',
    'os',
    'bigquery',
    'create_engine',
    'NullPool',
    'connection',
    'print',
    'len',
    'set',
    'train_test_split',
    *(function for function, _ in METRIC_FUNCTIONS.values()),
    *(cls for _, *classes, _ in ESTIMATORS.values() for cls in classes if cls is not None),
}


@dataclass
class _Generator:
    graph: ir.Graph
    imports: dict[str, set[str]] = field(default_factory=dict)
    modules: set[str] = field(default_factory=lambda: {'pandas as pd'})
    lines: list[str] = field(default_factory=list)
    # Python variable holding the value of each IR op.
    variables: dict[int, str] = field(default_factory=dict)
    # Eigrel name -> Python variable name.
    names: dict[str, str] = field(default_factory=dict)
    # Models wrapped in eigrel.runtime.EncodedLabelClassifier (XGBoost classifiers); registering
    # them declares eigrel as a requirement, since loading them imports that class.
    encoded: set[int] = field(default_factory=set)

    def generate(self, source_name: str) -> str:
        self.names = assign_names(self.graph, RESERVED, model_helpers=MODEL_HELPERS)
        for op in self.graph.ops:
            match op:
                case ir.Load():
                    self._load(op)
                case ir.Filter():
                    self._filter(op)
                case ir.Select():
                    self._select(op)
                case ir.Fill():
                    self._fill(op)
                case ir.DropMissing():
                    self._drop_missing(op)
                case ir.Train():
                    self._train(op)
                case ir.Evaluate():
                    self._evaluate(op)
                case ir.Register():
                    self._register(op)
                case ir.Predict():
                    self._predict(op)
        header = [
            f'"""Generated by eigrel {__version__} from {source_name}. Do not edit."""',
            '',
            *(f'import {module}' for module in sorted(self.modules)),
        ]
        for module in sorted(self.imports):
            header.append(f'from {module} import {", ".join(sorted(self.imports[module]))}')
        return '\n'.join([*header, *self.lines]) + '\n'

    def _load(self, op: ir.Load) -> None:
        name = self._bind(op)
        self._section(f'Dataset {op.name}')
        location = op.args[0]
        match op.format:
            case 'csv':
                read = f'pd.read_csv({location!r})'
            case 'parquet':
                read = f'pd.read_parquet({location!r})'
            case 'json':
                lines = str(location).lower().endswith(('.jsonl', '.ndjson'))
                read = f'pd.read_json({location!r}{", lines=True" if lines else ""})'
            case 'sql':
                # NullPool closes the connection as soon as the read is done.
                self._import('sqlalchemy', 'create_engine')
                self._import('sqlalchemy.pool', 'NullPool')
                engine = f'create_engine({self._source_value(location)}, poolclass=NullPool)'
                self.lines += [
                    f'with {engine}.connect() as connection:',
                    f'    {name} = pd.read_sql_query({sql.select_all(op)!r}, connection)',
                ]
                return
            case 'bigquery':
                self._import('google.cloud', 'bigquery')
                read = f'bigquery.Client().query({sql.select_all(op)!r}).to_dataframe()'
            case _:  # pragma: no cover
                raise AssertionError(f'semantic analysis rejects source {op.format!r}')
        self.lines.append(f'{name} = {read}')

    def _source_value(self, value: ir.SourceArg) -> str:
        if isinstance(value, ir.EnvVar):
            self.modules.add('os')
            return f'os.environ[{value.name!r}]'
        return repr(value)

    def _filter(self, op: ir.Filter) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        self.lines.append(f'{name} = {source}[{self._expr(op.condition, source)}]')

    def _select(self, op: ir.Select) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        self.lines.append(f'{name} = {source}[{list(op.columns)!r}]')

    def _fill(self, op: ir.Fill) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        self.lines.append(f'{name} = {source}.fillna({dict(op.values)!r})')

    def _drop_missing(self, op: ir.DropMissing) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        subset = '' if op.columns is None else f'subset={list(op.columns)!r}'
        self.lines.append(f'{name} = {source}.dropna({subset})')

    def _train(self, op: ir.Train) -> None:
        data = self.variables[op.input]
        model = self._bind(op)
        module, classifier, regressor, seeded = ESTIMATORS[op.algorithm]
        estimator = classifier if op.task == 'classification' else regressor
        assert estimator is not None, 'semantic analysis rejects unsupported tasks'
        self._import(module, estimator)
        self._import('sklearn.model_selection', 'train_test_split')
        self._import('sklearn.pipeline', 'Pipeline')
        self._import('sklearn.compose', 'ColumnTransformer')
        self._import('sklearn.preprocessing', 'OneHotEncoder')

        kwargs = [f'{PARAM_NAMES.get(key, key)}={value!r}' for key, value in op.params]
        if seeded:
            kwargs.append(f'random_state={op.seed}')
        self._section(f'Train {op.name}: {op.algorithm} ({op.task})')
        if op.time is not None:
            # Declared time column: train on the earliest rows, validate on the latest ones.
            self.lines.append(f"{model}_rows = {data}.sort_values({op.time!r}, kind='stable')")
            data = f'{model}_rows'
        if op.features is None:
            dropped = [op.target] if op.time is None else [op.target, op.time]
            features = f'{data}.drop(columns={dropped!r})'
        else:
            features = f'{data}[{list(op.features)!r}]'
        if op.time is not None:
            split = ['    shuffle=False,']
        else:
            stratify = 'None'
            if op.task == 'classification':
                stratify = f'{model}_y if {model}_y.value_counts().min() >= 2 else None'
            split = [f'    random_state={op.seed},', f'    stratify={stratify},']

        self.lines += [
            f'{model}_X = {features}',
            f'{model}_y = {data}[{op.target!r}]',
            f'{model}_X_train, {model}_X_test, {model}_y_train, {model}_y_test = train_test_split(',
            f'    {model}_X,',
            f'    {model}_y,',
            f'    test_size={op.validation},',
            *split,
            ')',
            # Text columns are one-hot encoded inside the model's pipeline, which is fitted on the
            # training split only, so the validation split cannot leak into the encoding and the
            # trained model accepts raw rows.
            (
                f"{model}_text = {model}_X.select_dtypes(include=['object', 'string', 'category'])"
                '.columns.tolist()'
            ),
            f'{model} = Pipeline([',
            '    (',
            "        'encode',",
            '        ColumnTransformer(',
            f"            [('text', OneHotEncoder(handle_unknown='ignore', sparse_output=False), {model}_text)],",
            "            remainder='passthrough',",
            '        ),',
            '    ),',
            f"    ('model', {self._estimator(op, estimator, kwargs)}),",
            '])',
            *self._fit(op, model),
            (
                f"print(f'{op.name}: {op.algorithm} {op.task},"
                f" trained on {{len({model}_X_train)}} rows, validated on {{len({model}_X_test)}}')"
            ),
        ]

    def _estimator(self, op: ir.Train, estimator: str, kwargs: list[str]) -> str:
        call = f'{estimator}({", ".join(kwargs)})'
        if op.algorithm == 'xgboost' and op.task == 'classification':
            # XGBoost needs labels 0..n-1; the wrapper encodes them inside the model, so it (and
            # any registered version of it) predicts the original classes.
            self._import('eigrel.runtime', 'EncodedLabelClassifier')
            self.encoded.add(op.id)
            return f'EncodedLabelClassifier({call})'
        return call

    def _fit(self, op: ir.Train, model: str) -> list[str]:
        return [f'{model}.fit({model}_X_train, {model}_y_train)']

    def _evaluate(self, op: ir.Evaluate) -> None:
        model = self.variables[op.model]
        self._section(f'Evaluate {op.name}')
        self.lines.append(f'{model}_pred = {model}.predict({model}_X_test)')
        if any(METRIC_FUNCTIONS[metric][1] for metric in op.metrics):
            self.lines.append(
                f"{model}_average = 'binary' if set({model}_y.unique()) <= {{0, 1}} else 'weighted'"
            )
        if 'auc' in op.metrics:
            self.lines += [
                f'{model}_proba = {model}.predict_proba({model}_X_test)',
                (
                    f'{model}_scores = {model}_proba[:, 1] if {model}_proba.shape[1] == 2'
                    f' else {model}_proba'
                ),
            ]
        self.lines.append(f'{model}_metrics = {{')
        for metric in op.metrics:
            function, averaged = METRIC_FUNCTIONS[metric]
            self._import('sklearn.metrics', function)
            args = [f'{model}_y_test', f'{model}_pred']
            if metric == 'auc':
                args = [f'{model}_y_test', f'{model}_scores', "multi_class='ovr'"]
            elif averaged:
                args += [f'average={model}_average', 'zero_division=0']
            self.lines.append(f'    {metric!r}: {function}({", ".join(args)}),')
        self.lines.append('}')
        width = max(len(metric) for metric in op.metrics)
        for metric in op.metrics:
            self.lines.append(f'print(f"  {metric:<{width}}  {{{model}_metrics[{metric!r}]:.4f}}")')

    def _predict(self, op: ir.Predict) -> None:
        model = self.variables[op.model]
        data = self.variables[op.input]
        train = self.graph.ops[op.model]
        assert isinstance(train, ir.Train)
        column = f'{train.target}_prediction'
        self._section(f'Predict with {op.name} on {self.graph.ops[op.input].name}')
        # The serving rows get the training data's fills, then go through the trained pipeline
        # (encoding included), with exactly the training features in the training order.
        source = f'{data}.fillna({dict(op.fills)!r})' if op.fills else f'{data}.copy()'
        self.lines += [
            f'{model}_input = {source}',
            f'{model}_scored = {model}_input.copy()',
            f'{model}_scored[{column!r}] = {model}.predict({model}_input[{model}_X_train.columns])',
        ]
        if train.task == 'classification':
            probability = f'{train.target}_probability'
            self.lines += [
                f'if len({model}.classes_) == 2:',
                f'    {model}_scored[{probability!r}] = {model}.predict_proba(',
                f'        {model}_input[{model}_X_train.columns]',
                '    )[:, 1]',
            ]
        if op.format == 'csv':
            write = f'to_csv({op.path!r}, index=False)'
        elif op.format == 'parquet':
            write = f'to_parquet({op.path!r}, index=False)'
        else:
            lines = op.path.lower().endswith(('.jsonl', '.ndjson'))
            write = f"to_json({op.path!r}, orient='records'{', lines=True' if lines else ''})"
        self.lines += [
            f'{model}_scored.{write}',
            f"print({f'{op.name}: predicted'!r}, len({model}_scored), 'rows into', {op.path!r})",
        ]

    def _register(self, op: ir.Register) -> None:
        model = self.variables[op.model]
        train = self.graph.ops[op.model]
        assert isinstance(train, ir.Train)
        self.modules.update({'logging', 'os', 'warnings'})
        params = {
            'algorithm': train.algorithm,
            'task': train.task,
            'target': train.target,
            'validation': train.validation,
            'seed': train.seed,
            **dict(train.params),
        }
        options = [
            model,
            "name='model'",
            f'registered_model_name={op.registered_name!r}',
            f'input_example={model}_X_train.head(5)',
        ]
        # The model is a scikit-learn pipeline (encoding + estimator). MLflow saves it with skops,
        # which asks which types to trust; the pipeline was trained just above, so its own types
        # are trusted.
        flavor = 'mlflow.sklearn'
        options.append(
            f'skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps({model}))'
        )
        if op.model in self.encoded:
            options.append(f"extra_pip_requirements=['eigrel=={__version__}']")
        message = f'{op.name}: registered in MLflow as {op.registered_name!r}, version'
        self._section(f'Register {op.name} in MLflow')
        self.lines += [
            '# Imported here so the hint MLflow prints on import can be turned off first.',
            "os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')",
            'import mlflow',
            'import skops.io',
            '',
            "logging.getLogger('mlflow').setLevel(logging.ERROR)",
            "warnings.filterwarnings('ignore', module='mlflow')",
        ]
        if op.experiment is not None:
            self.lines.append(f'mlflow.set_experiment({op.experiment!r})')
        self.lines += [
            f'with mlflow.start_run(run_name={op.name!r}):',
            f'    mlflow.log_params({params!r})',
        ]
        if op.evaluation is not None:
            self.lines.append(f'    mlflow.log_metrics({model}_metrics)')
        self.lines += [
            f'    {model}_registered = {flavor}.log_model(',
            *(f'        {option},' for option in options),
            '    )',
            f'print({message!r}, {model}_registered.registered_model_version)',
        ]

    def _expr(self, expr: ast.Expr, frame: str) -> str:
        match expr:
            case ast.Name(value=column):
                return f'{frame}[{column!r}]'
            case ast.IntLiteral(value=number) | ast.FloatLiteral(value=number):
                return repr(number)
            case ast.StringLiteral(value=text):
                return repr(text)
            case ast.BoolLiteral(value=flag):
                return repr(flag)
            case ast.Unary(op='not', operand=operand):
                return f'~({self._expr(operand, frame)})'
            case ast.Unary(op=op, operand=operand):
                return f'{op}({self._expr(operand, frame)})'
            case ast.Binary(op=op, left=left, right=right):
                python_op = BINARY_OPERATORS.get(op, op)
                return f'({self._expr(left, frame)} {python_op} {self._expr(right, frame)})'
        raise AssertionError(f'semantic analysis rejects {expr!r} in filters')  # pragma: no cover

    def _bind(self, op: ir.Op) -> str:
        self.variables[op.id] = self.names[op.name]
        return self.variables[op.id]

    def _import(self, module: str, name: str) -> None:
        self.imports.setdefault(module, set()).add(name)

    def _section(self, title: str) -> None:
        self.lines += ['', f'# {title}']


def generate(graph: ir.Graph, source_name: str = '<eigrel>') -> str:
    """Generate a standalone Python script from an IR graph."""
    return _Generator(graph).generate(source_name)
