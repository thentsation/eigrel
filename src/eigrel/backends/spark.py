"""Spark backend: turn the IR into a PySpark script that trains with Spark MLlib.

Files are read natively, `sql()` sources through JDBC and `bigquery()` through the
spark-bigquery connector; Spark downloads the JDBC driver or connector from Maven Central the
first time a program needs it.
"""

from dataclasses import dataclass, field

from eigrel import __version__
from eigrel.backends import Requirement
from eigrel.backends.naming import assign_names
from eigrel.compiler import ast, ir

# Database URL scheme -> (JDBC URL prefix, driver class, Maven package with the driver).
JDBC_DRIVERS = {
    'postgresql': (
        'jdbc:postgresql://',
        'org.postgresql.Driver',
        'org.postgresql:postgresql:42.7.13',
    ),
    'mysql': ('jdbc:mysql://', 'com.mysql.cj.jdbc.Driver', 'com.mysql:mysql-connector-j:26.7.0'),
    'mariadb': (
        'jdbc:mariadb://',
        'org.mariadb.jdbc.Driver',
        'org.mariadb.jdbc:mariadb-java-client:3.5.10',
    ),
    'sqlite': ('jdbc:sqlite:', 'org.sqlite.JDBC', 'org.xerial:sqlite-jdbc:3.53.4.0'),
}
BIGQUERY_PACKAGE = 'com.google.cloud.spark:spark-bigquery-with-dependencies_2.13:0.45.0'

# Eigrel algorithm -> (module, classifier, regressor, accepts seed)
ESTIMATORS: dict[str, tuple[str, str | None, str | None, bool]] = {
    'random_forest': (
        'pyspark.ml.{task}',
        'RandomForestClassifier',
        'RandomForestRegressor',
        True,
    ),
    'decision_tree': ('pyspark.ml.{task}', 'DecisionTreeClassifier', 'DecisionTreeRegressor', True),
    'gradient_boosting': ('pyspark.ml.{task}', 'GBTClassifier', 'GBTRegressor', True),
    'logistic_regression': ('pyspark.ml.classification', 'LogisticRegression', None, False),
    'linear_regression': ('pyspark.ml.regression', None, 'LinearRegression', False),
}
# Eigrel parameter -> MLlib parameter, per algorithm where names differ.
PARAM_NAMES = {
    'trees': 'numTrees',
    'max_depth': 'maxDepth',
    'min_samples_leaf': 'minInstancesPerNode',
    'learning_rate': 'stepSize',
    'max_iter': 'maxIter',
}
# Gradient boosting counts its trees as iterations.
GBT_PARAM_NAMES = {**PARAM_NAMES, 'trees': 'maxIter'}

# Metric -> (MLlib metric for 0/1 targets, MLlib metric otherwise). Binary targets report the
# metric for the positive class, like the Python backend's average='binary'.
CLASSIFICATION_METRICS = {
    'accuracy': ('accuracy', 'accuracy'),
    'precision': ('precisionByLabel', 'weightedPrecision'),
    'recall': ('recallByLabel', 'weightedRecall'),
    'f1': ('fMeasureByLabel', 'f1'),
}
REGRESSION_METRICS = {'mae': 'mae', 'mse': 'mse', 'rmse': 'rmse', 'r2': 'r2'}

BINARY_OPERATORS = {'and': '&', 'or': '|'}
LABEL = '__eigrel_label'
FEATURES = '__eigrel_features'

MODEL_HELPERS = (
    '_features',
    '_text',
    '_index',
    '_onehot',
    '_stages',
    '_data',
    '_train',
    '_test',
    '_pred',
    '_binary',
    '_metrics',
    '_auc',
)
DATASET_HELPERS = ('_jdbc', '_package')
RESERVED = {
    'os',
    'sys',
    'warnings',
    'F',
    'spark',
    'jdbc_options',
    'packages',
    'print',
    'len',
    'set',
    'dict',
    'Pipeline',
    'SparkSession',
    'StringIndexer',
    'OneHotEncoder',
    'VectorAssembler',
    'MulticlassClassificationEvaluator',
    'BinaryClassificationEvaluator',
    'RegressionEvaluator',
    *(cls for _, *classes, _ in ESTIMATORS.values() for cls in classes if cls is not None),
}

JDBC_HELPER = '''

def jdbc_options(url, table):
    """Spark JDBC options for a SQLAlchemy database URL."""
    parts = urlsplit(url)
    scheme = parts.scheme.split('+', 1)[0]
    if scheme not in JDBC_DRIVERS:
        raise SystemExit(f'eigrel: Spark cannot read {scheme} databases yet')
    prefix, driver, package = JDBC_DRIVERS[scheme]
    if scheme == 'sqlite':
        jdbc_url = prefix + url.split(':///', 1)[1]
    else:
        host = parts.hostname or 'localhost'
        port = f':{parts.port}' if parts.port else ''
        jdbc_url = f'{prefix}{host}{port}{parts.path}'
    options = {'url': jdbc_url, 'dbtable': table, 'driver': driver}
    if parts.username:
        options['user'] = unquote(parts.username)
    if parts.password:
        options['password'] = unquote(parts.password)
    return options, package'''


@dataclass
class _Generator:
    graph: ir.Graph
    source_name: str
    imports: dict[str, set[str]] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    variables: dict[int, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    # Datasets read through JDBC, whose options are resolved before Spark starts.
    jdbc: list[ir.Load] = field(default_factory=list)
    # Train op id -> task, for evaluate.
    trained: dict[int, ir.Task] = field(default_factory=dict)

    def generate(self) -> str:
        self.names = assign_names(self.graph, RESERVED, DATASET_HELPERS, MODEL_HELPERS)
        self._import('pyspark.sql', 'SparkSession')
        self._import('pyspark.sql', 'functions as F')
        for op in self.graph.ops:
            match op:
                case ir.Load():
                    self._load(op)
                case ir.Filter():
                    self._filter(op)
                case ir.Select():
                    self._select(op)
                case ir.Train():
                    self._train(op)
                case ir.Evaluate():
                    self._evaluate(op)
        return (
            '\n'.join([*self._header(), *self._session(), *self.lines, '', 'spark.stop()']) + '\n'
        )

    def _header(self) -> list[str]:
        lines = [
            (
                f'"""Generated by eigrel {__version__} from {self.source_name} for Apache Spark.'
                ' Do not edit."""'
            ),
            '',
            'import os',
            'import sys',
            'import warnings',
        ]
        if self.jdbc:
            self._import('urllib.parse', 'unquote')
            self._import('urllib.parse', 'urlsplit')
        for module in sorted(self.imports):
            lines.append(f'from {module} import {", ".join(sorted(self.imports[module]))}')
        if self.jdbc:
            lines += ['', f'JDBC_DRIVERS = {JDBC_DRIVERS!r}', *JDBC_HELPER.splitlines()[1:]]
        return lines

    def _session(self) -> list[str]:
        lines = [
            '',
            '# Spark workers must use this interpreter, and pandas 3 warnings from PySpark are noise.',
            "os.environ.setdefault('PYSPARK_PYTHON', sys.executable)",
            "warnings.filterwarnings('ignore', message='PySpark does not yet fully support pandas')",
        ]
        packages = []
        for load in self.jdbc:
            variable = self.names[load.name]
            url = _source_value(load.args[0])
            lines.append(
                f'{variable}_jdbc, {variable}_package = jdbc_options({url}, {load.args[1]!r})'
            )
            packages.append(f'{variable}_package')
        if any(op.format == 'bigquery' for op in self.graph.ops if isinstance(op, ir.Load)):
            packages.append(repr(BIGQUERY_PACKAGE))
        lines.append('')
        builder = f'SparkSession.builder.appName({f"eigrel: {self.source_name}"!r})'
        if packages:
            lines.append(f'packages = sorted({{{", ".join(packages)}}})')
            builder += ".config('spark.jars.packages', ','.join(packages))"
        lines += [f'spark = {builder}.getOrCreate()', "spark.sparkContext.setLogLevel('ERROR')"]
        return lines

    def _load(self, op: ir.Load) -> None:
        name = self._bind(op)
        location = op.args[0]
        match op.format:
            case 'csv':
                read = f'spark.read.csv({location!r}, header=True, inferSchema=True)'
            case 'parquet':
                read = f'spark.read.parquet({location!r})'
            case 'json':
                lines = str(location).lower().endswith(('.jsonl', '.ndjson'))
                options = '' if lines else ".option('multiLine', True)"
                read = f'spark.read{options}.json({location!r})'
            case 'sql':
                self.jdbc.append(op)
                read = f"spark.read.format('jdbc').options(**{name}_jdbc).load()"
            case 'bigquery':
                read = f"spark.read.format('bigquery').load({location!r})"
            case _:  # pragma: no cover
                raise AssertionError(f'semantic analysis rejects source {op.format!r}')
        self._section(f'Dataset {op.name}')
        self.lines.append(f'{name} = {read}')

    def _filter(self, op: ir.Filter) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        self.lines.append(f'{name} = {source}.filter({self._expr(op.condition)})')

    def _select(self, op: ir.Select) -> None:
        source = self.variables[op.input]
        name = self._bind(op)
        self.lines.append(f'{name} = {source}.select({", ".join(map(repr, op.columns))})')

    def _train(self, op: ir.Train) -> None:
        data = self.variables[op.input]
        model = self._bind(op)
        module, classifier, regressor, seeded = ESTIMATORS[op.algorithm]
        estimator = classifier if op.task == 'classification' else regressor
        assert estimator is not None, 'semantic analysis rejects unsupported tasks'
        package = 'classification' if op.task == 'classification' else 'regression'
        self._import(module.format(task=package), estimator)
        self._import('pyspark.ml', 'Pipeline')
        self._import('pyspark.ml.feature', 'OneHotEncoder')
        self._import('pyspark.ml.feature', 'StringIndexer')
        self._import('pyspark.ml.feature', 'VectorAssembler')

        names = GBT_PARAM_NAMES if op.algorithm == 'gradient_boosting' else PARAM_NAMES
        kwargs = [f'featuresCol={FEATURES!r}', f'labelCol={LABEL!r}']
        for key, value in op.params:
            if key == 'c':
                # scikit-learn's C is the inverse of the regularization strength.
                assert isinstance(value, int | float), 'semantic analysis checks c is a number'
                kwargs.append(f'regParam={1 / value!r}')
            else:
                kwargs.append(f'{names[key]}={value!r}')
        if seeded:
            kwargs.append(f'seed={op.seed}')
        if op.features is None:
            features = f'[c for c in {data}.columns if c != {op.target!r}]'
        else:
            features = repr(list(op.features))
        validation = op.validation

        self._section(f'Train {op.name}: {op.algorithm} ({op.task})')
        self.lines += [
            f'{model}_features = {features}',
            f"{model}_text = [c for c, t in {data}.select({model}_features).dtypes if t == 'string']",
            f'{model}_stages = []',
            f'if {model}_text:',
            f"    {model}_index = [c + '__index' for c in {model}_text]",
            f"    {model}_onehot = [c + '__onehot' for c in {model}_text]",
            f'    {model}_stages += [',
            f"        StringIndexer(inputCols={model}_text, outputCols={model}_index, handleInvalid='keep'),",
            f'        OneHotEncoder(inputCols={model}_index, outputCols={model}_onehot),',
            '    ]',
            f'    {model}_features = [c for c in {model}_features if c not in {model}_text] + {model}_onehot',
            f'{model}_stages.append(VectorAssembler(inputCols={model}_features, outputCol={FEATURES!r}))',
        ]
        if op.task == 'classification':
            self.lines += [
                f"if dict({data}.dtypes)[{op.target!r}] == 'string':",
                f'    {model}_stages.append(StringIndexer(inputCol={op.target!r}, outputCol={LABEL!r}))',
                f'    {model}_data = {data}',
                'else:',
                f"    {model}_data = {data}.withColumn({LABEL!r}, F.col({op.target!r}).cast('double'))",
                '# Like the Python backend, 0/1 targets report metrics for the positive class.',
                f"{model}_binary = dict({data}.dtypes)[{op.target!r}] != 'string' and {{",
                f'    row[0] for row in {model}_data.select({LABEL!r}).distinct().collect()',
                '} <= {0.0, 1.0}',
            ]
        else:
            self.lines.append(
                f"{model}_data = {data}.withColumn({LABEL!r}, F.col({op.target!r}).cast('double'))"
            )
        self.lines += [
            (
                f'{model}_train, {model}_test = {model}_data.randomSplit('
                f'[{1 - validation!r}, {validation!r}], seed={op.seed})'
            ),
            (
                f'{model} = Pipeline(stages=[*{model}_stages, {estimator}({", ".join(kwargs)})])'
                f'.fit({model}_train)'
            ),
            (
                f"print(f'{op.name}: {op.algorithm} {op.task},"
                f" trained on {{{model}_train.count()}} rows, validated on {{{model}_test.count()}}')"
            ),
        ]
        self.trained[op.id] = op.task

    def _evaluate(self, op: ir.Evaluate) -> None:
        model = self.variables[op.model]
        self._section(f'Evaluate {op.name}')
        self.lines.append(f'{model}_pred = {model}.transform({model}_test)')
        width = max(len(metric) for metric in op.metrics)
        if self.trained[op.model] == 'regression':
            self._import('pyspark.ml.evaluation', 'RegressionEvaluator')
            self.lines.append(f'{model}_metrics = RegressionEvaluator(labelCol={LABEL!r})')
            for metric in op.metrics:
                self.lines += [
                    f'{model}_metrics.setMetricName({REGRESSION_METRICS[metric]!r})',
                    f"print(f'  {metric:<{width}}  {{{model}_metrics.evaluate({model}_pred):.4f}}')",
                ]
            return
        self._import('pyspark.ml.evaluation', 'MulticlassClassificationEvaluator')
        self.lines.append(
            f'{model}_metrics = MulticlassClassificationEvaluator(labelCol={LABEL!r}, metricLabel=1.0)'
        )
        for metric in op.metrics:
            if metric == 'auc':
                self._import('pyspark.ml.evaluation', 'BinaryClassificationEvaluator')
                self.lines += [
                    f'if {model}_binary:',
                    f'    {model}_auc = BinaryClassificationEvaluator(labelCol={LABEL!r})',
                    f"    print(f'  {metric:<{width}}  {{{model}_auc.evaluate({model}_pred):.4f}}')",
                    'else:',
                    f"    print('  {metric:<{width}}  n/a (Spark computes AUC for 0/1 targets only)')",
                ]
                continue
            binary, other = CLASSIFICATION_METRICS[metric]
            name = (
                repr(binary) if binary == other else f'{binary!r} if {model}_binary else {other!r}'
            )
            self.lines += [
                f'{model}_metrics.setMetricName({name})',
                f"print(f'  {metric:<{width}}  {{{model}_metrics.evaluate({model}_pred):.4f}}')",
            ]

    def _expr(self, expr: ast.Expr) -> str:
        match expr:
            case ast.Name(value=column):
                return f'F.col({column!r})'
            case ast.IntLiteral(value=value) | ast.FloatLiteral(value=value):
                return f'F.lit({value!r})'
            case ast.StringLiteral(value=value) | ast.BoolLiteral(value=value):
                return f'F.lit({value!r})'
            case ast.Unary(op='not', operand=operand):
                return f'~({self._expr(operand)})'
            case ast.Unary(op=op, operand=operand):
                return f'{op}({self._expr(operand)})'
            case ast.Binary(op=op, left=left, right=right):
                python_op = BINARY_OPERATORS.get(op, op)
                return f'({self._expr(left)} {python_op} {self._expr(right)})'
        raise AssertionError(f'semantic analysis rejects {expr!r} in filters')  # pragma: no cover

    def _bind(self, op: ir.Op) -> str:
        self.variables[op.id] = self.names[op.name]
        return self.variables[op.id]

    def _import(self, module: str, name: str) -> None:
        self.imports.setdefault(module, set()).add(name)

    def _section(self, title: str) -> None:
        self.lines += ['', f'# {title}']


def runtime_requirements(graph: ir.Graph) -> list[Requirement]:
    """PySpark is the only Python module a Spark program needs; Spark itself also needs Java."""
    return [Requirement('pyspark', 'spark')]


def _source_value(value: ir.SourceArg) -> str:
    if isinstance(value, ir.EnvVar):
        return f'os.environ[{value.name!r}]'
    return repr(value)


def generate(graph: ir.Graph, source_name: str = '<eigrel>') -> str:
    """Generate a standalone PySpark script from an IR graph."""
    return _Generator(graph, source_name).generate()
