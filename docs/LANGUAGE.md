# Eigrel language reference (v0.7)

This describes the syntax the parser accepts and the rules the compiler checks before generating
code. Every rule below is reported at compile time, with the line and column of the problem.

## Lexical structure

- **Comments** start with `#` and run to the end of the line.
- **Whitespace and newlines** are insignificant; statements start with a keyword.
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`.
- **Integers**: `100`, `1_000`. **Floats**: `0.2`, `1e3`, `2.5E-4`.
- **Strings**: `"..."` or `'...'`, single line, escapes `\n \t \\ \" \'`.
- **Booleans**: `true`, `false`.

Reserved keywords: `dataset from transform filter select fill drop_missing features model train
evaluate register and or not true false`. Inside parameter blocks keywords may be used as parameter names
(e.g. `model = "llm"`).

## Statements

```text
program     := statement*

statement   := dataset | transform | features | model | train | evaluate | register

dataset     := 'dataset' IDENT 'from' call
transform   := 'transform' IDENT '{' transform_op* '}'
transform_op:= 'filter' expr
             | 'select' IDENT (',' IDENT)*
             | 'fill' IDENT '=' literal (',' IDENT '=' literal)*
             | 'drop_missing' (IDENT (',' IDENT)*)?
features    := 'features' IDENT '{' (IDENT ','?)* '}'
model       := 'model' IDENT '=' IDENT params?
train       := 'train' IDENT params
evaluate    := 'evaluate' IDENT params
register    := 'register' IDENT params?

params      := '{' (NAME '=' expr ','?)* '}'      # NAME is an identifier or keyword; no duplicates
```

## Expressions

From lowest to highest precedence:

| Level | Operators | Notes |
|---|---|---|
| or | `or` | left-associative |
| and | `and` | left-associative |
| not | `not` | prefix |
| comparison | `== != < <= > >=` | not chainable: write `a < x and x < b` |
| additive | `+ -` | left-associative |
| multiplicative | `* / %` | left-associative |
| unary | `-` | prefix |
| primary | literal, name, `f(args)`, `[items]`, `(expr)` | trailing commas allowed |

## Semantics

Statements run top to bottom, and names must be declared before they are used. Datasets and models
share one namespace, so a name can be declared only once.

### Data sources

| Source | Python backend | SQL backend | Extra |
|---|---|---|---|
| `csv("path.csv")` | `pandas.read_csv` | DuckDB `read_csv_auto` | `python` |
| `parquet("path.parquet")` | `pandas.read_parquet` | DuckDB `read_parquet` | `python` |
| `json("path.json")` | `pandas.read_json` (`.jsonl`/`.ndjson` read as JSON lines) | DuckDB `read_json_auto` | `python` |
| `sql(url, "table")` | `pandas.read_sql_query` over SQLAlchemy | the database's dialect | `sql` |
| `bigquery("project.dataset.table")` | `google.cloud.bigquery` client | BigQuery SQL | `bigquery` |

Paths are relative to the `.eig` file. `url` is any
[SQLAlchemy URL](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls), e.g.
`"sqlite:///data/shop.db"` or `"postgresql+psycopg://user@host/db"` (install the driver too). The
table may be schema-qualified (`"shop.orders"`).

Keep credentials out of the code with `env("NAME")`, which reads an environment variable when the
program runs. It is accepted wherever a connection URL is:

```eigrel
dataset orders from sql(env("DATABASE_URL"), "shop.orders")
```

`eigrel compile --target sql` emits one `SELECT` per dataset, with its filters and final projection,
quoted for its engine: DuckDB for files, BigQuery, MySQL/MariaDB, PostgreSQL, SQLite, or ANSI SQL
when the URL comes from `env()`. Models are not part of the SQL output.

### Transformations and columns

`transform` operations apply to the dataset in order. Until a `select`, the compiler does not know
which columns exist; after one, every column used later (in `filter`, `features` or `target`) must
be among the selected columns.

`filter` takes a boolean expression. Operands are type checked: arithmetic needs numbers, `and`,
`or` and `not` need booleans, and comparisons need values of the same type. Function calls and
lists are not allowed in filters.

### Missing values

`fill col = value, ...` replaces missing values in the listed columns with constants (numbers,
strings or booleans). `drop_missing a, b` removes rows where any listed column is missing;
`drop_missing` alone removes rows with a missing value in any column. Columns are checked like in
`select`. In SQL, `fill` becomes `COALESCE` and `drop_missing` becomes `IS NOT NULL`; SQL needs to
know the columns, so select them first when the compiler asks.

### Features

`features D { ... }` declares the input columns used to train on dataset `D`, once per dataset.
Without it, a model trains on every column except the target. Text columns are one-hot encoded.

### Models

| Algorithm | Tasks | Parameters |
|---|---|---|
| `random_forest` | classification, regression | `trees`, `max_depth`, `min_samples_leaf` |
| `decision_tree` | classification, regression | `max_depth`, `min_samples_leaf` |
| `gradient_boosting` | classification, regression | `trees`, `learning_rate`, `max_depth` |
| `xgboost` | classification, regression | `trees`, `learning_rate`, `max_depth` |
| `logistic_regression` | classification | `max_iter`, `c` |
| `linear_regression` | regression | none |

All parameters must be positive numbers; `trees`, `max_depth`, `min_samples_leaf` and `max_iter`
must be integers.

The **task** is decided by the compiler, in this order:

1. `task = classification` or `task = regression` in the model block;
2. the only task the algorithm supports;
3. the metrics the model is evaluated with anywhere in the program;
4. classification.

### Training

```text
train MODEL { target = COLUMN, data = DATASET, validation = 0.2, seed = 42 }
```

| Parameter | Required | Meaning |
|---|---|---|
| `target` | yes | Column to predict; must not be one of the features |
| `data` | no | Dataset to train on. If omitted: the only dataset with `features`, else the only dataset |
| `validation` | no | Fraction held out for evaluation, between 0 and 1 (default `0.2`) |
| `seed` | no | Random seed for the split and the model (default `42`) |

A model is trained once, and only after its data is declared. Classification splits are stratified
when every class has at least two rows.

Text features are one-hot encoded inside the model's pipeline, fitted on the training split only,
so the validation split never leaks into the encoding, categories unseen in training are ignored,
and the trained model accepts raw rows.

### Evaluation

`evaluate MODEL { metrics = [...] }` scores a trained model on its validation split. Without
`metrics`, classification reports accuracy, precision, recall and f1, and regression reports mae,
rmse and r2.

| Task | Metrics |
|---|---|
| classification | `accuracy`, `precision`, `recall`, `f1`, `auc` |
| regression | `mae`, `mse`, `rmse`, `r2` |

Precision, recall and f1 use the binary average for 0/1 targets and the weighted average otherwise.

### Registering models

`register MODEL { name = "...", experiment = "..." }` logs a trained model to MLflow and registers
it in the model registry: the training parameters, the metrics of the latest `evaluate` before it
(if any), and the model itself, which takes raw rows. Both parameters are optional (`name` defaults
to the model's name). MLflow uses `MLFLOW_TRACKING_URI`, or `mlflow.db` next to the program when it
is unset. Needs the `mlflow` extra.

## Plans

`eigrel plan` is the step between `check` and `run`. `check` never touches data; `plan` reads it:

1. It reads the schema of every source (DuckDB for files, SQLAlchemy for databases) and checks
   the program again against the real columns and types. A misspelled column, `age >= "18"` on a
   numeric column or `fill income = "none"` become errors with the exact location (code `schema`).
2. It compiles each step to SQL and runs counts inside the engine, so only numbers come back:
   rows after every `filter`, `fill` or `drop_missing`, missing values per feature, and the class
   counts of every classification target.
3. It reports findings, each with a severity, a stable code and a line:

| Code | Severity | Meaning |
|---|---|---|
| `source`, `probe` | error | a source cannot be read or queried |
| `schema` | error | the program does not match the data |
| `empty` | error | a dataset has no rows left |
| `target-missing` | error | the target has missing values; training would fail |
| `missing-values` | error | numeric features with missing values, for an algorithm or backend that rejects them |
| `text-target` | error | regression on a text target |
| `one-class` | error | the target has a single class |
| `capability` | error | the backend cannot do this (e.g. multiclass gradient boosting on Spark) |
| `continuous-target` | warning | classification on a numeric target with many distinct values |
| `imbalance` | warning | the smallest class is under 10% of the rows |
| `small-validation` | warning | fewer than 5 rows of a class are expected in the validation split |
| `no-stratify` | warning | a class has a single row, so the split cannot be stratified |
| `tiny` | warning | fewer than 30 training rows |
| `drift-removed`, `drift-type` | warning | columns removed or retyped since the recorded state |
| `drift-rows` | warning / info | the row count changed (warning above 50%) |
| `drift-added`, `drift-new` | info | new columns, or a dataset not in the recorded state |
| `not-probed` | info | BigQuery sources are not probed, because every query costs money |

`--save` writes `PROGRAM.eigstate` (each source's columns, native types, row count and a
fingerprint). Commit it next to the program: later plans compare against it and report drift.
The exit status is 1 when there are errors, and 2 with `--strict` when there are warnings.

## Spark backend

`eigrel run --target spark` and `eigrel compile --target spark` produce a PySpark script that trains
with Spark MLlib. It runs locally (`local[*]`) or under `spark-submit` on a cluster.

| Eigrel | Spark |
|---|---|
| `csv`, `parquet`, `json` | `spark.read` (CSV with header and schema inference; `.json` as multi-line JSON) |
| `sql(url, table)` | JDBC; the SQLAlchemy URL is converted at run time and the driver for PostgreSQL, MySQL, MariaDB or SQLite is downloaded from Maven Central |
| `bigquery(table)` | the spark-bigquery connector (downloaded from Maven Central) |
| `random_forest`, `decision_tree` | `RandomForest*`, `DecisionTree*` (`trees` → `numTrees`, `max_depth` → `maxDepth`, `min_samples_leaf` → `minInstancesPerNode`) |
| `gradient_boosting` | `GBT*` (`trees` → `maxIter`, `learning_rate` → `stepSize`) |
| `xgboost` | `SparkXGBClassifier`, `SparkXGBRegressor` from `xgboost.spark` |
| `logistic_regression` | `LogisticRegression` (`max_iter` → `maxIter`, `c` → `regParam = 1 / c`) |
| `fill`, `drop_missing` | `DataFrame.fillna`, `DataFrame.dropna` |
| `register` | `mlflow.spark.log_model` |
| `linear_regression` | `LinearRegression` |

Differences from the Python backend, so results are close but not identical:

- the validation split is random, not stratified, so its size varies slightly around `validation`;
- unset parameters use Spark's defaults (for example a maximum tree depth of 5);
- `gradient_boosting` classification supports two classes only;
- `auc` is reported for 0/1 targets only.

## Example

```eigrel
dataset users from bigquery("project.dataset.users")

transform users {
    filter age > 18 and income != 0
    select age, income, country, churned
}

model churn = gradient_boosting {
    trees = 200
    max_depth = 6
    learning_rate = 0.1
}

train churn {
    data = users
    target = churned
    validation = 0.2
}

evaluate churn {
    metrics = [accuracy, precision, recall, f1, auc]
}
```
