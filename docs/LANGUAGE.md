# Eigrel language reference (v0.5)

This describes the syntax the parser accepts and the rules the compiler checks before generating
code. Every rule below is reported at compile time, with the line and column of the problem.

## Lexical structure

- **Comments** start with `#` and run to the end of the line.
- **Whitespace and newlines** are insignificant; statements start with a keyword.
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`.
- **Integers**: `100`, `1_000`. **Floats**: `0.2`, `1e3`, `2.5E-4`.
- **Strings**: `"..."` or `'...'`, single line, escapes `\n \t \\ \" \'`.
- **Booleans**: `true`, `false`.

Reserved keywords: `dataset from transform filter select features model train evaluate and or not
true false`. Inside parameter blocks keywords may be used as parameter names
(e.g. `model = "llm"`).

## Statements

```text
program     := statement*

statement   := dataset | transform | features | model | train | evaluate

dataset     := 'dataset' IDENT 'from' call
transform   := 'transform' IDENT '{' transform_op* '}'
transform_op:= 'filter' expr
             | 'select' IDENT (',' IDENT)*
features    := 'features' IDENT '{' (IDENT ','?)* '}'
model       := 'model' IDENT '=' IDENT params?
train       := 'train' IDENT params
evaluate    := 'evaluate' IDENT params

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

```text
transform D {
    fill income = 0, city = "unknown"      // literals for missing values
    drop_missing age, purchases            // drop rows missing in these columns...
    drop_missing                            // ...or in any column
    filter age >= 18
    select age, income, purchases
}
```

`filter` takes a boolean expression. Operands are type checked: arithmetic needs numbers, `and`,
`or` and `not` need booleans, and comparisons need values of the same type. Function calls and
lists are not allowed in filters. `fill` takes `column = literal` pairs (numbers, strings or
booleans); `drop_missing` takes an optional column list.

### Features

`features D { ... }` declares the input columns used to train on dataset `D`, once per dataset.
Without it, a model trains on every column except the target. Categorical columns are one-hot
encoded by the trained model itself, so the same model transforms raw data later (see
Registration).

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

### Evaluation

`evaluate MODEL { metrics = [...] }` scores a trained model on its validation split. Without
`metrics`, classification reports accuracy, precision, recall and f1, and regression reports mae,
rmse and r2.

| Task | Metrics |
|---|---|
| classification | `accuracy`, `precision`, `recall`, `f1`, `auc` |
| regression | `mae`, `mse`, `rmse`, `r2` |

Precision, recall and f1 use the binary average for 0/1 targets and the weighted average otherwise.

## Registration

```text
register MODEL { name = "customer-churn", experiment = "eigrel-examples" }
```

Logs the trained model and its last evaluation to MLflow and registers a model version. `name`
defaults to the model's name; `experiment` is optional. Requires the `mlflow` package
(`pip install mlflow`); without a tracking server, runs are stored in `./mlruns` next to the
program.

The registered artifact is the whole pipeline — preprocessing and estimator trained together on
the training rows only — so the model version accepts the raw columns the program trained on and
applies every transform itself. Categories unseen in training are ignored instead of failing.
Python backend only; `register` is not available on Spark yet.

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
| `logistic_regression` | `LogisticRegression` (`max_iter` → `maxIter`, `c` → `regParam = 1 / c`) |
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
