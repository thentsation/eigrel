# Eigrel

**A compiler for tabular machine learning.** You declare the pipeline; the compiler checks it
before anything runs and generates the code for your laptop (pandas + scikit-learn) or your
cluster (Spark).

[![CI](https://github.com/thentsation/eigrel/actions/workflows/ci.yaml/badge.svg)](https://github.com/thentsation/eigrel/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/eigrel)](https://pypi.org/project/eigrel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/thentsation/eigrel/badge)](https://scorecard.dev/viewer/?uri=github.com/thentsation/eigrel)

> You don't review ML code; you review a plan. The compiler proves what it can, the backend runs it.

Eigrel describes datasets, cleaning, features, models, training, evaluation and registration in
one small language. Because the compiler sees the whole program, it rejects mistakes that a
notebook runs without complaint, and the same program compiles to Python, Spark or SQL.

```eigrel
dataset customers from csv("data/customers.csv")

transform customers {
    filter age >= 18
    select age, income, purchases, churned
}

features customers {
    age
    income
    purchases
}

model churn = random_forest {
    trees = 100
}

train churn {
    target = churned
}

evaluate churn {
    metrics = [accuracy, precision, recall, f1]
}
```

## The mistake a notebook misses

Put the target among the features by accident. In a notebook, nothing complains, and validation
looks perfect:

```python
X = customers[['age', 'income', 'purchases', 'churned']]   # churned is also the target
X_train, X_test, y_train, y_test = train_test_split(X, customers['churned'], stratify=customers['churned'], random_state=42)
RandomForestClassifier(random_state=42).fit(X_train, y_train).score(X_test, y_test)   # 1.0
```

The same pipeline in Eigrel ([`examples/mistakes/target_leakage.eig`](examples/mistakes/target_leakage.eig))
does not compile:

```text
$ eigrel check examples/mistakes/target_leakage.eig
error: target 'churned' is also declared as a feature of 'customers'
  --> examples/mistakes/target_leakage.eig:26:14
   |
26 |     target = churned
   |              ^
```

The honest model scores 0.80 on this data. Good engineers leak data silently; a compiler that sees
the whole pipeline does not. The generated code is held to the same standard: encoders are fitted
on the training split only, inside the model, so the validation split never leaks into them.

## Guarantees across the model's life

- **No leakage in the pipeline.** Encoders are fitted on the training split only, inside the
  model; the target can never be a feature.
- **Time-ordered validation.** `assumptions customers { time = signup_date }` validates on the
  latest rows; `eigrel plan` warns when dated data is split at random.
- **No train/serve skew.** `predict churn { data = new_customers, output = csv("scored.csv") }`
  must receive every training feature with the same types, and gets the training fills replayed;
  anything else is a compile error. Registered models predict the original classes.
- **Clear backend limits.** A [capability matrix](docs/LANGUAGE.md#backend-capabilities) says what
  each backend supports, and the plan checks programs against it.

See [`examples/serving.eig`](examples/serving.eig).

## See the consequences before running: `eigrel plan`

`eigrel plan` reads the data, checks the program against the real column names and types, and
shows what every step does to it, without training anything:

```text
$ eigrel plan examples/ml.eig
Plan for ml.eig (python backend)

dataset customers ← csv("data/customers.csv")
  columns: customer_id bigint, age bigint, income double, purchases bigint, churned bigint
  rows: 400
  filter (age >= 18)                           400 → 394 rows (-6)
  select age, income, purchases, churned       394 rows

train churn: random_forest classification on customers
  split: 315 train / 79 validation (20%, stratified)
  features: age, income, purchases
  target churned: 0 209 (53%), 1 185 (47%)
  missing values: none

✓ no problems found
state: no ml.eigstate yet; `eigrel plan --save` records the data schema so later plans can detect drift
```

When something would go wrong, the plan says so and points at the line:

```text
✗ error: line 9: features with missing values cannot be trained with logistic_regression: revenue (22); use fill or drop_missing first
! warning: line 9: class imbalance: 'enterprise' has 4 rows (1%) against 424 for 'basic'
! warning: line 9: less than one row(s) of class 'enterprise' expected in the validation split (20%); its metrics will be unstable
```

`eigrel plan --save` records each source's schema in `PROGRAM.eigstate`; commit it, and later
plans report drift (removed columns, changed types, row counts). `--json` gives the same plan to
tools and agents, `--target spark` applies Spark's limits, and `--strict` turns warnings into a
failing exit code for CI.

## Status

Eigrel 1.0 reads CSV, Parquet, JSON, SQL databases and BigQuery, cleans missing values, trains
scikit-learn, XGBoost or Spark MLlib models, scores new data with the training pipeline and
registers models in MLflow, compiling the same program to Python, Spark or SQL. `eigrel plan`
shows the consequences before anything runs, and `eigrel mcp` gives agents the whole loop. The
language, the CLI and the JSON outputs are stable: see [compatibility](docs/COMPATIBILITY.md) and
the [roadmap](docs/ROADMAP.md).

## Getting started

Install from PyPI (Python 3.13+). The `python` extra adds pandas and scikit-learn, which
`eigrel run` needs; add `sql` to read databases or `bigquery` for BigQuery
(`pip install "eigrel[python,sql]"`):

```bash
pip install "eigrel[python]"
eigrel init churn             # creates churn/main.eig and sample data
eigrel run churn/main.eig
```

```text
churn: random_forest classification, trained on 315 rows, validated on 79
  accuracy   0.7975
  precision  0.7442
  recall     0.8649
  f1         0.8000
```

### CLI

| Command | What it does |
|---|---|
| `eigrel run FILE [-t python\|spark]` | Compiles the program and runs it (default: Python) |
| `eigrel check FILE... [--json]` | Reports syntax and semantic errors with line and column (`--json`: machine-readable) |
| `eigrel plan FILE [-t python\|spark] [--json] [--save] [--strict]` | Checks the program against its data and shows row counts, classes, warnings and drift |
| `eigrel compile FILE [-t python\|spark\|sql] [-o PATH]` | Prints (or writes) the generated code |
| `eigrel ir FILE` | Prints the intermediate representation |
| `eigrel ast FILE` | Prints the syntax tree as JSON |
| `eigrel tokens FILE` | Prints the token stream |
| `eigrel mcp` | Serves check, plan, compile and run to agents over MCP (stdio) |
| `eigrel init NAME` | Creates a project with a starter program and sample data |

The compiler catches mistakes before anything runs, and points at the exact spot:

```text
error: column 'income' does not exist here; available columns: age, purchases, churned
 --> churn.eig:9:5
  |
9 |     income
  |     ^
```

### How it works

```text
source → lexer → parser → AST → semantic analysis → IR → Python backend → pandas + scikit-learn
```

`eigrel ir` shows the graph the backends work from:

```text
%0 = load csv("data/customers.csv")  # customers
%1 = filter %0 (age >= 18)  # customers
%2 = select %1 [age, income, purchases, churned]  # customers
%3 = train %2 random_forest(trees=100, max_depth=8) classification features=[age, income, purchases] target=churned validation=0.2 seed=42  # churn
%4 = evaluate %3 [accuracy, precision, recall, f1]  # churn
```

### Data sources

```eigrel
dataset customers from csv("data/customers.csv")
dataset events    from json("data/events.jsonl")
dataset orders    from sql(env("DATABASE_URL"), "shop.orders")
dataset users     from bigquery("my-project.analytics.users")
```

`eigrel compile --target sql` turns the data part of a program into a query for the engine each
source lives in:

```sql
-- dataset users (bigquery, bigquery dialect)
SELECT `age`, `income`, `country` FROM `project.dataset.users` WHERE ((`age` > 18) AND (`income` <> 0));
```

### Cleaning data, XGBoost and MLflow

```eigrel
transform customers {
    fill income = 0
    drop_missing age, purchases
}

model churn = xgboost {
    trees = 200
    learning_rate = 0.05
}

register churn {
    name = "customer-churn"
}
```

`register` logs the parameters, metrics and model to MLflow and registers a new version; the
registered model takes raw rows, because encoding is part of its pipeline. It needs the `xgboost`
and `mlflow` extras (`pip install "eigrel[python,xgboost,mlflow]"`; on macOS XGBoost also needs
OpenMP: `brew install libomp`). See
[`examples/mlflow.eig`](examples/mlflow.eig).

### Spark

The same program runs on Spark with `--target spark`, which needs the `spark` extra and Java 17 or
newer:

```bash
pip install "eigrel[python,spark]"
eigrel run --target spark churn/main.eig
eigrel compile --target spark churn/main.eig -o churn_spark.py   # e.g. for spark-submit
```

Files are read natively, `sql()` through JDBC and `bigquery()` through the spark-bigquery connector,
and models train with Spark MLlib. See [the Spark backend](docs/LANGUAGE.md#spark-backend) for how
it differs from the Python backend.

### Docker

The image on GitHub Container Registry (`linux/amd64` and `linux/arm64`) includes the `python` and
`sql` extras. Mount your project at `/work` and run as your own user, so the container can read your
files and anything it writes stays yours:

```bash
alias eigrel='docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" ghcr.io/thentsation/eigrel'
eigrel init churn
eigrel run churn/main.eig
```

Paths must be inside the current directory, since only it is mounted. To build the image locally,
use `make docker-build` and `make docker-run`.

### AI agents

Eigrel is designed to be written by AI agents and verified by the compiler. Point your agent at
[llms.txt](llms.txt): the complete grammar, the workflow and worked examples in one file. The
loop is `eigrel check --json` until `"ok": true` (every error has an exact line and column), then
`eigrel plan --json` against the real data, then `eigrel run`.

Agents that speak the Model Context Protocol get the same loop as tools. Install
`pip install "eigrel[mcp]"` and register the server with your MCP client:

```json
{
  "mcpServers": {
    "eigrel": { "command": "eigrel", "args": ["mcp"] }
  }
}
```

It exposes `check`, `plan`, `compile` and `run` (returning the same JSON as the CLI) and the
`eigrel://llms.txt` and `eigrel://capabilities` resources.

## Language

The full syntax is in [docs/LANGUAGE.md](docs/LANGUAGE.md). Examples live in [`examples/`](examples).

## Contributing

Bug reports, language proposals and pull requests are welcome. Read the
[contributing guide](CONTRIBUTING.md) to get set up, and note that this project follows a
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
