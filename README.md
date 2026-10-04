# Eigrel

**A programming language for Data, Machine Learning and AI.**

[![CI](https://github.com/thentsation/eigrel/actions/workflows/ci.yaml/badge.svg)](https://github.com/thentsation/eigrel/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/eigrel)](https://pypi.org/project/eigrel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/thentsation/eigrel/badge)](https://scorecard.dev/viewer/?uri=github.com/thentsation/eigrel)

> Write what you want. Let the compiler decide how to run it.

Eigrel is a declarative language for data engineering, machine learning and GenAI. You describe
datasets, transformations, features and models in one language; the compiler decides whether each
step becomes Python, SQL, Spark or something else.

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

## Status

Eigrel is at **v0.3 — Data**: programs read CSV, Parquet, JSON, SQL databases and BigQuery, are
checked for meaning, lowered into an intermediate representation and compiled to Python (pandas +
scikit-learn) or SQL, so `eigrel run` trains and evaluates real models. Next up: a Spark backend
(v0.4). See the [roadmap](docs/ROADMAP.md).

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
| `eigrel run FILE` | Compiles the program to Python and runs it |
| `eigrel check FILE...` | Reports syntax and semantic errors with line and column |
| `eigrel compile FILE [-t python\|sql] [-o PATH]` | Prints (or writes) the generated Python or SQL |
| `eigrel ir FILE` | Prints the intermediate representation |
| `eigrel ast FILE` | Prints the syntax tree as JSON |
| `eigrel tokens FILE` | Prints the token stream |
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

### Docker

```bash
make docker-build
make docker-run      # runs examples/ml.eig inside the container
```

## Language

The full syntax is in [docs/LANGUAGE.md](docs/LANGUAGE.md). Examples live in [`examples/`](examples).

## Contributing

Bug reports, language proposals and pull requests are welcome. Read the
[contributing guide](CONTRIBUTING.md) to get set up, and note that this project follows a
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
