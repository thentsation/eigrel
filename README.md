# Eigrel

**A programming language for Data, Machine Learning and AI.**

[![CI](https://github.com/thentsation/eigrel/actions/workflows/pipeline_python.yaml/badge.svg)](https://github.com/thentsation/eigrel/actions/workflows/pipeline_python.yaml)
[![PyPI](https://img.shields.io/pypi/v/eigrel)](https://pypi.org/project/eigrel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

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

Eigrel is at **v0.1 — Language**: the lexer, parser and syntax tree are done, and the CLI can
check files and print their tokens or AST. Nothing executes yet; the compiler, IR and Python backend
come in v0.2. See the [roadmap](docs/ROADMAP.md).

## Getting started

Install from PyPI (Python 3.13+):

```bash
pip install eigrel
eigrel init churn
eigrel check churn/main.eig
```

### From source

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install                     # creates .venv and installs eigrel in editable mode
.venv/bin/eigrel init churn      # creates churn/main.eig
.venv/bin/eigrel check churn/main.eig
```

### CLI

| Command | What it does |
|---|---|
| `eigrel check FILE...` | Parses each file and reports syntax errors with line and column |
| `eigrel ast FILE` | Prints the syntax tree as JSON |
| `eigrel tokens FILE` | Prints the token stream |
| `eigrel init NAME` | Creates a project with a starter `main.eig` |

Errors point at the exact spot:

```text
error: comparisons cannot be chained; combine them with 'and'
 --> pipeline.eig:4:21
  |
4 |     filter age > 18 < 30
  |                     ^
```

### Docker

```bash
make docker-build
make docker-run      # runs `eigrel check` on examples/ml.eig inside the container
```

## Language

The full syntax is in [docs/LANGUAGE.md](docs/LANGUAGE.md). Examples live in [`examples/`](examples).

## Contributing

Bug reports, language proposals and pull requests are welcome. Read the
[contributing guide](CONTRIBUTING.md) to get set up, and note that this project follows a
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
