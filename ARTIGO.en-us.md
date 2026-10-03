# Eigrel: a language for data, ML and AI

## The problem

A typical data and machine learning project mixes Python, SQL, Pandas, Spark, BigQuery, PyTorch,
MLflow, orchestrators, RAG frameworks, LLM APIs and infrastructure code. The result is lots of
boilerplate, hard-to-read pipelines, duplicated code, hand-made optimizations and tight coupling to
every tool.

## The idea

Eigrel puts a declarative layer above those technologies. You describe **what** you want (data,
transformations, features, models, pipelines) and the compiler decides **how** to run it.

```eigrel
dataset users from bigquery("users")

transform users {
    filter age > 18
    select age, income
}
```

Instead of pulling the whole table into Python and filtering there, the compiler can notice that
both operations fit inside BigQuery and generate:

```sql
SELECT age, income FROM users WHERE age > 18
```

Eigrel code describes intent, not implementation.

## Why a compiler and not another framework

The differentiator is not the syntax, it is the optimizer. With the whole pipeline in an IR, the
classic database and compiler optimizations apply: filter and projection pushdown, common
subexpression elimination, redundant transformation removal, backend selection and execution
planning.

## Where we are (v0.1)

The first version ships the compiler front end:

- A hand-written **lexer** that records the exact line and column of every token.
- A recursive-descent **parser** with operator precedence and error messages that point at the
  exact problem.
- An immutable **AST** (frozen dataclasses) that exports to JSON.
- A **CLI** with `check`, `ast`, `tokens` and `init`.

Pure Python, no runtime dependencies, 100% test coverage.

## Next

v0.2 brings semantic analysis, the Eigrel IR, the execution graph and the first backend (Python).
The full roadmap is in [docs/ROADMAP.md](docs/ROADMAP.md).
