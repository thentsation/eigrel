# Eigrel language reference (v0.2)

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

| Source | Python backend |
|---|---|
| `csv("path.csv")` | `pandas.read_csv` |
| `parquet("path.parquet")` | `pandas.read_parquet` (needs `pyarrow`) |
| `bigquery("project.dataset.table")` | accepted by the compiler; backend arrives in v0.3 |

Paths are relative to the `.eig` file.

### Transformations and columns

`transform` operations apply to the dataset in order. Until a `select`, the compiler does not know
which columns exist; after one, every column used later (in `filter`, `features` or `target`) must
be among the selected columns.

`filter` takes a boolean expression. Operands are type checked: arithmetic needs numbers, `and`,
`or` and `not` need booleans, and comparisons need values of the same type. Function calls and
lists are not allowed in filters.

### Features

`features D { ... }` declares the input columns used to train on dataset `D`, once per dataset.
Without it, a model trains on every column except the target. Text columns are one-hot encoded.

### Models

| Algorithm | Tasks | Parameters |
|---|---|---|
| `random_forest` | classification, regression | `trees`, `max_depth`, `min_samples_leaf` |
| `decision_tree` | classification, regression | `max_depth`, `min_samples_leaf` |
| `gradient_boosting` | classification, regression | `trees`, `learning_rate`, `max_depth` |
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
