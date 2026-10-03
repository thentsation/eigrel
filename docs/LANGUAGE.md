# Eigrel language reference (v0.1)

This describes the syntax the v0.1 parser accepts. Semantics (name resolution, types, execution)
arrive in v0.2.

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

## Example

```eigrel
dataset users from bigquery("project.dataset.users")

transform users {
    filter age > 18 and income != 0
    select age, income, country
}

model churn = xgboost {
    max_depth = 6
    learning_rate = 0.1
}

train churn {
    target = churned
    validation = 0.2
}

evaluate churn {
    metrics = [accuracy, precision, recall, f1, auc]
}
```
