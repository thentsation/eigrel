# Mistakes the compiler catches

Every program below is in [`examples/mistakes`](../examples/mistakes) and a test runs each one: if the
compiler stops rejecting it, CI fails. `check` needs no data; `plan` reads the real columns, types and rows.

## Target leakage

The target is also a feature. A notebook reports a perfect score because the model reads the answer.

```text
$ eigrel check examples/mistakes/target_leakage.eig
error: target 'churned' is also declared as a feature of 'customers'
  --> examples/mistakes/target_leakage.eig:26:14
   |
26 |     target = churned
   |              ^
```

## The target under another name

A column that is a copy of the target. The compiler cannot know from the program, but `plan` reads the data and sees the two columns are identical in every row. A notebook trains happily and reports a perfect score.

```text
$ eigrel plan examples/mistakes/duplicate_of_target.eig
Plan for examples/mistakes/duplicate_of_target.eig (python backend)

dataset accounts ← csv("../data/accounts.csv")
  columns: account_id bigint, age bigint, income double, tenure bigint, churned bigint, churn_flag bigint
  rows: 200
  fill income = 0                              200 rows

train churn: random_forest classification on accounts
  split: 160 train / 40 validation (20%, stratified)
  features: age, income, tenure, churn_flag
  target churned: 0 139 (70%), 1 61 (30%)
  missing values: none

✗ error: line 26: feature 'churn_flag' is identical to the target 'churned' in every row; the model would just read the answer
1 error(s), 0 warning(s)
```

## A row identifier used as a feature

`account_id` is different in every row, so a tree model can memorise it: great training numbers, nothing learned that carries over. A warning, so `--strict` is what fails CI.

```text
$ eigrel plan --strict examples/mistakes/id_feature.eig
Plan for examples/mistakes/id_feature.eig (python backend)

dataset accounts ← csv("../data/accounts.csv")
  columns: account_id bigint, age bigint, income double, tenure bigint, churned bigint, churn_flag bigint
  rows: 200

train churn: random_forest classification on accounts
  split: 160 train / 40 validation (20%, stratified)
  features: account_id, age, tenure
  target churned: 0 139 (70%), 1 61 (30%)
  missing values: none

! warning: line 21: feature 'account_id' is different in every row; it identifies rows instead of describing them, so the model can memorise it. Leave it out of features
0 error(s), 1 warning(s)
state: no id_feature.eigstate yet; `eigrel plan --save` records the data schema so later plans can detect drift
```

## A column an earlier step dropped

In a notebook, a `KeyError` after the load and the cleaning have already run.

```text
$ eigrel check examples/mistakes/unknown_column.eig
error: column 'income' does not exist here; available columns: age, purchases, churned
  --> examples/mistakes/unknown_column.eig:16:5
   |
16 |     income
   |     ^
```

## A metric for the wrong task

`accuracy` and `rmse` answer different questions; a notebook computes whichever you call and the number looks plausible.

```text
$ eigrel check examples/mistakes/wrong_metric.eig
error: metric 'rmse' is for regression, but 'churn' is a classification model
  --> examples/mistakes/wrong_metric.eig:26:26
   |
26 |     metrics = [accuracy, rmse]
   |                          ^
```

## Evaluating a model that was never trained

A `NameError`, or worse, a stale model from an earlier cell.

```text
$ eigrel check examples/mistakes/evaluate_before_train.eig
error: model 'churn' must be trained before it is evaluated
  --> examples/mistakes/evaluate_before_train.eig:19:1
   |
19 | evaluate churn {
   | ^
```

## Comparing a number with text

pandas raises deep in the stack, or compares as text and silently keeps the wrong rows. This one needs the real column types, so it is caught by `plan`.

```text
$ eigrel plan examples/mistakes/type_mismatch.eig
Plan for examples/mistakes/type_mismatch.eig (python backend)

dataset customers ← csv("../data/customers.csv")
  columns: customer_id bigint, age bigint, income double, purchases bigint, churned bigint

✗ error: line 11: cannot compare a number with a string
1 error(s), 0 warning(s)
```

## Training on a column with gaps

scikit-learn raises `Input contains NaN` at fit time, after the split. `plan` counts the missing values first.

```text
$ eigrel plan examples/mistakes/missing_values.eig
Plan for examples/mistakes/missing_values.eig (python backend)

dataset accounts ← csv("../data/accounts.csv")
  columns: account_id bigint, age bigint, income double, tenure bigint, churned bigint, churn_flag bigint
  rows: 200

train churn: logistic_regression classification on accounts
  split: 160 train / 40 validation (20%, stratified)
  features: age, income, tenure
  target churned: 0 139 (70%), 1 61 (30%)
  missing values: income 27

✗ error: line 20: features with missing values cannot be trained with logistic_regression: income (27); use fill or drop_missing first
1 error(s), 0 warning(s)
```

Found a mistake notebooks let through that Eigrel should catch? [Open an issue](https://github.com/thentsation/eigrel/issues/new?template=feature_request.yml): a good mistake is a small `.eig` file plus the sentence you wish the compiler said.
