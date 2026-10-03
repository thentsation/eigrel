import re

import pytest

from eigrel.compiler import SemanticError, compile_source, ir
from eigrel.compiler.tokens import Location

DATA = 'dataset d from csv("d.csv")\n'
SELECTED = DATA + 'transform d { select a, b, y }\n'


def ops(source: str) -> list[ir.Op]:
    return compile_source(source).ops


def train_op(source: str) -> ir.Train:
    op = next(op for op in ops(source) if isinstance(op, ir.Train))
    assert isinstance(op, ir.Train)
    return op


def test_lowers_a_full_pipeline() -> None:
    graph = ops(
        SELECTED
        + 'transform d { filter a > 1 }\n'
        + 'features d { a, b }\n'
        + 'model m = random_forest { trees = 10 }\n'
        + 'train m { target = y, validation = 0.25, seed = 7 }\n'
        + 'evaluate m { metrics = [accuracy, f1] }\n'
    )
    assert [type(op).__name__ for op in graph] == [
        'Load',
        'Select',
        'Filter',
        'Train',
        'Evaluate',
    ]
    train = graph[3]
    assert isinstance(train, ir.Train)
    assert (train.input, train.features, train.target) == (2, ('a', 'b'), 'y')
    assert (train.task, train.validation, train.seed) == ('classification', 0.25, 7)
    assert train.params == (('trees', 10),)
    assert graph[4] == ir.Evaluate(4, 'm', 3, ('accuracy', 'f1'))


def test_defaults() -> None:
    graph = ops(DATA + 'model m = random_forest\ntrain m { target = y }\nevaluate m {}')
    train, evaluate = graph[1], graph[2]
    assert isinstance(train, ir.Train) and isinstance(evaluate, ir.Evaluate)
    assert (train.features, train.validation, train.seed) == (None, 0.2, 42)
    assert evaluate.metrics == ('accuracy', 'precision', 'recall', 'f1')


@pytest.mark.parametrize(
    ('model', 'evaluate', 'task'),
    [
        ('random_forest', '[rmse, r2]', 'regression'),
        ('random_forest', '[auc]', 'classification'),
        ('linear_regression', '[mae]', 'regression'),
        ('logistic_regression', '[accuracy]', 'classification'),
        ('random_forest { task = regression }', '[r2]', 'regression'),
    ],
)
def test_task_inference(model: str, evaluate: str, task: str) -> None:
    source = DATA + f'model m = {model}\ntrain m {{ target = y }}\n'
    source += f'evaluate m {{ metrics = {evaluate} }}'
    assert train_op(source).task == task


def test_regression_default_metrics() -> None:
    graph = ops(DATA + 'model m = linear_regression\ntrain m { target = y }\nevaluate m {}')
    assert graph[-1] == ir.Evaluate(2, 'm', 1, ('mae', 'rmse', 'r2'))


def test_training_data_inference() -> None:
    two_datasets = DATA + 'dataset e from csv("e.csv")\nfeatures e { a }\nmodel m = random_forest\n'
    assert train_op(two_datasets + 'train m { target = y }').input == 1
    assert train_op(two_datasets + 'train m { data = d, target = y }').input == 0


def test_filter_type_checking_accepts_valid_expressions() -> None:
    graph = ops(
        DATA + 'transform d { filter not (a > 1 or b == "x") and c != true and -a * 2 % 3 <= 1.5 }'
    )
    assert isinstance(graph[1], ir.Filter)


@pytest.mark.parametrize(
    ('source', 'message', 'loc'),
    [
        (DATA + DATA, "'d' is already defined at line 1", (2, 1)),
        (DATA + 'model d = random_forest', "'d' is already defined at line 1", (2, 1)),
        ('dataset d from s3("x")', "unknown data source 's3'", (1, 16)),
        ('dataset d from csv("a", "b")', 'takes exactly one string argument', (1, 16)),
        ('dataset d from csv(1)', 'takes exactly one string argument', (1, 16)),
        ('transform d { select a }', "unknown dataset 'd'", (1, 1)),
        (
            'model m = random_forest\nfeatures m { a }',
            "unknown dataset 'm' (it is a model)",
            (2, 1),
        ),
        (SELECTED + 'transform d { filter z > 1 }', "column 'z' does not exist here", (3, 22)),
        (SELECTED + 'transform d { select a, z }', "column 'z' does not exist here", (3, 25)),
        (DATA + 'transform d { select a, a }', "column 'a' is listed twice", (2, 25)),
        (DATA + 'transform d { filter 1 }', 'must be a boolean expression, not a number', (2, 22)),
        (DATA + 'transform d { filter a + "x" > 1 }', "'+' expects a number", (2, 26)),
        (DATA + 'transform d { filter not 1 }', "'not' expects a bool", (2, 26)),
        (DATA + 'transform d { filter -"x" > 1 }', "'-' expects a number", (2, 23)),
        (DATA + 'transform d { filter 1 and a }', "'and' expects a bool", (2, 22)),
        (
            DATA + 'transform d { filter 1 == "x" }',
            'cannot compare a number with a string',
            (2, 24),
        ),
        (DATA + 'transform d { filter 1 < "x" }', 'cannot compare a number with a string', (2, 24)),
        (DATA + 'transform d { filter true < 1 }', "'<' cannot order booleans", (2, 27)),
        (DATA + 'transform d { filter f(a) }', 'function calls and lists', (2, 22)),
        (DATA + 'features d { a }\nfeatures d { b }', 'already declared', (3, 1)),
        (DATA + 'features d { }', 'at least one column', (2, 1)),
        ('model m = xgboost', "unknown algorithm 'xgboost'", (1, 1)),
        ('model m = random_forest { depth = 3 }', "has no parameter 'depth'", (1, 27)),
        ('model m = random_forest { trees = 0 }', "'trees' must be greater than 0", (1, 35)),
        ('model m = random_forest { trees = 1.5 }', "'trees' must be an integer", (1, 35)),
        ('model m = random_forest { trees = "x" }', "'trees' must be an integer", (1, 35)),
        ('model m = random_forest { task = cluster }', "not 'cluster'", (1, 34)),
        ('model m = random_forest { task = "x" }', 'task must be a name', (1, 34)),
        ('model m = linear_regression { task = classification }', 'does not support', (1, 38)),
        (
            DATA + 'model m = random_forest\ntrain m { target = y }\n'
            'evaluate m { metrics = [accuracy, r2] }',
            'both classification and regression metrics',
            (4, 35),
        ),
        ('train m { target = y }', "unknown model 'm'", (1, 1)),
        (DATA + 'train d { target = y }', "unknown model 'd' (it is a dataset)", (2, 1)),
        ('model m = random_forest\ntrain m { target = y }', 'none is declared yet', (2, 1)),
        (
            DATA + 'dataset e from csv("e.csv")\nmodel m = random_forest\ntrain m { target = y }',
            'cannot tell which dataset',
            (4, 1),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { data = x, target = y }',
            "unknown dataset 'x'",
            (3, 18),
        ),
        (DATA + 'model m = random_forest\ntrain m { }', 'train needs a target', (3, 1)),
        (
            DATA + 'model m = random_forest\ntrain m { target = 1 }',
            'target must be a name',
            (3, 20),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { epochs = 1 }',
            "train has no parameter 'epochs'",
            (3, 11),
        ),
        (
            SELECTED + 'model m = random_forest\ntrain m { target = z }',
            "column 'z' does not exist",
            (4, 20),
        ),
        (
            DATA + 'features d { a, y }\nmodel m = random_forest\ntrain m { target = y }',
            "target 'y' is also declared as a feature",
            (4, 20),
        ),
        (
            DATA + 'features d { a, z }\ntransform d { select a, y }\nmodel m = random_forest\n'
            'train m { target = y }',
            "feature column 'z' does not exist",
            (5, 1),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y, validation = 1 }',
            'between 0 and 1',
            (3, 36),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y, seed = -1 }',
            'must not be negative',
            (3, 30),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y, seed = 1.5 }',
            "'seed' must be an integer",
            (3, 30),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y }\ntrain m { target = y }',
            'already trained at line 3',
            (4, 1),
        ),
        (DATA + 'model m = random_forest\nevaluate m { }', 'must be trained before', (3, 1)),
        (
            DATA + 'model m = random_forest\ntrain m { target = y }\nevaluate m { metrics = [] }',
            'non-empty list',
            (4, 14),
        ),
        (
            DATA
            + 'model m = random_forest\ntrain m { target = y }\nevaluate m { metrics = ["f1"] }',
            'metrics must be names',
            (4, 25),
        ),
        (
            DATA
            + 'model m = random_forest\ntrain m { target = y }\nevaluate m { metrics = [speed] }',
            "unknown metric 'speed'",
            (4, 25),
        ),
        (
            DATA
            + 'model m = logistic_regression\ntrain m { target = y }\nevaluate m { metrics = [r2] }',
            "metric 'r2' is for regression, but 'm' is a classification model",
            (4, 25),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y }\n'
            'evaluate m { metrics = [f1, f1] }',
            "metric 'f1' is listed twice",
            (4, 29),
        ),
        (
            DATA + 'model m = random_forest\ntrain m { target = y }\nevaluate m { top = 1 }',
            "evaluate has no parameter 'top'",
            (4, 14),
        ),
    ],
)
def test_semantic_errors(source: str, message: str, loc: tuple[int, int]) -> None:
    with pytest.raises(SemanticError, match=re.escape(message)) as info:
        compile_source(source)
    assert info.value.loc == Location(*loc)


def test_negative_float_parameter_is_rejected() -> None:
    with pytest.raises(SemanticError, match='greater than 0'):
        compile_source('model m = gradient_boosting { learning_rate = -0.1 }')
