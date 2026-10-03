import re

import pytest

from eigrel.compiler import ParseError, ast, parse
from eigrel.compiler.tokens import Location


def single(source: str) -> ast.Statement:
    program = parse(source)
    assert len(program.statements) == 1
    return program.statements[0]


def expr(source: str) -> ast.Expr:
    statement = single(f'train m {{ x = {source} }}')
    assert isinstance(statement, ast.TrainStmt)
    return statement.params[0].value


L = Location(0, 0)  # locations are ignored by equality


def test_dataset() -> None:
    statement = single('dataset users from bigquery("project.dataset.users")')
    assert statement == ast.DatasetDecl(
        'users',
        ast.Call('bigquery', (ast.StringLiteral('project.dataset.users', loc=L),), loc=L),
        loc=L,
    )
    assert statement.loc == Location(1, 1)


def test_dataset_requires_a_call() -> None:
    with pytest.raises(ParseError, match='source call'):
        parse('dataset users from "users.csv"')


def test_transform() -> None:
    statement = single('transform users {\n filter age > 18\n select age, income\n}')
    assert statement == ast.TransformDecl(
        'users',
        (
            ast.FilterOp(
                ast.Binary('>', ast.Name('age', loc=L), ast.IntLiteral(18, loc=L), loc=L), loc=L
            ),
            ast.SelectOp((ast.Name('age', loc=L), ast.Name('income', loc=L)), loc=L),
        ),
        loc=L,
    )


def test_transform_rejects_unknown_operation() -> None:
    with pytest.raises(ParseError, match="expected 'filter', 'select'"):
        parse('transform users { drop age }')


def test_features_accepts_newlines_and_commas() -> None:
    statement = single('features customers {\n age\n income, purchases\n}')
    assert isinstance(statement, ast.FeaturesDecl)
    assert [c.value for c in statement.columns] == ['age', 'income', 'purchases']


def test_model_with_and_without_params() -> None:
    with_params = single('model churn = random_forest { trees = 100 }')
    assert with_params == ast.ModelDecl(
        'churn', 'random_forest', (ast.Param('trees', ast.IntLiteral(100, loc=L), loc=L),), loc=L
    )
    assert single('model churn = xgboost') == ast.ModelDecl('churn', 'xgboost', (), loc=L)


def test_train_and_evaluate() -> None:
    program = parse(
        'train churn { target = churned, validation = 0.2 }\n'
        'evaluate churn { metrics = [accuracy, f1,] }'
    )
    train, evaluate = program.statements
    assert train == ast.TrainStmt(
        'churn',
        (
            ast.Param('target', ast.Name('churned', loc=L), loc=L),
            ast.Param('validation', ast.FloatLiteral(0.2, loc=L), loc=L),
        ),
        loc=L,
    )
    assert evaluate == ast.EvaluateStmt(
        'churn',
        (
            ast.Param(
                'metrics',
                ast.ListExpr((ast.Name('accuracy', loc=L), ast.Name('f1', loc=L)), loc=L),
                loc=L,
            ),
        ),
        loc=L,
    )


def test_keywords_are_allowed_as_parameter_names() -> None:
    statement = single('train m { model = "llm" }')
    assert isinstance(statement, ast.TrainStmt)
    assert statement.params[0].name == 'model'


def test_duplicate_parameter() -> None:
    with pytest.raises(ParseError, match="duplicate parameter 'trees'") as info:
        parse('model m = rf {\n trees = 1\n trees = 2\n}')
    assert info.value.loc == Location(3, 2)


def test_bad_parameter_name() -> None:
    with pytest.raises(ParseError, match='expected a parameter name'):
        parse('train m { 1 = 2 }')


def test_operator_precedence() -> None:
    assert expr('a or b and not c') == ast.Binary(
        'or',
        ast.Name('a', loc=L),
        ast.Binary(
            'and', ast.Name('b', loc=L), ast.Unary('not', ast.Name('c', loc=L), loc=L), loc=L
        ),
        loc=L,
    )
    assert expr('1 + 2 * -3 > 4') == ast.Binary(
        '>',
        ast.Binary(
            '+',
            ast.IntLiteral(1, loc=L),
            ast.Binary(
                '*',
                ast.IntLiteral(2, loc=L),
                ast.Unary('-', ast.IntLiteral(3, loc=L), loc=L),
                loc=L,
            ),
            loc=L,
        ),
        ast.IntLiteral(4, loc=L),
        loc=L,
    )


def test_parentheses_and_literals() -> None:
    assert expr('(1 - 2) % 3') == ast.Binary(
        '%',
        ast.Binary('-', ast.IntLiteral(1, loc=L), ast.IntLiteral(2, loc=L), loc=L),
        ast.IntLiteral(3, loc=L),
        loc=L,
    )
    assert expr('true') == ast.BoolLiteral(True, loc=L)
    assert expr('false') == ast.BoolLiteral(False, loc=L)
    assert expr('f()') == ast.Call('f', (), loc=L)


def test_chained_comparison_is_rejected() -> None:
    with pytest.raises(ParseError, match='cannot be chained'):
        parse('transform t { filter 1 < x < 5 }')


@pytest.mark.parametrize(
    ('source', 'message', 'loc'),
    [
        ('customers', 'expected a statement', Location(1, 1)),
        ('dataset 1', 'expected a dataset name, found integer 1', Location(1, 9)),
        ('dataset d csv("x")', "expected 'from', found identifier 'csv'", Location(1, 11)),
        ('transform t', "expected '{', found end of file", Location(1, 12)),
        ('train m { x = }', "expected an expression, found '}'", Location(1, 15)),
        ('train m { x = [1 2] }', "expected ',' or ']', found integer 2", Location(1, 18)),
        ('train m { x = (1 }', "expected ')', found '}'", Location(1, 18)),
        ('model m rf', "expected '=', found identifier 'rf'", Location(1, 9)),
    ],
)
def test_error_messages_and_locations(source: str, message: str, loc: Location) -> None:
    with pytest.raises(ParseError, match=re.escape(message)) as info:
        parse(source)
    assert info.value.loc == loc


def test_examples_parse(examples_dir) -> None:  # type: ignore[no-untyped-def]
    for path in sorted(examples_dir.glob('*.eig')):
        assert parse(path.read_text()).statements, path.name


def test_to_dict_is_json_ready() -> None:
    data = ast.to_dict(parse('model m = rf { trees = 10 }'))
    assert data == {
        'node': 'Program',
        'statements': [
            {
                'node': 'ModelDecl',
                'name': 'm',
                'algorithm': 'rf',
                'params': [
                    {
                        'node': 'Param',
                        'name': 'trees',
                        'value': {
                            'node': 'IntLiteral',
                            'value': 10,
                            'loc': {'line': 1, 'column': 24},
                        },
                        'loc': {'line': 1, 'column': 16},
                    }
                ],
                'loc': {'line': 1, 'column': 1},
            }
        ],
        'loc': {'line': 1, 'column': 1},
    }
