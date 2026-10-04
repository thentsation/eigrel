from eigrel.compiler import ast, compile_source
from eigrel.compiler.ir import format_expr, format_graph
from eigrel.compiler.tokens import Location

L = Location(0, 0)


def test_format_graph() -> None:
    graph = compile_source(
        'dataset d from csv("d.csv")\n'
        'transform d { filter not (a > -1) or b == "x\\"y" and true }\n'
        'transform d { select a, b, y }\n'
        'model m = random_forest { trees = 5 }\n'
        'train m { target = y }\n'
        'model r = linear_regression\n'
        'train r { target = a, validation = 0.5 }\n'
        'evaluate m { metrics = [f1] }\n'
    )
    assert format_graph(graph).splitlines() == [
        '%0 = load csv("d.csv")  # d',
        '%1 = filter %0 ((not (a > (-1))) or ((b == "x\\"y") and true))  # d',
        '%2 = select %1 [a, b, y]  # d',
        (
            '%3 = train %2 random_forest(trees=5) classification features=[all but target]'
            ' target=y validation=0.2 seed=42  # m'
        ),
        (
            '%4 = train %2 linear_regression() regression features=[all but target]'
            ' target=a validation=0.5 seed=42  # r'
        ),
        '%5 = evaluate %3 [f1]  # m',
    ]


def test_format_sources() -> None:
    graph = compile_source(
        'dataset a from sql(env("DB_URL"), "customers")\ndataset b from bigquery("my-project.s.t")'
    )
    assert format_graph(graph).splitlines() == [
        '%0 = load sql(env("DB_URL"), "customers")  # a',
        '%1 = load bigquery("my-project.s.t")  # b',
    ]


def test_format_expr_covers_every_expression() -> None:
    expr = ast.Call(
        'f',
        (
            ast.ListExpr((ast.IntLiteral(1, loc=L), ast.FloatLiteral(2.5, loc=L)), loc=L),
            ast.BoolLiteral(False, loc=L),
            ast.StringLiteral('a\\b', loc=L),
        ),
        loc=L,
    )
    assert format_expr(expr) == 'f([1, 2.5], false, "a\\\\b")'
