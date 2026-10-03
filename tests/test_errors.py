from eigrel.compiler import ParseError, parse


def test_render_shows_line_and_caret() -> None:
    source = 'dataset users from csv("u.csv")\ntransform users\n'
    try:
        parse(source)
    except ParseError as exc:
        rendered = exc.render(source, 'pipeline.eig')
    assert rendered == (
        "error: expected '{', found end of file\n --> pipeline.eig:3:1\n  |\n3 | \n  | ^"
    )


def test_render_mid_line() -> None:
    source = 'model m rf'
    try:
        parse(source)
    except ParseError as exc:
        rendered = exc.render(source, 'm.eig')
    assert rendered.splitlines()[-2:] == ['1 | model m rf', '  |         ^']
