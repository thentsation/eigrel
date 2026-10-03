import pytest

from eigrel.compiler import LexError, tokenize
from eigrel.compiler.tokens import Location, TokenKind


def kinds(source: str) -> list[TokenKind]:
    return [token.kind for token in tokenize(source)]


def test_keywords_and_identifiers() -> None:
    tokens = tokenize('dataset customers from')
    assert [(t.kind, t.text) for t in tokens[:-1]] == [
        (TokenKind.KEYWORD, 'dataset'),
        (TokenKind.IDENT, 'customers'),
        (TokenKind.KEYWORD, 'from'),
    ]
    assert tokens[-1].kind is TokenKind.EOF


def test_numbers() -> None:
    tokens = tokenize('100 0.2 1e3 2.5E-4 1_000')
    assert [(t.kind, t.text) for t in tokens[:-1]] == [
        (TokenKind.INT, '100'),
        (TokenKind.FLOAT, '0.2'),
        (TokenKind.FLOAT, '1e3'),
        (TokenKind.FLOAT, '2.5E-4'),
        (TokenKind.INT, '1000'),
    ]


@pytest.mark.parametrize('source', ['12abc', '1_', '3e'])
def test_malformed_numbers(source: str) -> None:
    with pytest.raises(LexError, match='in number'):
        tokenize(source)


def test_strings_with_both_quotes_and_escapes() -> None:
    tokens = tokenize('"a\\"b" \'c\\nd\'')
    assert [t.text for t in tokens[:-1]] == ['a"b', 'c\nd']
    assert all(t.kind is TokenKind.STRING for t in tokens[:-1])


def test_unterminated_string_points_at_opening_quote() -> None:
    with pytest.raises(LexError, match='unterminated string') as info:
        tokenize('x = "abc\ny')
    assert info.value.loc == Location(1, 5)


def test_unknown_escape() -> None:
    with pytest.raises(LexError, match='unknown escape'):
        tokenize('"\\q"')


def test_operators_prefer_longest_match() -> None:
    assert kinds('<= >= == != < > =')[:-1] == [
        TokenKind.LE,
        TokenKind.GE,
        TokenKind.EQ,
        TokenKind.NE,
        TokenKind.LT,
        TokenKind.GT,
        TokenKind.ASSIGN,
    ]


def test_comments_and_locations() -> None:
    tokens = tokenize('# header\n  model x # trailing\n')
    assert [(t.text, t.loc) for t in tokens[:-1]] == [
        ('model', Location(2, 3)),
        ('x', Location(2, 9)),
    ]
    assert tokens[-1].loc == Location(3, 1)


def test_unexpected_character() -> None:
    with pytest.raises(LexError, match="unexpected character '@'") as info:
        tokenize('a\n  @')
    assert info.value.loc == Location(2, 3)
