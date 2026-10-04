from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    IDENT = 'identifier'
    KEYWORD = 'keyword'
    INT = 'integer'
    FLOAT = 'float'
    STRING = 'string'

    LBRACE = '{'
    RBRACE = '}'
    LPAREN = '('
    RPAREN = ')'
    LBRACKET = '['
    RBRACKET = ']'
    COMMA = ','
    ASSIGN = '='

    EQ = '=='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='

    PLUS = '+'
    MINUS = '-'
    STAR = '*'
    SLASH = '/'
    PERCENT = '%'

    EOF = 'end of file'


KEYWORDS = frozenset(
    {
        'dataset',
        'from',
        'transform',
        'filter',
        'select',
        'features',
        'model',
        'train',
        'evaluate',
        'register',
        'assumptions',
        'predict',
        'fill',
        'drop_missing',
        'and',
        'or',
        'not',
        'true',
        'false',
    }
)

# Longest match first, so '<=' wins over '<'.
OPERATORS: tuple[tuple[str, TokenKind], ...] = (
    ('==', TokenKind.EQ),
    ('!=', TokenKind.NE),
    ('<=', TokenKind.LE),
    ('>=', TokenKind.GE),
    ('<', TokenKind.LT),
    ('>', TokenKind.GT),
    ('=', TokenKind.ASSIGN),
    ('{', TokenKind.LBRACE),
    ('}', TokenKind.RBRACE),
    ('(', TokenKind.LPAREN),
    (')', TokenKind.RPAREN),
    ('[', TokenKind.LBRACKET),
    (']', TokenKind.RBRACKET),
    (',', TokenKind.COMMA),
    ('+', TokenKind.PLUS),
    ('-', TokenKind.MINUS),
    ('*', TokenKind.STAR),
    ('/', TokenKind.SLASH),
    ('%', TokenKind.PERCENT),
)


@dataclass(frozen=True)
class Location:
    line: int
    column: int


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    loc: Location

    def is_keyword(self, word: str) -> bool:
        return self.kind is TokenKind.KEYWORD and self.text == word

    def describe(self) -> str:
        if self.kind is TokenKind.EOF:
            return 'end of file'
        if self.kind in (TokenKind.IDENT, TokenKind.KEYWORD):
            return f"{self.kind.value} '{self.text}'"
        if self.kind in (TokenKind.INT, TokenKind.FLOAT, TokenKind.STRING):
            return f'{self.kind.value} {self.text}'
        return f"'{self.text}'"
