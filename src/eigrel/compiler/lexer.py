from eigrel.compiler.errors import LexError
from eigrel.compiler.tokens import KEYWORDS, OPERATORS, Location, Token, TokenKind

ESCAPES = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'"}


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_trivia()
            if self._at_end():
                tokens.append(Token(TokenKind.EOF, '', self._loc()))
                return tokens
            tokens.append(self._next_token())

    def _next_token(self) -> Token:
        char = self._peek()
        if char.isalpha() or char == '_':
            return self._identifier()
        if char.isdigit():
            return self._number()
        if char in ('"', "'"):
            return self._string()
        for text, kind in OPERATORS:
            if self.source.startswith(text, self.pos):
                loc = self._loc()
                self._advance(len(text))
                return Token(kind, text, loc)
        raise LexError(f'unexpected character {char!r}', self._loc())

    def _identifier(self) -> Token:
        loc = self._loc()
        start = self.pos
        while not self._at_end() and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()
        text = self.source[start : self.pos]
        kind = TokenKind.KEYWORD if text in KEYWORDS else TokenKind.IDENT
        return Token(kind, text, loc)

    def _number(self) -> Token:
        loc = self._loc()
        start = self.pos
        kind = TokenKind.INT
        self._digits()
        if self._peek() == '.' and self._peek(1).isdigit():
            kind = TokenKind.FLOAT
            self._advance()
            self._digits()
        if self._peek() in ('e', 'E'):
            offset = 2 if self._peek(1) in ('+', '-') else 1
            if self._peek(offset).isdigit():
                kind = TokenKind.FLOAT
                self._advance(offset)
                self._digits()
        if self._peek().isalpha() or self._peek() == '_':
            raise LexError(f'invalid character {self._peek()!r} in number', self._loc())
        return Token(kind, self.source[start : self.pos].replace('_', ''), loc)

    def _digits(self) -> None:
        while self._peek().isdigit() or (self._peek() == '_' and self._peek(1).isdigit()):
            self._advance()

    def _string(self) -> Token:
        loc = self._loc()
        quote = self._peek()
        self._advance()
        chars: list[str] = []
        while True:
            if self._at_end() or self._peek() == '\n':
                raise LexError('unterminated string', loc)
            char = self._peek()
            if char == quote:
                self._advance()
                return Token(TokenKind.STRING, ''.join(chars), loc)
            if char == '\\':
                escape_loc = self._loc()
                self._advance()
                escaped = self._peek()
                if escaped not in ESCAPES:
                    raise LexError(f'unknown escape sequence \\{escaped}', escape_loc)
                chars.append(ESCAPES[escaped])
                self._advance()
                continue
            chars.append(char)
            self._advance()

    def _skip_trivia(self) -> None:
        while not self._at_end():
            char = self._peek()
            if char == '#':
                while not self._at_end() and self._peek() != '\n':
                    self._advance()
            elif char.isspace():
                self._advance()
            else:
                return

    def _peek(self, offset: int = 0) -> str:
        index = self.pos + offset
        return self.source[index] if index < len(self.source) else ''

    def _advance(self, count: int = 1) -> None:
        for _ in range(count):
            if self.source[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _loc(self) -> Location:
        return Location(self.line, self.column)


def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()
