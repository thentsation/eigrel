from collections.abc import Callable

from eigrel.compiler import ast
from eigrel.compiler.errors import ParseError
from eigrel.compiler.lexer import tokenize
from eigrel.compiler.tokens import Token, TokenKind

COMPARISON_OPS = {
    TokenKind.EQ,
    TokenKind.NE,
    TokenKind.LT,
    TokenKind.LE,
    TokenKind.GT,
    TokenKind.GE,
}
ADDITIVE_OPS = {TokenKind.PLUS, TokenKind.MINUS}
MULTIPLICATIVE_OPS = {TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.statement_parsers: dict[str, Callable[[], ast.Statement]] = {
            'dataset': self._dataset,
            'transform': self._transform,
            'features': self._features,
            'model': self._model,
            'train': self._train,
            'evaluate': self._evaluate,
            'register': self._register,
        }

    def parse_program(self) -> ast.Program:
        loc = self._peek().loc
        statements: list[ast.Statement] = []
        while self._peek().kind is not TokenKind.EOF:
            statements.append(self._statement())
        return ast.Program(tuple(statements), loc=loc)

    # Statements

    def _statement(self) -> ast.Statement:
        token = self._peek()
        if token.kind is TokenKind.KEYWORD and token.text in self.statement_parsers:
            return self.statement_parsers[token.text]()
        expected = ', '.join(f"'{word}'" for word in self.statement_parsers)
        raise ParseError(f'expected a statement ({expected}), found {token.describe()}', token.loc)

    def _dataset(self) -> ast.DatasetDecl:
        loc = self._expect_keyword('dataset').loc
        name = self._expect(TokenKind.IDENT, 'a dataset name').text
        self._expect_keyword('from')
        source = self._primary()
        if not isinstance(source, ast.Call):
            raise ParseError(
                'expected a source call after \'from\', like csv("file.csv")', source.loc
            )
        return ast.DatasetDecl(name, source, loc=loc)

    def _transform(self) -> ast.TransformDecl:
        loc = self._expect_keyword('transform').loc
        dataset = self._expect(TokenKind.IDENT, 'a dataset name').text
        self._expect(TokenKind.LBRACE, "'{'")
        ops: list[ast.TransformOp] = []
        while not self._match(TokenKind.RBRACE):
            ops.append(self._transform_op())
        return ast.TransformDecl(dataset, tuple(ops), loc=loc)

    def _transform_op(self) -> ast.TransformOp:
        token = self._peek()
        if token.is_keyword('filter'):
            self._advance()
            return ast.FilterOp(self._expression(), loc=token.loc)
        if token.is_keyword('select'):
            self._advance()
            columns = [self._column()]
            while self._match(TokenKind.COMMA):
                columns.append(self._column())
            return ast.SelectOp(tuple(columns), loc=token.loc)
        if token.is_keyword('fill'):
            self._advance()
            values = [self._assignment()]
            while self._match(TokenKind.COMMA):
                values.append(self._assignment())
            return ast.FillOp(tuple(values), loc=token.loc)
        if token.is_keyword('drop_missing'):
            self._advance()
            columns = []
            if self._check(TokenKind.IDENT):
                columns.append(self._column())
                while self._match(TokenKind.COMMA):
                    columns.append(self._column())
            return ast.DropMissingOp(tuple(columns), loc=token.loc)
        raise ParseError(
            "expected 'filter', 'select', 'fill', 'drop_missing' or '}}',"
            f' found {token.describe()}',
            token.loc,
        )

    def _assignment(self) -> ast.Param:
        column = self._column()
        self._expect(TokenKind.ASSIGN, f"'=' after '{column.value}'")
        return ast.Param(column.value, self._expression(), loc=column.loc)

    def _features(self) -> ast.FeaturesDecl:
        loc = self._expect_keyword('features').loc
        dataset = self._expect(TokenKind.IDENT, 'a dataset name').text
        self._expect(TokenKind.LBRACE, "'{'")
        columns: list[ast.Name] = []
        while not self._match(TokenKind.RBRACE):
            columns.append(self._column())
            self._match(TokenKind.COMMA)
        return ast.FeaturesDecl(dataset, tuple(columns), loc=loc)

    def _model(self) -> ast.ModelDecl:
        loc = self._expect_keyword('model').loc
        name = self._expect(TokenKind.IDENT, 'a model name').text
        self._expect(TokenKind.ASSIGN, "'='")
        algorithm = self._expect(TokenKind.IDENT, 'an algorithm name').text
        params = self._param_block() if self._check(TokenKind.LBRACE) else ()
        return ast.ModelDecl(name, algorithm, params, loc=loc)

    def _train(self) -> ast.TrainStmt:
        loc = self._expect_keyword('train').loc
        model = self._expect(TokenKind.IDENT, 'a model name').text
        return ast.TrainStmt(model, self._param_block(), loc=loc)

    def _evaluate(self) -> ast.EvaluateStmt:
        loc = self._expect_keyword('evaluate').loc
        model = self._expect(TokenKind.IDENT, 'a model name').text
        return ast.EvaluateStmt(model, self._param_block(), loc=loc)

    def _register(self) -> ast.RegisterStmt:
        loc = self._expect_keyword('register').loc
        model = self._expect(TokenKind.IDENT, 'a model name').text
        params = self._param_block() if self._check(TokenKind.LBRACE) else ()
        return ast.RegisterStmt(model, params, loc=loc)

    def _param_block(self) -> tuple[ast.Param, ...]:
        self._expect(TokenKind.LBRACE, "'{'")
        params: list[ast.Param] = []
        seen: set[str] = set()
        while not self._match(TokenKind.RBRACE):
            key = self._peek()
            # Keywords are allowed as keys so blocks can grow (e.g. `model = "llm"`).
            if key.kind not in (TokenKind.IDENT, TokenKind.KEYWORD):
                raise ParseError(
                    f"expected a parameter name or '}}', found {key.describe()}", key.loc
                )
            if key.text in seen:
                raise ParseError(f"duplicate parameter '{key.text}'", key.loc)
            seen.add(key.text)
            self._advance()
            self._expect(TokenKind.ASSIGN, f"'=' after '{key.text}'")
            params.append(ast.Param(key.text, self._expression(), loc=key.loc))
            self._match(TokenKind.COMMA)
        return tuple(params)

    def _column(self) -> ast.Name:
        token = self._expect(TokenKind.IDENT, 'a column name')
        return ast.Name(token.text, loc=token.loc)

    # Expressions, lowest precedence first

    def _expression(self) -> ast.Expr:
        return self._or()

    def _or(self) -> ast.Expr:
        left = self._and()
        while self._peek().is_keyword('or'):
            op = self._advance()
            left = ast.Binary('or', left, self._and(), loc=op.loc)
        return left

    def _and(self) -> ast.Expr:
        left = self._not()
        while self._peek().is_keyword('and'):
            op = self._advance()
            left = ast.Binary('and', left, self._not(), loc=op.loc)
        return left

    def _not(self) -> ast.Expr:
        if self._peek().is_keyword('not'):
            op = self._advance()
            return ast.Unary('not', self._not(), loc=op.loc)
        return self._comparison()

    def _comparison(self) -> ast.Expr:
        left = self._additive()
        if self._peek().kind in COMPARISON_OPS:
            op = self._advance()
            left = ast.Binary(op.text, left, self._additive(), loc=op.loc)
            if self._peek().kind in COMPARISON_OPS:
                raise ParseError(
                    "comparisons cannot be chained; combine them with 'and'",
                    self._peek().loc,
                )
        return left

    def _additive(self) -> ast.Expr:
        left = self._multiplicative()
        while self._peek().kind in ADDITIVE_OPS:
            op = self._advance()
            left = ast.Binary(op.text, left, self._multiplicative(), loc=op.loc)
        return left

    def _multiplicative(self) -> ast.Expr:
        left = self._unary()
        while self._peek().kind in MULTIPLICATIVE_OPS:
            op = self._advance()
            left = ast.Binary(op.text, left, self._unary(), loc=op.loc)
        return left

    def _unary(self) -> ast.Expr:
        if self._check(TokenKind.MINUS):
            op = self._advance()
            return ast.Unary('-', self._unary(), loc=op.loc)
        return self._primary()

    def _primary(self) -> ast.Expr:
        token = self._advance()
        match token.kind:
            case TokenKind.INT:
                return ast.IntLiteral(int(token.text), loc=token.loc)
            case TokenKind.FLOAT:
                return ast.FloatLiteral(float(token.text), loc=token.loc)
            case TokenKind.STRING:
                return ast.StringLiteral(token.text, loc=token.loc)
            case TokenKind.KEYWORD if token.text in ('true', 'false'):
                return ast.BoolLiteral(token.text == 'true', loc=token.loc)
            case TokenKind.LBRACKET:
                return ast.ListExpr(self._sequence(TokenKind.RBRACKET, "']'"), loc=token.loc)
            case TokenKind.LPAREN:
                inner = self._expression()
                self._expect(TokenKind.RPAREN, "')'")
                return inner
            case TokenKind.IDENT:
                if self._match(TokenKind.LPAREN):
                    args = self._sequence(TokenKind.RPAREN, "')'")
                    return ast.Call(token.text, args, loc=token.loc)
                return ast.Name(token.text, loc=token.loc)
        raise ParseError(f'expected an expression, found {token.describe()}', token.loc)

    def _sequence(self, closing: TokenKind, closing_text: str) -> tuple[ast.Expr, ...]:
        """Comma-separated expressions up to `closing`, allowing a trailing comma."""
        items: list[ast.Expr] = []
        while not self._match(closing):
            items.append(self._expression())
            if not self._match(TokenKind.COMMA):
                self._expect(closing, f"',' or {closing_text}")
                break
        return tuple(items)

    # Token helpers

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        if token.kind is not TokenKind.EOF:
            self.pos += 1
        return token

    def _check(self, kind: TokenKind) -> bool:
        return self._peek().kind is kind

    def _match(self, kind: TokenKind) -> bool:
        if self._check(kind):
            self._advance()
            return True
        return False

    def _expect(self, kind: TokenKind, what: str) -> Token:
        token = self._peek()
        if token.kind is not kind:
            raise ParseError(f'expected {what}, found {token.describe()}', token.loc)
        return self._advance()

    def _expect_keyword(self, word: str) -> Token:
        token = self._peek()
        if not token.is_keyword(word):
            raise ParseError(f"expected '{word}', found {token.describe()}", token.loc)
        return self._advance()


def parse(source: str) -> ast.Program:
    return Parser(tokenize(source)).parse_program()
