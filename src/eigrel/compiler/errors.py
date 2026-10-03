from eigrel.compiler.tokens import Location


class EigrelError(Exception):
    """A compile-time error tied to a position in the source file."""

    def __init__(self, message: str, loc: Location) -> None:
        super().__init__(message)
        self.message = message
        self.loc = loc

    def render(self, source: str, filename: str) -> str:
        """Format the error with the offending line and a caret under the column."""
        lines = source.splitlines()
        text = lines[self.loc.line - 1] if 0 < self.loc.line <= len(lines) else ''
        gutter = ' ' * len(str(self.loc.line))
        return '\n'.join(
            [
                f'error: {self.message}',
                f'{gutter}--> {filename}:{self.loc.line}:{self.loc.column}',
                f'{gutter} |',
                f'{self.loc.line} | {text}',
                f'{gutter} | {" " * (self.loc.column - 1)}^',
            ]
        )


class LexError(EigrelError):
    pass


class ParseError(EigrelError):
    pass


class SemanticError(EigrelError):
    pass
