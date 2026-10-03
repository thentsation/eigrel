"""The Eigrel compiler: lexer, parser, semantic analysis and IR."""

from eigrel.compiler.errors import EigrelError, LexError, ParseError, SemanticError
from eigrel.compiler.ir import Graph
from eigrel.compiler.lexer import tokenize
from eigrel.compiler.parser import parse
from eigrel.compiler.semantic import analyze


def compile_source(source: str) -> Graph:
    """Parse and analyze Eigrel source, returning its IR graph."""
    return analyze(parse(source))


__all__ = [
    'EigrelError',
    'Graph',
    'LexError',
    'ParseError',
    'SemanticError',
    'analyze',
    'compile_source',
    'parse',
    'tokenize',
]
