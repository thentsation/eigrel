import argparse
import json
import sys
from pathlib import Path

from eigrel import __version__
from eigrel.compiler import EigrelError, parse, tokenize
from eigrel.compiler.ast import Program, to_dict

STARTER_PROGRAM = """\
# A first Eigrel pipeline: load data, pick features, train and evaluate a model.

dataset customers from csv("data/customers.csv")

transform customers {
    filter age >= 18
    select age, income, purchases, churned
}

features customers {
    age
    income
    purchases
}

model churn = random_forest {
    trees = 100
}

train churn {
    target = churned
}

evaluate churn {
    metrics = [accuracy, precision, recall, f1]
}
"""


def _load(path: str) -> tuple[str, str] | None:
    try:
        return Path(path).read_text(encoding='utf-8'), path
    except OSError as exc:
        print(f'error: cannot read {path}: {exc.strerror}', file=sys.stderr)
        return None


def _parse_file(path: str) -> Program | None:
    loaded = _load(path)
    if loaded is None:
        return None
    source, filename = loaded
    try:
        return parse(source)
    except EigrelError as exc:
        print(exc.render(source, filename), file=sys.stderr)
        return None


def cmd_check(args: argparse.Namespace) -> int:
    failed = 0
    for path in args.files:
        program = _parse_file(path)
        if program is None:
            failed += 1
        else:
            count = len(program.statements)
            noun = 'statement' if count == 1 else 'statements'
            print(f'ok: {path} ({count} {noun})')
    return 1 if failed else 0


def cmd_ast(args: argparse.Namespace) -> int:
    program = _parse_file(args.file)
    if program is None:
        return 1
    print(json.dumps(to_dict(program), indent=2))
    return 0


def cmd_tokens(args: argparse.Namespace) -> int:
    loaded = _load(args.file)
    if loaded is None:
        return 1
    source, filename = loaded
    try:
        tokens = tokenize(source)
    except EigrelError as exc:
        print(exc.render(source, filename), file=sys.stderr)
        return 1
    for token in tokens:
        print(f'{token.loc.line}:{token.loc.column}\t{token.kind.name}\t{token.text}')
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.name)
    if root.exists():
        print(f'error: {root} already exists', file=sys.stderr)
        return 1
    (root / 'data').mkdir(parents=True)
    (root / 'main.eig').write_text(STARTER_PROGRAM, encoding='utf-8')
    print(f'created {root}/main.eig')
    print(f'next: eigrel check {root}/main.eig')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='eigrel',
        description='Eigrel: a programming language for Data, Machine Learning and AI.',
    )
    parser.add_argument('--version', action='version', version=f'eigrel {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)

    check = commands.add_parser('check', help='parse files and report syntax errors')
    check.add_argument('files', nargs='+', metavar='FILE')
    check.set_defaults(handler=cmd_check)

    ast_cmd = commands.add_parser('ast', help='print the syntax tree of a file as JSON')
    ast_cmd.add_argument('file', metavar='FILE')
    ast_cmd.set_defaults(handler=cmd_ast)

    tokens = commands.add_parser('tokens', help='print the tokens of a file')
    tokens.add_argument('file', metavar='FILE')
    tokens.set_defaults(handler=cmd_tokens)

    init = commands.add_parser('init', help='create a new Eigrel project')
    init.add_argument('name', metavar='NAME')
    init.set_defaults(handler=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    result: int = handler(args)
    return result


if __name__ == '__main__':
    sys.exit(main())
