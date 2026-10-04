import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from eigrel import __version__
from eigrel.backends import python as python_backend
from eigrel.backends import sql as sql_backend
from eigrel.compiler import EigrelError, analyze, parse, tokenize
from eigrel.compiler.ast import to_dict
from eigrel.compiler.ir import Graph, format_graph
from eigrel.starter import PROGRAM, write_sample_customers


def _error(message: str) -> None:
    print(f'error: {message}', file=sys.stderr)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding='utf-8')
    except OSError as exc:
        _error(f'cannot read {path}: {exc.strerror}')
        return None


def _compile(path: str) -> Graph | None:
    """Parse and analyze a file, printing diagnostics on failure."""
    source = _read(path)
    if source is None:
        return None
    try:
        return analyze(parse(source))
    except EigrelError as exc:
        print(exc.render(source, path), file=sys.stderr)
        return None


BACKENDS = {'python': python_backend.generate, 'sql': sql_backend.generate}


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package such as google is missing
        return False


def cmd_check(args: argparse.Namespace) -> int:
    failed = 0
    for path in args.files:
        graph = _compile(path)
        if graph is None:
            failed += 1
        else:
            count = len(graph.ops)
            noun = 'operation' if count == 1 else 'operations'
            print(f'ok: {path} ({count} {noun})')
    return 1 if failed else 0


def cmd_ast(args: argparse.Namespace) -> int:
    source = _read(args.file)
    if source is None:
        return 1
    try:
        program = parse(source)
    except EigrelError as exc:
        print(exc.render(source, args.file), file=sys.stderr)
        return 1
    print(json.dumps(to_dict(program), indent=2))
    return 0


def cmd_tokens(args: argparse.Namespace) -> int:
    source = _read(args.file)
    if source is None:
        return 1
    try:
        tokens = tokenize(source)
    except EigrelError as exc:
        print(exc.render(source, args.file), file=sys.stderr)
        return 1
    for token in tokens:
        print(f'{token.loc.line}:{token.loc.column}\t{token.kind.name}\t{token.text}')
    return 0


def cmd_ir(args: argparse.Namespace) -> int:
    graph = _compile(args.file)
    if graph is None:
        return 1
    print(format_graph(graph))
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    graph = _compile(args.file)
    if graph is None:
        return 1
    code = BACKENDS[args.target](graph, Path(args.file).name)
    if args.output is None:
        print(code, end='')
    else:
        Path(args.output).write_text(code, encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    graph = _compile(args.file)
    if graph is None:
        return 1
    requirements = python_backend.runtime_requirements(graph)
    missing = [r for r in requirements if not _importable(r.module)]
    if missing:
        modules = ', '.join(r.module for r in missing)
        extras = ','.join(sorted({r.extra for r in missing}))
        _error(f'running needs {modules}; install them with: pip install "eigrel[{extras}]"')
        return 1
    code = python_backend.generate(graph, Path(args.file).name)
    # Paths inside the program are relative to the file, like imports in most languages.
    workdir = Path(args.file).resolve().parent
    return subprocess.run([sys.executable, '-c', code], cwd=workdir, check=False).returncode


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.name)
    if root.exists():
        _error(f'{root} already exists')
        return 1
    (root / 'data').mkdir(parents=True)
    (root / 'main.eig').write_text(PROGRAM, encoding='utf-8')
    write_sample_customers(root / 'data' / 'customers.csv')
    print(f'created {root}/main.eig and {root}/data/customers.csv')
    print(f'next: eigrel run {root}/main.eig')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='eigrel',
        description='Eigrel: a programming language for Data, Machine Learning and AI.',
    )
    parser.add_argument('--version', action='version', version=f'eigrel {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name: str, help: str, handler: object) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.set_defaults(handler=handler)
        return sub

    command('check', 'check files for syntax and semantic errors', cmd_check).add_argument(
        'files', nargs='+', metavar='FILE'
    )
    command('run', 'compile a file to Python and run it', cmd_run).add_argument(
        'file', metavar='FILE'
    )
    compile_cmd = command('compile', 'print the generated code (Python or SQL)', cmd_compile)
    compile_cmd.add_argument('file', metavar='FILE')
    compile_cmd.add_argument('-o', '--output', metavar='PATH', help='write the code to PATH')
    compile_cmd.add_argument(
        '-t',
        '--target',
        choices=sorted(BACKENDS),
        default='python',
        help='backend to generate code for (default: python)',
    )
    command('ir', 'print the intermediate representation', cmd_ir).add_argument(
        'file', metavar='FILE'
    )
    command('ast', 'print the syntax tree as JSON', cmd_ast).add_argument('file', metavar='FILE')
    command('tokens', 'print the tokens of a file', cmd_tokens).add_argument('file', metavar='FILE')
    command('init', 'create a new Eigrel project', cmd_init).add_argument('name', metavar='NAME')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.handler(args)
    return result


if __name__ == '__main__':
    sys.exit(main())
