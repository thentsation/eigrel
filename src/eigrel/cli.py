import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from eigrel import __version__, mcp, planner
from eigrel.backends import UnsupportedError
from eigrel.backends import python as python_backend
from eigrel.backends import spark as spark_backend
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


BACKENDS = {
    'python': python_backend.generate,
    'spark': spark_backend.generate,
    'sql': sql_backend.generate,
}
RUNNERS = {
    'python': (python_backend.generate, python_backend.runtime_requirements),
    'spark': (spark_backend.generate, spark_backend.runtime_requirements),
}


def _has_java() -> bool:
    """Whether a working Java runtime is available (macOS ships a stub `java` without one)."""
    java_home = os.environ.get('JAVA_HOME')
    java = str(Path(java_home) / 'bin' / 'java') if java_home else shutil.which('java')
    if java is None:
        return False
    try:
        result = subprocess.run([java, '-version'], capture_output=True, check=False, timeout=30)
    except OSError:
        return False
    return result.returncode == 0


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package such as google is missing
        return False


def cmd_check(args: argparse.Namespace) -> int:
    if args.json:
        reports = [_check_report(path) for path in args.files]
        ok = all(report['ok'] for report in reports)
        print(json.dumps({'eigrel': __version__, 'ok': ok, 'files': reports}, indent=2))
        return 0 if ok else 1
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


def _check_report(path: str) -> dict[str, object]:
    """Check one file and return a machine-readable report for `check --json`."""
    errors: list[dict[str, object]]
    try:
        source = Path(path).read_text(encoding='utf-8')
    except OSError as exc:
        errors = [{'stage': 'io', 'message': f'cannot read {path}: {exc.strerror}'}]
        return {'path': path, 'ok': False, 'operations': None, 'errors': errors}
    try:
        graph = analyze(parse(source))
    except EigrelError as exc:
        # LexError, ParseError or SemanticError -> 'lex', 'parse' or 'semantic'.
        stage = type(exc).__name__.removesuffix('Error').lower()
        errors = [
            {'stage': stage, 'message': exc.message, 'line': exc.loc.line, 'column': exc.loc.column}
        ]
        return {'path': path, 'ok': False, 'operations': None, 'errors': errors}
    count = len(graph.ops)
    return {'path': path, 'ok': True, 'operations': count, 'errors': []}


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
    try:
        code = BACKENDS[args.target](graph, Path(args.file).name)
    except UnsupportedError as exc:
        _error(str(exc))
        return 1
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
    generate, requirements = RUNNERS[args.target]
    missing = [r for r in requirements(graph) if not _importable(r.module)]
    if missing:
        modules = ', '.join(r.module for r in missing)
        extras = ','.join(sorted({r.extra for r in missing}))
        _error(f'running needs {modules}; install them with: pip install "eigrel[{extras}]"')
        return 1
    if args.target == 'spark' and not _has_java():
        _error('running on Spark needs Java 17 or newer; install a JDK and set JAVA_HOME')
        return 1
    code = generate(graph, Path(args.file).name)
    # Paths inside the program are relative to the file, like imports in most languages.
    workdir = Path(args.file).resolve().parent
    # Run from a real file rather than `python -c`: some libraries (PySpark under MLflow) exit
    # early when the main module has no file, and tracebacks point at real line numbers.
    with tempfile.TemporaryDirectory(prefix='eigrel-') as scratch:
        script = Path(scratch) / f'{Path(args.file).stem}_{args.target}.py'
        script.write_text(code, encoding='utf-8')
        return subprocess.run([sys.executable, str(script)], cwd=workdir, check=False).returncode


def cmd_plan(args: argparse.Namespace) -> int:
    plan = planner.build_plan(Path(args.file), args.target)
    if args.json:
        print(json.dumps(planner.to_json(plan), indent=2, default=str))
    else:
        print(planner.render_for(plan, sys.stdout.encoding), end='')
    if not plan.ok:
        return 1
    if args.save:
        path = planner.save_state(plan)
        if not args.json:
            print(f'saved {path}')
    if args.strict and any(f.severity == 'warning' for f in plan.findings):
        return 2
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    return mcp.main()


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

    check = command('check', 'check files for syntax and semantic errors', cmd_check)
    check.add_argument('files', nargs='+', metavar='FILE')
    check.add_argument(
        '--json', action='store_true', help='report the results as JSON (for tools and agents)'
    )
    run = command('run', 'compile a file and run it', cmd_run)
    run.add_argument('file', metavar='FILE')
    run.add_argument(
        '-t',
        '--target',
        choices=sorted(RUNNERS),
        default='python',
        help='backend to run on (default: python)',
    )
    plan = command('plan', 'show what a program will do to its data, before running it', cmd_plan)
    plan.add_argument('file', metavar='FILE')
    plan.add_argument(
        '-t',
        '--target',
        choices=sorted(RUNNERS),
        default='python',
        help='backend whose limits apply (default: python)',
    )
    plan.add_argument('--json', action='store_true', help='print the plan as JSON')
    plan.add_argument(
        '--save',
        action='store_true',
        help='record the data schema in FILE.eigstate for drift checks',
    )
    plan.add_argument(
        '--strict', action='store_true', help='exit with status 2 when there are warnings'
    )
    compile_cmd = command('compile', 'print the generated code (Python, Spark or SQL)', cmd_compile)
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
    command('mcp', 'serve check, plan and compile to AI agents over MCP (stdio)', cmd_mcp)
    command('init', 'create a new Eigrel project', cmd_init).add_argument('name', metavar='NAME')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.handler(args)
    return result


if __name__ == '__main__':
    sys.exit(main())
