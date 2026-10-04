import argparse
import json
import sys
from pathlib import Path

from eigrel import __version__, planner, runner
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


def cmd_check(args: argparse.Namespace) -> int:
    if args.json:
        report = runner.check_files(args.files)
        print(json.dumps(report, indent=2))
        return 0 if report['ok'] else 1
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
    problem = runner.missing_runtime(graph, args.target)
    if problem is not None:
        _error(problem)
        return 1
    return runner.execute(graph, args.file, args.target).returncode


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
    if not runner.importable('mcp'):
        _error('the MCP server needs the mcp package: pip install "eigrel[mcp]"')
        return 1
    from eigrel import server

    server.main()
    return 0


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
        choices=sorted(runner.RUNNERS),
        default='python',
        help='backend to run on (default: python)',
    )
    plan = command('plan', 'show what a program will do to its data, before running it', cmd_plan)
    plan.add_argument('file', metavar='FILE')
    plan.add_argument(
        '-t',
        '--target',
        choices=sorted(runner.RUNNERS),
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
    command('init', 'create a new Eigrel project', cmd_init).add_argument('name', metavar='NAME')
    command('mcp', 'serve check, plan, compile and run to agents over MCP (stdio)', cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.handler(args)
    return result


if __name__ == '__main__':
    sys.exit(main())
