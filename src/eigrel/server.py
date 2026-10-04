"""`eigrel mcp`: a Model Context Protocol server, so agents can check, plan, compile and run
Eigrel programs and get the same JSON the CLI prints.

Needs the `mcp` extra. The compiler never imports this module.
"""

import json
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from eigrel import __version__, planner, runner
from eigrel.backends import UnsupportedError
from eigrel.cli import BACKENDS
from eigrel.compiler import analyze, parse
from eigrel.compiler.catalog import BACKENDS as CAPABILITY_BACKENDS
from eigrel.compiler.catalog import CAPABILITIES

INSTRUCTIONS = """\
Eigrel is a small language for tabular machine learning. Write a program to a .eig file, then:
1. check it (fix every error at its line and column; repeat until ok is true),
2. plan it against its data (errors predict a failed run; report warnings to the human),
3. run it, or compile it to Python, Spark or SQL.
Read the eigrel://llms.txt resource for the complete grammar before writing a program."""

OUTPUT_LIMIT = 20_000


def grammar() -> str:
    """The complete language reference and agent workflow (llms.txt)."""
    return files('eigrel').joinpath('llms.txt').read_text(encoding='utf-8')


def capabilities() -> str:
    """What each backend supports, as JSON: {capability: {backend: support}}."""
    return json.dumps({'backends': list(CAPABILITY_BACKENDS), 'capabilities': CAPABILITIES})


def check(path: str | None = None, source: str | None = None) -> dict[str, Any]:
    """Check a program for syntax and semantic errors without reading any data.

    Pass the path of a .eig file, or its source text. Every error has a stage, a message and
    the exact line and column to fix.
    """
    if source is not None:
        return runner.check_source(source, path or '<source>')
    if path is None:
        return _usage('check needs a path or a source')
    return runner.check_file(path)


def plan(path: str, target: Literal['python', 'spark'] = 'python') -> dict[str, Any]:
    """Check a program against its data and report what every step will do, without training.

    Returns datasets with row counts per step, trainings with class counts and splits,
    predictions, and findings with a severity, a stable code and a line.
    """
    return planner.to_json(planner.build_plan(Path(path), target))


def compile_program(
    path: str | None = None,
    source: str | None = None,
    target: Literal['python', 'spark', 'sql'] = 'python',
) -> dict[str, Any]:
    """Compile a program to Python (pandas + scikit-learn), Spark (PySpark + MLlib) or SQL."""
    if source is None:
        if path is None:
            return _usage('compile needs a path or a source')
        try:
            source = Path(path).read_text(encoding='utf-8')
        except OSError as exc:
            return _usage(f'cannot read {path}: {exc.strerror}')
    report = runner.check_source(source, path or '<source>')
    if not report['ok']:
        return report
    try:
        code = BACKENDS[target](analyze(parse(source)), Path(path or 'program.eig').name)
    except UnsupportedError as exc:
        return _usage(str(exc))
    return {'ok': True, 'target': target, 'code': code}


def run(
    path: str, target: Literal['python', 'spark'] = 'python', timeout_seconds: int = 600
) -> dict[str, Any]:
    """Compile a program and run it next to its file; returns the exit code and its output."""
    report = runner.check_file(path)
    if not report['ok']:
        return report
    graph = analyze(parse(Path(path).read_text(encoding='utf-8')))
    problem = runner.missing_runtime(graph, target)
    if problem is not None:
        return _usage(problem)
    try:
        result = runner.execute(graph, path, target, capture=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return _usage(f'the program did not finish within {timeout_seconds} seconds')
    return {
        'ok': result.returncode == 0,
        'exit_code': result.returncode,
        'stdout': result.stdout[-OUTPUT_LIMIT:],
        'stderr': result.stderr[-OUTPUT_LIMIT:],
    }


def _usage(message: str) -> dict[str, Any]:
    return {'ok': False, 'errors': [{'stage': 'usage', 'message': message}]}


def build_server() -> MCPServer:
    server: MCPServer = MCPServer('eigrel', instructions=INSTRUCTIONS, version=__version__)
    server.tool()(check)
    server.tool()(plan)
    server.tool(name='compile')(compile_program)
    server.tool()(run)
    server.resource('eigrel://llms.txt', mime_type='text/plain')(grammar)
    server.resource('eigrel://capabilities', mime_type='application/json')(capabilities)
    return server


def main() -> None:
    build_server().run('stdio')


if __name__ == '__main__':
    main()
