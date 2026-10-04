"""Checking and running programs: shared by the CLI and the MCP server."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from eigrel import __version__
from eigrel.backends import python as python_backend
from eigrel.backends import spark as spark_backend
from eigrel.compiler import EigrelError, Graph, analyze, parse

RUNNERS = {
    'python': (python_backend.generate, python_backend.runtime_requirements),
    'spark': (spark_backend.generate, spark_backend.runtime_requirements),
}


def check_source(source: str, path: str) -> dict[str, Any]:
    """The `check --json` report for one program's source."""
    try:
        graph = analyze(parse(source))
    except EigrelError as exc:
        # LexError, ParseError or SemanticError -> 'lex', 'parse' or 'semantic'.
        stage = type(exc).__name__.removesuffix('Error').lower()
        error = {
            'stage': stage,
            'message': exc.message,
            'line': exc.loc.line,
            'column': exc.loc.column,
        }
        return {'path': path, 'ok': False, 'operations': None, 'errors': [error]}
    return {'path': path, 'ok': True, 'operations': len(graph.ops), 'errors': []}


def check_file(path: str) -> dict[str, Any]:
    """The `check --json` report for one file."""
    try:
        source = Path(path).read_text(encoding='utf-8')
    except OSError as exc:
        errors = [{'stage': 'io', 'message': f'cannot read {path}: {exc.strerror}'}]
        return {'path': path, 'ok': False, 'operations': None, 'errors': errors}
    return check_source(source, path)


def check_files(paths: list[str]) -> dict[str, Any]:
    reports = [check_file(path) for path in paths]
    return {'eigrel': __version__, 'ok': all(r['ok'] for r in reports), 'files': reports}


def importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package such as google is missing
        return False


def has_java() -> bool:
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


def missing_runtime(graph: Graph, target: str) -> str | None:
    """Why a program cannot run on this machine, or None when it can."""
    _, requirements = RUNNERS[target]
    missing = [r for r in requirements(graph) if not importable(r.module)]
    if missing:
        modules = ', '.join(r.module for r in missing)
        extras = ','.join(sorted({r.extra for r in missing}))
        return f'running needs {modules}; install them with: pip install "eigrel[{extras}]"'
    if target == 'spark' and not has_java():
        return 'running on Spark needs Java 17 or newer; install a JDK and set JAVA_HOME'
    return None


def execute(
    graph: Graph, path: str, target: str, capture: bool = False, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Generate the program's code and run it next to the program, as `eigrel run` does."""
    generate, _ = RUNNERS[target]
    code = generate(graph, Path(path).name)
    # Paths inside the program are relative to the file, like imports in most languages.
    workdir = Path(path).resolve().parent
    # Run from a real file rather than `python -c`: some libraries (PySpark under MLflow) exit
    # early when the main module has no file, and tracebacks point at real line numbers.
    with tempfile.TemporaryDirectory(prefix='eigrel-') as scratch:
        script = Path(scratch) / f'{Path(path).stem}_{target}.py'
        script.write_text(code, encoding='utf-8')
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            check=False,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
