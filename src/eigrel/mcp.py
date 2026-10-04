"""A Model Context Protocol server: lets agents check, plan and compile Eigrel programs.

Speaks MCP over stdio (newline-delimited JSON-RPC 2.0) with the standard library only, so
`eigrel mcp` works wherever `eigrel` does. The tools never run a pipeline or train anything:
`eigrel run` stays a decision for a human.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from eigrel import __version__, planner
from eigrel.backends import UnsupportedError
from eigrel.backends import python as python_backend
from eigrel.backends import spark as spark_backend
from eigrel.backends import sql as sql_backend
from eigrel.compiler import EigrelError, analyze, parse
from eigrel.compiler.ir import Graph, format_graph

PROTOCOL_VERSION = '2025-06-18'
GENERATORS = {
    'python': python_backend.generate,
    'spark': spark_backend.generate,
    'sql': sql_backend.generate,
}

_SOURCE_OR_PATH = {
    'source': {'type': 'string', 'description': 'Eigrel program text.'},
    'path': {
        'type': 'string',
        'description': 'Path of a .eig file. Used when `source` is not given.',
    },
}
_TARGET = {
    'type': 'string',
    'enum': sorted(GENERATORS),
    'default': 'python',
    'description': 'Backend to generate code for.',
}

TOOLS: list[dict[str, Any]] = [
    {
        'name': 'eigrel_check',
        'description': (
            'Check an Eigrel program for syntax and semantic errors without reading any data. '
            'Returns ok and a list of errors, each with stage, message, line and column. '
            'Fix the first error and check again.'
        ),
        'inputSchema': {'type': 'object', 'properties': _SOURCE_OR_PATH},
    },
    {
        'name': 'eigrel_plan',
        'description': (
            'Read the real data and show what the program will do before it runs: columns and '
            'types, rows kept by every step, split sizes, class balance, and findings (errors '
            'predict a failed run, warnings are judgement calls) plus schema drift against the '
            'saved .eigstate. Trains nothing.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path of the .eig file.'},
                'target': {
                    'type': 'string',
                    'enum': ['python', 'spark'],
                    'default': 'python',
                    'description': 'Backend whose limits apply.',
                },
            },
            'required': ['path'],
        },
    },
    {
        'name': 'eigrel_compile',
        'description': 'Generate the code for a valid program: Python, Spark or SQL.',
        'inputSchema': {
            'type': 'object',
            'properties': {**_SOURCE_OR_PATH, 'target': _TARGET},
        },
    },
    {
        'name': 'eigrel_ir',
        'description': 'Show the operation graph (intermediate representation) of a program.',
        'inputSchema': {'type': 'object', 'properties': _SOURCE_OR_PATH},
    },
]


class ToolError(Exception):
    """A tool call that failed in a way the agent should read about and fix."""


def _program(arguments: dict[str, Any]) -> tuple[str, str]:
    """The program text and the name to report it under."""
    source = arguments.get('source')
    if isinstance(source, str):
        return source, '<source>'
    path = arguments.get('path')
    if not isinstance(path, str):
        raise ToolError('pass either `source` (the program text) or `path` (a .eig file)')
    try:
        return Path(path).read_text(encoding='utf-8'), path
    except OSError as exc:
        raise ToolError(f'cannot read {path}: {exc.strerror}') from exc


def _graph(arguments: dict[str, Any]) -> Graph:
    source, name = _program(arguments)
    try:
        return analyze(parse(source))
    except EigrelError as exc:
        raise ToolError(exc.render(source, name)) from exc


def _check(arguments: dict[str, Any]) -> Any:
    source, name = _program(arguments)
    try:
        graph = analyze(parse(source))
    except EigrelError as exc:
        stage = type(exc).__name__.removesuffix('Error').lower()
        error = {'stage': stage, 'message': exc.message, 'line': exc.loc.line}
        error['column'] = exc.loc.column
        return {
            'ok': False,
            'path': name,
            'errors': [error],
            'rendered': exc.render(source, name),
        }
    return {'ok': True, 'path': name, 'operations': len(graph.ops), 'errors': []}


def _plan(arguments: dict[str, Any]) -> Any:
    path = arguments.get('path')
    if not isinstance(path, str):
        raise ToolError('eigrel_plan needs `path`: it reads the data the program points at')
    target = arguments.get('target', 'python')
    if target not in ('python', 'spark'):
        raise ToolError("target must be 'python' or 'spark'")
    return planner.to_json(planner.build_plan(Path(path), target))


def _compile(arguments: dict[str, Any]) -> Any:
    target = arguments.get('target', 'python')
    if target not in GENERATORS:
        raise ToolError(f'target must be one of: {", ".join(sorted(GENERATORS))}')
    graph = _graph(arguments)
    _, name = _program(arguments)
    try:
        return GENERATORS[target](graph, Path(name).name)
    except UnsupportedError as exc:
        raise ToolError(str(exc)) from exc


def _ir(arguments: dict[str, Any]) -> Any:
    return format_graph(_graph(arguments))


HANDLERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    'eigrel_check': _check,
    'eigrel_plan': _plan,
    'eigrel_compile': _compile,
    'eigrel_ir': _ir,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one tool and wrap the outcome as an MCP tool result."""
    handler = HANDLERS.get(name)
    if handler is None:
        return {'content': [{'type': 'text', 'text': f'unknown tool {name}'}], 'isError': True}
    try:
        result = handler(arguments)
    except ToolError as exc:
        return {'content': [{'type': 'text', 'text': str(exc)}], 'isError': True}
    text = result if isinstance(result, str) else json.dumps(result, indent=2, default=str)
    return {'content': [{'type': 'text', 'text': text}], 'isError': False}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one JSON-RPC message; notifications get no answer."""
    method = message.get('method')
    request_id = message.get('id')
    if request_id is None:  # a notification such as notifications/initialized
        return None

    def reply(result: Any) -> dict[str, Any]:
        return {'jsonrpc': '2.0', 'id': request_id, 'result': result}

    if method == 'initialize':
        return reply(
            {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'eigrel', 'version': __version__},
                'instructions': (
                    'Write the program, run eigrel_check until ok, then eigrel_plan to see what '
                    'it does to the real data before anything runs.'
                ),
            }
        )
    if method == 'ping':
        return reply({})
    if method == 'tools/list':
        return reply({'tools': TOOLS})
    if method == 'tools/call':
        params = message.get('params') or {}
        arguments = params.get('arguments') or {}
        return reply(call_tool(str(params.get('name')), arguments))
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'error': {'code': -32601, 'message': f'method not found: {method}'},
    }


def serve(stdin: TextIO, stdout: TextIO) -> None:
    """Read one JSON message per line from stdin and answer on stdout until it closes."""
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = {
                'jsonrpc': '2.0',
                'id': None,
                'error': {'code': -32700, 'message': 'parse error'},
            }
        else:
            response = handle(message) if isinstance(message, dict) else None
        if response is not None:
            stdout.write(json.dumps(response) + '\n')
            stdout.flush()


def main() -> int:
    serve(sys.stdin, sys.stdout)
    return 0
