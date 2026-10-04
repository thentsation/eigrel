import io
import json
from pathlib import Path
from typing import Any

from eigrel import mcp
from eigrel.cli import main

VALID = """\
dataset customers from csv("data/customers.csv")
features customers { age income purchases }
model churn = random_forest { trees = 10 }
train churn { target = churned }
"""
LEAK = VALID.replace('purchases }', 'purchases churned }')


def rpc(method: str, params: dict[str, Any] | None = None, id: int = 1) -> dict[str, Any]:
    message: dict[str, Any] = {'jsonrpc': '2.0', 'id': id, 'method': method}
    if params is not None:
        message['params'] = params
    response = mcp.handle(message)
    assert response is not None
    return response


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = rpc('tools/call', {'name': name, 'arguments': arguments})['result']
    return result


def test_initialize_announces_the_tools_capability() -> None:
    result = rpc('initialize', {'protocolVersion': '2025-06-18'})['result']
    assert result['capabilities'] == {'tools': {}}
    assert result['serverInfo']['name'] == 'eigrel'


def test_notifications_get_no_answer() -> None:
    assert mcp.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


def test_tools_are_listed_with_schemas() -> None:
    tools = rpc('tools/list')['result']['tools']
    assert {t['name'] for t in tools} == {
        'eigrel_check',
        'eigrel_plan',
        'eigrel_compile',
        'eigrel_ir',
    }
    assert all(t['inputSchema']['type'] == 'object' for t in tools)


def test_check_accepts_a_valid_program() -> None:
    result = call('eigrel_check', {'source': VALID})
    assert result['isError'] is False
    assert json.loads(result['content'][0]['text']) == {
        'ok': True,
        'path': '<source>',
        'operations': 2,
        'errors': [],
    }


def test_check_reports_the_leak_with_its_position() -> None:
    report = json.loads(call('eigrel_check', {'source': LEAK})['content'][0]['text'])
    assert report['ok'] is False
    [error] = report['errors']
    assert error['stage'] == 'semantic'
    assert "target 'churned' is also declared as a feature" in error['message']
    assert (error['line'], error['column']) == (4, 24)
    assert 'error:' in report['rendered']


def test_compile_generates_code_for_every_target() -> None:
    python = call('eigrel_compile', {'source': VALID})['content'][0]['text']
    assert 'RandomForestClassifier' in python
    sql = call('eigrel_compile', {'source': VALID, 'target': 'sql'})['content'][0]['text']
    assert 'SELECT' in sql


def test_compile_of_an_invalid_program_is_a_tool_error() -> None:
    result = call('eigrel_compile', {'source': LEAK})
    assert result['isError'] is True
    assert 'also declared as a feature' in result['content'][0]['text']


def test_ir_shows_the_graph() -> None:
    assert '%0 = load csv' in call('eigrel_ir', {'source': VALID})['content'][0]['text']


def test_plan_reads_the_data(examples_dir: Path) -> None:
    result = call('eigrel_plan', {'path': str(examples_dir / 'ml.eig')})
    plan = json.loads(result['content'][0]['text'])
    assert plan['ok'] is True


def test_plan_needs_a_path() -> None:
    assert call('eigrel_plan', {'source': VALID})['isError'] is True


def test_missing_input_and_unknown_tools_are_tool_errors(tmp_path: Path) -> None:
    assert call('eigrel_check', {})['isError'] is True
    assert call('eigrel_check', {'path': str(tmp_path / 'nope.eig')})['isError'] is True
    assert call('nope', {})['isError'] is True


def test_unknown_methods_are_json_rpc_errors() -> None:
    assert rpc('nope')['error']['code'] == -32601


def test_serve_answers_line_by_line_and_survives_garbage() -> None:
    lines = [
        json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}),
        'not json',
        '',
        json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}),
        json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}),
    ]
    out = io.StringIO()
    mcp.serve(io.StringIO('\n'.join(lines) + '\n'), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get('id') for r in replies] == [1, None, 2]
    assert replies[1]['error']['code'] == -32700


def test_the_cli_has_an_mcp_command(monkeypatch: Any) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO(''))
    assert main(['mcp']) == 0
