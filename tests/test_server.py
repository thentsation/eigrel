import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eigrel import runner, server
from eigrel.cli import main


def write(tmp_path: Path, source: str, name: str = 'p.eig') -> str:
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_check_tool() -> None:
    assert server.check(source='model m = random_forest')['ok'] is True
    report = server.check(source='model m = rf')
    assert report['errors'][0]['stage'] == 'semantic'
    assert report['errors'][0]['line'] == 1
    assert server.check()['errors'][0]['stage'] == 'usage'


def test_check_tool_reads_files(tmp_path: Path) -> None:
    assert server.check(path=write(tmp_path, 'model m = random_forest'))['ok'] is True
    assert server.check(path=str(tmp_path / 'missing.eig'))['errors'][0]['stage'] == 'io'


def test_plan_tool(examples_dir: Path) -> None:
    data = server.plan(str(examples_dir / 'ml.eig'))
    assert data['ok'] is True
    assert [s['rows'] for s in data['datasets'][0]['steps']] == [400, 394, 394]
    assert server.plan(str(examples_dir / 'ml.eig'), 'spark')['backend'] == 'spark'


def test_compile_tool(tmp_path: Path, examples_dir: Path) -> None:
    python = server.compile_program(path=str(examples_dir / 'ml.eig'))
    assert python['ok'] and 'RandomForestClassifier' in python['code']
    assert (
        'SELECT' in server.compile_program(path=str(examples_dir / 'ml.eig'), target='sql')['code']
    )
    spark = server.compile_program(source='dataset d from csv("d.csv")', target='spark')
    assert 'SparkSession' in spark['code']
    assert server.compile_program(source='model m = rf')['errors'][0]['stage'] == 'semantic'
    assert server.compile_program()['errors'][0]['stage'] == 'usage'
    assert (
        'cannot read'
        in server.compile_program(path=str(tmp_path / 'x.eig'))['errors'][0]['message']
    )
    unsupported = server.compile_program(
        source='dataset d from sql("sqlite://", "t")\ntransform d { drop_missing }', target='sql'
    )
    assert 'drop_missing' in unsupported['errors'][0]['message']


def test_run_tool(tmp_path: Path, examples_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = server.run(str(examples_dir / 'ml.eig'))
    assert result['ok'] and result['exit_code'] == 0
    assert 'churn: random_forest classification' in result['stdout']
    assert server.run(write(tmp_path, 'model m = rf'))['errors'][0]['stage'] == 'semantic'

    failing = write(
        tmp_path,
        'dataset d from csv("missing.csv")\nmodel m = random_forest\ntrain m { target = y }',
        'f.eig',
    )
    result = server.run(failing)
    assert result['ok'] is False and result['exit_code'] != 0 and result['stderr']

    monkeypatch.setattr(runner, 'importable', lambda module: module != 'sklearn')
    assert 'eigrel[python]' in server.run(str(examples_dir / 'ml.eig'))['errors'][0]['message']
    monkeypatch.undo()

    def too_slow(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd='python', timeout=1)

    monkeypatch.setattr(runner, 'execute', too_slow)
    assert (
        'did not finish within 1 seconds'
        in server.run(str(examples_dir / 'ml.eig'), timeout_seconds=1)['errors'][0]['message']
    )


def test_resources() -> None:
    assert server.grammar().startswith('# Eigrel')
    capabilities = json.loads(server.capabilities())
    assert capabilities['backends'] == ['python', 'spark', 'sql']
    assert capabilities['capabilities']['predict']['sql'] == 'no'


def test_server_lists_tools_and_resources() -> None:
    app = server.build_server()

    async def inspect() -> tuple[list[str], list[str]]:
        tools = [t.name for t in await app.list_tools()]
        resources = [str(r.uri) for r in await app.list_resources()]
        return tools, resources

    tools, resources = asyncio.run(inspect())
    assert tools == ['check', 'plan', 'compile', 'run']
    assert resources == ['eigrel://llms.txt', 'eigrel://capabilities']


def test_eigrel_mcp_over_stdio(examples_dir: Path) -> None:
    """A real MCP client talks to `eigrel mcp` over stdio, the way agents do."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def session() -> tuple[str, list[str], dict[str, object]]:
        params = StdioServerParameters(command=sys.executable, args=['-m', 'eigrel', 'mcp'])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            info = await client.initialize()
            tools = [t.name for t in (await client.list_tools()).tools]
            result = await client.call_tool('check', {'path': str(examples_dir / 'ml.eig')})
            return info.server_info.name, tools, json.loads(result.content[0].text)

    name, tools, report = asyncio.run(session())
    assert name == 'eigrel'
    assert tools == ['check', 'plan', 'compile', 'run']
    assert report['ok'] is True


def test_mcp_command_needs_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, 'importable', lambda module: module != 'mcp')
    assert main(['mcp']) == 1
    assert 'pip install "eigrel[mcp]"' in capsys.readouterr().err


def test_mcp_command_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    started = []
    monkeypatch.setattr(server, 'main', lambda: started.append(True))
    assert main(['mcp']) == 0
    assert started == [True]


def test_main_runs_over_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    transports = []

    class Fake:
        def run(self, transport: str) -> None:
            transports.append(transport)

    monkeypatch.setattr(server, 'build_server', lambda: Fake())
    server.main()
    assert transports == ['stdio']
