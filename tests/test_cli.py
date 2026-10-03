import json
import runpy
from pathlib import Path

import pytest

from eigrel import __version__
from eigrel.cli import main


def write(tmp_path: Path, source: str, name: str = 'main.eig') -> str:
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_check_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, 'dataset d from csv("d.csv")\nmodel m = rf')
    assert main(['check', path]) == 0
    assert capsys.readouterr().out == f'ok: {path} (2 statements)\n'


def test_check_reports_errors_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = write(tmp_path, 'model m rf', 'bad.eig')
    good = write(tmp_path, 'model m = rf', 'good.eig')
    assert main(['check', bad, good]) == 1
    out, err = capsys.readouterr()
    assert f'ok: {good}' in out
    assert f'{bad}:1:9' in err


def test_check_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['check', str(tmp_path / 'nope.eig')]) == 1
    assert 'cannot read' in capsys.readouterr().err


def test_ast(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, 'model m = rf')
    assert main(['ast', path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['statements'][0]['node'] == 'ModelDecl'


def test_ast_with_error(tmp_path: Path) -> None:
    assert main(['ast', write(tmp_path, 'oops')]) == 1


def test_tokens(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, 'model m')
    assert main(['tokens', path]) == 0
    assert capsys.readouterr().out.splitlines() == [
        '1:1\tKEYWORD\tmodel',
        '1:7\tIDENT\tm',
        '1:8\tEOF\t',
    ]


def test_tokens_with_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['tokens', write(tmp_path, '"open')]) == 1
    assert 'unterminated string' in capsys.readouterr().err
    assert main(['tokens', str(tmp_path / 'missing.eig')]) == 1


def test_init_creates_a_project_that_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / 'churn'
    assert main(['init', str(project)]) == 0
    assert (project / 'data').is_dir()
    assert main(['check', str(project / 'main.eig')]) == 0
    assert main(['init', str(project)]) == 1
    assert 'already exists' in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(['--version'])
    assert info.value.code == 0
    assert capsys.readouterr().out == f'eigrel {__version__}\n'


def test_python_dash_m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('sys.argv', ['eigrel', 'check', write(tmp_path, 'model m = rf')])
    with pytest.raises(SystemExit) as info:
        runpy.run_module('eigrel', run_name='__main__')
    assert info.value.code == 0
