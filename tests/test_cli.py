import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from eigrel import __version__
from eigrel.cli import main

VALID = 'dataset d from csv("d.csv")\nmodel m = random_forest'


def write(tmp_path: Path, source: str, name: str = 'main.eig') -> str:
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_check_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, VALID)
    assert main(['check', path]) == 0
    assert capsys.readouterr().out == f'ok: {path} (1 operation)\n'


def test_check_reports_errors_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = write(tmp_path, 'model m random_forest', 'bad.eig')
    unknown = write(tmp_path, 'model m = rf', 'unknown.eig')
    good = write(tmp_path, VALID, 'good.eig')
    assert main(['check', bad, unknown, good]) == 1
    out, err = capsys.readouterr()
    assert f'ok: {good}' in out
    assert f'{bad}:1:9' in err
    assert "unknown algorithm 'rf'" in err


def test_check_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['check', str(tmp_path / 'nope.eig')]) == 1
    assert 'cannot read' in capsys.readouterr().err


def test_ast(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['ast', write(tmp_path, VALID)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['statements'][1]['node'] == 'ModelDecl'


def test_ast_does_not_need_valid_semantics(tmp_path: Path) -> None:
    assert main(['ast', write(tmp_path, 'model m = rf')]) == 0
    assert main(['ast', write(tmp_path, 'oops')]) == 1
    assert main(['ast', str(tmp_path / 'missing.eig')]) == 1


def test_tokens(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['tokens', write(tmp_path, 'model m')]) == 0
    assert capsys.readouterr().out.splitlines() == [
        '1:1\tKEYWORD\tmodel',
        '1:7\tIDENT\tm',
        '1:8\tEOF\t',
    ]


def test_tokens_with_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['tokens', write(tmp_path, '"open')]) == 1
    assert 'unterminated string' in capsys.readouterr().err
    assert main(['tokens', str(tmp_path / 'missing.eig')]) == 1


def test_ir(examples_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['ir', str(examples_dir / 'ml.eig')]) == 0
    out = capsys.readouterr().out
    assert out.startswith('%0 = load csv("data/customers.csv")  # customers\n')
    assert '%4 = evaluate %3 [accuracy, precision, recall, f1]  # churn' in out


def test_ir_with_error(tmp_path: Path) -> None:
    assert main(['ir', write(tmp_path, 'train m { target = y }')]) == 1


def test_compile_to_stdout_and_file(
    examples_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(['compile', str(examples_dir / 'ml.eig')]) == 0
    code = capsys.readouterr().out
    assert 'RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)' in code

    output = tmp_path / 'ml.py'
    assert main(['compile', str(examples_dir / 'ml.eig'), '-o', str(output)]) == 0
    assert output.read_text() == code
    assert capsys.readouterr().out == f'wrote {output}\n'


def test_compile_to_sql(examples_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['compile', '--target', 'sql', str(examples_dir / 'pipeline.eig')]) == 0
    out = capsys.readouterr().out
    assert '-- dataset users (bigquery, bigquery dialect)' in out
    assert 'FROM `project.dataset.users` WHERE ((`age` > 18) AND (`income` <> 0));' in out


def test_run_names_the_missing_extra(
    examples_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # google-cloud-bigquery is not a test dependency, so the BigQuery example cannot run.
    assert main(['run', str(examples_dir / 'pipeline.eig')]) == 1
    err = capsys.readouterr().err
    assert 'running needs google.cloud.bigquery, db_dtypes' in err
    assert 'pip install "eigrel[bigquery]"' in err


def test_compile_with_semantic_error(tmp_path: Path) -> None:
    assert main(['compile', write(tmp_path, 'model m = rf')]) == 1


def test_run_trains_and_evaluates(examples_dir: Path, capfd: pytest.CaptureFixture[str]) -> None:
    assert main(['run', str(examples_dir / 'ml.eig')]) == 0
    out = capfd.readouterr().out
    assert 'churn: random_forest classification, trained on 315 rows, validated on 79' in out
    assert '  accuracy ' in out


def test_run_reports_compile_errors(tmp_path: Path) -> None:
    assert main(['run', write(tmp_path, 'model m = rf')]) == 1


def test_run_reads_a_sql_database(examples_dir: Path, capfd: pytest.CaptureFixture[str]) -> None:
    assert main(['run', str(examples_dir / 'sql.eig')]) == 0
    assert (
        'churn: logistic_regression classification, trained on 300 rows' in capfd.readouterr().out
    )


def test_compile_to_spark(examples_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['compile', '-t', 'spark', str(examples_dir / 'ml.eig')]) == 0
    assert 'from pyspark.ml.classification import RandomForestClassifier' in capsys.readouterr().out


def test_run_on_spark_needs_pyspark_and_java(
    examples_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from eigrel import runner

    monkeypatch.setattr(runner, 'importable', lambda module: module != 'pyspark')
    assert main(['run', '-t', 'spark', str(examples_dir / 'ml.eig')]) == 1
    assert 'pip install "eigrel[spark]"' in capsys.readouterr().err

    monkeypatch.setattr(runner, 'importable', lambda module: True)
    monkeypatch.setattr(runner, 'has_java', lambda: False)
    assert main(['run', '-t', 'spark', str(examples_dir / 'ml.eig')]) == 1
    assert 'needs Java 17 or newer' in capsys.readouterr().err


def test_has_java(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    from eigrel import runner

    monkeypatch.setenv('JAVA_HOME', str(tmp_path / 'missing'))
    assert runner.has_java() is False  # the binary does not exist

    monkeypatch.delenv('JAVA_HOME')
    monkeypatch.setattr(runner.shutil, 'which', lambda name: None)
    assert runner.has_java() is False

    monkeypatch.setattr(runner.shutil, 'which', lambda name: '/usr/bin/java')
    monkeypatch.setattr(
        runner.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, returncode=0)
    )
    assert runner.has_java() is True


def test_compile_reports_unsupported_sql(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, 'dataset d from sql("sqlite://", "t")\ntransform d { drop_missing }')
    assert main(['compile', '-t', 'sql', path]) == 1
    assert 'error: SQL needs the column names for drop_missing' in capsys.readouterr().err


def test_run_names_the_xgboost_and_mlflow_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from eigrel import runner

    monkeypatch.setattr(runner, 'importable', lambda module: module not in ('xgboost', 'mlflow'))
    path = write(
        tmp_path,
        'dataset d from csv("d.csv")\nmodel m = xgboost\ntrain m { target = y }\nregister m',
    )
    assert main(['run', path]) == 1
    assert 'pip install "eigrel[mlflow,xgboost]"' in capsys.readouterr().err


def test_run_without_runtime_dependencies(
    examples_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str) -> object:
        return None if name == 'sklearn' else real_find_spec(name)

    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)
    assert main(['run', str(examples_dir / 'ml.eig')]) == 1
    assert 'running needs sklearn; install them with: pip install "eigrel[python]"' in (
        capsys.readouterr().err
    )


def test_init_creates_a_project_that_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / 'churn'
    assert main(['init', str(project)]) == 0
    assert (project / 'data' / 'customers.csv').is_file()
    assert main(['check', str(project / 'main.eig')]) == 0
    assert main(['init', str(project)]) == 1
    assert 'already exists' in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(['--version'])
    assert info.value.code == 0
    assert capsys.readouterr().out == f'eigrel {__version__}\n'


def test_python_dash_m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('sys.argv', ['eigrel', 'check', write(tmp_path, VALID)])
    with pytest.raises(SystemExit) as info:
        runpy.run_module('eigrel', run_name='__main__')
    assert info.value.code == 0


def test_check_json_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write(tmp_path, VALID)
    assert main(['check', '--json', path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        'eigrel': __version__,
        'ok': True,
        'files': [{'path': path, 'ok': True, 'operations': 1, 'errors': []}],
    }


def test_check_json_reports_positioned_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = write(tmp_path, VALID, 'good.eig')
    bad = write(tmp_path, 'model m = rf', 'bad.eig')
    broken = write(tmp_path, 'model m random_forest', 'broken.eig')
    assert main(['check', '--json', bad, broken, good]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data['eigrel'] == __version__
    assert data['ok'] is False
    semantic, parse, ok = data['files']
    assert semantic['ok'] is False
    assert semantic['errors'][0]['stage'] == 'semantic'
    assert semantic['errors'][0]['message'].startswith("unknown algorithm 'rf'")
    assert semantic['errors'][0]['line'] == 1
    assert semantic['errors'][0]['column'] == 1
    assert parse['ok'] is False
    assert parse['errors'][0]['stage'] == 'parse'
    assert parse['errors'][0]['line'] == 1
    assert parse['errors'][0]['column'] == 9
    assert ok == {'path': good, 'ok': True, 'operations': 1, 'errors': []}


def test_check_json_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = str(tmp_path / 'nope.eig')
    assert main(['check', '--json', missing]) == 1
    data = json.loads(capsys.readouterr().out)
    report = data['files'][0]
    assert report['ok'] is False
    assert report['errors'][0]['stage'] == 'io'
    assert report['errors'][0]['message'].startswith(f'cannot read {missing}')
