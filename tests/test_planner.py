import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from eigrel.cli import main
from eigrel.planner import build_plan, render, save_state, to_json
from eigrel.probe import ProbeError, duckdb_type, python_type


def leads(tmp_path: Path, rows: int = 200) -> Path:
    lines = ['id,age,income,city,plan']
    for i in range(rows):
        age = '' if i % 23 == 0 else str(18 + i % 50)
        income = '' if i % 9 == 0 else str(1000 + i * 13)
        city = '' if i % 7 == 0 else ['rio', 'sp', 'bh'][i % 3]
        plan = 'enterprise' if i % 40 == 0 else ('pro' if i % 3 == 0 else 'basic')
        lines.append(f'{i},{age},{income},{city},{plan}')
    path = tmp_path / 'leads.csv'
    path.write_text('\n'.join(lines) + '\n')
    return path


def program(tmp_path: Path, body: str, name: str = 'p.eig') -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


PIPELINE = (
    'dataset leads from csv("leads.csv")\n'
    'transform leads {{ filter age >= 18 }}\n'
    'model m = {algorithm}\n'
    'train m {{ target = plan }}\n'
)


def codes(plan_findings: list) -> list[str]:  # type: ignore[type-arg]
    return [f.code for f in plan_findings]


def test_plan_of_the_example_matches_the_run(examples_dir: Path) -> None:
    plan = build_plan(examples_dir / 'ml.eig')
    assert plan.ok and not plan.findings
    dataset = plan.datasets[0]
    assert [s.rows for s in dataset.steps] == [400, 394, 394]
    train = plan.trains[0]
    assert (train.rows, train.train_rows, train.validation_rows) == (394, 315, 79)
    assert train.stratified is True
    assert train.classes == [(0, 209), (1, 185)]
    text = render(plan)
    assert 'filter (age >= 18)' in text and '400 → 394 rows (-6)' in text
    assert '✓ no problems found' in text


def test_missing_values_imbalance_and_small_validation(tmp_path: Path) -> None:
    leads(tmp_path)
    plan = build_plan(program(tmp_path, PIPELINE.format(algorithm='logistic_regression')))
    assert codes(plan.findings) == [
        'missing-values',
        'id-feature',
        'imbalance',
        'small-validation',
    ]
    assert not plan.ok
    error = plan.findings[0]
    assert error.loc is not None and error.loc.line == 4
    assert 'income (22)' in error.message
    assert plan.trains[0].missing == {'id': 0, 'age': 0, 'income': 22, 'city': 27, 'plan': 0}

    # scikit-learn trees handle missing values, so only the warnings remain.
    plan = build_plan(program(tmp_path, PIPELINE.format(algorithm='random_forest')))
    assert plan.ok and codes(plan.findings) == ['id-feature', 'imbalance', 'small-validation']

    # On Spark, VectorAssembler rejects nulls for every algorithm.
    plan = build_plan(program(tmp_path, PIPELINE.format(algorithm='random_forest')), 'spark')
    assert codes(plan.findings)[0] == 'missing-values'
    assert plan.trains[0].stratified is False


def test_target_problems(tmp_path: Path) -> None:
    (tmp_path / 'd.csv').write_text(
        'x,y,label,score\n'
        + ''.join(
            f'{i},{"" if i % 10 == 0 else i * 2},{"a" if i < 20 else "b"},{i * 1.5}\n'
            for i in range(40)
        )
    )
    plan = build_plan(
        program(
            tmp_path,
            'dataset d from csv("d.csv")\n'
            'model a = random_forest\ntrain a { target = y }\n'
            'model b = random_forest { task = regression }\ntrain b { data = d, target = label }\n'
            'model c = decision_tree\ntrain c { data = d, target = score }\n',
        )
    )
    assert codes(plan.findings) == [
        'target-missing',
        'continuous-target',
        'text-target',
        'continuous-target',
    ]


def test_one_class_tiny_and_no_stratify(tmp_path: Path) -> None:
    (tmp_path / 'one.csv').write_text('x,y\n' + ''.join(f'{i},1\n' for i in range(10)))
    (tmp_path / 'rare.csv').write_text('x,y\n' + ''.join(f'{i},{int(i == 0)}\n' for i in range(40)))
    plan = build_plan(
        program(
            tmp_path,
            'dataset one from csv("one.csv")\nmodel a = random_forest\ntrain a { target = y }\n'
            'dataset rare from csv("rare.csv")\nmodel b = random_forest\n'
            'train b { data = rare, target = y }\n',
        )
    )
    assert codes(plan.findings) == [
        'tiny',
        'one-class',
        'no-stratify',
        'imbalance',
        'small-validation',
    ]
    assert 'less than one row(s)' in plan.findings[-1].message


def test_empty_dataset_and_gbt_on_spark(tmp_path: Path) -> None:
    leads(tmp_path)
    plan = build_plan(
        program(
            tmp_path,
            'dataset leads from csv("leads.csv")\ntransform leads { filter age > 1000 }\n',
        )
    )
    assert codes(plan.findings) == ['empty']
    empty = program(tmp_path, 'dataset e from csv("empty.csv")\n', 'empty.eig')
    (tmp_path / 'empty.csv').write_text('a,b\n')
    assert codes(build_plan(empty).findings) == ['empty']
    gbt = program(
        tmp_path,
        'dataset leads from csv("leads.csv")\ntransform leads { drop_missing\n select age, plan }\n'
        'model m = gradient_boosting\ntrain m { target = plan }\n',
        'gbt.eig',
    )
    assert 'capability' in codes(build_plan(gbt, 'spark').findings)


def test_schema_contract_errors(tmp_path: Path) -> None:
    leads(tmp_path)
    for body, message in [
        ('transform leads { filter age >= "18" }', 'cannot compare a number with a string'),
        ('transform leads { select age, incme }', "column 'incme' does not exist here"),
        ('transform leads { fill income = "none" }', "cannot fill the number column 'income'"),
        ('transform leads { fill city = 0 }', "cannot fill the string column 'city'"),
    ]:
        plan = build_plan(program(tmp_path, f'dataset leads from csv("leads.csv")\n{body}\n'))
        assert codes(plan.findings) == ['schema'], body
        assert plan.findings[0].message.startswith(message)
        assert plan.findings[0].loc is not None and plan.findings[0].loc.line == 2


def test_syntax_and_semantic_errors_stop_the_plan(tmp_path: Path) -> None:
    assert codes(build_plan(program(tmp_path, 'model m rf')).findings) == ['parse']
    assert codes(build_plan(program(tmp_path, 'model m = rf')).findings) == ['semantic']
    assert codes(build_plan(tmp_path / 'missing.eig').findings) == ['io']


def test_sources_that_cannot_be_probed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NOPE_URL', raising=False)
    plan = build_plan(
        program(
            tmp_path,
            'dataset a from sql(env("NOPE_URL"), "t")\ndataset b from csv("missing.csv")\n'
            'dataset c from bigquery("my-project.s.t")\n'
            'dataset d from sql("sqlite:///shop.db", "nope")\n'
            'dataset e from sql("nosuchdriver://x", "t")\n',
        )
    )
    assert [(f.severity, f.code, f.loc.line if f.loc else None) for f in plan.findings] == [
        ('error', 'source', 1),
        ('error', 'source', 2),
        ('info', 'not-probed', 3),
        ('error', 'source', 4),
        ('error', 'source', 5),
    ]


def test_sql_database_source(tmp_path: Path, examples_dir: Path) -> None:
    import shutil

    shutil.copy(examples_dir / 'data' / 'customers.db', tmp_path / 'customers.db')
    with closing(sqlite3.connect(tmp_path / 'customers.db')) as connection, connection:
        connection.execute(
            'CREATE TABLE tags (id INTEGER, tag TEXT, ok BOOLEAN, at DATETIME, misc)'
        )
        connection.execute("INSERT INTO tags VALUES (1, 'a', 1, '2026-01-01', NULL)")
    plan = build_plan(
        program(
            tmp_path,
            'dataset c from sql("sqlite:///customers.db", "customers")\n'
            'transform c { fill income = 0\n select age, income, churned\n drop_missing }\n'
            'model m = random_forest\ntrain m { target = churned }\n'
            'dataset t from sql("sqlite:///customers.db", "tags")\n',
        )
    )
    assert plan.ok, plan.findings
    assert [s.rows for s in plan.datasets[0].steps] == [400, 400, 400, 400]
    assert plan.trains[0].classes == [(0, 211), (1, 189)]
    assert [(c.name, c.type) for c in plan.datasets[1].columns or ()] == [
        ('id', 'number'),
        ('tag', 'string'),
        ('ok', 'bool'),
        ('at', 'unknown'),
        ('misc', 'unknown'),
    ]


def test_known_schema_lets_sqlite_probe_fills(tmp_path: Path, examples_dir: Path) -> None:
    """SQLite cannot fill columns behind SELECT *, but the plan knows the columns."""
    import shutil

    shutil.copy(examples_dir / 'data' / 'customers.db', tmp_path / 'customers.db')
    plan = build_plan(
        program(
            tmp_path,
            'dataset c from sql("sqlite:///customers.db", "customers")\n'
            'transform c { fill income = 0\n drop_missing\n select age, income, churned }\n'
            'model m = random_forest\ntrain m { target = churned }\n',
        )
    )
    assert plan.ok and not plan.findings
    assert [s.rows for s in plan.datasets[0].steps] == [400, 400, 400, 400]


def test_state_and_drift(tmp_path: Path) -> None:
    path = leads(tmp_path)
    source = program(tmp_path, 'dataset leads from csv("leads.csv")\n')
    plan = build_plan(source)
    assert not plan.state_exists
    state = save_state(plan)
    saved = json.loads(state.read_text())
    assert saved['datasets']['leads']['rows'] == 200
    assert len(saved['datasets']['leads']['fingerprint']) == 16
    assert build_plan(source).findings == []

    text = path.read_text().replace('id,age,income,city,plan', 'id,age,revenue,city,plan')
    path.write_text(
        text.replace('\n1,', '\n1.5,') + ''.join(f'{i},30,1,sp,basic\n' for i in range(300))
    )
    plan = build_plan(source)
    assert plan.state_exists
    assert codes(plan.findings) == ['drift-removed', 'drift-type', 'drift-added', 'drift-rows']
    assert plan.findings[3].severity == 'warning'

    source.write_text('dataset leads from csv("leads.csv")\ndataset again from csv("leads.csv")\n')
    assert 'drift-new' in codes(build_plan(source).findings)
    state.write_text('not json')
    assert codes(build_plan(source).findings) == ['state']


def test_json_output_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    leads(tmp_path)
    good = program(tmp_path, PIPELINE.format(algorithm='random_forest'), 'good.eig')
    assert main(['plan', str(good)]) == 0
    assert main(['plan', '--strict', str(good)]) == 2
    capsys.readouterr()
    assert main(['plan', '--json', str(good)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['ok'] is True and data['backend'] == 'python'
    assert data['datasets'][0]['steps'][1] == {
        'op': 'filter (age >= 18)',
        'rows': 191,
        'columns': ['id', 'age', 'income', 'city', 'plan'],
        'line': 2,
    }
    training = data['trainings'][0]
    assert training['classes'][0] == {'value': 'basic', 'rows': 124}
    assert {f['code'] for f in data['findings']} == {'id-feature', 'imbalance', 'small-validation'}
    assert data['findings'][0]['line'] == 4

    bad = program(tmp_path, PIPELINE.format(algorithm='logistic_regression'), 'bad.eig')
    assert main(['plan', '--save', str(bad)]) == 1
    assert not (tmp_path / 'bad.eigstate').exists()
    assert main(['plan', '--save', str(good)]) == 0
    assert 'saved' in capsys.readouterr().out
    assert main(['plan', '--save', '--json', str(good)]) == 0


def test_type_mappings() -> None:
    assert duckdb_type('DECIMAL(10,2)') == 'number'
    assert duckdb_type('VARCHAR') == 'string'
    assert duckdb_type('BOOLEAN') == 'bool'
    assert duckdb_type('TIMESTAMP') == 'unknown'
    from decimal import Decimal

    assert python_type(Decimal) == 'number'
    assert python_type(None) == 'unknown'
    assert issubclass(ProbeError, Exception)


def test_bigquery_datasets_are_planned_without_probing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    leads(tmp_path)
    source = program(
        tmp_path,
        'dataset users from bigquery("my-project.s.users")\n'
        'transform users { filter age > 18 }\n'
        'model m = random_forest\ntrain m { target = plan }\n'
        'dataset leads from csv("leads.csv")\n',
    )
    plan = build_plan(source)
    assert plan.ok and codes(plan.findings) == ['not-probed']
    assert plan.datasets[0].steps[1].rows is None
    assert plan.trains[0].rows is None and plan.trains[0].features is None
    assert to_json(plan)['datasets'][0]['columns'] is None
    assert 'filter (age > 18)' in render(plan)
    state = json.loads(save_state(plan).read_text())
    assert list(state['datasets']) == ['leads']


def test_database_url_from_the_environment(
    tmp_path: Path, examples_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    shutil.copy(examples_dir / 'data' / 'customers.db', tmp_path / 'customers.db')
    monkeypatch.setenv('SHOP_URL', 'sqlite:///customers.db')
    plan = build_plan(program(tmp_path, 'dataset c from sql(env("SHOP_URL"), "customers")\n'))
    assert plan.ok and plan.datasets[0].steps[0].rows == 400


def test_probe_failures_become_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eigrel import planner

    leads(tmp_path)
    source = program(tmp_path, PIPELINE.format(algorithm='random_forest'))

    def broken(engine: object, query: str) -> int:
        raise ProbeError('connection lost')

    monkeypatch.setattr(planner, 'count', broken)
    plan = build_plan(source)
    assert [(f.code, f.loc.line if f.loc else None) for f in plan.findings] == [
        ('probe', 1),
        ('probe', 2),
        ('probe', 4),
    ]
    assert plan.findings[0].message == 'leads: connection lost'


def test_engines_report_their_own_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from eigrel import probe

    (tmp_path / 'bad.parquet').write_text('not parquet')
    plan = build_plan(program(tmp_path, 'dataset d from parquet("bad.parquet")\n'))
    assert codes(plan.findings) == ['source']

    engine = probe.DatabaseEngine('sqlite://', tmp_path)
    with pytest.raises(ProbeError):
        engine.rows('SELECT FROM nowhere')

    class Empty:
        def get_columns(self, table: str, schema: str | None = None) -> list[object]:
            return []

    monkeypatch.setattr('sqlalchemy.inspect', lambda engine: Empty())
    from eigrel.compiler import ir

    load = ir.Load(0, 'd', 'sql', ('sqlite://', 'gone'))
    with pytest.raises(ProbeError, match='table gone does not exist'):
        engine.columns(load)

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        probe.importlib.util,
        'find_spec',
        lambda name: None if name in ('duckdb', 'sqlalchemy') else real_find_spec(name),
    )
    with pytest.raises(ProbeError, match=r'eigrel\[python\]'):
        probe.DuckDBEngine(tmp_path)
    with pytest.raises(ProbeError, match=r'eigrel\[sql\]'):
        probe.DatabaseEngine('sqlite://', tmp_path)


def test_database_types_without_a_python_type_are_unknown() -> None:
    from eigrel.probe import _python_type

    class Opaque:
        @property
        def python_type(self) -> type:
            raise NotImplementedError

    assert _python_type(Opaque()) is None
    assert python_type(_python_type(Opaque())) == 'unknown'


def test_ascii_fallback_for_limited_encodings(examples_dir: Path) -> None:
    from eigrel.planner import render_for

    plan = build_plan(examples_dir / 'ml.eig')
    assert '→' in render_for(plan, 'utf-8')
    text = render_for(plan, 'cp1252')
    assert '400 -> 394 rows' in text and 'ok no problems found' in text
    text.encode('cp1252')
    assert render_for(plan, 'no-such-codec').isascii()


def test_features_that_cannot_help(tmp_path: Path) -> None:
    rows = [f'{i},{i % 5},{i % 2},7,{"a" if i % 3 else "b"},{i % 2}' for i in range(60)]
    (tmp_path / 'f.csv').write_text('customer_id,age,flag,const,name,churned\n' + '\n'.join(rows))
    plan = build_plan(
        program(
            tmp_path,
            'dataset f from csv("f.csv")\n'
            'features f { customer_id\n age\n flag\n const\n name }\n'
            'model m = random_forest\ntrain m { target = churned }\n',
        )
    )
    found = {(f.code, f.message.split("'")[1]) for f in plan.findings}
    assert found == {
        ('id-feature', 'customer_id'),
        ('constant-feature', 'const'),
        ('duplicate-of-target', 'flag'),
    }
    assert not plan.ok
    assert plan.trains[0].distinct_values['const'] == 1


def test_unique_values_are_only_identifiers_when_named_or_typed_like_one(tmp_path: Path) -> None:
    (tmp_path / 'u.csv').write_text(
        'income,email,y\n' + ''.join(f'{1000 + i * 7},u{i}@x.com,{i % 2}\n' for i in range(60))
    )
    plan = build_plan(
        program(
            tmp_path,
            'dataset u from csv("u.csv")\nmodel m = random_forest\ntrain m { target = y }\n',
        )
    )
    # A unique number can be a legitimate measurement; a unique text column never is.
    assert [(f.code, f.message.split("'")[1]) for f in plan.findings] == [('id-feature', 'email')]
