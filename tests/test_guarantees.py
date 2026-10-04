"""0.8 guarantees: time-ordered splits, predict reusing the training pipeline, decoded labels."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from eigrel.backends import python as python_backend
from eigrel.backends import spark as spark_backend
from eigrel.compiler import ParseError, SemanticError, analyze, compile_source, ir, parse
from eigrel.compiler.catalog import BACKENDS, CAPABILITIES, handles_missing_values
from eigrel.compiler.ir import format_graph
from eigrel.compiler.tokens import Location
from eigrel.planner import build_plan, render, to_json

DATA = 'dataset d from csv("d.csv")\ntransform d { select day, age, income, city, y }\n'


def test_parse_assumptions_and_predict() -> None:
    assumptions, predict = parse(
        'assumptions d { time = day }\npredict m { data = n, output = csv("s.csv") }'
    ).statements
    assert assumptions.dataset == 'd' and assumptions.params[0].name == 'time'  # type: ignore[union-attr]
    assert predict.model == 'm' and [p.name for p in predict.params] == ['data', 'output']  # type: ignore[union-attr]
    with pytest.raises(ParseError, match="expected '{'"):
        parse('assumptions d')


def test_time_and_predict_are_lowered() -> None:
    graph = compile_source(
        DATA + 'transform d { fill income = 0, city = "x" }\n'
        'assumptions d { time = day }\nmodel m = random_forest\ntrain m { target = y }\n'
        'dataset n from csv("n.csv")\ntransform n { select age, income, city }\n'
        'predict m { data = n, output = parquet("out.parquet") }'
    )
    train = next(op for op in graph.ops if isinstance(op, ir.Train))
    predict = graph.ops[-1]
    assert train.time == 'day'
    assert predict == ir.Predict(
        predict.id,
        'm',
        train.id,
        predict.id - 1,
        (('income', 0), ('city', 'x')),
        'parquet',
        'out.parquet',
    )
    assert format_graph(graph).splitlines()[-1] == (
        f'%{predict.id} = predict %{train.id} on %{predict.id - 1}'
        ' fill [income = 0, city = "x"] into parquet("out.parquet")  # m'
    )
    assert 'time=day' in format_graph(graph)


def test_fills_on_columns_that_are_not_features_are_not_replayed() -> None:
    graph = compile_source(
        DATA + 'transform d { fill income = 0, city = "x" }\nfeatures d { age, income }\n'
        'model m = random_forest\ntrain m { target = y }\n'
        'predict m { data = d, output = json("out.jsonl") }'
    )
    predict = graph.ops[-1]
    assert isinstance(predict, ir.Predict) and predict.fills == (('income', 0),)


@pytest.mark.parametrize(
    ('source', 'message', 'loc'),
    [
        ('assumptions z { time = day }', "unknown dataset 'z'", (3, 1)),
        ('assumptions d { }', 'assumptions needs a time column', (3, 1)),
        ('assumptions d { time = "day" }', 'time must be a name', (3, 24)),
        ('assumptions d { time = when }', "column 'when' does not exist here", (3, 24)),
        ('assumptions d { order = day }', "assumptions has no parameter 'order'", (3, 17)),
        (
            'assumptions d { time = day }\nassumptions d { time = age }',
            'already declared at line 3',
            (4, 1),
        ),
        (
            'model m = random_forest\npredict m { data = d, output = csv("s.csv") }',
            'must be trained before it predicts',
            (4, 1),
        ),
        (
            'model m = random_forest\ntrain m { target = y }\npredict m { data = d }',
            'predict needs data and output',
            (5, 1),
        ),
        (
            'model m = random_forest\ntrain m { target = y }\npredict m { data = d, output = "s.csv" }',
            'output must be a file',
            (5, 32),
        ),
        (
            'model m = random_forest\ntrain m { target = y }\npredict m { data = d, output = csv("") }',
            'output must be a file',
            (5, 32),
        ),
        (
            'model m = random_forest\ntrain m { target = y }\npredict m { data = x, output = csv("s.csv") }',
            "unknown dataset 'x'",
            (5, 20),
        ),
        (
            (
                'model m = random_forest\ntrain m { target = y }\ndataset n from csv("n.csv")\n'
                'transform n { select age, income }\npredict m { data = n, output = csv("s.csv") }'
            ),
            "'n' lacks features the model was trained on: day, city",
            (7, 20),
        ),
    ],
)
def test_assumption_and_predict_errors(source: str, message: str, loc: tuple[int, int]) -> None:
    with pytest.raises(SemanticError, match=re.escape(message)) as info:
        compile_source(DATA + source)
    assert info.value.loc == Location(*loc)


def test_schemas_make_time_and_skew_checks_precise() -> None:
    program = parse('dataset d from csv("d.csv")\nassumptions d { time = city }\n')
    schemas = {'d': (('city', 'string'),), 'n': (('age', 'number'),)}
    with pytest.raises(SemanticError, match="time column 'city' is text"):
        analyze(program, schemas)
    program = parse(
        'dataset d from csv("d.csv")\nmodel m = random_forest\ntrain m { target = y }\n'
        'dataset n from csv("n.csv")\npredict m { data = n, output = csv("s.csv") }'
    )
    schemas = {'d': (('age', 'number'), ('y', 'number')), 'n': (('age', 'string'),)}
    with pytest.raises(SemanticError, match="feature 'age' is a number in the training data"):
        analyze(program, schemas)


def test_capability_matrix_matches_the_docs(examples_dir: Path) -> None:
    docs = (examples_dir.parent / 'docs' / 'LANGUAGE.md').read_text(encoding='utf-8')
    section = docs[docs.index('## Backend capabilities') :]
    rows = re.findall(
        r'^\| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$', section, re.MULTILINE
    )
    table = {name: dict(zip(BACKENDS, values, strict=True)) for name, *values in rows[1:]}
    assert table == CAPABILITIES
    assert handles_missing_values('python', 'xgboost')
    assert not handles_missing_values('spark', 'random_forest')


def test_python_time_split_and_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = ['day,age,income,city,plan']
    for i in range(120):
        income = '' if i % 9 == 0 else str(1000 + i * 7)
        rows.append(
            f'2025-{1 + i // 28:02d}-{1 + i % 28:02d},{20 + i % 40},{income},{["rio", "sp"][i % 2]},{"pro" if i % 40 > 25 else "basic"}'
        )
    (tmp_path / 'leads.csv').write_text('\n'.join(rows) + '\n')
    (tmp_path / 'new.csv').write_text('age,income,city\n30,,rio\n50,9000,recife\n')
    source = (
        'dataset leads from csv("leads.csv")\ntransform leads { fill income = 0 }\n'
        'assumptions leads { time = day }\n'
        'model m = xgboost { trees = 10 }\ntrain m { target = plan }\n'
        'evaluate m { metrics = [accuracy] }\n'
        'dataset incoming from csv("new.csv")\n'
        'predict m { data = incoming, output = csv("scored.csv") }\n'
        'predict m { data = incoming, output = json("ages.jsonl") }\n'
        'predict m { data = incoming, output = parquet("scored.parquet") }\n'
    )
    code = python_backend.generate(compile_source(source))
    assert "m_rows = leads.sort_values('day', kind='stable')" in code
    assert "m_X = m_rows.drop(columns=['plan', 'day'])" in code
    assert 'shuffle=False' in code
    assert 'EncodedLabelClassifier(XGBClassifier(' in code
    monkeypatch.chdir(tmp_path)
    exec(compile(code, 'generated.py', 'exec'), {'__name__': '__main__'})  # noqa: S102
    scored = pd.read_csv(tmp_path / 'scored.csv')
    assert list(scored.columns) == ['age', 'income', 'city', 'plan_prediction', 'plan_probability']
    assert set(scored['plan_prediction']) <= {'basic', 'pro'}
    assert scored['income'].tolist() == [0.0, 9000.0]  # the training fill was replayed
    ages = [json.loads(line) for line in (tmp_path / 'ages.jsonl').read_text().splitlines()]
    assert 'plan_prediction' in ages[0]
    assert len(pd.read_parquet(tmp_path / 'scored.parquet')) == 2


def test_registered_xgboost_declares_eigrel() -> None:
    code = python_backend.generate(
        compile_source(
            'dataset d from csv("d.csv")\nmodel m = xgboost\ntrain m { target = y }\nregister m'
        )
    )
    assert "extra_pip_requirements=['eigrel==" in code


def test_runtime_classifier_decodes_labels() -> None:
    import numpy as np
    from xgboost import XGBClassifier

    from eigrel.runtime import EncodedLabelClassifier

    X = pd.DataFrame({'a': np.arange(60) % 7})
    y = np.where(X['a'] > 3, 'gold', 'bronze')
    model = EncodedLabelClassifier(XGBClassifier(n_estimators=5)).fit(X, y)
    assert list(model.classes_) == ['bronze', 'gold']
    assert set(model.predict(X)) <= {'bronze', 'gold'}
    assert model.predict_proba(X).shape == (60, 2)


def test_spark_time_split_predict_and_decoding_code() -> None:
    code = spark_backend.generate(
        compile_source(
            'dataset d from csv("d.csv")\ntransform d { fill n = 0 }\n'
            'assumptions d { time = day }\nmodel m = random_forest\ntrain m { target = y }\n'
            'dataset e from csv("e.csv")\npredict m { data = e, output = parquet("p") }\n'
            'model r = linear_regression\ntrain r { data = d, target = n }\n'
            'predict r { data = e, output = csv("r.csv") }\n'
        )
    )
    assert "F.percent_rank().over(Window.orderBy('day'))" in code
    assert "m_features = [c for c in d.columns if c not in ('y', 'day')]" in code
    assert (
        "IndexToString(inputCol='prediction', outputCol='y_prediction', labels=m_labels.labels)"
        in code
    )
    assert "m_input = e.fillna({'n': 0})" in code
    assert (
        "m_scored.select(*m_input.columns, 'y_prediction').write.mode('overwrite').parquet('p')"
        in code
    )
    assert "r_scored = r_scored.withColumn('n_prediction', F.col('prediction'))" in code
    assert ".option('header', True).csv('r.csv')" in code
    compile(code, 'generated.py', 'exec')


@pytest.mark.spark
def test_spark_time_split_and_predict_run(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    rows = ['day,age,city,plan']
    for i in range(80):
        rows.append(
            f'2025-{1 + i // 28:02d}-{1 + i % 28:02d},{20 + i % 40},{["rio", "sp"][i % 2]},{"pro" if i % 40 > 25 else "basic"}'
        )
    (tmp_path / 'leads.csv').write_text('\n'.join(rows) + '\n')
    (tmp_path / 'new.csv').write_text('age,city\n30,rio\n50,recife\n')
    program = (
        'dataset leads from csv("leads.csv")\nassumptions leads { time = day }\n'
        'model m = decision_tree\ntrain m { target = plan }\n'
        'dataset incoming from csv("new.csv")\n'
        'predict m { data = incoming, output = csv("scored") }\n'
    )
    script = tmp_path / 'run.py'
    script.write_text(spark_backend.generate(compile_source(program)))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, 'PYSPARK_PYTHON': sys.executable},
        timeout=900,
    )
    assert 'm: decision_tree classification, trained on 64 rows, validated on 16' in result.stdout
    scored = pd.concat(pd.read_csv(p) for p in (tmp_path / 'scored').glob('part-*.csv'))
    assert set(scored['plan_prediction']) <= {'basic', 'pro'} and len(scored) == 2


def test_plan_reports_time_periods_predictions_and_skew(tmp_path: Path) -> None:
    rows = ['day,age,income,city,plan']
    for i in range(100):
        income = '' if i % 9 == 0 else str(1000 + i)
        rows.append(
            f'2025-{1 + i // 28:02d}-{1 + i % 28:02d},{20 + i % 40},{income},{["rio", "sp"][i % 2]},{"pro" if i % 4 == 0 else "basic"}'
        )
    (tmp_path / 'leads.csv').write_text('\n'.join(rows) + '\n')
    (tmp_path / 'new.csv').write_text('age,income,city\n30,,rio\n50,1,recife\n')
    (tmp_path / 'p.eig').write_text(
        'dataset leads from csv("leads.csv")\nassumptions leads { time = day }\n'
        'model m = random_forest\ntrain m { target = plan }\n'
        'dataset incoming from csv("new.csv")\n'
        'predict m { data = incoming, output = csv("s.csv") }\n'
    )
    plan = build_plan(tmp_path / 'p.eig')
    assert plan.ok
    train = plan.trains[0]
    assert train.features == ('age', 'income', 'city')
    assert train.stratified is False
    assert str(train.train_period[0]) == '2025-01-01'  # type: ignore[index]
    assert str(train.validation_period[1]) == '2025-04-16'  # type: ignore[index]
    prediction = plan.predictions[0]
    assert prediction.rows == 2 and prediction.unseen == {'city': ['recife']}
    assert [f.code for f in plan.findings] == ['unseen-categories']
    text = render(plan)
    assert 'by day: train 2025-01-01 … ' in text and 'predict m on incoming: 2 rows' in text
    data = to_json(plan)
    assert data['predictions'][0]['unseen'] == {'city': ['recife']}
    assert data['trainings'][0]['time'] == 'day'

    # logistic regression cannot score the missing income of the serving rows.
    (tmp_path / 'lr.eig').write_text(
        (tmp_path / 'p.eig')
        .read_text()
        .replace('random_forest', 'logistic_regression')
        .replace(
            'assumptions leads { time = day }\n',
            'transform leads { drop_missing income }\nassumptions leads { time = day }\n',
        )
    )
    codes = [f.code for f in build_plan(tmp_path / 'lr.eig').findings]
    assert 'missing-values' in codes and 'unseen-categories' in codes
    # Implicit features are only known from the data: the plan catches serving data without them.
    (tmp_path / 'skew.eig').write_text(
        'dataset leads from csv("leads.csv")\nmodel r = linear_regression\n'
        'train r { target = age }\ndataset incoming from csv("new.csv")\n'
        'predict r { data = incoming, output = csv("s.csv") }\n'
    )
    skew = build_plan(tmp_path / 'skew.eig').findings
    assert [f.code for f in skew] == ['schema']
    assert 'lacks features the model was trained on: day, plan' in skew[0].message
    # Without the time assumption, the plan warns about a random split over dated rows.
    (tmp_path / 'r.eig').write_text(
        'dataset leads from csv("leads.csv")\nmodel m = random_forest\ntrain m { target = plan }\n'
    )
    assert [f.code for f in build_plan(tmp_path / 'r.eig').findings] == ['random-split']


def test_plan_time_problems(tmp_path: Path) -> None:
    (tmp_path / 'd.csv').write_text(
        'day,x,y\n' + ''.join(f'{"" if i == 5 else i // 10},{i},{i % 2}\n' for i in range(50))
    )
    (tmp_path / 'p.eig').write_text(
        'dataset d from csv("d.csv")\nassumptions d { time = day }\n'
        'model m = random_forest\ntrain m { target = y }\n'
    )
    assert [f.code for f in build_plan(tmp_path / 'p.eig').findings] == ['time-missing']
    (tmp_path / 'p.eig').write_text(
        'dataset d from csv("d.csv")\ntransform d { drop_missing day }\n'
        'assumptions d { time = day }\nmodel m = random_forest\n'
        'train m { target = y, validation = 0.25 }\n'
    )
    assert [f.code for f in build_plan(tmp_path / 'p.eig').findings] == ['time-overlap']


def test_new_customers_are_deterministic(tmp_path: Path, examples_dir: Path) -> None:
    from eigrel.starter import write_new_customers

    write_new_customers(tmp_path / 'new.csv')
    assert (tmp_path / 'new.csv').read_text() == (
        examples_dir / 'data' / 'new_customers.csv'
    ).read_text()


def test_plan_edge_cases_for_time_and_predict(
    tmp_path: Path, examples_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eigrel import planner
    from eigrel.probe import ProbeError

    # One row: nothing is left to train on, so no periods are computed.
    (tmp_path / 'one.csv').write_text('day,x,y\n1,1,1\n')
    (tmp_path / 'one.eig').write_text(
        'dataset d from csv("one.csv")\nassumptions d { time = day }\n'
        'model m = random_forest\ntrain m { target = y }\n'
    )
    assert build_plan(tmp_path / 'one.eig').trains[0].train_period is None

    # Predicting on BigQuery data is planned without probing; empty serving data is an error.
    (tmp_path / 'd.csv').write_text('x,y\n' + ''.join(f'{i},{i % 2}\n' for i in range(60)))
    (tmp_path / 'p.eig').write_text(
        'dataset d from csv("d.csv")\nmodel m = random_forest\ntrain m { target = y }\n'
        'dataset bq from bigquery("my-project.s.t")\n'
        'predict m { data = bq, output = csv("a.csv") }\n'
        'dataset none from csv("d.csv")\ntransform none { filter x > 1000 }\n'
        'predict m { data = none, output = csv("b.csv") }\n'
    )
    plan = build_plan(tmp_path / 'p.eig')
    assert [f.code for f in plan.findings] == ['not-probed', 'empty', 'empty']
    assert plan.predictions[0].rows is None

    # The serving example replays the training fill, and says so.
    text = render(build_plan(examples_dir / 'serving.eig'))
    assert 'replays training fills: income = 0' in text

    # Probe failures while planning a prediction become findings.
    calls = {'n': 0}
    real_count = planner.count

    def flaky(engine: object, query: str) -> int:
        calls['n'] += 1
        if calls['n'] > 2:
            raise ProbeError('connection lost')
        return real_count(engine, query)  # type: ignore[arg-type]

    monkeypatch.setattr(planner, 'count', flaky)
    (tmp_path / 'q.eig').write_text(
        'dataset d from csv("d.csv")\nmodel m = random_forest\ntrain m { target = y }\n'
        'predict m { data = d, output = csv("c.csv") }\n'
    )
    assert [f.code for f in build_plan(tmp_path / 'q.eig').findings][-1] == 'probe'
