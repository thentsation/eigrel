import csv
from pathlib import Path

import pytest

from eigrel.backends.python import Requirement, generate, runtime_requirements
from eigrel.compiler import compile_source


def run(source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Compile Eigrel to Python and execute it inside tmp_path, returning the code."""
    code = generate(compile_source(source), 'test.eig')
    monkeypatch.chdir(tmp_path)
    # Executing the generated program is the point of these tests.
    exec(compile(code, 'generated.py', 'exec'), {'__name__': '__main__'})  # noqa: S102
    return code


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """Rows whose label and value are learnable from x, plus a text column."""
    path = tmp_path / 'data.csv'
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'z', 'color', 'binary', 'kind', 'value'])
        for i in range(120):
            x = i % 30
            writer.writerow(
                [
                    x,
                    (i * 7) % 11,
                    ['red', 'blue'][i % 2],
                    int(x > 14),
                    ['a', 'b', 'c'][x // 10],
                    x * 2.5,
                ]
            )
    return path


def test_binary_classification_with_all_metrics(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset d from csv("data.csv")\n'
        'transform d { filter x >= -1 and not (color == "green") and true }\n'
        'features d { x, z, color }\n'
        'model m = gradient_boosting { trees = 20, learning_rate = 0.1, max_depth = 2 }\n'
        'train m { target = binary, seed = 1 }\n'
        'evaluate m { metrics = [accuracy, precision, recall, f1, auc] }\n',
        tmp_path,
        monkeypatch,
    )
    out = capsys.readouterr().out
    assert 'm: gradient_boosting classification, trained on 96 rows, validated on 24' in out
    assert '  accuracy   1.0000' in out
    assert '  auc        1.0000' in out
    assert "d = d[(((d['x'] >= -(1)) & ~((d['color'] == 'green'))) & True)]" in code
    assert "OneHotEncoder(handle_unknown='ignore', sparse_output=False)" in code


def test_multiclass_logistic_regression(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run(
        'dataset d from csv("data.csv")\n'
        'transform d { select x, kind }\n'
        'model m = logistic_regression { max_iter = 2000, c = 10 }\n'
        'train m { target = kind }\n'
        'evaluate m { metrics = [precision, auc] }\n',
        tmp_path,
        monkeypatch,
    )
    out = capsys.readouterr().out
    assert 'm: logistic_regression classification' in out
    assert '  precision ' in out and '  auc ' in out


def test_regression_without_features_block(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset d from csv("data.csv")\n'
        'transform d { select x, value }\n'
        'model m = linear_regression\n'
        'train m { target = value, validation = 0.25 }\n'
        'evaluate m { metrics = [mae, mse, rmse, r2] }\n'
        'model t = decision_tree { task = regression, max_depth = 3, min_samples_leaf = 2 }\n'
        'train t { target = value }\n'
        'evaluate t {}\n',
        tmp_path,
        monkeypatch,
    )
    out = capsys.readouterr().out
    assert 'm: linear_regression regression, trained on 90 rows, validated on 30' in out
    assert '  r2    1.0000' in out
    assert "m_X = d.drop(columns=['value'])" in code
    assert 'LinearRegression()' in code
    assert 'stratify=None' in code
    assert 'DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)' in code


def test_python_keywords_and_reserved_names_are_renamed(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset pd from csv("data.csv")\n'
        'model class = random_forest { trees = 5, min_samples_leaf = 1 }\n'
        'train class { target = binary }\n',
        tmp_path,
        monkeypatch,
    )
    assert "pd_ = pd.read_csv('data.csv')" in code
    assert 'class_ = Pipeline([' in code
    assert 'class: random_forest classification' in capsys.readouterr().out


def test_names_never_clash_with_helpers_or_imports(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset m_X from csv("data.csv")\n'
        'dataset accuracy_score from csv("data.csv")\n'
        'model m = random_forest { trees = 5 }\n'
        'train m { data = m_X, target = binary }\n'
        'model m_ = decision_tree\n'
        'train m_ { data = accuracy_score, target = binary }\n'
        'evaluate m { metrics = [accuracy] }\n'
        'evaluate m_ { metrics = [accuracy] }\n',
        tmp_path,
        monkeypatch,
    )
    assert "m_X = pd.read_csv('data.csv')" in code
    assert "accuracy_score_ = pd.read_csv('data.csv')" in code
    assert 'm_ = Pipeline([' in code
    assert 'm__ = Pipeline([' in code
    assert capsys.readouterr().out.count('  accuracy  ') == 2


def test_file_sources(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pandas as pd

    frame = pd.read_csv(data)
    frame.to_parquet(tmp_path / 'data.parquet')
    frame.to_json(tmp_path / 'data.json', orient='records')
    frame.to_json(tmp_path / 'data.jsonl', orient='records', lines=True)
    code = run(
        'dataset p from parquet("data.parquet")\n'
        'dataset j from json("data.json")\n'
        'dataset l from json("data.jsonl")\n'
        'model m = random_forest { trees = 5 }\n'
        'train m { data = l, target = binary }\n'
        'evaluate m { metrics = [accuracy] }\n',
        tmp_path,
        monkeypatch,
    )
    assert "p = pd.read_parquet('data.parquet')" in code
    assert "j = pd.read_json('data.json')" in code
    assert "l = pd.read_json('data.jsonl', lines=True)" in code
    assert 'trained on 96 rows' in capsys.readouterr().out


def test_sql_source_with_env_url(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlite3
    from contextlib import closing

    import pandas as pd

    with closing(sqlite3.connect(tmp_path / 'shop.db')) as connection:
        pd.read_csv(data).to_sql('rows', connection, index=False)
    monkeypatch.setenv('SHOP_DB', 'sqlite:///shop.db')
    code = run(
        'dataset d from sql(env("SHOP_DB"), "rows")\n'
        'transform d { filter x > 5 }\n'
        'model m = decision_tree\n'
        'train m { target = binary }\n',
        tmp_path,
        monkeypatch,
    )
    assert 'import os\n' in code
    assert 'from sqlalchemy import create_engine' in code
    assert (
        "with create_engine(os.environ['SHOP_DB'], poolclass=NullPool).connect() as connection:\n"
        """    d = pd.read_sql_query('SELECT * FROM "rows"', connection)"""
    ) in code
    assert 'trained on 76 rows' in capsys.readouterr().out


def test_bigquery_source(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs against a stand-in BigQuery client that records the query it receives."""
    import sys
    import types

    import pandas as pd

    queries: list[str] = []

    class Client:
        def query(self, sql: str) -> 'Client':
            queries.append(sql)
            return self

        def to_dataframe(self) -> pd.DataFrame:
            return pd.read_csv(data)

    google = types.ModuleType('google')
    cloud = types.ModuleType('google.cloud')
    bigquery = types.ModuleType('google.cloud.bigquery')
    bigquery.Client = Client  # type: ignore[attr-defined]
    cloud.bigquery = bigquery  # type: ignore[attr-defined]
    google.cloud = cloud  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.cloud', cloud)
    monkeypatch.setitem(sys.modules, 'google.cloud.bigquery', bigquery)

    code = run(
        'dataset d from bigquery("my-project.shop.rows")\n'
        'transform d { select x, binary }\n'
        'model m = random_forest { trees = 5 }\n'
        'train m { target = binary }\n',
        tmp_path,
        monkeypatch,
    )
    assert 'from google.cloud import bigquery' in code
    assert queries == ['SELECT * FROM `my-project.shop.rows`']
    assert 'trained on 96 rows' in capsys.readouterr().out


def test_runtime_requirements() -> None:
    def modules(source: str) -> list[str]:
        return [r.module for r in runtime_requirements(compile_source(source))]

    assert modules('dataset d from csv("d.csv")') == ['pandas']
    assert modules(
        'dataset d from parquet("d.parquet")\nmodel m = random_forest\ntrain m { target = y }'
    ) == ['pandas', 'sklearn', 'pyarrow']
    assert modules('dataset d from sql("sqlite://", "t")') == ['pandas', 'sqlalchemy']
    assert runtime_requirements(compile_source('dataset d from bigquery("my-project.a.b")'))[
        1:
    ] == [
        Requirement('google.cloud.bigquery', 'bigquery'),
        Requirement('db_dtypes', 'bigquery'),
    ]


def test_generated_header() -> None:
    code = generate(compile_source('dataset d from csv("d.csv")'), 'x.eig')
    assert code.startswith('"""Generated by eigrel ')
    assert 'from x.eig. Do not edit."""\n\nimport pandas as pd\n' in code


@pytest.fixture
def messy(tmp_path: Path) -> Path:
    """Rows with missing values, a text feature and a text target."""
    path = tmp_path / 'messy.csv'
    rows = ['x,city,plan,amount']
    for i in range(90):
        x = '' if i % 15 == 0 else str(i % 30)
        city = '' if i % 20 == 0 else ['rio', 'sp', 'bh'][i % 3]
        plan = ['basic', 'pro', 'max'][(i % 30) // 10]
        amount = '' if i % 25 == 0 else str(i * 1.5)
        rows.append(f'{x},{city},{plan},{amount}')
    path.write_text('\n'.join(rows) + '\n')
    return path


def test_fill_and_drop_missing(
    messy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset d from csv("messy.csv")\n'
        'transform d { fill city = "unknown", amount = 0\n drop_missing x }\n'
        'model m = random_forest { trees = 10 }\n'
        'train m { target = plan }\n'
        'evaluate m { metrics = [accuracy] }\n',
        tmp_path,
        monkeypatch,
    )
    assert "d = d.fillna({'city': 'unknown', 'amount': 0})" in code
    assert "d = d.dropna(subset=['x'])" in code
    assert 'trained on 67 rows' in capsys.readouterr().out
    assert 'pd.get_dummies' not in code


def test_drop_missing_on_any_column(
    messy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset d from csv("messy.csv")\ntransform d { drop_missing }\n'
        'model m = decision_tree\ntrain m { target = plan }\n',
        tmp_path,
        monkeypatch,
    )
    assert 'd = d.dropna()' in code
    assert 'trained on' in capsys.readouterr().out


def test_xgboost_with_text_target_and_features(
    messy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        'dataset d from csv("messy.csv")\n'
        'transform d { drop_missing x }\n'
        'model m = xgboost { trees = 20, max_depth = 3, learning_rate = 0.3 }\n'
        'train m { target = plan }\n'
        'evaluate m { metrics = [accuracy, f1, auc] }\n'
        'model r = xgboost { task = regression }\ntrain r { data = d, target = x }\nevaluate r {}\n',
        tmp_path,
        monkeypatch,
    )
    assert 'EncodedLabelClassifier(XGBClassifier(' in code
    assert 'm_pred = m.predict(m_X_test)' in code
    assert 'r.fit(r_X_train, r_y_train)' in code
    out = capsys.readouterr().out
    assert 'm: xgboost classification' in out and 'r: xgboost regression' in out
    assert '  auc       ' in out


def test_register_in_mlflow(
    messy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlflow
    import pandas as pd

    tracking = f'sqlite:///{tmp_path / "mlflow.db"}'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tracking)
    code = run(
        'dataset d from csv("messy.csv")\n'
        'transform d { fill city = "unknown"\n drop_missing }\n'
        'model m = random_forest { trees = 5 }\n'
        'train m { target = plan }\n'
        'evaluate m { metrics = [accuracy] }\n'
        'register m { name = "plans", experiment = "tests" }\n'
        'model r = linear_regression\ntrain r { data = d, target = amount }\nregister r\n',
        tmp_path,
        monkeypatch,
    )
    assert "mlflow.set_experiment('tests')" in code
    assert code.count('mlflow.log_metrics(') == 1  # r is registered without an evaluation
    out = capsys.readouterr().out
    assert "m: registered in MLflow as 'plans', version 1" in out

    mlflow.set_tracking_uri(tracking)
    model = mlflow.pyfunc.load_model('models:/plans/1')
    # The registered pipeline takes raw rows, including a city it never saw.
    raw = pd.DataFrame({'x': [3.0], 'city': ['recife'], 'amount': [10.0]})
    assert list(model.predict(raw)) == ['basic']
    run_data = mlflow.get_run(mlflow.MlflowClient().get_model_version('plans', '1').run_id).data
    assert run_data.params['algorithm'] == 'random_forest'
    assert 'accuracy' in run_data.metrics


def test_requirements_for_xgboost_and_mlflow() -> None:
    graph = compile_source(
        'dataset d from csv("d.csv")\nmodel m = xgboost\ntrain m { target = y }\nregister m'
    )
    assert [r.module for r in runtime_requirements(graph)][-2:] == ['xgboost', 'mlflow']
