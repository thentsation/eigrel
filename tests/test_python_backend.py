import csv
from pathlib import Path

import pytest

from eigrel.backends.python import BackendError, generate
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
    assert 'pd.get_dummies(' in code


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
    assert "pd.get_dummies(d.drop(columns=['value']))" in code
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
    assert 'class_ = RandomForestClassifier(' in code
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
    assert 'm_ = RandomForestClassifier(' in code
    assert 'm__ = DecisionTreeClassifier(' in code
    assert capsys.readouterr().out.count('  accuracy  ') == 2


def test_parquet_and_unsupported_sources() -> None:
    code = generate(compile_source('dataset d from parquet("d.parquet")'))
    assert "d = pd.read_parquet('d.parquet')" in code
    assert code.startswith('"""Generated by eigrel ')
    with pytest.raises(BackendError, match="cannot load bigquery sources yet \\(dataset 'u'\\)"):
        generate(compile_source('dataset u from bigquery("p.d.u")'))
