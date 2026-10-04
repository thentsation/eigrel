from pathlib import Path

import pytest

from eigrel.cli import main


def test_every_example_checks(examples_dir: Path) -> None:
    paths = sorted(str(p) for p in examples_dir.glob('*.eig'))
    assert paths
    assert main(['check', *paths]) == 0


def test_mistakes_are_rejected_where_the_readme_says(
    examples_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = examples_dir / 'mistakes' / 'target_leakage.eig'
    assert main(['check', str(path)]) == 1
    err = capsys.readouterr().err
    assert "error: target 'churned' is also declared as a feature of 'customers'" in err
    assert ':26:14' in err
    readme = (examples_dir.parent / 'README.md').read_text(encoding='utf-8')
    assert 'examples/mistakes/target_leakage.eig:26:14' in readme


# example -> (command, exit code, text the diagnostic must contain)
MISTAKES = {
    'unknown_column': ('check', 1, "column 'income' does not exist here"),
    'wrong_metric': ('check', 1, "metric 'rmse' is for regression"),
    'evaluate_before_train': ('check', 1, 'must be trained before it is evaluated'),
    'type_mismatch': ('plan', 1, 'cannot compare a number with a string'),
    'missing_values': ('plan', 1, 'cannot be trained with logistic_regression: income (27)'),
}


@pytest.mark.parametrize('name', sorted(MISTAKES))
def test_every_mistake_in_the_gallery_is_rejected(
    name: str, examples_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command, code, message = MISTAKES[name]
    assert main([command, str(examples_dir / 'mistakes' / f'{name}.eig')]) == code
    captured = capsys.readouterr()
    assert message in captured.out + captured.err


def test_the_gallery_covers_every_mistake_file_and_its_page(examples_dir: Path) -> None:
    files = {p.stem for p in (examples_dir / 'mistakes').glob('*.eig')}
    assert files == set(MISTAKES) | {'target_leakage'}
    page = (examples_dir.parent / 'docs' / 'MISTAKES.md').read_text(encoding='utf-8')
    for name in files:
        assert f'examples/mistakes/{name}.eig' in page
