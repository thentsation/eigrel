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
    readme = (examples_dir.parent / 'README.md').read_text()
    assert 'examples/mistakes/target_leakage.eig:26:14' in readme
