from pathlib import Path

from eigrel.compiler import compile_source
from eigrel.starter import PROGRAM, write_sample_customers


def test_starter_program_compiles() -> None:
    assert len(compile_source(PROGRAM).ops) == 5


def test_sample_data_is_deterministic_and_matches_the_example(
    tmp_path: Path, examples_dir: Path
) -> None:
    generated = tmp_path / 'customers.csv'
    write_sample_customers(generated)
    assert generated.read_text() == (examples_dir / 'data' / 'customers.csv').read_text()
    assert generated.read_text().splitlines()[0] == 'customer_id,age,income,purchases,churned'
