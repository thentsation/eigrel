from pathlib import Path

from eigrel.compiler import compile_source
from eigrel.starter import (
    PROGRAM,
    sample_customers,
    write_sample_customers,
    write_sample_database,
)


def test_starter_program_compiles() -> None:
    assert len(compile_source(PROGRAM).ops) == 5


def test_sample_data_is_deterministic_and_matches_the_example(
    tmp_path: Path, examples_dir: Path
) -> None:
    generated = tmp_path / 'customers.csv'
    write_sample_customers(generated)
    assert generated.read_text() == (examples_dir / 'data' / 'customers.csv').read_text()
    assert generated.read_text().splitlines()[0] == 'customer_id,age,income,purchases,churned'


def test_sample_database_matches_the_generator_and_the_example(
    tmp_path: Path, examples_dir: Path
) -> None:
    import sqlite3
    from contextlib import closing

    database = tmp_path / 'customers.db'
    write_sample_database(database)
    write_sample_database(database)  # rewriting replaces the file instead of failing
    for path in (database, examples_dir / 'data' / 'customers.db'):
        with closing(sqlite3.connect(path)) as connection:
            rows = connection.execute('SELECT * FROM customers ORDER BY customer_id').fetchall()
        assert rows == list(sample_customers())
