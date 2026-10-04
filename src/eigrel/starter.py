"""The starter project created by `eigrel init`."""

import csv
import math
import random
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

PROGRAM = """\
# A first Eigrel pipeline: load data, pick features, train and evaluate a model.
# Run it with: eigrel run main.eig

dataset customers from csv("data/customers.csv")

transform customers {
    filter age >= 18
    select age, income, purchases, churned
}

features customers {
    age
    income
    purchases
}

model churn = random_forest {
    trees = 100
}

train churn {
    target = churned
}

evaluate churn {
    metrics = [accuracy, precision, recall, f1]
}
"""


COLUMNS = ('customer_id', 'age', 'income', 'purchases', 'churned')


def sample_customers(rows: int = 400, seed: int = 7) -> Iterator[tuple[int, int, float, int, int]]:
    """Synthetic churn data; the same seed always produces the same rows."""
    rng = random.Random(seed)
    for customer_id in range(1, rows + 1):
        age = rng.randint(16, 75)
        income = round(max(800, rng.gauss(1200 + 70 * age, 900)), 2)
        purchases = max(0, int(rng.gauss(12, 6)))
        # Fewer purchases and a lower income make churn more likely.
        score = 2.0 - 0.18 * purchases - 0.0005 * (income - 4200) + rng.gauss(0, 0.4)
        churned = int(1 / (1 + math.exp(-score)) > 0.5)
        yield customer_id, age, income, purchases, churned


def write_sample_customers(path: Path) -> None:
    """Write the sample churn data as CSV."""
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(COLUMNS)
        writer.writerows(sample_customers())


def write_sample_database(path: Path) -> None:
    """Write the sample churn data as a SQLite database with a `customers` table."""
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, age INTEGER,'
            ' income REAL, purchases INTEGER, churned INTEGER)'
        )
        connection.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?)', sample_customers())
