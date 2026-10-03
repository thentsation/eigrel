"""The starter project created by `eigrel init`."""

import csv
import math
import random
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


def write_sample_customers(path: Path, rows: int = 400, seed: int = 7) -> None:
    """Write a synthetic churn dataset; the same seed always produces the same file."""
    rng = random.Random(seed)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['customer_id', 'age', 'income', 'purchases', 'churned'])
        for customer_id in range(1, rows + 1):
            age = rng.randint(16, 75)
            income = round(max(800, rng.gauss(1200 + 70 * age, 900)), 2)
            purchases = max(0, int(rng.gauss(12, 6)))
            # Fewer purchases and a lower income make churn more likely.
            score = 2.0 - 0.18 * purchases - 0.0005 * (income - 4200) + rng.gauss(0, 0.4)
            churned = int(1 / (1 + math.exp(-score)) > 0.5)
            writer.writerow([customer_id, age, income, purchases, churned])
