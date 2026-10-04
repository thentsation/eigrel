"""Variable names for generated code that never clash with each other or with library names."""

import keyword
from collections.abc import Collection

from eigrel.compiler import ir


def assign_names(
    graph: ir.Graph,
    reserved: Collection[str],
    dataset_helpers: tuple[str, ...] = (),
    model_helpers: tuple[str, ...] = (),
) -> dict[str, str]:
    """Map each Eigrel name to a Python variable.

    Datasets and models also get helper variables named with the given suffixes (`churn_X`,
    `churn_pred`, ...); no variable may equal another one's helper, a keyword or a reserved name.
    """
    models = {op.name for op in graph.ops if isinstance(op, ir.Train)}
    names: dict[str, str] = {}
    taken: set[str] = set()
    for op in graph.ops:
        if op.name in names:
            continue
        helpers = model_helpers if op.name in models else dataset_helpers
        variable = op.name
        while (
            keyword.iskeyword(variable)
            or variable in reserved
            or variable in taken
            or any(variable + suffix in taken for suffix in helpers)
        ):
            variable += '_'
        taken.add(variable)
        taken.update(variable + suffix for suffix in helpers)
        names[op.name] = variable
    return names
