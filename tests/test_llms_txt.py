"""The llms.txt agent contract: every Eigrel snippet in it must stay valid."""

import re
from pathlib import Path

from eigrel.compiler import parse

LLMS = Path(__file__).parent.parent / 'llms.txt'


def eigrel_blocks() -> list[str]:
    text = LLMS.read_text(encoding='utf-8')
    return re.findall(r'```eigrel\n(.*?)```', text, re.DOTALL)


def test_every_kind_of_statement_is_documented_and_parses() -> None:
    blocks = eigrel_blocks()
    assert len(blocks) == 7  # dataset, transform, features, model, train, evaluate, register
    for block in blocks:
        parse(block)  # syntax only: blocks reference each other's datasets and models


def test_blocks_use_hash_comments_only() -> None:
    for block in eigrel_blocks():
        assert '//' not in block, 'Eigrel comments are #, not //'
