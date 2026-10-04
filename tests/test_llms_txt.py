"""The llms.txt agent contract: every Eigrel snippet in it must stay valid."""

import re
from pathlib import Path

from eigrel.compiler import compile_source, parse

ROOT = Path(__file__).parent.parent
LLMS = ROOT / 'llms.txt'


def blocks(section: str) -> list[str]:
    text = LLMS.read_text(encoding='utf-8')
    start = text.index(f'## {section}\n')
    end = text.find('\n## ', start + 1)
    return re.findall(r'```eigrel\n(.*?)```', text[start:end], re.DOTALL)


def test_every_kind_of_statement_is_documented_and_parses() -> None:
    statements = blocks('Language reference')
    # dataset, transform, features, model, train, evaluate, register, assumptions, predict
    assert len(statements) == 9
    for block in statements:
        parse(block)  # syntax only: blocks reference each other's datasets and models


def test_examples_are_complete_valid_programs() -> None:
    examples = blocks('Examples')
    assert len(examples) == 3
    for program in examples:
        assert compile_source(program).ops


def test_blocks_use_hash_comments_only() -> None:
    for block in blocks('Language reference') + blocks('Examples'):
        assert '//' not in block, 'Eigrel comments are #, not //'


def test_the_packaged_copy_is_identical() -> None:
    packaged = ROOT / 'src' / 'eigrel' / 'llms.txt'
    assert packaged.read_text(encoding='utf-8') == LLMS.read_text(encoding='utf-8')
