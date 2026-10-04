"""The 1.x contracts: JSON outputs match the published schemas in docs/schemas."""

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from eigrel import planner, runner
from eigrel.planner import build_plan, save_state, to_json

SCHEMAS = Path(__file__).parent.parent / 'docs' / 'schemas'


def validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMAS / f'{name}.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_check_json_matches_its_schema(tmp_path: Path, examples_dir: Path) -> None:
    (tmp_path / 'bad.eig').write_text('model m = rf')
    (tmp_path / 'broken.eig').write_text('model m random_forest')
    report = runner.check_files(
        [
            str(examples_dir / 'ml.eig'),
            str(tmp_path / 'bad.eig'),
            str(tmp_path / 'broken.eig'),
            str(tmp_path / 'missing.eig'),
        ]
    )
    assert report['format'] == 1
    validator('check').validate(report)


@pytest.mark.parametrize('example', ['ml', 'regression', 'sql', 'mlflow', 'serving', 'pipeline'])
def test_plan_json_matches_its_schema(example: str, examples_dir: Path) -> None:
    data = json.loads(json.dumps(to_json(build_plan(examples_dir / f'{example}.eig')), default=str))
    assert data['format'] == 1
    validator('plan').validate(data)


def test_failing_plans_match_the_schema(tmp_path: Path) -> None:
    (tmp_path / 'd.csv').write_text(
        'day,x,y\n' + ''.join(f'2025-01-{1 + i % 28:02d},{i},{i % 2}\n' for i in range(30))
    )
    (tmp_path / 'p.eig').write_text(
        'dataset d from csv("d.csv")\nmodel m = logistic_regression\ntrain m { target = y }\n'
        'dataset e from csv("missing.csv")\n'
    )
    data = json.loads(json.dumps(to_json(build_plan(tmp_path / 'p.eig')), default=str))
    assert data['ok'] is False
    validator('plan').validate(data)


def test_state_file_matches_its_schema(examples_dir: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(examples_dir, tmp_path / 'examples')
    plan = build_plan(tmp_path / 'examples' / 'ml.eig')
    state = json.loads(save_state(plan).read_text(encoding='utf-8'))
    assert state['format'] == 1
    validator('eigstate').validate(state)


def test_every_finding_code_is_in_the_schema() -> None:
    source = Path(planner.__file__).read_text(encoding='utf-8')
    emitted = set(re.findall(r"'(?:error|warning|info)',\s*'([a-z-]+)'", source))
    emitted |= set(re.findall(r"severity, '([a-z-]+)'", source))
    emitted -= {'error', 'warning', 'info'}  # the Severity type itself
    emitted |= {'io', 'lex', 'parse', 'semantic'}  # stages reported as codes
    schema = json.loads((SCHEMAS / 'plan.schema.json').read_text(encoding='utf-8'))
    codes = set(schema['$defs']['finding']['properties']['code']['enum'])
    assert emitted - codes == set()
    assert 'not-probed' in emitted and 'drift-rows' in emitted
