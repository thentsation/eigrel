# Compatibility

Eigrel follows [semantic versioning](https://semver.org/) from 1.0. Within 1.x, a program that
`eigrel check` accepts keeps being accepted, and tools that read Eigrel's JSON keep working.

## Stable in 1.x

| Area | Promise |
|---|---|
| Language | Valid programs stay valid with the same meaning. Keywords are not added in a way that breaks existing identifiers without a deprecation release first. |
| CLI | Commands (`check`, `plan`, `run`, `compile`, `ir`, `ast`, `tokens`, `init`, `mcp`), their flags and their exit codes keep working. |
| Exit codes | `0` success; `1` errors (syntax, semantics, plan errors, failed runs); `2` warnings under `plan --strict`. |
| JSON | `check --json`, `plan --json` and `.eigstate` files are [format 1](schemas/). Fields may be added; none is removed or changes meaning. A change that would break readers ships as a new `format`. |
| Finding codes | The codes in the plan schema keep their meaning. New codes may be added; severities only become stricter in a minor release when the old behaviour let a run fail. |
| MCP | The `check`, `plan`, `compile` and `run` tools and the `eigrel://llms.txt` and `eigrel://capabilities` resources keep their names and argument names. |

## Not covered

- The text of generated Python, Spark or SQL code, and the human-readable output of `plan`,
  `check` and `run`: they are for people and may improve in any release. Parse the JSON instead.
- Model results: new library versions may change scores slightly.
- The `ir`, `ast` and `tokens` commands, which show compiler internals.
- Python APIs under `eigrel.*` other than `eigrel.runtime`, which registered models import.

## Deprecations

A feature to be removed first produces a warning (a plan finding with an `info` or `warning`
severity, or a message on standard error) for at least one minor release, with the replacement.
Removal happens only in a major release.
