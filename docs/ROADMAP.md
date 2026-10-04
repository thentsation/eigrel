# Roadmap

Eigrel aims to be to tabular machine learning what Terraform is to infrastructure: you declare
intent, the tool shows the consequences before anything runs, and the same program runs on
different providers. Nobody adopts a language for its syntax; they adopt it for what the tool
does around it.

| Terraform | Eigrel |
|---|---|
| HCL | `.eig` programs |
| Providers | Backends: Python (pandas + scikit-learn), Spark (PySpark + MLlib), SQL ✅ |
| Validation | `eigrel check` (`--json` for tools and agents) ✅ |
| `terraform plan` | `eigrel plan`: row counts, schemas, class balance and warnings before running ✅ |
| State and drift | `.eigstate`: schema fingerprints, drift detection ✅ |
| Review in pull requests | A pipeline change is a reviewable diff with proven consequences |

## Released

| Version | Theme | Scope |
|---|---|---|
| 0.1 | Language | Lexer, parser, AST, CLI |
| 0.2 | Compiler | Semantic analysis, Eigrel IR, Python backend |
| 0.3 | Data | CSV, Parquet, JSON, SQL databases, BigQuery, `env()`, SQL backend |
| 0.4 | Spark | Spark backend: PySpark for data, MLlib for training |
| 0.5 – 0.6 | ML | `fill`, `drop_missing`, XGBoost, MLflow `register`, leak-free encoding pipeline, Spark parity, `check --json`, [`llms.txt`](../llms.txt) |
| 0.7 | Plan | `eigrel plan`: schema inference, schema contracts, row counts, class balance, findings with codes, `--json`, `.eigstate` with drift detection |
| 0.8 | Guarantees | `predict` that reuses the training pipeline (skew is a compile error), `assumptions { time }` with time-ordered splits and `random-split` warnings, a backend capability matrix, labels decoded inside registered models |
| 0.9 | Agents | `eigrel mcp`: an MCP server with check, plan, compile and run tools and the grammar as a resource; worked examples in `llms.txt` |
| 1.0 | Stable | Grammar, CLI, exit codes and JSON formats frozen under semantic versioning ([compatibility](COMPATIBILITY.md)); JSON Schemas in [`schemas/`](schemas/) checked by tests |

## After 1.0

Ideas, in no particular order; none is promised:

- An editor extension (syntax highlighting, `check` and `plan` diagnostics inline).
- More backends behind the same programs, starting where the capability matrix says "no".
- Cross-validation (`train { folds = 5 }`) and hyperparameter search, reported by `plan`.
- Remote state for `.eigstate`, so CI and teammates share drift baselines.

## Out of scope

- GenAI and agents as language features: they are not uniform the way tabular data is, so a small
  language would get in the way.
- A general optimizer phase: projection pushdown happens where it matters, in the probes behind
  `eigrel plan`.

## Design notes

The guarantees come from restraint: the smaller the language, the more the compiler can prove.
Strong guarantees (a column exists, the target is not a feature, the serving path reuses the
training pipeline) are errors. Checks that depend on declarations (temporal leakage needs a
declared time column) say so.
