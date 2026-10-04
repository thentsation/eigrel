# Roadmap

| Version | Theme | Scope | Status |
|---|---|---|---|
| v0.1 | Language | Lexer, parser, AST, basic types, datasets, transformations, CLI | ✅ done |
| v0.2 | Compiler | Semantic analysis, Eigrel IR, execution graph, Python backend | ✅ done |
| v0.3 | Data | CSV, Parquet, JSON, SQL databases, BigQuery, `env()` secrets, SQL backend | ✅ done |
| v0.4 | Spark | Spark backend: PySpark for data operations, Spark MLlib for training | ✅ done |
| v0.5 | ML | Features, models, training, evaluation, MLflow (`register`) | |
| v0.6 | Optimizer | Filter/projection pushdown, CSE, DAG optimization, backend selection, execution planning | |
| v0.7 | GenAI | Embeddings, vector stores, RAG (`knowledge`, `assistant`), LLMs, evaluation | |
| v0.8 | Agents | Tools, agents, workflows, multi-agent pipelines | |
| v1.0 | Platform | Language, compiler, IR, optimizer, runtime, backends, CLI, package manager, registry, VS Code extension, docs | |

## Target architecture

```text
Eigrel source
   │
   ▼
Lexer → Parser → AST → Semantic analysis → Eigrel IR → Optimizer
                                                          │
                                   ┌──────────────────────┼──────────────────────┐
                                   ▼                      ▼                      ▼
                              Python ✅               SQL ✅                Spark ✅
```

Later backends: PyTorch, MLIR, BigQuery, cloud runtimes.

## CLI, planned

```bash
eigrel init my-project   # ✅ v0.1 (sample data since v0.2)
eigrel check             # ✅ v0.2 (syntax and semantics)
eigrel run               # ✅ v0.2
eigrel compile file.eig  # ✅ v0.2 (--target sql since v0.3)
eigrel ir file.eig       # ✅ v0.2
eigrel build             # planned
eigrel fmt               # planned
eigrel add xgboost       # package manager, v1.0
```

## Future syntax sketches

These are design notes, not yet accepted by the parser.

```eigrel
register churn                      # MLflow

knowledge company_docs {
    source = "gs://company/docs"
    embedding = "text-embedding-model"
    vectorstore = "pgvector"
}

assistant support {
    retrieve company_docs { top = 5 }
    generate { model = "llm" }
}

agent analyst {
    tools = [sql, python, search]
    model = "llm"
    goal = "analyze customer behavior"
}

deploy churn {
    runtime = cloud_run
    replicas = 3
}
```
