# Eigrel

**Uma linguagem de programação para Dados, Machine Learning e IA.**

> Escreva o que você quer. Deixe o compilador decidir como executar.

🇺🇸 [Read in English](README.md)

Eigrel é uma linguagem declarativa para engenharia de dados, machine learning e GenAI. Você descreve
datasets, transformações, features e modelos numa linguagem só; o compilador decide se cada etapa
vira Python, SQL, Spark ou outra coisa.

```eigrel
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
```

## Status

O Eigrel está na **v0.1 — Language**: lexer, parser e árvore sintática prontos, e a CLI já valida
arquivos e mostra tokens ou AST. Nada executa ainda; compilador, IR e backend Python chegam na
v0.2. Veja o [roadmap](docs/ROADMAP.md).

## Começando

Requer Python 3.13+ e [uv](https://docs.astral.sh/uv/).

```bash
make install                     # cria a .venv e instala o eigrel em modo editável
.venv/bin/eigrel init churn      # cria churn/main.eig
.venv/bin/eigrel check churn/main.eig
```

### CLI

| Comando | O que faz |
|---|---|
| `eigrel check ARQUIVO...` | Faz o parse de cada arquivo e aponta erros de sintaxe com linha e coluna |
| `eigrel ast ARQUIVO` | Mostra a árvore sintática em JSON |
| `eigrel tokens ARQUIVO` | Mostra a sequência de tokens |
| `eigrel init NOME` | Cria um projeto com um `main.eig` inicial |

Os erros apontam o lugar exato:

```text
error: comparisons cannot be chained; combine them with 'and'
 --> pipeline.eig:4:21
  |
4 |     filter age > 18 < 30
  |                     ^
```

### Docker

```bash
make docker-build
make docker-run      # roda `eigrel check` em examples/ml.eig dentro do container
```

## Linguagem

A sintaxe completa está em [docs/LANGUAGE.md](docs/LANGUAGE.md). Exemplos ficam em
[`examples/`](examples).

## Desenvolvimento

```bash
make test            # pytest
make coverage        # pytest com o gate de 90% de cobertura
make lint            # ruff check + checagem de formatação
make typecheck       # mypy
make check-examples  # eigrel check examples/*.eig
```

Os commits seguem [Conventional Commits](https://www.conventionalcommits.org/); releases e
changelog são gerados pelo semantic-release.

## Licença

Licenciado sob a [Apache License 2.0](LICENSE).
