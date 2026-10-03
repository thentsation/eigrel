# Eigrel: uma linguagem para dados, ML e IA

## O problema

Um projeto típico de dados e machine learning mistura Python, SQL, Pandas, Spark, BigQuery,
PyTorch, MLflow, orquestradores, frameworks de RAG, APIs de LLM e código de infraestrutura. O
resultado é muito boilerplate, pipelines difíceis de ler, código duplicado, otimizações feitas à
mão e um acoplamento forte com cada ferramenta.

## A proposta

O Eigrel coloca uma camada declarativa acima dessas tecnologias. Você descreve **o que** quer
(dados, transformações, features, modelos, pipelines) e o compilador decide **como** executar.

```eigrel
dataset users from bigquery("users")

transform users {
    filter age > 18
    select age, income
}
```

Em vez de trazer a tabela inteira para Python e filtrar lá, o compilador pode perceber que as duas
operações cabem no próprio BigQuery e gerar:

```sql
SELECT age, income FROM users WHERE age > 18
```

O código Eigrel descreve a intenção, não a implementação.

## Por que um compilador, e não mais um framework

O diferencial não é a sintaxe, é o otimizador. Com o pipeline inteiro representado numa IR, dá para
aplicar otimizações clássicas de banco de dados e compiladores: pushdown de filtros e projeções,
eliminação de subexpressões comuns, remoção de transformações redundantes, escolha de backend e
planejamento de execução.

## Onde estamos (v0.1)

A primeira versão entrega o front end do compilador:

- **Lexer** escrito à mão, com localização exata (linha e coluna) de cada token.
- **Parser** de descida recursiva, com precedência de operadores e mensagens de erro que apontam o
  ponto exato do problema.
- **AST** imutável (dataclasses congeladas) e exportável para JSON.
- **CLI** com `check`, `ast`, `tokens` e `init`.

Tudo em Python puro, sem dependências de runtime, com 100% de cobertura de testes.

## Próximos passos

A v0.2 traz análise semântica, a Eigrel IR, o grafo de execução e o primeiro backend (Python). O
roadmap completo está em [docs/ROADMAP.md](docs/ROADMAP.md).
