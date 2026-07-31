# density — Design Doc

Data: 2026-07-31 · Status: aprovado (conversa Gilson + Claude)

## Visão

Ferramenta open-source de RAG production-ready para busca inteligente e sumarização
sobre documentos longos (PDF, TXT, MD). Diferencial: **avaliação rigorosa integrada**
— o sistema mede a qualidade do próprio retrieval e da própria geração (RAGAS +
métricas próprias), e cada decisão técnica é justificada com números de benchmark.

Em uma frase: *um motor de perguntas-e-respostas sobre documentos que prova, com
métricas, que suas respostas são confiáveis.*

Objetivos paralelos: aprendizado prático de RAG (Gilson) e peça de portfólio
(GitHub + LinkedIn) com relatório de benchmark comparando estratégias.

## Não-objetivos (por enquanto)

- Interface web / API HTTP (a biblioteca permite; não é o foco do MVP)
- Multi-tenancy, auth, deploy em nuvem
- Suporte a formatos além de PDF/TXT/MD no MVP
- Otimização de performance antes de haver medição

## Arquitetura

**Padrão: Ports & Adapters (hexagonal leve).** Toda dependência externa fica atrás
de um `Protocol`: `Chunker`, `EmbeddingProvider`, `VectorStore`, `Reranker`,
`LLMProvider`. Justificativa: a tese do projeto é *comparar* estratégias
(pgvector vs Qdrant, OpenAI vs BGE, chunking fixo vs recursivo, Cohere vs
cross-encoder local) — sem interfaces cada comparação é um refactor; com elas é
uma linha de config no comando `benchmark`.

**Contratos Pydantic em `models.py`** são a espinha dorsal. Cada módulo é uma
função de um contrato para outro:

```
ingestion:  arquivo           -> Document
chunking:   Document          -> list[Chunk]
embedding:  list[Chunk]       -> list[EmbeddedChunk]
storage:    EmbeddedChunk     -> pgvector (e busca -> list[RetrievalResult])
retrieval:  query             -> list[RetrievalResult]   (dense + sparse + RRF)
reranking:  list[RetrievalResult] -> list[RetrievalResult] (reordenado)
generation: query + contexto  -> Answer (com citações)
evaluation: golden dataset    -> EvalResult (RAGAS + próprias)
```

Os modelos são criados **na etapa em que se tornam necessários** (YAGNI) —
etapa 0 define apenas `Document` e `Chunk`.

**Regras de dependência:** módulos de estágio não se importam entre si; importam
apenas `models.py` e seus próprios Protocols. `pipeline.py` é o único composition
root (monta `Indexer` e `QueryEngine`). CLI chama só o pipeline.

**Organização por estágio do pipeline** (não por camada técnica): o projeto também
é material didático — quem estuda "chunking" encontra tudo em `chunking/`.

```
src/density/
├── config.py       # Pydantic Settings (env DENSITY_*)
├── models.py       # contratos
├── ingestion/  chunking/  embedding/  storage/
├── retrieval/  reranking/  generation/  evaluation/
├── pipeline.py     # composition root
└── cli/            # Typer: ingest, query, ask, eval, benchmark
```

`src/` layout (testes rodam contra o pacote instalado); `tests/` (funciona?)
separado de `benchmarks/` (qual estratégia é melhor?); `docs/` para specs.

## Decisões de stack e porquês

| Escolha | Racional |
|---|---|
| Python 3.11+, venv pinado em 3.12 | ecossistema ML (torch/sentence-transformers) ainda não é confiável em 3.14 |
| uv | resolver rápido, lockfile, gerencia o próprio Python; padrão emergente 2025+ |
| pgvector (pg17) | zero custo, um só sistema; `tsvector` dá o lado sparse do híbrido de graça na mesma query. Comparação com Qdrant vira experimento futuro |
| text-embedding-3-small | barato, forte baseline; BGE local entra depois via adapter |
| Reranker local primeiro (bge-reranker via sentence-transformers) | grátis, sem API key, roda em CPU para top-50; Cohere vira segundo adapter e comparação de benchmark |
| Full-text do Postgres para sparse | sem infra extra; nota honesta: `ts_rank` ≠ BM25 — comparação com `rank_bm25` puro é experimento do relatório |
| RAGAS | padrão de mercado; métricas separam falha de retrieval vs falha de geração |
| Typer + Rich, Pydantic v2, pytest, ruff | DX moderna, validação nos contratos |

## Roadmap (cada etapa termina rodando e demonstrável)

| # | Entrega | Conceito novo |
|---|---|---|
| 0 | Scaffolding: uv, ruff, pytest, docker-compose pgvector, esqueleto + models base | tooling |
| 1 | Ingestão PDF/TXT/MD + chunking fixo e recursivo + CLI `ingest` | chunking (tokens, overlap, fronteiras) |
| 2 | Embeddings OpenAI + schema pgvector + gravação | embeddings, cosseno, dimensões |
| 3 | Busca dense top-k + CLI `query` | kNN, índices ANN (HNSW/IVFFlat) |
| 4 | Geração fundamentada com citações + CLI `ask` | grounding, orçamento de contexto |
| 5 | Golden dataset (~20-30 perguntas) + RAGAS + CLI `eval` | faithfulness, relevancy, context precision/recall |
| 6 | Híbrido (FTS + RRF), medido | sparse vs dense, RRF |
| 7 | Reranking cross-encoder local, medido | bi-encoder vs cross-encoder |
| 8 | Benchmark sistemático + relatório com tabelas/gráficos | metodologia de experimento |
| 9 | README final, modo Chain of Density, servidor MCP | MCP |

**Decisão-chave de ordem:** avaliação (etapa 5) vem ANTES de híbrido e reranking,
para que toda melhoria entre com número antes/depois — "reranking subiu context
precision de X para Y" é a narrativa do projeto.

## Estratégia de testes

- `tests/` unit: sem rede, sem docker — contratos, chunkers, fusão RRF, montagem de prompt
- `tests/` integration (marker `integration`): exigem o Postgres do compose
- Chamadas a APIs pagas: sempre atrás de Protocol → fakes determinísticos nos testes
- `benchmarks/` fora do pytest: experimentos com custo e não-determinismo

## Riscos conhecidos

- Custo de API em eval/benchmark (RAGAS chama LLM-as-judge) → datasets pequenos e curados, cache de respostas
- Variância de LLM-as-judge → fixar modelo judge + reportar desvio entre runs
- Parsing de PDF ruim contamina tudo → validar extração visualmente na etapa 1 antes de seguir
