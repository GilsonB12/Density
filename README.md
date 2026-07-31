# density

> Production-ready RAG with rigorous built-in evaluation — ask questions about
> long documents and get answers **backed by metrics**, not vibes.

**Status: 🚧 under active development** — following the roadmap below, one
measured step at a time.

## Why another RAG tool?

Most RAG projects stop at "it answers questions". **density** treats evaluation
as a first-class feature: every retrieval and generation decision (chunking
strategy, hybrid search, reranking) is validated with RAGAS metrics on a golden
dataset, and the benchmark report is part of the repo.

The name comes from [Chain of Density](https://doi.org/10.1007/978-3-031-79032-4_26)
— a summarization technique the author researched and published on.

## Planned architecture

```
Indexing:  file ─▶ ingestion ─▶ chunking ─▶ embedding ─▶ pgvector
Query:     question ─▶ hybrid retrieval (dense + BM25) ─▶ rerank ─▶ grounded answer
Evaluation: golden dataset ─▶ RAGAS (faithfulness, relevancy, context precision/recall)
```

## Quick start (dev)

```bash
docker compose up -d   # Postgres + pgvector
uv sync                # install deps into .venv
uv run density --help
uv run pytest
```

## Roadmap

- [x] 0 — Scaffolding (uv, ruff, pytest, docker-compose pgvector)
- [ ] 1 — Ingestion (PDF/TXT/MD) + chunking strategies
- [ ] 2 — Embeddings + pgvector storage
- [ ] 3 — Dense retrieval (`density query`)
- [ ] 4 — Grounded generation with citations (`density ask`)
- [ ] 5 — Evaluation harness: RAGAS + golden dataset (`density eval`)
- [ ] 6 — Hybrid search (dense + sparse, RRF fusion)
- [ ] 7 — Cross-encoder reranking
- [ ] 8 — Benchmark report: chunking & retrieval strategies compared
- [ ] 9 — Chain of Density summarization mode + MCP server

## License

MIT
