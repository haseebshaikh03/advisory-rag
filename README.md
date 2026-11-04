# advisory-rag

Retrieval-augmented question answering over Python security advisories, with an evaluation harness that measures which retrieval strategy actually works.

Ask "Which packages had unsafe pickle deserialization bugs?" or "What is CVE-2026-112941 about?" and get back the matching GitHub security advisories, plus an answer from Claude that cites the exact advisory text it used.

## What it does

1. **Ingest.** Pulls reviewed PyPI advisories from the GitHub Advisory Database, splits each one into overlapping chunks that each carry the advisory id, package, severity and weakness type, embeds them locally with `BAAI/bge-small-en-v1.5` and stores them in Postgres with pgvector.
2. **Retrieve.** Three retrievers over the same table: vector search (cosine distance on an HNSW index), keyword search (Postgres full-text with `ts_rank_cd`), and hybrid search that merges the two with Reciprocal Rank Fusion. Exact CVE and GHSA ids in the question are routed to a direct lookup first.
3. **Answer.** Each retrieved chunk goes to Claude as a separate document with citations turned on. The prompt restricts the answer to those documents and asks the model to say so when they do not contain the answer.
4. **Evaluate.** Builds 180 labelled questions from the advisories themselves and scores every retriever on recall@5 and MRR@10.

## Results

300 advisories, 1,383 chunks, 180 questions. Each cell is recall@5 / MRR@10.

| Retriever | id lookups | package + weakness | symptom | all |
|---|---|---|---|---|
| vector | 0.03 / 0.01 | 0.97 / 0.84 | 1.00 / 0.98 | 0.67 / 0.61 |
| keyword | 0.02 / 0.02 | 0.88 / 0.72 | 1.00 / 0.94 | 0.63 / 0.56 |
| hybrid (RRF) | 0.05 / 0.01 | 0.98 / 0.88 | 1.00 / 0.97 | 0.68 / 0.62 |
| hybrid + id routing | 1.00 / 1.00 | 0.98 / 0.88 | 1.00 / 0.97 | 0.99 / 0.95 |

What the numbers say:

- Embeddings cannot find an advisory by its CVE number. "CVE-2026-112941" carries almost no meaning for an embedding model, so vector search returns whatever is closest in topic.
- Plain keyword search fails on ids too, for a different reason. Every advisory contains the tokens "cve" and "2026", and `ts_rank_cd` has no inverse document frequency, so the one rare token that matters gets drowned out.
- Hybrid search beats either retriever alone on the package and weakness questions, which is the case RRF is built for.
- Routing exact identifiers to a lookup fixes the id questions completely and lifts overall recall@5 from 0.68 to 0.99.

Limits of this eval, stated plainly: the symptom questions reuse the advisory summary with the package name removed, so they are easier than real user questions and every retriever scores close to 1.0 on them. The next step is a set of paraphrased questions written by hand or by a model, and an answer-level check that the cited text supports each claim.

## Run it

Needs Docker, Python 3.12+ and the GitHub CLI (`gh auth login`).

```bash
docker compose up -d --wait
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # .venv/bin on Linux and macOS
python -m rag.fetch --pages 3      # 300 advisories into data/advisories.jsonl
python -m rag.ingest               # chunk, embed, load into pgvector
python -m eval.make_questions
python -m eval.run_eval
pytest -q
```

Serve the API:

```bash
uvicorn rag.api:app --reload
curl "localhost:8000/search?q=pickle+deserialization+remote+code+execution&mode=hybrid&n=3"
```

`POST /ask` needs `ANTHROPIC_API_KEY`. `/search` works without it.

## Layout

| Path | Purpose |
|---|---|
| `rag/fetch.py` | GitHub Advisory Database download, cursor pagination, normalization |
| `rag/chunk.py` | Header plus overlapping word windows per advisory |
| `rag/embed.py` | Local ONNX embeddings, query instruction prefix for BGE |
| `rag/db.py` | Schema: pgvector column with HNSW index, generated tsvector with GIN index |
| `rag/ingest.py` | Embedding cache keyed by model and text hash, bulk load |
| `rag/search.py` | Vector, keyword, RRF hybrid, id routing, per-advisory dedupe |
| `rag/answer.py` | Claude call with citable document blocks and refusal handling |
| `rag/api.py` | FastAPI endpoints |
| `eval/` | Question generation and retriever scoring |
| `tests/` | Chunking, fusion and answer-parsing tests |
