# Study guide

Read this top to bottom with the code open next to it. Each section names the file to look at. By the end you should be able to explain every design choice in this repo, and answer the interview questions at the bottom without notes.

## 1. What RAG is, in one paragraph

A language model only knows what was in its training data. Retrieval-augmented generation (RAG) fixes that at question time. You keep your own documents in a searchable store, find the few passages most relevant to the question, paste them into the prompt, and tell the model to answer only from them. Two halves: **retrieval** (find the right text) and **generation** (write the answer from it). Almost every RAG failure is a retrieval failure, which is why most of this repo is about retrieval and measuring it.

## 2. Chunking (`rag/chunk.py`)

Documents are split into chunks because:

- an embedding of a whole long document blurs many topics into one vector, so it matches nothing well
- the prompt has a token budget, and you want to send only the relevant part

Choices made here:

- **Word windows of 180 with 40 overlap.** Overlap means a sentence cut at a boundary still appears whole in the neighbouring chunk.
- **A header on every chunk** (summary, ids, package, severity, CWE). A chunk from the middle of an advisory says "the attacker can overwrite files" but never names the package. Without the header, retrieval finds it and the model cannot tell which package it is about. This is often called contextual chunking.
- **Code blocks removed.** Proof-of-concept code is long and adds noise to embeddings.

Try it: change `CHUNK_WORDS` in `rag/config.py` to 80 or 400, rerun ingest and eval, and watch the numbers move.

## 3. Embeddings (`rag/embed.py`)

An embedding model turns text into a vector (here 384 numbers) so that texts with similar meaning end up close together. Closeness is measured with **cosine similarity**, the angle between two vectors.

- `bge-small-en-v1.5` runs locally on CPU through ONNX (the `fastembed` library), so there is no API key and no cost.
- BGE models were trained with an instruction in front of queries ("Represent this sentence for searching relevant passages: ") but not in front of documents. Queries and documents are different kinds of text, and this asymmetry is common in retrieval models.
- Embedding is the slow step (about 3 minutes for 1,400 chunks on a laptop), so `rag/ingest.py` caches vectors keyed by a hash of the model name plus the text.

## 4. The vector database (`rag/db.py`)

pgvector adds a `vector` column type to Postgres. You could use Pinecone, Qdrant, Weaviate or Chroma instead. Postgres was chosen because:

- the advisories, their metadata and their vectors live in one database with normal SQL joins and transactions
- the same table also holds a full-text index, so hybrid search is one database, not two systems to keep in sync

Two indexes:

- **HNSW** on the embedding (`vector_cosine_ops`). Without an index, every query compares against every row. HNSW is an approximate nearest-neighbour graph: much faster, very slightly less exact. `<=>` is the cosine distance operator.
- **GIN** on a generated `tsvector` column for full-text search. `to_tsvector('english', ...)` lowercases, drops stop words like "the" and stems words ("vulnerabilities" becomes "vulner").

## 5. Three retrievers and fusion (`rag/search.py`)

- **Vector search** finds meaning: "loading untrusted pickles" matches "unsafe deserialization" with no shared words. It is weak on exact strings like ids and version numbers.
- **Keyword search** finds exact terms and rare words. It is weak on paraphrases.
- **Hybrid search** runs both and merges the lists with **Reciprocal Rank Fusion**: each result scores `1 / (60 + rank)` in each list it appears in, summed. RRF uses ranks, not raw scores, so you never have to put cosine similarity (0 to 1) and `ts_rank_cd` (unbounded) on the same scale. Something ranked high by both lists beats something ranked first by only one.
- **Id routing.** A regex spots CVE and GHSA ids and looks them up exactly before any fuzzy search. The eval shows why: without it, id questions score near zero in every retriever.
- **Dedupe per advisory.** One advisory can have many chunks. Returning five chunks of the same advisory wastes the context window, so only the best chunk per advisory is kept.

## 6. Generation (`rag/answer.py`)

- Each chunk is sent as its own `document` block with `citations: {enabled: true}`. Claude's response then comes back in pieces, each carrying the exact quoted text and which document it came from. That makes answers checkable.
- The system prompt says to use only the documents and to say when they do not cover the question. This is the main guard against hallucination.
- `stop_reason == "refusal"` is checked before reading the answer, and server-side fallbacks are turned on so a safety decline is retried on a suitable model.
- `effort: low` because this is short question answering, not deep reasoning.

## 7. Evaluation (`eval/`)

You cannot improve retrieval you do not measure. The eval:

- builds questions from the data with known correct answers (no model needed)
- **recall@5**: did a correct advisory appear in the top 5 at all
- **MRR@10** (mean reciprocal rank): 1 if the correct advisory is first, 0.5 if second, 0.33 if third, 0 if not in the top 10. It rewards ranking, not just presence
- splits results by question type, because an average hides the fact that one type fails completely

Be honest about the eval's limits (see the README). The symptom questions reuse summary wording, so they are too easy. Real systems add paraphrased questions and grade the final answer too, for example by checking that every cited quote actually supports the claim next to it.

## 8. Things to try yourself

1. Run `python -m eval.run_eval` with `CHUNK_WORDS` at 80 and at 400.
2. Change `RRF_K` from 60 to 10 and see whether hybrid gets better or worse.
3. Swap the embedding model in `rag/config.py` to `BAAI/bge-base-en-v1.5` (768 dimensions, change `EMBED_DIM` too) and compare.
4. Write 20 questions by hand in your own words and add them as a new `kind`. Watch the scores drop. That is the real-world number.
5. Add a reranker: take the top 20 from hybrid and reorder them with a cross-encoder (fastembed has `TextCrossEncoder`). Measure whether MRR goes up.

## 9. Interview questions you should be able to answer

- Why chunk at all, and what goes wrong with chunks that are too small or too large?
- Why did vector search fail on CVE ids, and how did you fix it?
- What is Reciprocal Rank Fusion and why use ranks instead of scores?
- Why pgvector instead of a dedicated vector database? When would you switch?
- What does HNSW trade away for speed?
- How do you stop the model from making things up?
- How do you know your retrieval is good? What metrics, and what are their limits?
- What would you add next? (reranking, paraphrased eval questions, answer faithfulness grading, metadata filters such as severity or package, incremental ingest)
