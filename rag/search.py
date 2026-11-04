"""Three retrievers over the same chunks table, so they can be compared in the eval.

- vector:  cosine similarity on embeddings (good at meaning, weak on exact ids)
- keyword: Postgres full-text search ranked with ts_rank_cd (good at ids and rare terms)
- hybrid:  both, merged with Reciprocal Rank Fusion
"""
import re
from dataclasses import dataclass

import psycopg

from .embed import embed_query

RRF_K = 60  # standard constant from the RRF paper; dampens the weight of rank 1


@dataclass
class Hit:
    chunk_id: int
    advisory_id: str
    text: str
    score: float


def vector_search(conn: psycopg.Connection, query: str, k: int = 20) -> list[Hit]:
    q = embed_query(query)
    rows = conn.execute(
        "SELECT id, advisory_id, text, 1 - (embedding <=> %s) AS sim "
        "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (q, q, k),
    ).fetchall()
    return [Hit(*r) for r in rows]


def keyword_search(conn: psycopg.Connection, query: str, k: int = 20) -> list[Hit]:
    # websearch_to_tsquery ANDs the words, which is too strict for natural questions,
    # so the terms are OR-ed and ts_rank_cd rewards chunks that match more of them.
    rows = conn.execute(
        """
        WITH q AS (
          SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS tsq
          FROM unnest(to_tsvector('english', %s)) AS t(lexeme, positions, weights)
        )
        SELECT c.id, c.advisory_id, c.text, ts_rank_cd(c.tsv, q.tsq) AS rank
        FROM chunks c, q
        WHERE q.tsq IS NOT NULL AND c.tsv @@ q.tsq
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return [Hit(*r) for r in rows]


def rrf(*ranked_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).

    Uses only ranks, never raw scores, so cosine similarity and ts_rank do not
    need to be on the same scale.
    """
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(h.chunk_id, h)
    fused = sorted(scores, key=scores.get, reverse=True)
    return [Hit(c, by_id[c].advisory_id, by_id[c].text, scores[c]) for c in fused]


def hybrid_search(conn: psycopg.Connection, query: str, k: int = 20) -> list[Hit]:
    return rrf(vector_search(conn, query, k), keyword_search(conn, query, k))[:k]


def dedupe_advisories(hits: list[Hit], n: int) -> list[Hit]:
    """Keep the best chunk per advisory, so the top n are n different advisories."""
    seen, out = set(), []
    for h in hits:
        if h.advisory_id not in seen:
            seen.add(h.advisory_id)
            out.append(h)
        if len(out) == n:
            break
    return out


ID_PATTERN = re.compile(r"\b(CVE-\d{4}-\d{4,}|GHSA(?:-[23456789cfghjmpqrvwx]{4}){3})\b", re.I)


def id_lookup(conn: psycopg.Connection, query: str) -> list[Hit]:
    """Exact match on CVE or GHSA ids found in the query.

    Neither retriever handles ids well. Embeddings see "CVE-2026-112941" as a vague
    string of digits, and every advisory contains the tokens "cve" and "2026", so
    full-text ranking cannot tell them apart either. Production RAG systems route
    exact identifiers to a lookup like this one before the fuzzy search.
    """
    ids = [m.upper() for m in ID_PATTERN.findall(query)]
    if not ids:
        return []
    rows = conn.execute(
        "SELECT c.id, c.advisory_id, c.text, 1.0 FROM chunks c JOIN advisories a ON a.id = c.advisory_id "
        "WHERE (upper(a.cve) = ANY(%s) OR upper(a.id) = ANY(%s)) AND c.ord = 0",
        (ids, ids),
    ).fetchall()
    return [Hit(*r) for r in rows]


RETRIEVERS = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}


def search(conn: psycopg.Connection, query: str, mode: str = "hybrid", n: int = 5, use_ids: bool = True) -> list[Hit]:
    hits = RETRIEVERS[mode](conn, query, k=max(20, n * 4))
    if use_ids:
        hits = id_lookup(conn, query) + hits
    return dedupe_advisories(hits, n)
