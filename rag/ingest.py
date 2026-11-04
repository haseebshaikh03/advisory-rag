"""Load data/advisories.jsonl into Postgres: chunk, embed, insert.

    python -m rag.ingest
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .chunk import chunk_advisory
from .config import CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL
from .db import connect, init_schema
from .embed import embed_documents

DATA = Path(__file__).resolve().parent.parent / "data" / "advisories.jsonl"
CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings_cache.npz"


def load() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cached_embeddings(texts: list[str]) -> list[np.ndarray]:
    """Embedding on CPU is the slow step, so vectors are cached by text hash.

    Re-running ingest after changing only the schema or the search code then takes
    seconds. The model name is part of the key, so switching models re-embeds.
    """
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    keys = [hashlib.sha256(f"{EMBED_MODEL}\n{t}".encode()).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        for i, v in zip(missing, embed_documents([texts[i] for i in missing])):
            cache[keys[i]] = v
        np.savez(CACHE, **cache)
    print(f"{len(texts) - len(missing)} embeddings from cache, {len(missing)} computed")
    return [cache[k] for k in keys]


def main() -> None:
    advisories = load()
    chunks = [c for a in advisories for c in chunk_advisory(a, CHUNK_WORDS, CHUNK_OVERLAP)]
    t0 = time.time()
    vectors = cached_embeddings([c["text"] for c in chunks])
    print(f"embedded {len(chunks)} chunks from {len(advisories)} advisories in {time.time() - t0:.1f}s")

    conn = connect()
    init_schema(conn)
    with conn.transaction():
        conn.execute("TRUNCATE advisories CASCADE")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO advisories (id, cve, summary, severity, packages, url, published) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(a["id"], a["cve"], a["summary"], a["severity"], a["packages"], a["url"], a["published"]) for a in advisories],
            )
            cur.executemany(
                "INSERT INTO chunks (advisory_id, ord, text, embedding) VALUES (%s, %s, %s, %s)",
                [(c["advisory_id"], c["ord"], c["text"], v) for c, v in zip(chunks, vectors)],
            )
    print("done")


if __name__ == "__main__":
    main()
