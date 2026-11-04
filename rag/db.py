import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS advisories (
    id        text PRIMARY KEY,
    cve       text,
    summary   text NOT NULL,
    severity  text,
    packages  text[] NOT NULL,
    url       text,
    published timestamptz
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    advisory_id text NOT NULL REFERENCES advisories(id) ON DELETE CASCADE,
    ord         int  NOT NULL,
    text        text NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    -- Postgres full-text index for the keyword half of hybrid search
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (advisory_id, ord)
);

-- HNSW: approximate nearest-neighbour index for cosine distance
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
