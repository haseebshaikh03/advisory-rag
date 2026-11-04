import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5433/rag")

# Small, fast English embedding model that runs locally on CPU through ONNX.
# 384 dimensions, so the vector column below must match.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

# Chunking: advisories are short, so most fit in one or two chunks.
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-opus-5-5")
