"""Local embeddings with fastembed (ONNX on CPU, no API key).

BGE models expect a short instruction in front of queries but not in front of
documents, which is why there are two functions.
"""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from .config import EMBED_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def model() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return list(model().embed(texts, batch_size=32))


def embed_query(text: str) -> np.ndarray:
    return next(iter(model().embed([QUERY_PREFIX + text])))
