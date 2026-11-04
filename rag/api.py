"""HTTP API.

    uvicorn rag.api:app --reload
    GET  /search?q=...&mode=hybrid&n=5   retrieval only, no API key needed
    POST /ask {"question": "...", "n": 5}  retrieval + cited answer from Claude
"""
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .answer import answer
from .db import connect
from .search import search

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["conn"] = connect()
    yield
    state["conn"].close()


app = FastAPI(title="advisory-rag", lifespan=lifespan)


@app.get("/search")
def search_endpoint(q: str, mode: Literal["vector", "keyword", "hybrid"] = "hybrid", n: int = Query(5, ge=1, le=20)):
    hits = search(state["conn"], q, mode, n)
    return [{"advisory_id": h.advisory_id, "score": round(h.score, 4), "text": h.text} for h in hits]


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    n: int = Field(5, ge=1, le=10)


@app.post("/ask")
def ask(req: AskRequest):
    hits = search(state["conn"], req.question, "hybrid", req.n)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise HTTPException(503, "Set ANTHROPIC_API_KEY to enable answers. /search works without it.")
    result = answer(req.question, hits)
    result["retrieved"] = [h.advisory_id for h in hits]
    return result
