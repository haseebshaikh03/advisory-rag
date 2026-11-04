"""Generation step: answer a question from retrieved chunks, with citations.

Each retrieved chunk is passed to Claude as its own document block with citations
enabled, so every claim in the answer points back to the exact text it came from.
The model is told to answer only from those documents and to say so when they do
not contain the answer, which is the main defence against hallucination in RAG.
"""
import anthropic

from .config import ANSWER_MODEL
from .search import Hit

SYSTEM = (
    "You answer questions about Python package security advisories. "
    "Use only the documents provided. If they do not contain the answer, say that "
    "the retrieved advisories do not cover it. Name advisory ids and package versions exactly."
)


def answer(question: str, hits: list[Hit]) -> dict:
    if not hits:
        return {"answer": "No advisories were retrieved for this question.", "citations": []}

    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.text},
            "title": h.advisory_id,
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=ANSWER_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        # On a safety decline, the API reruns the request on a suitable model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    )
    if response.stop_reason == "refusal":
        return {"answer": "The model declined to answer this question.", "citations": []}

    text, citations = [], []
    for block in response.content:
        if block.type != "text":
            continue
        text.append(block.text)
        for c in getattr(block, "citations", None) or []:
            citations.append({"advisory_id": c.document_title, "quote": c.cited_text})
    return {"answer": "".join(text).strip(), "citations": citations}
