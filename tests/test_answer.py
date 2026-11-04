from types import SimpleNamespace as NS

from rag import answer as answer_mod
from rag.search import Hit


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(monkeypatch, response):
    messages = FakeMessages(response)
    monkeypatch.setattr(answer_mod.anthropic, "Anthropic", lambda: NS(beta=NS(messages=messages)))
    return messages


def test_each_hit_becomes_a_citable_document(monkeypatch):
    resp = NS(stop_reason="end_turn", content=[
        NS(type="text", text="Versions before 1.2.0 are affected.",
           citations=[NS(document_title="GHSA-a", cited_text="demo < 1.2.0")]),
    ])
    messages = fake_client(monkeypatch, resp)
    out = answer_mod.answer("which versions?", [Hit(1, "GHSA-a", "demo < 1.2.0", 1.0), Hit(2, "GHSA-b", "other", 0.5)])

    docs = [b for b in messages.kwargs["messages"][0]["content"] if b["type"] == "document"]
    assert [d["title"] for d in docs] == ["GHSA-a", "GHSA-b"]
    assert all(d["citations"] == {"enabled": True} for d in docs)
    assert out["citations"] == [{"advisory_id": "GHSA-a", "quote": "demo < 1.2.0"}]


def test_refusal_is_reported_not_returned_as_an_answer(monkeypatch):
    fake_client(monkeypatch, NS(stop_reason="refusal", content=[]))
    out = answer_mod.answer("q", [Hit(1, "GHSA-a", "t", 1.0)])
    assert "declined" in out["answer"]


def test_no_hits_skips_the_model_call(monkeypatch):
    messages = fake_client(monkeypatch, None)
    out = answer_mod.answer("q", [])
    assert messages.kwargs is None and out["citations"] == []
