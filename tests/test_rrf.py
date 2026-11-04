from rag.search import Hit, dedupe_advisories, rrf


def h(cid, aid="A"):
    return Hit(cid, aid, f"chunk {cid}", 0.0)


def test_item_found_by_both_retrievers_wins():
    vector = [h(1), h(2), h(3)]
    keyword = [h(3), h(4)]
    fused = rrf(vector, keyword)
    assert fused[0].chunk_id == 3  # rank 3 + rank 1 beats rank 1 alone


def test_rrf_ignores_raw_score_scale():
    a = [Hit(1, "A", "", 0.99), Hit(2, "B", "", 0.98)]
    b = [Hit(2, "B", "", 1500.0), Hit(1, "A", "", 3.0)]
    fused = rrf(a, b)
    assert {f.chunk_id for f in fused} == {1, 2}
    assert fused[0].score == fused[1].score  # symmetric ranks, so a tie


def test_dedupe_keeps_best_chunk_per_advisory():
    hits = [h(1, "A"), h(2, "A"), h(3, "B"), h(4, "C")]
    out = dedupe_advisories(hits, 2)
    assert [x.advisory_id for x in out] == ["A", "B"]
    assert out[0].chunk_id == 1
