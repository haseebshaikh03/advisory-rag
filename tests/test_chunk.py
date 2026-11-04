import pytest

from rag.chunk import chunk_advisory, clean, windows

ADV = {
    "id": "GHSA-test-0000-0000",
    "cve": "CVE-2026-0001",
    "summary": "demo: path traversal in upload handler",
    "description": "",
    "severity": "high",
    "packages": ["demo"],
    "affected": ["demo < 1.2.0 (fixed in 1.2.0)"],
    "cwes": [{"id": "CWE-22", "name": "Path Traversal"}],
}


def test_windows_cover_every_word_with_overlap():
    words = [str(i) for i in range(500)]
    parts = windows(words, 180, 40)
    assert parts[0][0] == "0" and parts[-1][-1] == "499"
    assert parts[1][0] == "140"  # step = size - overlap
    assert all(len(p) <= 180 for p in parts)


def test_short_text_is_one_window():
    assert windows(["a", "b"], 180, 40) == [["a", "b"]]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        windows(["a"] * 10, 5, 5)


def test_every_chunk_carries_the_header():
    adv = {**ADV, "description": " ".join(["word"] * 400)}
    chunks = chunk_advisory(adv, 180, 40)
    assert len(chunks) == 3
    for c in chunks:
        assert "GHSA-test-0000-0000 / CVE-2026-0001" in c["text"]
        assert "Path Traversal" in c["text"]


def test_advisory_without_description_still_gets_a_chunk():
    chunks = chunk_advisory(ADV)
    assert len(chunks) == 1 and chunks[0]["ord"] == 0


def test_clean_drops_code_blocks():
    assert "rm -rf" not in clean("before ```bash\nrm -rf /\n``` after")
