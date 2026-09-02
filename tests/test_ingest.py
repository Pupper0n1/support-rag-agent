from __future__ import annotations

from scripts.ingest_kb import CHUNK_CHARS, _chunk


def test_short_text_is_one_chunk() -> None:
    assert _chunk("short") == ["short"]


def test_long_text_splits_on_paragraphs_with_overlap() -> None:
    para = "word " * 150  # ~750 chars
    text = "\n\n".join([para.strip()] * 4)
    chunks = _chunk(text)
    assert len(chunks) >= 2
    assert all(len(c) <= CHUNK_CHARS + 200 for c in chunks)
    # Overlap: the tail of chunk n appears at the head of chunk n+1.
    assert chunks[1].startswith(chunks[0][-50:][:20])
