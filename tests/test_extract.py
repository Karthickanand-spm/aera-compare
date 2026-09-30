"""Offline tests for extract.py: no Claude calls, only the plain-Python helpers."""

from pathlib import Path

import pytest

from aera import extract
from aera.ingest import Payload
from aera.rfx import load_rfx

RFX = load_rfx(Path(__file__).resolve().parent.parent / "data" / "sample" / "rfx.json")


def test_blanks_become_none_but_zero_and_false_stay():
    raw = {"a": "", "b": "  ", "c": 0, "d": False, "e": [{"f": ""}, "x"], "g": "text"}
    assert extract.blanks_to_none(raw) == {
        "a": None, "b": None, "c": 0, "d": False, "e": [{"f": None}, "x"], "g": "text"
    }


def test_cost_usd_uses_config_prices(monkeypatch):
    monkeypatch.setattr(
        extract, "PRICE_USD_PER_MTOK",
        {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2},
    )
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000,
             "cache_write_tokens": 0, "cache_read_tokens": 500_000}
    assert extract.cost_usd(usage) == pytest.approx(2.0 + 1.0 + 0.1)


def test_rfx_context_lists_every_line():
    text = extract.rfx_context(RFX)
    for ln in RFX.lines:
        assert f"\n{ln.line_id} | {ln.description} |" in text


def _payload(sha="abc123"):
    return Payload("f.docx", "word", sha, "hello", [{"type": "text", "text": "hello"}])


def test_cache_hit_skips_claude(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    calls = []

    def fake_call(system, content, schema):
        calls.append(1)
        return {"vendor": "X", "lines": [{"rfx_line_id": 1}], "not_quoted": []}, {
            "model": "m", "input_tokens": 1, "output_tokens": 1,
            "cache_write_tokens": 0, "cache_read_tokens": 0,
        }

    monkeypatch.setattr(extract, "_call_claude", fake_call)
    first = extract.extract_reply(_payload(), RFX)
    second = extract.extract_reply(_payload(), RFX)
    assert len(calls) == 1
    assert first["from_cache"] is False and second["from_cache"] is True
    assert second["runs"][0]["lines"][0]["source_file"] == "f.docx"

    extract.extract_reply(_payload(), RFX, force=True)  # Re-extract ignores the cache
    assert len(calls) == 2


def test_images_are_extracted_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(extract, "_call_claude", lambda s, c, m: (
        {"vendor": "X", "lines": [], "not_quoted": []},
        {"model": "m", "input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0},
    ))
    photo = Payload("p.jpg", "image", "img1", None, [])
    assert len(extract.extract_reply(photo, RFX)["runs"]) == 2


def test_stale_cache_version_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    (tmp_path / "abc123.json").write_text(
        '{"extractor_version": -1, "extraction_type": "reply", "rfx_id": "%s"}' % RFX.rfx_id,
        encoding="utf-8",
    )
    assert extract._read_cache("abc123", "reply", RFX.rfx_id) is None
