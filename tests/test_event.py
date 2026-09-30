import pytest

import aera.event as event_mod
import aera.extract as extract_mod
from aera.event import CERTIFICATE, REPLY, add_reply, load_sample_event, reextract_all


@pytest.fixture
def no_api(monkeypatch):
    """Any real Claude call fails the test."""
    def boom(*args, **kwargs):
        raise AssertionError("Claude was called")
    monkeypatch.setattr(extract_mod, "_call_claude", boom)


def test_sample_event_loads_from_cache_without_api_calls(no_api):
    event = load_sample_event()
    assert event.errors == []
    assert len(event.replies) == 5
    assert len(event.certificates) == 5
    assert all(e["from_cache"] for e in event.replies + event.certificates)
    roles = {f.role for f in event.files.values()}
    assert roles == {REPLY, CERTIFICATE}
    # Text formats keep their text for snippet checks; PDFs and photos don't have any.
    kinds = {f.kind: f.text for f in event.files.values()}
    assert kinds["excel"] and kinds["word"] and kinds["email"]
    assert kinds["pdf"] is None and kinds["image"] is None


def fake_extraction(vendor, sha):
    return {"source_file": "x", "sha256": sha, "file_kind": "email", "cost_usd": 0.01,
            "runs": [{"vendor": vendor, "lines": [], "not_quoted": []}]}


def test_add_reply_adds_then_replaces_same_vendor(no_api, monkeypatch):
    event = load_sample_event()
    calls = []

    def fake_extract(payload, rfx, force):
        calls.append(force)
        return {**fake_extraction("New Vendor Ltd", payload.sha256), "source_file": payload.filename}

    monkeypatch.setattr(event_mod, "extract_reply", fake_extract)
    assert add_reply(event, "new.eml", b"From: a@b.c\n\nRate Rs 5") == (True, None)
    assert len(event.replies) == 6
    assert calls == [True]  # uploads always run live

    added, note = add_reply(event, "new_v2.eml", b"From: a@b.c\n\nRate Rs 4")
    assert added and "Replaced" in note
    assert len(event.replies) == 6
    assert event.replies[-1]["source_file"] == "new_v2.eml"


def test_add_reply_same_file_twice_is_skipped(no_api, monkeypatch):
    event = load_sample_event()
    monkeypatch.setattr(event_mod, "extract_reply",
                        lambda payload, rfx, force: fake_extraction("V", payload.sha256))
    data = b"From: a@b.c\n\nRate Rs 5"
    add_reply(event, "v.eml", data)
    added, note = add_reply(event, "v_copy.eml", data)
    assert not added and "already in the comparison" in note
    assert len(event.replies) == 6


def test_add_reply_rejects_unsupported_type(no_api):
    event = load_sample_event()
    with pytest.raises(ValueError, match="Unsupported"):
        add_reply(event, "quote.txt", b"hello")


def test_replaced_upload_is_not_brought_back_by_reextract(no_api, monkeypatch):
    event = load_sample_event()
    monkeypatch.setattr(event_mod, "extract_reply", lambda payload, rfx, force: {
        **fake_extraction("New Vendor Ltd", payload.sha256), "source_file": payload.filename})
    add_reply(event, "new.eml", b"From: a@b.c\n\nRate Rs 5")
    add_reply(event, "new_v2.eml", b"From: a@b.c\n\nRate Rs 4")
    assert "new.eml" not in event.files
    assert "new_v2.eml" in event.files


def test_failed_reextract_keeps_earlier_extractions(no_api):
    # no_api makes every Claude call fail, like a network outage.
    event = load_sample_event()
    fresh = reextract_all(event)
    assert len(fresh.replies) == len(event.replies)
    assert len(fresh.certificates) == len(event.certificates)
    assert len(fresh.errors) == len(event.files)
    assert all("Kept the earlier extraction" in e for e in fresh.errors)
