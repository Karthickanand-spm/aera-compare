import pytest

import aera.event as event_mod
import aera.extract as extract_mod
from aera.compare import vendor_name
from aera.event import (
    CERTIFICATE, REPLY, NotAQuote, add_reply, load_sample_event, quote_problem, reextract_all,
)


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
            "quote_check": {"is_quote_for_rfx": True, "not_a_quote_reason": ""},
            "runs": [{"vendor": vendor, "lines": [{"rfx_line_id": 1}], "not_quoted": []}]}


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


# ---------- Uploads that aren't quotes ----------

def _not_a_quote(payload, rfx, force):
    return {"source_file": payload.filename, "sha256": payload.sha256, "file_kind": "pdf", "cost_usd": 0.01,
            "quote_check": {"is_quote_for_rfx": False, "not_a_quote_reason": "it is a canteen lunch menu."},
            "runs": [{"vendor": "Canteen", "lines": [], "not_quoted": []}]}


def test_upload_judged_not_a_quote_is_not_added(no_api, monkeypatch):
    event = load_sample_event()
    vendors_before = [vendor_name(e) for e in event.replies]
    files_before = set(event.files)
    monkeypatch.setattr(event_mod, "extract_reply", _not_a_quote)

    with pytest.raises(NotAQuote, match="it is a canteen lunch menu$"):
        add_reply(event, "menu.eml", b"From: canteen@x.y\n\nThali Rs 80")
    # Every page reads the event, so the vendor appears nowhere.
    assert [vendor_name(e) for e in event.replies] == vendors_before
    assert set(event.files) == files_before


def test_add_anyway_adds_it_from_the_cache(no_api, monkeypatch):
    event = load_sample_event()
    forces = []

    def fake(payload, rfx, force):
        forces.append(force)
        return _not_a_quote(payload, rfx, force)

    monkeypatch.setattr(event_mod, "extract_reply", fake)
    data = b"From: canteen@x.y\n\nThali Rs 80"
    with pytest.raises(NotAQuote):
        add_reply(event, "menu.eml", data)
    assert add_reply(event, "menu.eml", data, force=False, allow_non_quote=True) == (True, None)
    assert vendor_name(event.replies[-1]) == "Canteen"
    assert forces == [True, False]  # the override reuses the cached extraction


def test_empty_extraction_is_rejected_by_code_even_if_claude_says_quote(no_api, monkeypatch):
    event = load_sample_event()
    monkeypatch.setattr(event_mod, "extract_reply", lambda payload, rfx, force: {
        **fake_extraction("Someone", payload.sha256), "runs": [{"vendor": "Someone", "lines": [],
                                                                 "not_quoted": []}]})
    with pytest.raises(NotAQuote, match="no RFx lines"):
        add_reply(event, "blank.eml", b"From: a@b.c\n\nHello")
    assert len(event.replies) == 5


def _run(**kw):
    return {"quote_check": {"is_quote_for_rfx": True, "not_a_quote_reason": ""},
            "runs": [{"vendor": "V", "lines": [], "not_quoted": [], **kw}]}


def test_quote_problem_accepts_terms_or_answers_without_lines():
    rfx = load_sample_event().rfx
    unclear = {"value": "unclear", "source_snippet": ""}
    assert quote_problem(_run(commercial_terms={"freight": unclear, "payment_days": {"value": 45}}), rfx) is None
    assert quote_problem(_run(questionnaire={"lead_time_days": {"value": 10}}), rfx) is None
    assert quote_problem(_run(not_quoted=[{"rfx_line_id": 3, "source_snippet": "We can't make line 3"}]), rfx) is None


def test_quote_problem_ignores_unanswered_fields_and_unknown_lines():
    rfx = load_sample_event().rfx
    run = _run(lines=[{"rfx_line_id": 999}], not_quoted=[{"rfx_line_id": 1, "source_snippet": ""}],
               commercial_terms={"freight": {"value": "unclear"}, "validity": {"value": ""}, "discounts": []},
               questionnaire={"iso9001_claimed": {"value": None}, "test_report_per_batch": {"value": "not_stated"}})
    assert quote_problem(run, rfx) is not None


def test_quote_check_saying_no_wins_even_if_lines_were_matched():
    rfx = load_sample_event().rfx
    ext = {"quote_check": {"is_quote_for_rfx": False, "not_a_quote_reason": ""},
           "runs": [{"lines": [{"rfx_line_id": 1}]}]}
    assert "doesn't offer prices" in quote_problem(ext, rfx)


def test_extract_reply_runs_the_quote_check_and_caches_it(monkeypatch, tmp_path):
    rfx = load_sample_event().rfx
    calls = []

    def fake_claude(system, content, schema):
        calls.append(schema.__name__)
        if schema is extract_mod.QuoteCheck:
            return {"is_quote_for_rfx": False, "not_a_quote_reason": "it is a menu"}, _usage()
        return {"vendor": "Canteen", "lines": [], "not_quoted": []}, _usage()

    monkeypatch.setattr(extract_mod, "_call_claude", fake_claude)
    monkeypatch.setattr(extract_mod, "CACHE_DIR", tmp_path)
    payload = event_mod.load_reply_bytes("menu.eml", b"From: c@x.y\n\nThali Rs 80")
    ext = extract_mod.extract_reply(payload, rfx, force=True)
    assert calls == ["QuoteCheck", "VendorExtraction"]
    assert ext["quote_check"]["is_quote_for_rfx"] is False
    assert quote_problem(extract_mod.extract_reply(payload, rfx), rfx) == "it is a menu"  # from the cache
    assert len(calls) == 2


def _usage():
    return {"model": "m", "input_tokens": 100, "output_tokens": 10, "cache_write_tokens": 0,
            "cache_read_tokens": 0}


def test_old_cached_extraction_without_the_flag_uses_the_code_check():
    rfx = load_sample_event().rfx
    assert quote_problem({"runs": [{"vendor": "V", "lines": [{"rfx_line_id": 1}]}]}, rfx) is None
