from dataclasses import replace
from datetime import date

import pytest

from aera import clarify
from aera.analyst import AnalystError
from aera.clarify import (
    ClarifyError, all_open_items, by_severity, draft_email, draft_prompt, format_due, open_items, reply_by,
    ticked_by_default, vendors_mentioned,
)
from aera.compare import HIGH, LOW, MEDIUM, USE_EXTRACTED, buyer_decision, compare
from aera.rfx import RFx, RfxLine

FX = {"USD": 94.50}


def make_rfx(n_lines: int = 3) -> RFx:
    lines = [
        RfxLine(line_id=i, description=f"Carton {i}", ply="3-ply", board_spec="150 GSM",
                size="100 x 100 x 100 mm", print="Plain", annual_qty=1000, uom="pcs",
                nominal_weight_g=100.0)
        for i in range(1, n_lines + 1)
    ]
    return RFx(rfx_id="T-1", buyer="Buyer Co", title="Test", issued="2026-09-16", due="2026-09-25",
               terms={"validity": "12 months"},
               quality_bar={"iso9001_valid_on": "2026-09-16", "max_defect_rate_pct": 2.0,
                            "test_report_per_batch": True},
               questionnaire=[], lines=lines)


def line(line_id, price, raw=None, **extra):
    raw = raw if raw is not None else f"{price}"
    return {"rfx_line_id": line_id, "raw_price_text": raw, "price": price, "currency": "INR",
            "unit_basis": "per_piece", "pack_size": None, "vendor_qty_text": None,
            "vendor_line_total_text": None, "quoted_spec_if_different": None, "source_snippet": raw,
            "page": None, "interpretation_note": None, "is_ambiguous": False, "candidate_group": None,
            "ambiguity_reason": None, **extra}


def questionnaire(**overrides):
    q = {
        "iso9001_claimed": {"value": True, "source_snippet": "ISO yes"},
        "cert_valid_until_claimed": {"value": None, "iso_date": None, "source_snippet": None},
        "defect_rate_pct": {"value": 1.0, "raw_text": "1%", "source_snippet": "1%"},
        "test_report_per_batch": {"value": "yes", "source_snippet": "yes"},
        "capacity_t_month": {"value": 500.0, "raw_text": "500 T", "source_snippet": "500 T"},
        "lead_time_days": {"value": 10.0, "raw_text": "10 days", "source_snippet": "10 days"},
    }
    q.update(overrides)
    return q


def reply(lines, vendor="Acme Boxes", not_quoted=None, freight="included", freight_words=None,
          q=None, payment_days=30.0):
    run = {
        "vendor": vendor, "lines": lines, "not_quoted": not_quoted or [],
        "commercial_terms": {
            "freight": {"value": freight, "source_snippet": freight_words},
            "payment_days": {"value": payment_days, "raw_text": None, "source_snippet": None},
            "validity": {"value": None, "source_snippet": None}, "discounts": [], "other_conditions": [],
        },
        "questionnaire": q or questionnaire(),
    }
    return {"source_file": "acme.eml", "file_kind": "email", "runs": [run]}


def cert(holder="Acme Boxes", valid_until="2028-01-01"):
    return {"source_file": "acme_cert.pdf",
            "runs": [{"holder": holder, "standard": "ISO 9001:2015",
                      "valid_until": {"value": valid_until, "iso_date": valid_until,
                                      "source_snippet": valid_until}}]}


def items_for(ext, certs=None, rfx=None, decisions=None, missing=None):
    rfx = rfx or make_rfx()
    certs = [cert()] if certs is None else certs
    comparison, _ = compare(rfx, {}, [ext], certs, FX, "2026-09-25", None, decisions)
    return open_items(rfx, comparison, ext, certs, missing)


def keys(items):
    return [i["key"] for i in items]


# ---------- Dates ----------

def test_reply_by_skips_the_weekend():
    assert reply_by(date(2026, 10, 1)) == date(2026, 10, 6)  # Thursday -> Tuesday
    assert reply_by(date(2026, 10, 2)) == date(2026, 10, 7)  # Friday -> Wednesday
    assert reply_by(date(2026, 10, 3)) == date(2026, 10, 7)  # Saturday -> Wednesday
    assert reply_by(date(2026, 10, 5)) == date(2026, 10, 8)  # Monday -> Thursday


def test_format_due():
    assert format_due(date(2026, 10, 6)) == "Tuesday 6 October 2026"


# ---------- Open items ----------

def test_clean_vendor_has_nothing_to_clarify():
    assert items_for(reply([line(1, 10), line(2, 11), line(3, 12)])) == []


def test_not_quoted_lines_with_the_vendors_words():
    ext = reply([line(1, 10)], not_quoted=[
        {"rfx_line_id": 2, "reason": "not in range", "source_snippet": "Item 2 not in our range"}])
    items = items_for(ext)
    # Line 3 was skipped silently (medium); line 2 was explicitly declined (low), so it sorts last.
    assert keys(items) == ["not_quoted:3", "not_quoted:2"]
    assert [i["severity"] for i in items] == [MEDIUM, LOW]
    assert items[1]["text"] == "Line 2 (Carton 2): not quoted"
    assert items[1]["vendor_words"] == "Item 2 not in our range"
    assert items[0]["vendor_words"] is None


def test_spec_difference_names_both_specs():
    ext = reply([line(1, 10, quoted_spec_if_different="120 GSM instead of 150"), line(2, 11), line(3, 12)])
    [spec] = items_for(ext)
    assert spec["key"] == "spec:1"
    assert "120 GSM instead of 150" in spec["text"]
    assert "the RFx asks for 3-ply, 150 GSM, 100 x 100 x 100 mm, Plain" in spec["text"]


def test_unusable_price_is_asked_about():
    ext = reply([line(1, 10, raw="Rs 10 per lot", unit_basis="other"), line(2, 11), line(3, 12)])
    [price] = items_for(ext)
    assert price["key"] == "price:1"
    assert "'Rs 10 per lot'" in price["text"]


def test_ambiguous_line_lists_both_readings_until_confirmed():
    ext = reply([line(1, 10, raw="10 for the 3-ply", is_ambiguous=True, candidate_group="g"),
                 line(1, 12, raw="same as before", is_ambiguous=True, candidate_group="g"),
                 line(2, 11), line(3, 12)])
    [amb] = items_for(ext)
    assert amb["key"] == "ambiguous:1"
    assert "'10 for the 3-ply'" in amb["text"] and "'same as before'" in amb["text"]

    decided = {(1, "Acme Boxes"): buyer_decision(USE_EXTRACTED, "2026-10-01 10:00 UTC")}
    assert items_for(ext, decisions=decided) == []


def test_freight_extra_and_unclear():
    full = [line(1, 10), line(2, 11), line(3, 12)]
    [extra] = items_for(reply(full, freight="extra", freight_words="Freight extra at actuals"))
    assert extra["key"] == "freight" and "Freight is extra" in extra["text"]
    assert extra["vendor_words"] == "Freight extra at actuals"
    [unclear] = items_for(reply(full, freight="unclear"))
    assert "unclear" in unclear["text"]
    assert items_for(reply(full, freight=None))[0]["key"] == "freight"


def test_missing_and_on_request_answers():
    q = questionnaire(capacity_t_month={"value": None, "raw_text": None, "source_snippet": None},
                      test_report_per_batch={"value": "on_request", "source_snippet": "Reports on request"})
    items = items_for(reply([line(1, 10), line(2, 11), line(3, 12)], q=q, payment_days=None))
    # The test report is in the quality bar, so it blocks the decision and comes first.
    assert keys(items) == ["answer:test_report_per_batch", "answer:capacity_t_month", "answer:payment_days"]
    assert [i["severity"] for i in items] == [HIGH, MEDIUM, MEDIUM]
    assert items[0]["text"].startswith("Only 'on request'")
    assert items[0]["vendor_words"] == "Reports on request"


def test_quality_bar_answer_is_medium_when_not_required():
    rfx = replace(make_rfx(), quality_bar={"test_report_per_batch": False})
    q = questionnaire(test_report_per_batch={"value": "on_request", "source_snippet": "on request"},
                      defect_rate_pct={"value": None, "raw_text": None, "source_snippet": None})
    items = items_for(reply([line(1, 10), line(2, 11), line(3, 12)], q=q), rfx=rfx)
    assert {i["key"]: i["severity"] for i in items} == {
        "answer:defect_rate_pct": MEDIUM, "answer:test_report_per_batch": MEDIUM}


def test_certificate_expiring_mid_contract():
    [c] = items_for(reply([line(1, 10), line(2, 11), line(3, 12)]), certs=[cert(valid_until="2027-08-31")])
    assert c["key"] == "certificate"
    assert "expires on 31 August 2027, during the 12-month contract" in c["text"]
    assert c["severity"] == LOW  # more than 6 months into the contract


def test_certificate_expiring_soon_is_high():
    [c] = items_for(reply([line(1, 10), line(2, 11), line(3, 12)]), certs=[cert(valid_until="2026-12-31")])
    assert c["severity"] == HIGH


def test_certificate_expired():
    [c] = items_for(reply([line(1, 10), line(2, 11), line(3, 12)]), certs=[cert(valid_until="2026-03-15")])
    assert "expired on 15 March 2026" in c["text"]
    assert c["severity"] == HIGH


def test_no_certificate_attached():
    [c] = items_for(reply([line(1, 10), line(2, 11), line(3, 12)]), certs=[])
    assert c["key"] == "answer:iso9001"
    assert "not attached" in c["text"]
    assert c["severity"] == HIGH


def test_items_sorted_high_first_and_low_unticked():
    ext = reply([line(1, 10, quoted_spec_if_different="120 GSM")], freight="unclear",
                not_quoted=[{"rfx_line_id": 2, "reason": "", "source_snippet": "No line 2"}])
    items = items_for(ext, certs=[cert(valid_until="2027-08-31")],
                      missing=["Acme's freight rate per kg"])
    assert [(i["key"].split(":")[0], i["severity"]) for i in items] == [
        ("spec", HIGH),
        ("not_quoted", MEDIUM), ("freight", MEDIUM), ("analyst", MEDIUM),  # line 3 skipped silently
        ("not_quoted", LOW), ("certificate", LOW),
    ]
    assert [ticked_by_default(i) for i in items] == [True, True, True, True, False, False]


def test_by_severity_keeps_order_within_a_severity():
    a, b, c = (clarify.item("x", t, severity=s) for t, s in (("a", LOW), ("b", HIGH), ("c", LOW)))
    assert [i["text"] for i in by_severity([a, b, c])] == ["b", "a", "c"]


def test_unknown_severity_is_rejected():
    with pytest.raises(ValueError):
        clarify.item("x", "text", severity="urgent")


def test_analyst_missing_data_added_once():
    items = items_for(reply([line(1, 10), line(2, 11), line(3, 12)]),
                      missing=["Acme's freight rate per kg", "acme's freight rate per kg", " "])
    assert [i["text"] for i in items] == ["Acme's freight rate per kg"]
    assert items[0]["kind"] == "analyst"


def test_all_open_items_keyed_by_vendor():
    rfx = make_rfx()
    a = reply([line(1, 10), line(2, 11), line(3, 12)], vendor="Acme Boxes")
    b = reply([line(1, 10)], vendor="Bolt Cartons")
    certs = [cert("Acme Boxes"), cert("Bolt Cartons")]
    comparison, _ = compare(rfx, {}, [a, b], certs, FX, "2026-09-25")
    out = all_open_items(rfx, comparison, [a, b], certs, {"Bolt Cartons": ["Bolt's freight"]})
    assert out["Acme Boxes"] == []
    assert keys(out["Bolt Cartons"])[:2] == ["not_quoted:2", "not_quoted:3"]
    assert out["Bolt Cartons"][-1]["text"] == "Bolt's freight"


def test_vendors_mentioned():
    names = ["Ganesh Packaging Industries", "Deccan Corrupack Pvt Ltd"]
    assert vendors_mentioned("Ganesh's freight charge per kg", names) == ["Ganesh Packaging Industries"]
    assert vendors_mentioned("freight for every vendor", names) == []


# ---------- Drafting (fake Claude) ----------

class FakeClaude:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, system, messages, schema, max_tokens, purpose):
        self.calls.append({"system": system, "messages": messages, "purpose": purpose})
        return self.response, {"cost_usd": 0.001, "purpose": purpose}


TODAY = date(2026, 10, 1)
DUE = "Tuesday 6 October 2026"
ITEMS = [clarify.item("not_quoted", "Line 2 (Carton 2): not quoted", key="not_quoted:2"),
         clarify.item("freight", "Freight is extra", "freight extra", key="freight")]
REPLY_TEXT = "Our rates are below. Freight extra.\nWarm regards,\nV. Patil\nAcme Boxes"


def response(body=f"Please quote line 2 and your freight charge. Please reply by {DUE}.",
             name="V. Patil", snippet="Warm regards,\nV. Patil"):
    return {"contact_name": name, "contact_snippet": snippet, "subject": "RFx T-1: clarification",
            "body": body}


def test_prompt_groups_items_by_severity():
    items = [clarify.item("certificate", "Cert expires later", severity=LOW, key="certificate"),
             clarify.item("spec", "Line 1 (Carton 1): spec differs", "120 GSM", "spec:1", HIGH),
             clarify.item("answer", "Not provided: lead time", severity=MEDIUM)]
    prompt = draft_prompt(make_rfx(), "Acme Boxes", by_severity(items), DUE)
    blocking = prompt.index("BLOCKING THE DECISION")
    needed = prompt.index("ALSO NEEDED")
    records = prompt.index("FOR OUR RECORDS")
    assert blocking < prompt.index("1. Line 1 (Carton 1): spec differs") < needed
    assert needed < prompt.index("2. Not provided: lead time") < records < prompt.index("3. Cert expires later")
    assert "Before we can finalise, we need" in prompt and "Also, for our records" in prompt


def test_prompt_skips_empty_groups():
    prompt = draft_prompt(make_rfx(), "Acme Boxes", [clarify.item("x", "Lead time", severity=MEDIUM)], DUE)
    assert "ALSO NEEDED" in prompt
    assert "BLOCKING" not in prompt and "FOR OUR RECORDS" not in prompt


SEVERITY_ITEMS = [clarify.item("cert", "Cert expires later", severity=LOW, key="certificate"),
                  clarify.item("spec", "Line 1 (Carton 1): spec differs", severity=HIGH, key="spec:1")]


def test_draft_in_the_right_order_has_no_order_notes():
    body = (f"Before we can finalise, we need your price for line 1 in the RFx spec. "
            f"Also, for our records, please share renewal plans. Please reply by {DUE}.")
    d = draft_email(make_rfx(), "Acme Boxes", SEVERITY_ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body=body)))
    assert d.item_keys == ["spec:1", "certificate"]  # sorted high first
    assert not any("Before we can finalise" in n or "for our records" in n.casefold() or "reorder" in n
                   for n in d.notes)


def test_draft_missing_lead_ins_or_wrong_order_is_flagged():
    d = draft_email(make_rfx(), "Acme Boxes", SEVERITY_ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body=f"Please quote line 1. Reply by {DUE}.")))
    assert any("Before we can finalise" in n for n in d.notes)
    assert any("Also, for our records" in n for n in d.notes)

    backwards = (f"Also, for our records, please share renewal plans. Before we can finalise, we need "
                 f"line 1 in the RFx spec. Reply by {DUE}.")
    d = draft_email(make_rfx(), "Acme Boxes", SEVERITY_ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body=backwards)))
    assert any("reorder" in n for n in d.notes)


def test_draft_uses_checked_contact_and_code_greeting():
    fake = FakeClaude(response())
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [{"type": "text", "text": REPLY_TEXT}],
                    REPLY_TEXT, "email", TODAY, call=fake)
    assert d.text.startswith("Dear V. Patil,\n\n")
    assert d.text.endswith("Regards,\nPurchase team\nBuyer Co")
    assert d.due == DUE and d.contact_name == "V. Patil"
    assert d.item_keys == ["not_quoted:2", "freight"]
    assert any("found in the reply" in n for n in d.notes)
    # Claude saw the reply, every ticked item with the vendor's words, and the due date.
    content = fake.calls[0]["messages"][0]["content"]
    assert content[0]["text"] == REPLY_TEXT
    prompt = content[-1]["text"]
    assert "1. Line 2 (Carton 2): not quoted" in prompt and "Vendor's words: 'freight extra'" in prompt
    assert f"Reply by: {DUE}" in prompt
    assert fake.calls[0]["purpose"] == "clarification"


def test_contact_not_in_reply_text_falls_back_to_team():
    d = draft_email(make_rfx(), "ACME BOXES", ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(name="R. Sharma", snippet="R. Sharma")))
    assert d.text.startswith("Dear Acme Boxes team,")
    assert d.contact_name is None
    assert any("could not be found" in n for n in d.notes)


def test_contact_from_photo_is_used_but_flagged():
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [], None, "image", TODAY, call=FakeClaude(response()))
    assert d.text.startswith("Dear V. Patil,")
    assert any("read from the photo" in n for n in d.notes)


def test_no_contact_name():
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(name="", snippet="")))
    assert d.text.startswith("Dear Acme Boxes team,")


def test_missing_due_date_is_added_by_code():
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body="Please quote line 2 and freight.")))
    assert f"Please reply by {DUE}." in d.text
    assert any("code added" in n for n in d.notes)


def test_unmentioned_line_and_long_draft_are_flagged():
    long_body = "Please send your freight charge. " + "word " * 160 + f"Reply by {DUE}."
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body=long_body)))
    assert any("does not mention line(s) 2" in n for n in d.notes)
    assert any("over the 150-word limit" in n for n in d.notes)


def test_markdown_is_stripped():
    d = draft_email(make_rfx(), "Acme Boxes", ITEMS, [], REPLY_TEXT, "email", TODAY,
                    call=FakeClaude(response(body=f"**Line 2**: please quote. Reply by {DUE}.")))
    assert "**" not in d.text


def test_nothing_ticked_raises():
    with pytest.raises(ClarifyError):
        draft_email(make_rfx(), "Acme Boxes", [], [], None, None, TODAY, call=FakeClaude(response()))


def test_api_problem_becomes_clarify_error():
    def failing(*args):
        raise AnalystError("Rate limited by the Claude API. Wait a minute and try again.")
    with pytest.raises(ClarifyError, match="Rate limited"):
        draft_email(make_rfx(), "Acme Boxes", ITEMS, [], None, None, TODAY, call=failing)
