import json
import math
from datetime import date
from pathlib import Path

from aera.rfx_builder import (
    MAX_FOLLOW_UPS, build_messages, cap_questions, checklist, empty_draft, guard_numbers, is_complete,
    load_vendor_list, next_turn, numbers_in_text, parse_reply, reply_text, rows_to_lines, strip_tags,
    tagged_answers, to_rfx_json,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def _line(**kw):
    base = dict.fromkeys(("description", "ply", "board_spec", "size", "print", "annual_qty", "uom",
                          "nominal_weight_g"))
    return {**base, **kw}


def _complete_draft():
    d = empty_draft()
    d["lines"] = [_line(description="Carton", board_spec="150 GSM", size="300 x 200 x 100 mm", annual_qty=5000)]
    d["terms"]["delivery"] = "Pune plant"
    d["quality_bar"]["iso9001_required"] = True
    return d


# ---------- numbers_in_text ----------

def test_numbers_plain_and_indian_commas():
    assert {73000.0, 110000.0} <= numbers_in_text("73000 of these and 1,10,000 of those")


def test_numbers_with_multipliers():
    found = numbers_in_text("about 50k irons, 1.1 lakh grinders and 2 million fans")
    assert {50000.0, 110000.0, 2000000.0} <= found


def test_units_are_not_multipliers():
    found = numbers_in_text("each box is 128g and 50 kg per bundle")
    assert 128.0 in found and 50.0 in found and 50000.0 not in found


# ---------- guard_numbers ----------

def test_guard_keeps_stated_quantity():
    new = empty_draft()
    new["lines"] = [_line(description="Iron carton", annual_qty=73000)]
    checked, notes = guard_numbers(new, empty_draft(), ["We need 73,000 iron cartons a year"])
    assert checked["lines"][0]["annual_qty"] == 73000
    assert notes == []


def test_guard_blanks_invented_quantity_never_zero():
    new = empty_draft()
    new["lines"] = [_line(description="Iron carton", annual_qty=25000)]
    checked, notes = guard_numbers(new, empty_draft(), ["1 lakh cartons split across 4 products"])
    assert checked["lines"][0]["annual_qty"] is None
    assert "25000" in notes[0]


def test_guard_keeps_numbers_already_in_draft_from_table_edits():
    old = empty_draft()
    old["lines"] = [_line(description="Iron carton", annual_qty=4200, nominal_weight_g=128.0)]
    new = json.loads(json.dumps(old))
    checked, notes = guard_numbers(new, old, ["make the print 2-colour"])
    assert checked["lines"][0]["annual_qty"] == 4200
    assert checked["lines"][0]["nominal_weight_g"] == 128.0
    assert notes == []


def test_guard_blanks_invented_weight_and_defect_rate():
    new = empty_draft()
    new["lines"] = [_line(description="Carton", nominal_weight_g=190.0)]
    new["quality_bar"]["max_defect_rate_pct"] = 2.0
    checked, notes = guard_numbers(new, empty_draft(), ["low defects please"])
    assert checked["lines"][0]["nominal_weight_g"] is None
    assert checked["quality_bar"]["max_defect_rate_pct"] is None
    assert len(notes) == 2


# ---------- rows_to_lines ----------

def test_rows_to_lines_blank_cells_become_none_not_zero():
    rows = [{"description": "Carton", "ply": "", "board_spec": None, "size": " 300 x 200 ",
             "print": float("nan"), "annual_qty": float("nan"), "uom": "pcs", "nominal_weight_g": 128}]
    [line] = rows_to_lines(rows)
    assert line["ply"] is None and line["board_spec"] is None and line["print"] is None
    assert line["annual_qty"] is None
    assert line["size"] == "300 x 200"
    assert line["nominal_weight_g"] == 128.0


def test_rows_to_lines_drops_empty_rows_and_makes_qty_int():
    rows = [{"description": None, "annual_qty": float("nan")}, {"description": "Pad", "annual_qty": 5000.0}]
    lines = rows_to_lines(rows)
    assert len(lines) == 1
    assert lines[0]["annual_qty"] == 5000 and isinstance(lines[0]["annual_qty"], int)


# ---------- checklist ----------

def test_empty_draft_is_incomplete():
    assert not is_complete(empty_draft())


def test_complete_draft():
    assert is_complete(_complete_draft())


def test_checklist_names_lines_missing_qty():
    d = _complete_draft()
    d["lines"].append(_line(description="Pad", board_spec="120 GSM", size="1000 x 1200 mm"))
    items = {name: (ok, detail) for name, ok, detail in checklist(d)}
    assert items["Every line has a quantity"] == (False, "missing on line 2")
    assert not is_complete(d)


def test_checklist_needs_quality_bar_and_delivery():
    d = _complete_draft()
    d["quality_bar"]["iso9001_required"] = False
    d["terms"]["delivery"] = "  "
    items = {name: ok for name, ok, _ in checklist(d)}
    assert not items["Quality bar set"]
    assert not items["Delivery location set"]


# ---------- export ----------

def test_export_matches_sample_rfx_shape():
    out = to_rfx_json(_complete_draft(), date(2026, 10, 1))
    sample = json.loads((SAMPLE / "rfx.json").read_text(encoding="utf-8"))
    assert set(sample) <= set(out)
    assert set(sample["lines"][0]) == set(out["lines"][0])
    assert set(sample["terms"]) == set(out["terms"])
    assert out["lines"][0]["line_id"] == 1
    assert out["quality_bar"]["iso9001_valid_on"] == "2026-10-01"


def test_export_keeps_unknowns_null():
    d = _complete_draft()
    out = to_rfx_json(d, date(2026, 10, 1))
    assert out["lines"][0]["nominal_weight_g"] is None
    assert out["quality_bar"]["max_defect_rate_pct"] is None
    assert out["terms"]["payment"] is None


def test_vendor_list_loads():
    vendors = load_vendor_list(SAMPLE / "vendors.json")
    assert vendors and all(v["name"] and "@" in v["email"] for v in vendors)


# ---------- a whole turn, with a fake Claude ----------

def test_build_messages_puts_draft_on_latest_buyer_message():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "notes": ["x"]},
               {"role": "user", "content": "50k cartons"}]
    msgs = build_messages(history, empty_draft())
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert "notes" not in msgs[1]
    assert msgs[2]["content"].startswith("50k cartons") and "<current_draft>" in msgs[2]["content"]


def test_next_turn_checks_claudes_numbers():
    def fake_claude(system, messages, schema, max_tokens, purpose):
        draft = empty_draft()
        draft["lines"] = [_line(description="Iron carton", annual_qty=50000, size=""),
                          _line(description="Kettle carton", annual_qty=12345)]
        return {"summary": " Added two lines. ", "questions": [_q("Q1", "How many kettle cartons?")],
                "rfx_draft": draft}, {"cost_usd": 0.0}

    history = [{"role": "user", "content": "50k iron cartons and some kettle cartons"}]
    summary, questions, draft, notes, usage = next_turn(history, empty_draft(), call=fake_claude)
    assert summary == "Added two lines."
    assert [q["text"] for q in questions] == ["How many kettle cartons?"]
    assert draft["lines"][0]["annual_qty"] == 50000
    assert draft["lines"][0]["size"] is None
    assert draft["lines"][1]["annual_qty"] is None
    assert len(notes) == 1 and "Kettle carton" in notes[0]
    assert not math.isnan(usage["cost_usd"])


# ---------- Structured questions ----------

def _q(qid, text, field="line 1 annual_qty", example="60,000", propose=False):
    return {"id": qid, "text": text, "example": example, "field": field, "allow_vendors_propose": propose}


def test_parse_reply_with_three_questions():
    parsed = {"summary": "Added the kettle carton line. ",
              "questions": [_q("Q1", "How many kettle cartons a year?"),
                            _q("Q2", "Inside size (L x W x H mm)?", "line 1 size", "310 x 150 x 140"),
                            _q("Q3", "Board grade?", "line 1 board_spec", "150/120/150 GSM", propose=True)],
              "rfx_draft": empty_draft()}
    summary, questions = parse_reply(parsed)
    assert summary == "Added the kettle carton line."
    assert [q["id"] for q in questions] == ["Q1", "Q2", "Q3"]
    assert questions[0]["example"] == "60,000" and questions[1]["field"] == "line 1 size"
    assert [q["allow_vendors_propose"] for q in questions] == [False, False, True]
    assert reply_text(summary, questions).splitlines()[1] == "Q1 (line 1 annual_qty): How many kettle cartons a year?"


def test_parse_reply_enforces_max_three_questions():
    parsed = {"summary": "s", "questions": [_q(f"Q{i}", f"Question {i}?") for i in range(1, 6)]}
    _, questions = parse_reply(parsed)
    assert MAX_FOLLOW_UPS == 3
    assert [q["text"] for q in questions] == ["Question 1?", "Question 2?", "Question 3?"]


def test_cap_questions_drops_blanks_and_renumbers():
    qs = cap_questions([_q("Q1", "  "), _q("Q7", "How many?"), _q("Q2", "Size?")])
    assert [(q["id"], q["text"]) for q in qs] == [("Q1", "How many?"), ("Q2", "Size?")]


def test_tagged_answers_string():
    qs = [_q("Q1", "How many kettle cartons a year?"),
          _q("Q2", "Inside size?", "line 1 size"),
          _q("Q3", "Board grade?", "line 1 board_spec", propose=True)]
    out = tagged_answers(qs, {"Q1": " 60,000 ", "Q2": "", "Q3": ""}, {"Q3": True})
    assert out == "Q1 (line 1 annual_qty): 60,000 / Q2: skipped / Q3 (line 1 board_spec): vendors to propose"


def test_tagged_answers_propose_keeps_typed_note():
    qs = [_q("Q1", "Board grade?", "line 1 board_spec", propose=True)]
    expected = "Q1 (line 1 board_spec): vendors to propose; at least BF 18"
    assert tagged_answers(qs, {"Q1": "at least BF 18"}, {"Q1": True}) == expected


def test_tags_are_not_read_as_stated_numbers():
    text = strip_tags("Q1 (line 1 annual_qty): 60,000 / Q2: skipped")
    assert numbers_in_text(text) == {60000.0}


def test_parse_reply_keeps_only_first_of_a_combined_question():
    parsed = {"summary": "s", "questions": [
        _q("Q1", "What are the kettle carton dimensions (L x W x H, mm)? And how many plies?", "line 1 size"),
        _q("Q2", "Board grade?", "line 1 board_spec")]}
    _, questions = parse_reply(parsed)
    assert [q["text"] for q in questions] == ["What are the kettle carton dimensions (L x W x H, mm)?",
                                              "Board grade?"]
    assert questions[0]["field"] == "line 1 size"
