import json
import math
from datetime import date
from pathlib import Path

from aera.rfx_builder import (
    build_messages, checklist, empty_draft, guard_numbers, is_complete, load_vendor_list, next_turn,
    numbers_in_text, rows_to_lines, to_rfx_json,
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
        return {"reply_to_buyer": " How many kettle cartons? ", "rfx_draft": draft}, {"cost_usd": 0.0}

    history = [{"role": "user", "content": "50k iron cartons and some kettle cartons"}]
    reply, draft, notes, usage = next_turn(history, empty_draft(), call=fake_claude)
    assert reply == "How many kettle cartons?"
    assert draft["lines"][0]["annual_qty"] == 50000
    assert draft["lines"][0]["size"] is None
    assert draft["lines"][1]["annual_qty"] is None
    assert len(notes) == 1 and "Kettle carton" in notes[0]
    assert not math.isnan(usage["cost_usd"])
