"""Offline tests for analyst.py: the code sandbox, money formatting, and the
classify -> compute -> write -> validate pipeline.

Claude is replaced by a fake caller, so no API calls are made.
"""

import io
import json

import pandas as pd
import pytest

from aera import analyst
from aera.analyst import (
    AnalystData, CodeError, annual_weight_kg, check_code, display_table, format_inr, money_header,
    run_code,
)


def _data() -> AnalystData:
    df = pd.DataFrame([
        {"rfx_line_id": 1, "vendor": "A", "display_name": "Alpha", "price_inr_per_piece": 10.0,
         "label": "Comparable", "included_in_totals": True, "needs_review": False,
         "buyer_confirmed": False, "assumptions": []},
        {"rfx_line_id": 2, "vendor": "A", "display_name": "Alpha", "price_inr_per_piece": None,
         "label": "Not quoted", "included_in_totals": False, "needs_review": False,
         "buyer_confirmed": False, "assumptions": []},
        {"rfx_line_id": 1, "vendor": "B", "display_name": "Beta", "price_inr_per_piece": 12.5,
         "label": "Comparable with assumption", "included_in_totals": True, "needs_review": True,
         "buyer_confirmed": False, "assumptions": ["USD at 94.5"]},
    ])
    vendors = pd.DataFrame([
        {"vendor": "A", "display_name": "Alpha", "quality_status": "PASS", "freight": "included"},
        {"vendor": "B", "display_name": "Beta", "quality_status": "UNCLEAR", "freight": "extra"},
        {"vendor": "C", "display_name": "Gamma", "quality_status": "FAIL", "freight": "included"},
    ])
    rfx_lines = pd.DataFrame([{"line_id": 1, "description": "Box", "annual_qty": 1000,
                               "uom": "pcs", "nominal_weight_g": 300.0},
                              {"line_id": 2, "description": "Lid", "annual_qty": 500,
                               "uom": "pcs", "nominal_weight_g": 100.0}])
    last_year = pd.DataFrame([{"line_id": 1, "price_inr_per_piece": 11.0}])
    return AnalystData(df, vendors, rfx_lines, last_year, {"USD": 94.5}, "2026-09-25")


# ---------- Sandbox: what is rejected ----------

@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "from os import path\nresult = 1",
    "result = open('x.txt').read()",
    "result = eval('1+1')",
    "result = __import__('os')",
    "result = df.__class__",
    "result = (1).__class__.__subclasses__()",
    "result = '{0.__class__}'.format(df)",
    "p = pd\nresult = p.read_csv('x.csv')",
    "result = pd.read_csv('http://example.com/x.csv')",
    "df.to_csv('out.csv')\nresult = 1",
    "result = df.query('price_inr_per_piece > 1')",
    "result = getattr(df, 'shape')",
    "try:\n    result = 1\nexcept:\n    pass",
    "result = (x for x in []).gi_frame",
])
def test_unsafe_code_is_rejected(code):
    with pytest.raises(CodeError):
        check_code(code)


def test_ordinary_pandas_code_is_allowed():
    code = (
        "m = df[df['included_in_totals']].merge(rfx_lines, left_on='rfx_line_id', right_on='line_id')\n"
        "m['annual_cost_inr'] = m['annual_qty'] * m['price_inr_per_piece']\n"
        "result = m.groupby('display_name', as_index=False).agg("
        "annual_cost_inr=('annual_cost_inr', 'sum'), lines_in_total=('rfx_line_id', 'count')).round(2)"
    )
    result = run_code(code, _data().namespace())
    assert result.to_dict(orient="records") == [
        {"display_name": "Alpha", "annual_cost_inr": 10000.0, "lines_in_total": 1},
        {"display_name": "Beta", "annual_cost_inr": 12500.0, "lines_in_total": 1},
    ]


def test_code_must_assign_result():
    with pytest.raises(CodeError, match="result"):
        run_code("x = 1", _data().namespace())


def test_runtime_error_is_reported_with_its_type():
    with pytest.raises(CodeError, match="KeyError"):
        run_code("result = df['no_such_column']", _data().namespace())


def test_slow_code_is_stopped_even_if_it_catches_exceptions():
    code = "while True:\n    try:\n        x = 1\n    except Exception:\n        pass"
    with pytest.raises(CodeError, match="longer than"):
        run_code(code, _data().namespace(), timeout=0.3)


def test_code_cannot_change_the_original_data():
    data = _data()
    run_code("df.loc[0, 'price_inr_per_piece'] = 0\ndf.loc[2, 'assumptions'].append('hacked')\n"
             "vendors.drop(columns=['vendor'], inplace=True)\nresult = 1", data.namespace())
    assert data.df.loc[0, "price_inr_per_piece"] == 10.0
    assert data.df.loc[2, "assumptions"] == ["USD at 94.5"]
    assert "vendor" in data.vendors.columns


def test_missing_price_stays_missing_in_the_namespace():
    result = run_code("result = df['price_inr_per_piece'].isna().sum()", _data().namespace())
    assert result == 1


# ---------- Money format ----------

@pytest.mark.parametrize("value, expected", [
    (26_700_000, "₹2.67 crore"),
    (10_000_000, "₹1.00 crore"),
    (1_234_567_890_123, "₹1,23,456.79 crore"),
    (9_999_999, "₹1.00 crore"),  # 99.99999 lakh rounds to 100 lakh: say crore, not "100.00 lakh"
    (9_960_000, "₹99.60 lakh"),
    (431_000, "₹4.31 lakh"),
    (100_000, "₹1.00 lakh"),
    (99_999.6, "₹1.00 lakh"),  # rounds up to 1 lakh
    (99_999, "₹99,999"),
    (35_780, "₹35,780"),
    (35_780.5, "₹35,781"),  # half rounds up
    (999, "₹999"),
    (0, "₹0"),
    (-431_000, "-₹4.31 lakh"),
    (-0.4, "₹0"),  # no "-₹0"
])
def test_format_inr_amounts(value, expected):
    assert format_inr(value) == expected


@pytest.mark.parametrize("value, expected", [
    (5.14, "₹5.14"),
    (5.145, "₹5.15"),  # half up, not banker's rounding
    (5, "₹5.00"),
    (0.65, "₹0.65"),
    (1_234_567.5, "₹12,34,567.50"),  # never lakh/crore for a per-unit price
    (-2.5, "-₹2.50"),
])
def test_format_inr_per_unit_keeps_two_decimals(value, expected):
    assert format_inr(value, per_unit=True) == expected


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, float("inf"), "abc"])
def test_format_inr_missing_is_a_dash_never_zero(value):
    assert format_inr(value) == "—"
    assert format_inr(value, per_unit=True) == "—"


def test_money_headers_follow_the_column_suffix():
    assert money_header("annual_cost_inr") == "annual_cost (₹)"
    assert money_header("price_inr_per_piece") == "price (₹/piece)"
    assert money_header("breakeven_freight_inr_per_kg") == "breakeven_freight (₹/kg)"
    assert money_header("lines_in_total") == "lines_in_total"
    assert money_header("savings_pct") == "savings_pct"


def test_display_table_formats_only_money_columns_and_keeps_missing_visible():
    table = pd.DataFrame({"display_name": ["Alpha", "Beta"], "annual_cost_inr": [26_700_000, None],
                          "price_inr_per_piece": [5.14, 6.0], "rate_inr_per_kg": [0.65, None],
                          "lines_in_total": [30, 25]})
    shown = display_table(table)
    assert list(shown.columns) == ["display_name", "annual_cost (₹)", "price (₹/piece)",
                                   "rate (₹/kg)", "lines_in_total"]
    assert shown["annual_cost (₹)"].tolist() == ["₹2.67 crore", "—"]
    assert shown["price (₹/piece)"].tolist() == ["₹5.14", "₹6.00"]
    assert shown["rate (₹/kg)"].tolist() == ["₹0.65/kg", "—"]
    assert shown["lines_in_total"].tolist() == [30, 25]
    assert table["annual_cost_inr"].iloc[0] == 26_700_000  # the original stays numeric


# ---------- Breakeven weight ----------

def test_annual_weight_kg_is_qty_times_nominal_weight():
    lines = _data().rfx_lines  # line 1: 1000 x 300 g, line 2: 500 x 100 g
    assert annual_weight_kg(lines, [1]) == 300.0
    assert annual_weight_kg(lines, [1, 2, 2]) == 350.0  # duplicates counted once


def test_annual_weight_kg_refuses_missing_weight_or_unknown_lines():
    lines = _data().rfx_lines
    lines.loc[1, "nominal_weight_g"] = None
    with pytest.raises(ValueError, match="no nominal weight"):
        annual_weight_kg(lines, [2])
    with pytest.raises(ValueError, match="no RFx line"):
        annual_weight_kg(lines, [99])


def test_generated_code_can_use_annual_weight_kg_for_a_freight_breakeven():
    code = ("gap_inr = 2500.0\n"
            "result = pd.DataFrame([{'gap_inr': gap_inr, "
            "'breakeven_freight_inr_per_kg': round(gap_inr / annual_weight_kg([1]), 2)}])")
    result = run_code(code, _data().namespace())
    assert result["breakeven_freight_inr_per_kg"].iloc[0] == pytest.approx(8.33)

# ---------- Rule: user's examples for format_inr ----------

@pytest.mark.parametrize("value, expected", [
    (36_400_000, "₹3.64 crore"),
    (1_203_000, "₹12.03 lakh"),
    (210_000, "₹2.10 lakh"),
])
def test_format_inr_examples_from_the_brief(value, expected):
    assert analyst.format_inr(value) == expected


# ---------- Rule: direction in words, not signs ----------

@pytest.mark.parametrize("saving, pct, expected", [
    (1_203_000, 3.2, "saves ₹12.03 lakh (3.2%)"),
    (-210_000, -0.6, "costs ₹2.10 lakh (0.6%) more"),
    (-36_400_000, None, "costs ₹3.64 crore more"),
    (35_780, None, "saves ₹35,780"),
    (1_000, 3.25, "saves ₹1,000 (3.3%)"),  # percent rounds half up to 1 decimal
    (0.2, 0.0, "no change"),
    (None, 3.2, "—"),  # missing is never "no change"
    (float("nan"), None, "—"),
])
def test_describe_change_gives_direction_in_words(saving, pct, expected):
    assert analyst.describe_change(saving, pct) == expected


def test_display_table_words_savings_and_merges_the_percent_column():
    table = pd.DataFrame({"display_name": ["Alpha", "Beta"], "split_saving_inr": [1_203_000, -210_000],
                          "split_saving_pct": [3.2, -0.6], "lines_won": [21, 9]})
    shown = display_table(table)
    assert list(shown.columns) == ["display_name", "split_saving (₹)", "lines_won"]
    assert shown["split_saving (₹)"].tolist() == ["saves ₹12.03 lakh (3.2%)", "costs ₹2.10 lakh (0.6%) more"]
    assert not any("-" in str(v) for v in shown["split_saving (₹)"])


def test_excel_headers_explain_the_sign_of_savings():
    assert analyst.excel_header("split_saving_inr") == "split_saving (₹, positive = saves)"
    assert analyst.excel_header("split_saving_pct") == "split_saving_pct (positive = saves)"
    assert analyst.excel_header("annual_cost_inr") == "annual_cost (₹)"


@pytest.mark.parametrize("text, found", [
    ("It changes by -₹4.31 lakh.", ["-₹4"]),
    ("That is −2.1% on last year.", ["−2.1%"]),
    ("A swing of - 3% overall.", ["- 3%"]),
    ("It costs ₹2.10 lakh (0.6%) more.", []),
    ("For the 2025-26 contract, lines 3-5.", []),
])
def test_bare_minus_signs_on_money_and_percent_are_found(text, found):
    assert analyst.bare_minus_signs(text) == found


def test_numbers_not_in_the_source_are_found_allowing_for_rounding():
    source = json.dumps({"total": "₹3.64 crore", "lines": "25 of 30 lines", "rate": "₹0.6512/kg"},
                        ensure_ascii=False)
    assert analyst.numbers_not_in("₹3.64 crore over 25 lines, about ₹0.65/kg.", source) == []
    assert analyst.numbers_not_in("₹3.65 crore over 26 lines.", source) == ["3.65", "26"]


def test_unformatted_large_numbers_are_found():
    assert analyst.unformatted_numbers("It costs 36375440 a year, or 3,63,75,440.") == ["36375440", "3,63,75,440"]
    assert analyst.unformatted_numbers("It costs ₹3.64 crore; ₹12,34,567.50 per piece; 99,999 pieces.") == []


# ---------- The pipeline, with a fake Claude ----------

class FakeClaude:
    """Returns queued responses in order and records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, messages, schema, max_tokens, purpose):
        self.calls.append({"system": system, "messages": messages, "purpose": purpose, "schema": schema})
        usage = {"model": "fake", "input_tokens": 100, "output_tokens": 10, "cache_write_tokens": 0,
                 "cache_read_tokens": 0, "cost_usd": 0.001, "purpose": purpose}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, usage


def _cls(intent, **params):
    base = {"intent": intent, "vendors": [], "include_failed": False, "cap": None, "line_ids": [],
            "fx_change_pct": None, "freight_rate_inr_per_kg": None, "wants_chart": False, "refusal_reply": ""}
    return {**base, **params}


def _writes(text):
    return {"answer": text}


@pytest.fixture(scope="module")
def sample() -> AnalystData:
    from aera.compare import compare
    from aera.config import FX_DATE, FX_RATES
    from aera.event import load_sample_event
    event = load_sample_event()
    comparison, summary = compare(event.rfx, event.last_year, event.replies, event.certificates,
                                  FX_RATES, FX_DATE, event.texts, {})
    return analyst.analyst_data(comparison, summary, event.rfx, event.last_year, FX_RATES, FX_DATE)


SAMPLE_NAMES = ["Deccan Corrupack Pvt Ltd", "Indus Packaging Solutions", "Sahyadri Boxes & Cartons",
                "Harbourline Packaging (EOU)", "Ganesh Packaging Industries"]
LEFT_OUT = " Sahyadri (failed quality) and Harbourline (quality unclear) were left out."
GOOD_SPLIT = ("Saves ₹12.03 lakh (3.2%) before freight. Giving each line to the cheapest vendor that passed quality "
              "costs ₹3.64 crore a year: Ganesh wins 21 lines, Indus 7 and Deccan 2. Ganesh hasn't quoted freight; "
              "the saving shrinks as freight rises and is gone at about ₹3.50/kg, so I can draft a clarification "
              "asking Ganesh to quote freight." + LEFT_OUT)


def test_classify_is_one_small_call_with_names_and_lines_but_no_prices(sample):
    fake = FakeClaude(_cls("award_split"), _writes(GOOD_SPLIT))
    analyst.ask("Best split of the award?", sample, call=fake)
    first = fake.calls[0]
    assert first["purpose"] == "classify" and first["schema"] is analyst.Classification
    content = first["messages"][0]["content"]
    assert "Ganesh Packaging Industries" in content and "Shipper carton" in content
    assert "price_inr_per_piece" not in content and "₹7.22" not in content
    assert content.endswith("## Buyer's question\nBest split of the award?")


def test_award_split_is_computed_by_code_and_a_good_answer_is_kept(sample):
    fake = FakeClaude(_cls("award_split"), _writes(GOOD_SPLIT))
    answer = analyst.ask("My VP wants the best split of the award. How much do we save?", sample, call=fake)
    assert [c["purpose"] for c in fake.calls] == ["classify", "summary"]  # no code written by Claude
    assert answer.text == GOOD_SPLIT and answer.wording_notes == []
    assert answer.tag == "award split, cheapest per line across 3 vendors"
    assert "award_split(data, scope)" in answer.code and "def award_split" in answer.code
    assert [sv["vendor"] for sv in answer.sensitivity] == ["Ganesh Packaging Industries"]
    assert answer.data_sufficient is False and "freight" in answer.missing_data[0]


def test_the_writer_sees_only_the_facts_and_this_intents_rules(sample):
    fake = FakeClaude(_cls("award_split"), _writes(GOOD_SPLIT))
    analyst.ask("Split?", sample, call=fake)
    writer = fake.calls[1]
    sent = writer["messages"][0]["content"]
    assert sent.startswith("FACTS:")
    facts = json.loads(sent[len("FACTS:"):])
    assert facts["total_annual_cost"] == "₹3.64 crore" and facts["question"] == "Split?"
    assert writer["system"] == analyst.WRITER_RULES + analyst.INTENT_RULES["award_split"]
    assert "36375440" not in sent  # raw numbers never reach the writer


def test_an_answer_that_drops_the_freight_figure_is_rebuilt_from_the_template(sample):
    fake = FakeClaude(_cls("award_split"),
                      _writes("Saves ₹12.03 lakh (3.2%) before freight. The split costs ₹3.64 crore." + LEFT_OUT))
    answer = analyst.ask("Split?", sample, call=fake)
    assert answer.text == analyst.template_answer("award_split", answer.facts)
    assert "₹3.50/kg" in answer.text and answer.text.startswith("Saves ₹12.03 lakh (3.2%) before freight.")
    assert any("rebuilt" in n and "₹3.50/kg" in n for n in answer.wording_notes)


def test_vendor_totals_answer_without_the_common_lines_count_is_rebuilt(sample):
    fake = FakeClaude(_cls("vendor_totals", wants_chart=True),
                      _writes("Ganesh is cheapest at ₹2.67 crore." + LEFT_OUT))
    answer = analyst.ask("Chart total annual cost by vendor", sample, call=fake)
    assert answer.tag == "vendor totals, like for like on 25 lines"
    assert answer.answer_type == "chart" and set(answer.table["lines_in_total"]) == {25}
    assert answer.text.startswith("On the 25 of 30 lines, the ones all 3 compared vendors quoted comparably,")
    assert any("common-lines count" in n for n in answer.wording_notes)


def test_a_freight_sentence_leaking_into_vendor_totals_is_never_kept(sample):
    leaked = ("On the 25 lines all 3 vendors quoted, Ganesh is cheapest at ₹2.67 crore. Saves ₹12.03 lakh (3.2%) "
              "before freight." + LEFT_OUT)
    fake = FakeClaude(_cls("vendor_totals"), _writes(leaked))
    answer = analyst.ask("Total cost per vendor", sample, call=fake)
    assert "freight" not in answer.text.lower() and "₹12.03 lakh" not in answer.text
    assert answer.sensitivity == [] and answer.price_risks == []


def test_include_failed_warning_must_open_the_answer(sample):
    fake = FakeClaude(_cls("award_split", vendors=["Sahyadri Boxes & Cartons", "Deccan Corrupack Pvt Ltd"],
                           include_failed=True),
                      _writes("Sahyadri wins 26 lines and Deccan 4, for ₹3.50 crore a year."))
    answer = analyst.ask("Split between Sahyadri and Deccan, include Sahyadri even though it failed", sample, call=fake)
    assert answer.text.startswith("Includes Sahyadri Boxes & Cartons (failed quality: ISO 9001 certificate expired")
    assert any("quality warning" in n for n in answer.wording_notes)


def test_add_vendors_adds_the_named_vendor_to_the_quality_passed_ones(sample):
    sahyadri = "Sahyadri Boxes & Cartons"
    cls = _cls("award_split", vendors=[sahyadri], include_failed=True, add_vendors=True)
    a = analyst.compute("award_split", cls, sample)
    passed = list(sample.vendors.loc[sample.vendors["quality_status"] == "PASS", "display_name"])
    considered = [n for n in sample.vendors["display_name"] if n in a.facts["vendors_considered"]]
    assert sorted(considered) == sorted(passed + [sahyadri])
    assert a.facts["quality_warning"].startswith(f"Includes {sahyadri} (failed quality")
    assert a.excluded == [{"display_name": "Harbourline Packaging (EOU)", "reason": "quality unclear"}]


def test_named_vendors_without_add_vendors_limit_the_answer_to_them(sample):
    sahyadri = "Sahyadri Boxes & Cartons"
    a = analyst.compute("award_split", _cls("award_split", vendors=[sahyadri], include_failed=True), sample)
    assert a.facts["vendors_considered"] == sahyadri


def test_recommendation_is_three_views_worded_from_facts(sample):
    text = ("It depends what matters most: Ganesh is the cheapest single vendor but hasn't quoted freight, Indus "
            "carries the least risk but costs ₹4.31 lakh more on the 25 common lines, and the cheapest split uses "
            "Ganesh, Indus and Deccan." + LEFT_OUT + " Tell me what matters most (lowest price, lowest risk or "
            "fewest vendors) and I'll work it out.")
    fake = FakeClaude(_cls("recommendation"), _writes(text))
    answer = analyst.ask("Which vendor is best?", sample, call=fake)
    assert answer.text == text
    assert list(answer.table["view"]) == ["Cheapest single vendor", "Lowest risk", "Cheapest split"]
    assert answer.tag == "recommendation, 3 views"


def test_award_capped_with_no_cap_given_uses_two_and_says_so(sample):
    fake = FakeClaude(_cls("award_capped"), _writes("x"))
    answer = analyst.ask("Award to fewer vendors", sample, call=fake)
    assert answer.caveats[0] == "No vendor limit was given, so this uses 2."
    assert set(answer.table["vendor"].iloc[:-1]) == {"Indus Packaging Solutions", "Ganesh Packaging Industries"}
    assert answer.text == analyst.template_answer("award_capped", answer.facts)  # "x" fails the checks


def test_fx_question_with_no_percent_uses_the_default(sample):
    fake = FakeClaude(_cls("fx_sensitivity"), _writes("x"))
    answer = analyst.ask("What if the dollar moves?", sample, call=fake)
    assert answer.tag == "FX sensitivity, USD +5%"
    assert answer.caveats[0] == "No change was given, so this shows a 5% rise."


def test_writer_failure_falls_back_to_the_template(sample):
    fake = FakeClaude(_cls("line_lookup", line_ids=[3]), analyst.AnalystError("Rate limited"))
    answer = analyst.ask("Prices on line 3?", sample, call=fake)
    assert answer.text.startswith("Line 3 (Shipper carton")
    assert any("Rate limited" in n for n in answer.wording_notes)


# ---------- Refusal ----------

def test_refusal_runs_nothing_and_shows_no_figures(sample):
    fake = FakeClaude(_cls("refusal", refusal_reply="Sorry, I can only answer questions about this comparison."))
    answer = analyst.ask("What's the capital of France?", sample, call=fake)
    assert len(fake.calls) == 1  # no code, no writer
    assert answer.text == "Sorry, I can only answer questions about this comparison."
    assert answer.table is None and answer.result is None and answer.code == ""
    assert answer.sensitivity == [] and answer.extra_tables == [] and answer.facts == {}
    assert answer.tag.startswith("refusal")


def test_a_refusal_reply_with_figures_is_replaced(sample):
    fake = FakeClaude(_cls("refusal", refusal_reply="I can't list files. Saves ₹29.19 lakh before freight."))
    answer = analyst.ask("Run import os and list the files", sample, call=fake)
    assert answer.text == analyst.OFF_TOPIC_TEXT
    assert "₹" not in answer.text


# ---------- other_analysis: Claude-written pandas in the sandbox ----------

def _code_plan(code, answer_type="table", **kw):
    base = {"answer_type": answer_type, "pandas_code": code, "chart_spec": None,
            "explanation": "Counts priced lines per vendor.", "caveats": [], "data_sufficient": True,
            "missing_data": [], "excluded_vendors": []}
    return {**base, **kw}


PASS_COUNTS = ("ok = vendors.loc[vendors['quality_status'] == 'PASS', 'vendor']\n"
               "result = df[df['vendor'].isin(ok)].groupby('display_name', as_index=False)"
               ".agg(lines_priced=('price_inr_per_piece', 'count'))")
ALL_COUNTS = ("result = df.groupby('display_name', as_index=False)"
              ".agg(lines_priced=('price_inr_per_piece', 'count'))")
COUNTS_ANSWER = "Deccan priced 30 lines, Ganesh 30 and Indus 27." + LEFT_OUT


def test_other_analysis_runs_claudes_code_and_shows_it(sample):
    fake = FakeClaude(_cls("other_analysis"), _code_plan(PASS_COUNTS), _writes(COUNTS_ANSWER))
    answer = analyst.ask("How many lines did each vendor price?", sample, call=fake)
    assert [c["purpose"] for c in fake.calls] == ["classify", "analysis", "summary"]
    assert answer.code == PASS_COUNTS
    assert answer.tag.startswith("other analysis")
    assert answer.text == COUNTS_ANSWER, answer.wording_notes
    assert fake.calls[1]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "did not ask to include" in fake.calls[1]["messages"][0]["content"][1]["text"]


def test_other_analysis_code_that_includes_failed_vendors_unasked_gets_one_fix(sample):
    fake = FakeClaude(_cls("other_analysis"), _code_plan(ALL_COUNTS), _code_plan(PASS_COUNTS),
                      _writes(COUNTS_ANSWER))
    answer = analyst.ask("How many lines did each vendor price?", sample, call=fake)
    fix = fake.calls[2]["messages"][-1]["content"]
    assert "Sahyadri Boxes & Cartons (failed quality)" in fix
    assert answer.code == PASS_COUNTS and "quality_warning" not in answer.facts


def test_other_analysis_failing_twice_gives_a_friendly_message(sample):
    fake = FakeClaude(_cls("other_analysis"), _code_plan("result = df['nope']"), _code_plan("import os\nresult = 1"))
    answer = analyst.ask("Something odd", sample, call=fake)
    assert answer.error == analyst.FRIENDLY_FAIL
    assert len(answer.code_errors) == 2 and len(fake.calls) == 3


def test_other_analysis_with_an_unformatted_big_number_is_rebuilt(sample):
    code = ("m = df[df['included_in_totals'] & (df['display_name'] == 'Deccan Corrupack Pvt Ltd')]"
            ".merge(rfx_lines, left_on='rfx_line_id', right_on='line_id')\n"
            "result = pd.DataFrame([{'annual_cost_inr': round((m['annual_qty'] * m['price_inr_per_piece']).sum(), 2)}])")
    fake = FakeClaude(_cls("other_analysis"), _code_plan(code, answer_type="text"),
                      _writes("Deccan costs 39790260 a year."))
    answer = analyst.ask("Deccan's annual cost?", sample, call=fake)
    assert "39790260" not in answer.text
    assert any("unformatted large numbers" in n for n in answer.wording_notes)


# ---------- The validator, intent by intent (no API calls) ----------

INTENT_PARAMS = {
    "award_split": {}, "award_capped": {"cap": 2}, "vendor_totals": {}, "recommendation": {},
    "landed_cost": {}, "line_lookup": {"line_ids": [3, 14]}, "fx_sensitivity": {"fx_change_pct": 3},
    "quality_risk": {},
}


@pytest.mark.parametrize("intent", list(INTENT_PARAMS))
def test_every_template_passes_its_own_checks(sample, intent):
    for include_failed in (False, True):
        facts = analyst.compute(intent, _cls(intent, include_failed=include_failed, **INTENT_PARAMS[intent]), sample).facts
        text = analyst.template_answer(intent, facts)
        assert analyst.check_answer(intent, text, facts, "", SAMPLE_NAMES) == []
        if facts.get("quality_warning"):
            assert text.startswith(facts["quality_warning"])


def _facts(sample, intent, **params):
    return analyst.compute(intent, _cls(intent, **{**INTENT_PARAMS[intent], **params}), sample).facts


@pytest.mark.parametrize("intent, text, problem", [
    ("award_split", "Saves ₹12.03 lakh (3.2%) before freight; ₹3.64 crore a year. Ganesh hasn't quoted freight."
                    + LEFT_OUT, "freight figure"),
    ("award_capped", "Capped at 2, Ganesh covers all 30 lines for ₹3.64 crore." + LEFT_OUT,
     "chosen vendor Indus Packaging Solutions not named"),
    ("vendor_totals", "Ganesh is cheapest at ₹2.67 crore." + LEFT_OUT, "common-lines count"),
    ("recommendation", "Ganesh is cheapest and the split uses three vendors." + LEFT_OUT,
     "lowest-risk vendor Indus Packaging Solutions not named"),
    ("landed_cost", "Ganesh wins 21 lines; the saving is gone at about ₹3.50/kg." + LEFT_OUT,
     "the saving before freight"),
    ("line_lookup", "On line 3 Ganesh is cheapest at ₹7.22 per piece." + LEFT_OUT, "line 14 not mentioned"),
    ("fx_sensitivity", "No line changes hands." + LEFT_OUT, "rate change"),
    ("quality_risk", "Sahyadri failed quality and Ganesh has two high risks.", "Harbourline Packaging (EOU) not named"),
])
def test_each_intents_required_facts_are_checked(sample, intent, text, problem):
    problems = analyst.check_answer(intent, text, _facts(sample, intent), "", SAMPLE_NAMES)
    assert any(problem in p for p in problems), problems


def test_excluded_vendors_must_be_named(sample):
    facts = _facts(sample, "vendor_totals")
    text = "On the 25 lines all 3 vendors quoted, Ganesh is cheapest at ₹2.67 crore. Sahyadri (failed quality) was left out."
    assert analyst.check_answer("vendor_totals", text, facts, "", SAMPLE_NAMES) == [
        "left-out vendor Harbourline Packaging (EOU) not named"]


def test_made_up_numbers_and_minus_signs_fail(sample):
    facts = _facts(sample, "vendor_totals")
    text = "On the 25 lines, Ganesh is cheapest at ₹2.60 crore, -₹4.31 lakh against Indus." + LEFT_OUT
    problems = analyst.check_answer("vendor_totals", text, facts, "", SAMPLE_NAMES)
    assert any("minus signs" in p for p in problems)
    assert any("numbers not in the facts: 2.60" in p for p in problems)


def test_numbers_from_the_question_are_allowed(sample):
    facts = _facts(sample, "line_lookup", line_ids=[3])
    text = ("Line 3: Ganesh is cheapest at ₹7.22 per piece (quoted per kg, box weight 190 g; freight extra), ₹0.77 per piece (₹65,450 a year) below Indus."
            + LEFT_OUT + " That covers the 1 line you asked about.")
    assert analyst.check_answer("line_lookup", text, facts, "what about 1 line, line 3?", SAMPLE_NAMES) == []


def test_refusal_check_allows_no_figures():
    assert analyst.check_answer("refusal", "I can only help with this comparison.", {}) == []
    assert analyst.check_answer("refusal", "It saves ₹12.03 lakh.", {}) == ["a refusal must not contain figures"]
    assert analyst.check_answer("refusal", "There are 5 vendors.", {}) == ["a refusal must not contain figures"]


# ---------- Excel export ----------

def test_excel_export_has_the_table_an_about_sheet_and_the_freight_table(sample):
    answer = analyst.ask("Split?", sample, call=FakeClaude(_cls("award_split"), _writes(GOOD_SPLIT)))
    sheets = pd.read_excel(io.BytesIO(analyst.to_excel_bytes(answer)), sheet_name=None)
    assert sheets["Answer"]["annual_cost (₹)"].iloc[-1] == pytest.approx(36_375_440)
    about = dict(zip(sheets["About"]["item"], sheets["About"]["value"]))
    assert about["Answered as"] == "award split, cheapest per line across 3 vendors"
    assert any(name.startswith("Freight Ganesh") for name in sheets)
