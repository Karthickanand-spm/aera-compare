"""Offline tests for analyst.py: the code sandbox, the number check and the ask() flow.

Claude is replaced by a fake caller, so no API calls are made.
"""

import io

import pandas as pd
import pytest

from aera import analyst
from aera.analyst import (
    AnalystData, CodeError, annual_weight_kg, check_code, display_table, format_inr, money_header,
    run_code, unsupported_numbers,
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


# ---------- Number check ----------

def test_numbers_from_the_result_are_accepted_with_separators_and_rounding():
    result = pd.DataFrame({"display_name": ["Alpha"], "annual_cost_inr": [123456.789]})
    text = "Alpha costs INR 1,23,456.79 a year, about 123457 in round numbers."
    assert unsupported_numbers(text, result) == []


def test_numbers_the_model_made_up_are_flagged():
    result = pd.DataFrame({"display_name": ["Alpha", "Beta"], "annual_cost_inr": [10000.0, 12500.0]})
    text = "Alpha is cheaper by INR 2,500, saving 20% across 2 vendors."
    assert unsupported_numbers(text, result) == ["2,500", "20"]


def test_a_negative_result_written_in_words_is_accepted():
    result = pd.DataFrame({"display_name": ["Alpha"], "savings_inr": [-414733.4]})
    assert unsupported_numbers("Alpha is dearer than last year by 414733.40 INR.", result) == []


def test_numbers_in_the_question_and_text_cells_are_accepted():
    result = pd.DataFrame({"description": ["5-ply box"], "price": [12.5]})
    assert unsupported_numbers("Line 3's 5-ply box is 12.5.", result, question="What about line 3?") == []


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


def test_formatted_money_in_the_answer_counts_as_in_the_result():
    result = pd.DataFrame({"gap_inr": [431_234.0], "breakeven_freight_inr_per_kg": [0.6512]})
    text = "If Beta's freight is above ₹0.65/kg (about ₹4.31 lakh a year on these lines), Alpha is cheaper."
    assert unsupported_numbers(text, result) == []
    assert unsupported_numbers("The gap is ₹5.00 lakh.", result) == ["5.00"]


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


# ---------- ask() with a fake Claude ----------

def _plan(code, answer_type="table", chart_spec=None, caveats=None, sufficient=True, missing=None,
          intent="line_lookup", asked_for_flagged=False, explanation="Adds up annual cost per vendor.",
          vendors_compared=None):
    return {"intent": intent, "vendors_compared": vendors_compared or [],
            "buyer_asked_for_flagged_vendors": asked_for_flagged,
            "answer_type": answer_type, "pandas_code": code, "chart_spec": chart_spec,
            "explanation": explanation, "caveats": caveats or [],
            "data_sufficient": sufficient, "missing_data": missing or []}


class FakeClaude:
    """Returns queued responses in order and records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, messages, schema, max_tokens, purpose):
        self.calls.append({"system": system, "messages": messages, "purpose": purpose})
        usage = {"model": "fake", "input_tokens": 100, "output_tokens": 10, "cache_write_tokens": 0,
                 "cache_read_tokens": 0, "cost_usd": 0.001, "purpose": purpose}
        return self.responses.pop(0), usage


GOOD_CODE = ("m = df[df['included_in_totals']].merge(rfx_lines, left_on='rfx_line_id', right_on='line_id')\n"
             "m['annual_cost_inr'] = m['annual_qty'] * m['price_inr_per_piece']\n"
             "result = m.groupby('display_name', as_index=False)['annual_cost_inr'].sum().round(2)")


def test_ask_runs_code_and_words_the_answer_from_the_result():
    fake = FakeClaude(_plan(GOOD_CODE), {"answer": "Alpha costs INR 10,000 a year and Beta 12,500."})
    answer = analyst.ask("Total cost per vendor?", _data(), call=fake)

    assert answer.error is None
    assert answer.answer_type == "table"
    assert answer.table["annual_cost_inr"].tolist() == [10000.0, 12500.0]
    assert answer.unchecked_numbers == []
    assert [u["purpose"] for u in answer.usages] == ["analysis", "summary"]
    assert answer.cost_usd == pytest.approx(0.002)
    # The summary call sees the computed result, money already formatted by code.
    assert "₹12,500" in fake.calls[1]["messages"][0]["content"]
    # Beta's price needs review and is unconfirmed: code adds that caveat itself.
    assert any("Beta line 1" in c for c in answer.caveats)


def test_context_sent_to_claude_has_rules_columns_vendors_and_fx():
    fake = FakeClaude(_plan(GOOD_CODE), {"answer": "ok"})
    analyst.ask("Anything", _data(), call=fake)
    sent = fake.calls[0]
    assert "Missing is never zero" in sent["system"]
    context_block, question_block = sent["messages"][0]["content"]
    for expected in ("price_inr_per_piece", "UNCLEAR", "1 USD = ₹94.5", "2026-09-25", "nominal_weight_g"):
        assert expected in context_block["text"]
    assert context_block["cache_control"] == {"type": "ephemeral"}  # reused by follow-up questions
    assert question_block["text"] == "## Buyer's question\nAnything"


def test_failed_code_is_sent_back_once_and_the_fix_is_used():
    fake = FakeClaude(_plan("result = df['nope']"), _plan(GOOD_CODE), {"answer": "Fixed."})
    answer = analyst.ask("Total cost per vendor?", _data(), call=fake)

    assert answer.error is None
    assert len(answer.code_errors) == 1 and "KeyError" in answer.code_errors[0]
    retry_messages = fake.calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert "KeyError" in retry_messages[-1]["content"]
    assert answer.code == GOOD_CODE


def test_two_failures_give_a_friendly_message_and_no_summary_call():
    fake = FakeClaude(_plan("result = df['nope']"), _plan("import os\nresult = 1"))
    answer = analyst.ask("Total cost per vendor?", _data(), call=fake)

    assert answer.error == analyst.FRIENDLY_FAIL
    assert len(answer.code_errors) == 2
    assert len(fake.calls) == 2
    assert answer.code == "import os\nresult = 1"  # still shown under "Show the working"


def test_insufficient_data_without_code_skips_running_and_summary():
    fake = FakeClaude(_plan("", answer_type="text", sufficient=False, missing=["Freight costs"]))
    answer = analyst.ask("Delivered cost?", _data(), call=fake)

    assert answer.data_sufficient is False
    assert answer.missing_data == ["Freight costs"]
    assert answer.text == "Adds up annual cost per vendor."
    assert len(fake.calls) == 1


def test_bad_chart_spec_falls_back_to_a_table():
    plan = _plan(GOOD_CODE, answer_type="chart", chart_spec={"x": "display_name", "y": "nope", "kind": "bar"})
    answer = analyst.ask("Chart it", _data(), call=FakeClaude(plan, {"answer": "ok"}))
    assert answer.answer_type == "table"
    assert answer.chart_spec is None
    assert any("Could not draw the chart" in c for c in answer.caveats)


def test_good_chart_spec_is_kept():
    plan = _plan(GOOD_CODE, answer_type="chart",
                 chart_spec={"x": "display_name", "y": "annual_cost_inr", "kind": "bar"})
    answer = analyst.ask("Chart it", _data(), call=FakeClaude(plan, {"answer": "ok"}))
    assert answer.answer_type == "chart"


def test_scalar_result_becomes_a_text_answer():
    plan = _plan("result = int(df['label'].eq('Not quoted').sum())")
    answer = analyst.ask("How many not quoted?", _data(), call=FakeClaude(plan, {"answer": "1 line."}))
    assert answer.answer_type == "text"
    assert answer.table is None
    assert answer.result == 1


def test_made_up_number_in_the_summary_is_flagged():
    fake = FakeClaude(_plan(GOOD_CODE), {"answer": "Alpha saves INR 2,500 over Beta."})
    answer = analyst.ask("Who is cheaper?", _data(), call=fake)
    assert answer.unchecked_numbers == ["2,500"]


def test_excel_export_has_the_table_and_an_about_sheet():
    fake = FakeClaude(_plan(GOOD_CODE, caveats=["Beta is UNCLEAR"]), {"answer": "ok"})
    answer = analyst.ask("Total cost per vendor?", _data(), call=fake)
    sheets = pd.read_excel(io.BytesIO(analyst.to_excel_bytes(answer)), sheet_name=None)
    assert sheets["Answer"]["annual_cost (₹)"].tolist() == [10000.0, 12500.0]  # numbers, ₹ in header
    assert "Beta is UNCLEAR" in sheets["About"]["value"].tolist()


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


SPLIT_CODE = ("result = pd.DataFrame([{'display_name': 'Alpha', 'split_saving_inr': -210000.0, "
              "'split_saving_pct': -0.6}])")


def test_summary_with_a_minus_sign_is_rewritten_once():
    fake = FakeClaude(_plan(SPLIT_CODE), {"answer": "Alpha changes cost by -₹2.10 lakh (-0.6%)."},
                      {"answer": "Alpha costs ₹2.10 lakh (0.6%) more."})
    answer = analyst.ask("Split?", _data(), call=fake)
    assert answer.text == "Alpha costs ₹2.10 lakh (0.6%) more."
    assert [u["purpose"] for u in answer.usages] == ["analysis", "summary", "summary"]
    assert "minus signs" in fake.calls[2]["messages"][0]["content"]
    # The summary call was given the saving already worded, not a signed number.
    assert "costs ₹2.10 lakh (0.6%) more" in fake.calls[1]["messages"][0]["content"]
    assert answer.wording_notes == []


def test_a_minus_sign_that_survives_the_rewrite_is_noted():
    fake = FakeClaude(_plan(SPLIT_CODE), {"answer": "Down -₹2.10 lakh."}, {"answer": "Still -₹2.10 lakh."})
    answer = analyst.ask("Split?", _data(), call=fake)
    assert any("Minus sign still" in n for n in answer.wording_notes)


# ---------- Rule: name excluded vendors and why ----------

QUALITY_CODE = ("ok = vendors.loc[vendors['quality_status'] == 'PASS', 'vendor']\n"
                "result = df[df['vendor'].isin(ok)].groupby('display_name', as_index=False)"
                "['price_inr_per_piece'].sum()")


def test_vendors_removed_by_a_quality_filter_are_found_by_code_with_reasons():
    data = _data()
    result = run_code(QUALITY_CODE, data.namespace())
    excluded = analyst.excluded_vendors([], QUALITY_CODE, result, data.vendors)
    assert excluded == [{"display_name": "Beta", "reason": "quality unclear"},
                        {"display_name": "Gamma", "reason": "failed quality"}]


def test_claudes_excluded_list_is_kept_but_unknown_names_are_dropped():
    data = _data()
    excluded = analyst.excluded_vendors(
        [{"display_name": "gamma", "reason": "did not quote line 1"},
         {"display_name": "Nobody Ltd", "reason": "made up"}], "result = 1", 1, data.vendors)
    assert excluded == [{"display_name": "Gamma", "reason": "did not quote line 1"}]


# ---------- Post-check: rules the writer cannot skip ----------

SAMPLE_VENDORS = ["Deccan Corrupack Pvt Ltd", "Ganesh Packaging Industries", "Indus Packaging Solutions",
                  "Sahyadri Boxes & Cartons", "Harbourline Packaging (EOU)"]
SAMPLE_EXCLUDED = [{"display_name": "Sahyadri Boxes & Cartons", "reason": "failed quality"},
                   {"display_name": "Harbourline Packaging (EOU)", "reason": "quality unclear"}]
SAMPLE_RISK = {"vendor": "Ganesh Packaging Industries", "kind": "freight", "must_include": ["₹3.19/kg"],
               "text": "Ganesh Packaging Industries wins 21 lines but hasn't quoted freight; above ₹3.19/kg "
                       "(about ₹16.23 lakh a year on those lines) the saving shrinks"}


def test_post_check_appends_the_left_out_sentence_when_the_answer_skips_exclusions():
    answer = ("Awarding each line to the cheapest vendor that cleared quality saves ₹12.03 lakh (3.2%) "
              "against last year, but Ganesh Packaging Industries wins 21 lines and hasn't quoted freight; "
              "above ₹3.19/kg the saving shrinks.")
    text, notes = analyst.post_check(answer, [SAMPLE_RISK], SAMPLE_EXCLUDED, [], SAMPLE_VENDORS)
    assert text == answer + (" Left out: Sahyadri Boxes & Cartons (failed quality) and "
                             "Harbourline Packaging (EOU) (quality unclear).")
    assert any(n.startswith("Code added: Left out:") for n in notes)


def test_post_check_appends_the_risk_sentence_when_the_answer_skips_it():
    answer = "Awarding each line to the cheapest qualified vendor saves ₹12.03 lakh (3.2%) against last year."
    text, _ = analyst.post_check(answer, [SAMPLE_RISK], [], [], SAMPLE_VENDORS)
    assert text.endswith("Ganesh Packaging Industries wins 21 lines but hasn't quoted freight; above "
                         "₹3.19/kg (about ₹16.23 lakh a year on those lines) the saving shrinks.")


def test_post_check_needs_the_breakeven_not_just_the_word_freight():
    answer = "Saves ₹12.03 lakh, but Ganesh hasn't quoted freight."
    text, _ = analyst.post_check(answer, [SAMPLE_RISK], [], [], SAMPLE_VENDORS)
    assert "above ₹3.19/kg" in text


def test_post_check_leaves_a_complete_answer_alone_and_accepts_short_names():
    answer = ("Saves ₹12.03 lakh (3.2%), but Ganesh wins 21 lines and hasn't quoted freight; above ₹3.19/kg "
              "the saving shrinks. Sahyadri (failed quality) and Harbourline (quality unclear) were left out.")
    assert analyst.post_check(answer, [SAMPLE_RISK], SAMPLE_EXCLUDED, [], SAMPLE_VENDORS) == (answer, [])


def test_post_check_formats_big_bare_numbers_that_match_money():
    money = [(1_203_000.0, "amount"), (39_790_260.0, "amount")]
    text, notes = analyst.post_check("It saves 1203000 and Deccan costs 3,97,90,260 a year over 30 lines.",
                                     [], [], money, SAMPLE_VENDORS)
    assert text == "It saves ₹12.03 lakh and Deccan costs ₹3.98 crore a year over 30 lines."
    assert len(notes) == 2


def test_post_check_leaves_small_and_already_formatted_numbers_and_notes_unknown_big_ones():
    text, notes = analyst.post_check("Lines 12345 and ₹1234567.50 per piece; weight 875032 kg.",
                                     [], [], [], SAMPLE_VENDORS)
    assert text == "Lines 12345 and ₹1234567.50 per piece; weight 875032 kg."
    assert notes == ["Unformatted number left in the answer: 875032"]


# ---------- Price risks worked out in code ----------

def _beta_wins_line_1() -> AnalystData:
    data = _data()
    data.df.loc[2, "price_inr_per_piece"] = 8.0  # Beta 8.00 vs Alpha 10.00 on line 1 (1000 pcs, 300 g)
    return data


def _two_line_data() -> AnalystData:
    """Tiny award with a known freight breakeven.

    2 lines, 1,000 pieces each, 500 g a box (0.5 kg), last year ₹10.00 on both.
    A (freight extra): ₹8.00 and ₹9.00.   B (freight included): ₹11.00 and ₹10.50.
    Freight r ₹/kg adds 0.5r to each A price.
    - Before freight A wins both: cost 17,000 vs last year 20,000, saves ₹3,000 (15.0%).
    - While A keeps both lines the saving is 3,000 - 1,000r, so it is gone at r = ₹3.00/kg
      (line 2 is then a tie at ₹10.50; the saving is exactly zero either way).
    - Line 2 moves to B above ₹3/kg, line 1 above ₹6/kg.
    """
    rows = []
    for vendor, prices in (("A", (8.0, 9.0)), ("B", (11.0, 10.5))):
        for line_id, price in zip((1, 2), prices):
            rows.append({"rfx_line_id": line_id, "vendor": vendor, "display_name": vendor,
                         "price_inr_per_piece": price, "label": "Comparable", "included_in_totals": True,
                         "needs_review": False, "buyer_confirmed": False, "assumptions": []})
    df = pd.DataFrame(rows)
    vendors = pd.DataFrame([
        {"vendor": "A", "display_name": "A", "quality_status": "PASS", "freight": "extra"},
        {"vendor": "B", "display_name": "B", "quality_status": "PASS", "freight": "included"},
    ])
    rfx_lines = pd.DataFrame([{"line_id": i, "description": f"Box {i}", "annual_qty": 1000, "uom": "pcs",
                               "nominal_weight_g": 500.0} for i in (1, 2)])
    last_year = pd.DataFrame([{"line_id": 1, "price_inr_per_piece": 10.0},
                              {"line_id": 2, "price_inr_per_piece": 10.0}])
    return AnalystData(df, vendors, rfx_lines, last_year, {"USD": 94.5}, "2026-09-25")


def test_freight_sensitivity_on_a_two_line_award_with_a_known_breakeven():
    s = analyst.freight_sensitivity(_two_line_data(), "A", ["A", "B"])
    assert s["zero_rate"] == 3.0
    table = s["table"].set_index("rate_inr_per_kg")
    assert table.loc[0, "saving_inr"] == 3000.0 and table.loc[0, "saving_pct"] == 15.0
    assert table.loc[0, "lines_won"] == 2
    # The saving shrinks from the first rupee of freight, not only above some rate.
    assert table.loc[0.5, "saving_inr"] == 2500.0
    assert table.loc[1, "saving_inr"] == 2000.0
    assert table.loc[2, "saving_inr"] == 1000.0
    # At ₹4/kg line 2 has moved to B (10.50 < 11.00); A keeps line 1 at 10.00 (< B's 11.00).
    assert table.loc[4, "lines_won"] == 1
    assert table.loc[4, "total_inr"] == 20500.0 and table.loc[4, "saving_inr"] == -500.0
    assert list(s["table"]["rate_inr_per_kg"]) == [0, 0.5, 1, 2, 3, 4]


def test_freight_risk_uses_the_sensitivity_wording():
    data = _two_line_data()
    result = pd.DataFrame({"display_name": ["A", "B"], "total_inr": [1.0, 2.0]})
    [risk] = analyst.price_risks(result, data)
    assert risk["headline"] == "Saves ₹3,000 (15.0%) before freight"
    assert risk["text"] == ("A wins 2 lines but hasn't quoted freight; the saving shrinks as freight rises "
                            "and is gone at about ₹3.00/kg (table below)")
    assert risk["must_include"] == ["₹3.00/kg", "before freight"]
    assert risk["sensitivity"]["zero_rate"] == 3.0


def test_post_check_appends_the_template_sentence_when_freight_is_skipped():
    data = _two_line_data()
    [risk] = analyst.price_risks(pd.DataFrame({"display_name": ["A", "B"], "total_inr": [1.0, 2.0]}), data)
    text, _ = analyst.post_check("The award saves ₹3,000 (15.0%).", [risk], [], [], ["A", "B"])
    assert text == ("The award saves ₹3,000 (15.0%). Saves ₹3,000 (15.0%) before freight. A wins 2 lines but "
                    "hasn't quoted freight; the saving shrinks as freight rises and is gone at about "
                    "₹3.00/kg (table below).")


def test_post_check_accepts_a_risk_stated_after_a_semicolon_in_the_vendors_sentence():
    risk = {"vendor": "Ganesh Packaging Industries", "kind": "ambiguous", "must_include": [],
            "text": "1 of the 21 lines Ganesh Packaging Industries wins is priced at the higher of two readings"}
    answer = "Ganesh wins 21 lines; 1 of those lines is priced at the higher of two readings until you confirm."
    assert analyst.post_check(answer, [risk], [], [], SAMPLE_VENDORS) == (answer, [])


def test_when_other_vendors_take_all_its_lines_the_saving_has_a_floor():
    data = _beta_wins_line_1()  # Beta 8.00 vs Alpha 10.00 on line 1, last year 11.00; Beta loses it above ₹6.67/kg
    result = pd.DataFrame({"display_name": ["Alpha", "Beta"], "total_inr": [1.0, 2.0]})
    [risk] = analyst.price_risks(result, data)
    s = risk["sensitivity"]
    assert s["zero_rate"] is None
    assert s["lines_won_at_max"] == 0 and s["saving_at_max_inr"] == 1000.0  # Alpha at 10.00 still saves
    assert risk["text"].endswith("so it never falls below ₹1,000 (table below)")


def test_a_lone_freight_extra_vendor_has_its_saving_gone_at_a_rate():
    data = _beta_wins_line_1()
    [risk] = analyst.price_risks(pd.DataFrame({"display_name": ["Beta"], "total_inr": [2.0]}), data)
    # Only Beta: 1,000 x (11.00 - 8.00) = 3,000 saving; 0.3 kg a box, so gone at 3.00 / 0.3 = ₹10.00/kg
    assert risk["sensitivity"]["zero_rate"] == 10.0
    assert "gone at about ₹10.00/kg" in risk["text"]


def test_missing_box_weight_is_reported_not_treated_as_zero():
    data = _two_line_data()
    data.rfx_lines.loc[0, "nominal_weight_g"] = None
    [risk] = analyst.price_risks(pd.DataFrame({"display_name": ["A", "B"], "total_inr": [1.0, 2.0]}), data)
    assert risk["sensitivity"] is None
    assert "no box weight for line(s) [1]" in risk["text"]


def test_ambiguous_risk_counts_only_lines_the_vendor_wins():
    data = _beta_wins_line_1()
    data.df.loc[2, "assumptions"] = [analyst.AMBIGUOUS_ASSUMPTION]
    result = pd.DataFrame({"display_name": ["Alpha", "Beta"], "total_inr": [1.0, 2.0]})
    kinds = [r["kind"] for r in analyst.price_risks(result, data)]
    assert kinds == ["freight", "ambiguous"]


def test_no_risk_for_a_vendor_that_wins_nothing_or_is_not_in_the_result():
    data = _data()  # Alpha (freight included) is cheaper on line 1, so Beta wins nothing
    both = pd.DataFrame({"display_name": ["Alpha", "Beta"], "total_inr": [1.0, 2.0]})
    assert analyst.price_risks(both, data) == []
    only_alpha = pd.DataFrame({"display_name": ["Alpha"], "total_inr": [1.0]})
    assert analyst.price_risks(only_alpha, _beta_wins_line_1()) == []


def test_facts_hold_only_formatted_strings():
    result = pd.DataFrame({"display_name": ["Total"], "lines_won": [30], "annual_cost_inr": [36_400_000.0],
                           "saving_inr": [1_203_000.0], "saving_pct": [3.2], "share_pct": [-0.6],
                           "weight_kg": [875032.0]})
    answer = analyst.Answer(question="q", asked_at="now", excluded_vendors=SAMPLE_EXCLUDED,
                            price_risks=[SAMPLE_RISK])
    facts = analyst.build_facts(answer, result)
    row = facts["result_rows"][0]
    assert row == {"display_name": "Total", "lines_won": "30", "annual_cost (₹)": "₹3.64 crore",
                   "saving (₹)": "saves ₹12.03 lakh (3.2%)", "share_pct": "-0.6%", "weight_kg": "8,75,032"}
    assert all(isinstance(v, str) for v in row.values())
    assert facts["savings_in_words"] == ["Total: saves ₹12.03 lakh (3.2%)"]
    assert facts["excluded_vendors"] == ["Sahyadri Boxes & Cartons (failed quality)",
                                         "Harbourline Packaging (EOU) (quality unclear)"]
    assert facts["price_risks"] == [SAMPLE_RISK["text"]]


def test_ask_sends_only_facts_and_fixes_an_answer_that_skips_the_rules():
    data = _beta_wins_line_1()
    fake = FakeClaude(_plan(QUALITY_CODE), {"answer": "Alpha and Beta are priced at 10 and 8 per piece."})
    answer = analyst.ask("Prices for vendors that cleared quality?", data, call=fake)
    sent = fake.calls[1]["messages"][0]["content"]
    assert sent.startswith("FACTS:")
    assert fake.calls[1]["system"] == analyst.WRITER_RULES
    # QUALITY_CODE keeps only PASS vendors, so only Alpha is in the result: Beta and Gamma are left out.
    assert answer.text.endswith("Left out: Beta (quality unclear) and Gamma (failed quality).")
    assert any(n.startswith("Code added") for n in answer.wording_notes)


# ---------- Off-topic and non-cost questions get nothing added ----------

def _award_data() -> AnalystData:
    """Two lines. On price alone Gamma (FAIL) wins line 1 and Beta (UNCLEAR, freight extra) line 2;
    Alpha (PASS) is dearest on both."""
    prices = {"alpha_co": (10.0, 5.0), "beta_co": (9.0, 3.5), "gamma_co": (7.0, 4.0)}
    names = {"alpha_co": "Alpha", "beta_co": "Beta", "gamma_co": "Gamma"}
    rows = [{"rfx_line_id": line, "vendor": v, "display_name": names[v], "price_inr_per_piece": p,
             "label": "Comparable", "included_in_totals": True, "needs_review": False,
             "buyer_confirmed": False, "assumptions": []}
            for v, ps in prices.items() for line, p in zip((1, 2), ps)]
    vendors = pd.DataFrame([
        {"vendor": "alpha_co", "display_name": "Alpha", "quality_status": "PASS", "freight": "included",
         "quality_reasons": ["PASS: defect rate 1.2% is within the 2% limit"]},
        {"vendor": "beta_co", "display_name": "Beta", "quality_status": "UNCLEAR", "freight": "extra",
         "quality_reasons": ["PASS: ISO 9001 certificate valid until 2027-05-20 (beta_iso.pdf)",
                             "UNCLEAR: defect rate not provided"]},
        {"vendor": "gamma_co", "display_name": "Gamma", "quality_status": "FAIL", "freight": "included",
         "quality_reasons": ["FAIL: ISO 9001 certificate expired 2026-03-15 (gamma_iso.pdf)",
                             "FAIL: defect rate 2.8% is above the 2% limit",
                             "PASS: test report supplied with every batch"]},
    ])
    rfx_lines = pd.DataFrame([{"line_id": i, "description": f"Box {i}", "annual_qty": 1000, "uom": "pcs",
                               "nominal_weight_g": 500.0} for i in (1, 2)])
    last_year = pd.DataFrame([{"line_id": 1, "price_inr_per_piece": 11.0},
                              {"line_id": 2, "price_inr_per_piece": 6.0}])
    return AnalystData(pd.DataFrame(rows), vendors, rfx_lines, last_year, {"USD": 94.5}, "2026-09-25")


def _award_code(keep: str) -> str:
    """Cheapest counted price per line among the vendors `keep` (a boolean mask on df) allows."""
    return (f"m = df[df['included_in_totals'] & ({keep})].merge(rfx_lines, left_on='rfx_line_id', right_on='line_id')\n"
            "w = m.sort_values('price_inr_per_piece').groupby('rfx_line_id', as_index=False).first()\n"
            "w['annual_cost_inr'] = (w['annual_qty'] * w['price_inr_per_piece']).round(2)\n"
            "result = w.groupby('display_name', as_index=False).agg("
            "lines_won=('rfx_line_id', 'count'), annual_cost_inr=('annual_cost_inr', 'sum'))")


PASS_ONLY = "df['vendor'].isin(vendors[vendors['quality_status'] == 'PASS']['vendor'])"
AWARD_ALL_CODE = _award_code("df['vendor'].notna()")
AWARD_PASS_CODE = _award_code(PASS_ONLY)
AWARD_WITH_GAMMA_CODE = _award_code(f"{PASS_ONLY} | (df['vendor'] == 'gamma_co')")
OFF_TOPIC_PLAN = dict(answer_type="text", intent="refusal",
                      explanation="I can only answer questions about this comparison.")


def _has_money_or_freight(answer) -> bool:
    return ("₹" in answer.text or "freight" in answer.text.lower() or bool(answer.sensitivity)
            or bool(answer.price_risks) or answer.table is not None)


def test_off_topic_question_runs_no_code_and_adds_no_money_or_freight():
    # Even if Claude wrote award code anyway, it is not run and nothing is appended.
    fake = FakeClaude(_plan(AWARD_ALL_CODE, **OFF_TOPIC_PLAN))
    answer = analyst.ask("What's the capital of France?", _award_data(), call=fake)
    assert len(fake.calls) == 1  # no writer call
    assert answer.result is None
    assert answer.text == "I can only answer questions about this comparison."
    assert not _has_money_or_freight(answer)


def test_refused_request_never_shows_money_even_if_the_explanation_has_some():
    plan = _plan("", **{**OFF_TOPIC_PLAN, "explanation": "I can't list files. Saves ₹29.19 lakh before freight."})
    answer = analyst.ask("Run import os and list the files", _award_data(), call=FakeClaude(plan))
    assert answer.text == analyst.OFF_TOPIC_TEXT
    assert not _has_money_or_freight(answer)


def test_non_cost_question_gets_no_award_saving_or_freight_facts():
    # Beta (freight extra) wins line 1 in the result, but the question is not about cost.
    fake = FakeClaude(_plan(GOOD_CODE, intent="line_lookup"), {"answer": "Alpha and Beta quoted line 1."})
    answer = analyst.ask("Which vendors quoted line 1?", _beta_wins_line_1(), call=fake)
    assert answer.price_risks == [] and answer.sensitivity == []
    assert answer.facts["headline_before_freight"] is None
    assert answer.text == "Alpha and Beta quoted line 1."


# ---------- Default awards use quality PASS vendors only ----------

def test_default_split_excludes_fail_and_unclear_vendors():
    excluded = [{"display_name": "Beta", "reason": "quality unclear"},
                {"display_name": "Gamma", "reason": "failed quality"}]
    fake = FakeClaude(_plan(AWARD_ALL_CODE, intent="award_split"),
                      {**_plan(AWARD_PASS_CODE, intent="award_split"), "excluded_vendors": excluded},
                      {"answer": "Alpha wins both lines. Beta (quality unclear) and Gamma (failed quality) were left out."})
    answer = analyst.ask("What's the best split of the award?", _award_data(), call=fake)
    fix = fake.calls[1]["messages"][-1]["content"]
    assert "Beta (quality unclear)" in fix and "Gamma (failed quality)" in fix
    assert list(answer.table["display_name"]) == ["Alpha"]
    assert answer.quality_warnings == []
    assert not answer.text.startswith("Includes")


def test_a_flagged_vendor_that_survives_the_fix_is_kept_only_with_a_warning_first():
    fake = FakeClaude(_plan(AWARD_ALL_CODE, intent="award_split"),
                      _plan(AWARD_ALL_CODE, intent="award_split"), {"answer": "Gamma wins both lines."})
    answer = analyst.ask("What's the best split of the award?", _award_data(), call=fake)
    # Both flagged vendors win a line: one warning each, before anything else.
    assert answer.text.startswith("Includes Beta, whose quality is unclear: defect rate not provided; "
                                  "not a valid award option until that is confirmed. "
                                  "Includes Gamma, which failed quality:")
    assert "Kept, with a warning" in answer.code_errors[-1]


def test_explicit_include_request_puts_the_quality_warning_in_the_first_sentence():
    fake = FakeClaude(_plan(AWARD_WITH_GAMMA_CODE, intent="award_split"), {"answer": "Gamma wins both lines."})
    answer = analyst.ask("Split the award, and include gamma this time", _award_data(), call=fake)
    assert len(fake.calls) == 2  # asked for by name: no fix round
    first = answer.text.split(". ")[0] + "."
    assert first == ("Includes Gamma, which failed quality: ISO 9001 certificate expired 2026-03-15; "
                     "defect rate 2.8% is above the 2% limit; not a valid award option.")
    assert "Gamma wins both lines." in answer.text
    assert answer.unchecked_numbers == []


def test_unclear_vendor_asked_for_as_a_group_gets_its_own_warning():
    code = _award_code(f"{PASS_ONLY} | (df['vendor'] == 'beta_co')")
    fake = FakeClaude(_plan(code, intent="award_split", asked_for_flagged=True), {"answer": "Beta wins both lines."})
    answer = analyst.ask("Split across every vendor, even the unclear ones", _award_data(), call=fake)
    assert answer.text.startswith("Includes Beta, whose quality is unclear: defect rate not provided; "
                                  "not a valid award option until that is confirmed.")



def test_other_intent_is_words_only_with_no_money_or_table():
    fake = FakeClaude(_plan(GOOD_CODE, intent="other"),
                      {"answer": "Two vendors replied with prices. Alpha totals ₹10,000."})
    answer = analyst.ask("How many vendors sent prices?", _data(), call=fake)
    assert answer.text == "Two vendors replied with prices."
    assert answer.table is None and answer.answer_type == "text"


# ---------- Vendor totals: like for like, no award sentence ----------

def _gamma_skips_line_2() -> AnalystData:
    """Gamma (FAIL) quotes only line 1, so its total over its own lines looks cheapest."""
    data = _award_data()
    gamma_2 = (data.df["vendor"] == "gamma_co") & (data.df["rfx_line_id"] == 2)
    data.df.loc[gamma_2, ["price_inr_per_piece", "label", "included_in_totals"]] = [None, "Not quoted", False]
    return data


def test_chart_of_totals_uses_common_lines_and_has_no_award_sentence():
    writer = "On the 1 of 2 lines all three vendors quoted, Gamma is cheapest; the second table shows full totals."
    fake = FakeClaude(_plan("", intent="vendor_totals", answer_type="chart"), {"answer": writer})
    answer = analyst.ask("Show me a chart of total annual cost by vendor", _gamma_skips_line_2(), call=fake)

    assert answer.text == writer  # nothing appended: no award, saving or freight sentence
    assert "before freight" not in answer.text
    assert answer.price_risks == [] and answer.sensitivity == [] and answer.quality_warnings == []
    assert answer.chart_spec == {"x": "vendor", "y": "like_for_like_total_inr", "kind": "bar"}
    assert answer.table.to_dict(orient="records") == [
        {"vendor": "Gamma (failed quality)", "like_for_like_total_inr": 7000.0, "lines_in_total": 1},
        {"vendor": "Beta (quality unclear)", "like_for_like_total_inr": 9000.0, "lines_in_total": 1},
        {"vendor": "Alpha", "like_for_like_total_inr": 10000.0, "lines_in_total": 1},
    ]
    (title, full), = answer.extra_tables
    assert "don't compare" in title
    assert full.set_index("vendor")["lines_in_total"].to_dict() == {
        "Alpha": 2, "Beta (quality unclear)": 2, "Gamma (failed quality)": 1}
    assert answer.facts["like_for_like_lines"] == "1 of 2 lines, the ones all 3 compared vendors quoted"
    assert "vendor_totals(data" in answer.code  # the app's code is shown under the answer


def test_vendor_totals_for_chosen_vendors_lists_the_rest_as_left_out():
    fake = FakeClaude(_plan("", intent="vendor_totals", vendors_compared=["alpha", "Beta"]),
                      {"answer": "On the 2 lines both vendors quoted, Beta is cheaper."})
    answer = analyst.ask("Total cost for Alpha and Beta", _gamma_skips_line_2(), call=fake)
    assert list(answer.table["lines_in_total"]) == [2, 2]
    assert answer.excluded_vendors == [{"display_name": "Gamma", "reason": "failed quality"}]
    assert answer.text.endswith("Left out: Gamma (failed quality).")


def test_no_common_line_gives_full_totals_with_a_caveat_not_a_like_for_like_claim():
    data = _gamma_skips_line_2()
    alpha_1 = (data.df["vendor"] == "alpha_co") & (data.df["rfx_line_id"] == 1)
    data.df.loc[alpha_1, ["price_inr_per_piece", "label", "included_in_totals"]] = [None, "Not quoted", False]
    fake = FakeClaude(_plan("", intent="vendor_totals", answer_type="chart"), {"answer": "No like-for-like total."})
    answer = analyst.ask("Total annual cost by vendor", data, call=fake)
    assert answer.answer_type == "table" and answer.extra_tables == []
    assert "total_on_own_lines_inr" in answer.table.columns
    assert any("no like-for-like total" in c for c in answer.caveats)


# ---------- On the real sample event (Claude faked, no API calls) ----------

@pytest.fixture(scope="module")
def sample_data() -> AnalystData:
    from aera.compare import compare
    from aera.config import FX_DATE, FX_RATES
    from aera.event import load_sample_event
    event = load_sample_event()
    comparison, summary = compare(event.rfx, event.last_year, event.replies, event.certificates,
                                  FX_RATES, FX_DATE, event.texts, {})
    return analyst.analyst_data(comparison, summary, event.rfx, event.last_year, FX_RATES, FX_DATE)


VP_SPLIT_CODE = """ok = vendors[vendors['quality_status'] == 'PASS']['vendor']
m = df[df['included_in_totals'] & df['vendor'].isin(ok)].merge(rfx_lines, left_on='rfx_line_id', right_on='line_id')
w = m.sort_values(['rfx_line_id', 'price_inr_per_piece', 'display_name']).groupby('rfx_line_id', as_index=False).first()
ly = last_year.set_index('line_id')['price_inr_per_piece']
w['annual_cost_inr'] = w['annual_qty'] * w['price_inr_per_piece']
w['last_year_cost_inr'] = w['annual_qty'] * w['rfx_line_id'].map(ly)
g = w.groupby('display_name', as_index=False).agg(lines_won=('rfx_line_id', 'count'),
    annual_cost_inr=('annual_cost_inr', 'sum'), last_year_cost_inr=('last_year_cost_inr', 'sum'))
total = pd.DataFrame([{'display_name': 'Total split award', 'lines_won': g['lines_won'].sum(),
    'annual_cost_inr': g['annual_cost_inr'].sum(), 'last_year_cost_inr': g['last_year_cost_inr'].sum()}])
g = pd.concat([g, total], ignore_index=True)
g['saving_inr'] = g['last_year_cost_inr'] - g['annual_cost_inr']
g['saving_pct'] = (g['saving_inr'] / g['last_year_cost_inr'] * 100).round(1)
result = g.round(2)"""


def test_vp_split_still_gives_the_award_saving_freight_table_and_exclusions(sample_data):
    fake = FakeClaude(_plan(VP_SPLIT_CODE, intent="award_split"), {"answer": "Here is the split."})
    answer = analyst.ask("My VP wants the best split of the award across vendors. What is it and how "
                         "much do we save against last year?", sample_data, call=fake)
    shown = display_table(answer.table).set_index("display_name")
    assert shown.loc["Total split award", "annual_cost (₹)"] == "₹3.64 crore"
    assert shown.loc["Total split award", "saving (₹)"] == "saves ₹12.03 lakh (3.2%)"
    assert answer.facts["headline_before_freight"] == "Saves ₹12.03 lakh (3.2%) before freight"
    assert "Saves ₹12.03 lakh (3.2%) before freight." in answer.text  # the post-check adds it for award_split
    assert [sv["vendor"] for sv in answer.sensitivity] == ["Ganesh Packaging Industries"]
    assert {e["display_name"]: e["reason"] for e in answer.excluded_vendors} == {
        "Sahyadri Boxes & Cartons": "failed quality", "Harbourline Packaging (EOU)": "quality unclear"}


def test_which_is_best_returns_the_views_table_and_asks_what_matters(sample_data):
    fake = FakeClaude(_plan("result = 1", intent="recommendation"))
    answer = analyst.ask("Which vendor is best?", sample_data, call=fake)
    assert len(fake.calls) == 1  # worded by code, no writer call
    assert answer.result is answer.table  # Claude's code was not run
    views = answer.table.set_index("view")
    assert "Cheapest split" in views.index and 2 <= len(views) <= 3
    assert all(views["caveat"].str.len() > 0)
    assert display_table(answer.table).set_index("view").loc["Cheapest split", "annual_cost (₹)"] == "₹3.64 crore"
    assert answer.text.endswith("Tell me what matters most (lowest price, lowest risk or fewest vendors) "
                                "and I'll work it out.")
    assert "Sahyadri Boxes & Cartons (failed quality)" in answer.text
    assert answer.sensitivity == [] and answer.price_risks == []
    assert "Sahyadri" not in " ".join(answer.table["vendors"])


def test_chart_of_totals_on_sample_data_compares_on_common_lines_only(sample_data):
    fake = FakeClaude(_plan("", intent="vendor_totals", answer_type="chart"), {"answer": "Totals by vendor."})
    answer = analyst.ask("Show me a chart of total annual cost by vendor", sample_data, call=fake)
    assert answer.table["lines_in_total"].nunique() == 1  # every bar covers the same lines
    assert "Sahyadri Boxes & Cartons (failed quality)" in set(answer.table["vendor"])
    assert "before freight" not in answer.text and answer.sensitivity == []
