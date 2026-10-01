"""Display helpers: one colour per status everywhere, and the workflow stepper."""

from aera.compare import COMPARABLE, FAIL, NOT_COMPARABLE, NOT_QUOTED, PASS, UNCLEAR, WITH_ASSUMPTION
from aera.ui import (
    STEPS, badge_color, badge_md, has_high_risk, header_html, rfx_status, severity_counts_text, sidebar_sections,
    split_headline, status_text_md, stepper_html, tint,
)
from ui.state import (
    ASK_HISTORY, AWARD_CONFIRMED, EVENT, RFX_SENT_LOG, SENT_LOG, completed_steps,
)


def test_quality_badges():
    assert badge_color(PASS) == "green"
    assert badge_color(FAIL) == "red"
    assert badge_color(UNCLEAR) == "orange"


def test_risk_badges_any_case():
    for severity, color in (("high", "red"), ("medium", "orange"), ("low", "gray")):
        assert badge_color(severity) == color
        assert badge_color(severity.upper()) == color


def test_every_comparability_label_has_its_own_colour():
    colors = [badge_color(label) for label in (COMPARABLE, WITH_ASSUMPTION, NOT_COMPARABLE, NOT_QUOTED)]
    assert colors == ["green", "blue", "red", "gray"]


def test_unknown_status_is_gray():
    assert badge_color("something new") == "gray"
    assert badge_color(None) == "gray"


def test_markdown_badge_and_widget_label_share_the_colour():
    assert badge_md("HIGH") == ":red-badge[HIGH]"
    assert status_text_md("HIGH") == ":red[**HIGH**]"
    assert badge_md("Quality PASS", PASS) == ":green-badge[Quality PASS]"


def test_badge_text_cannot_break_out_of_the_badge():
    assert badge_md("a]b") == r":gray-badge[a\]b]"


def test_tint_uses_badge_colour():
    assert tint(NOT_COMPARABLE) == tint(FAIL)
    assert tint(UNCLEAR).startswith("background-color: rgba(245, 158, 11")


def test_stepper_marks_current_and_done():
    html = stepper_html("Decide", {"Create RFx", "Compare"})
    for step in STEPS:
        assert step in html
    assert html.count('aria-current="step"') == 1
    assert 'class="aera-step current"' in html  # Decide: current, not done
    assert html.count("aera-step done") == 2
    assert html.count("✓") == 2


def test_completed_steps_from_session_state():
    empty = {EVENT: None, ASK_HISTORY: [], AWARD_CONFIRMED: None, RFX_SENT_LOG: [], SENT_LOG: []}
    assert completed_steps(empty) == set()

    full = {EVENT: object(), ASK_HISTORY: ["answer"], AWARD_CONFIRMED: {"confirmed_at": "now"},
            RFX_SENT_LOG: [{"Vendor": "x"}], SENT_LOG: [{"Vendor": "x"}]}
    assert completed_steps(full) == set(STEPS)


def test_setup_pages_show_only_the_session_section():
    assert sidebar_sections("Create RFx") == ("Session",)
    assert sidebar_sections("Start here") == ("Session",)


def test_event_pages_show_all_three_sections():
    for page in ("Compare", "Ask", "Decide", "Clarify"):
        assert sidebar_sections(page) == ("Event", "Assumptions", "Session")


def test_headline_is_the_first_sentence():
    assert split_headline("E is cheapest. It saves 4% on last year.") == ("E is cheapest.", "It saves 4% on last year.")
    assert split_headline("Is that right? Yes!") == ("Is that right?", "Yes!")


def test_headline_ignores_decimals_and_stops_at_a_line_break():
    assert split_headline("Saving is ₹1.5 lakh. Details below.") == ("Saving is ₹1.5 lakh.", "Details below.")
    assert split_headline("Totals by vendor\n- A: ₹2 lakh") == ("Totals by vendor", "- A: ₹2 lakh")


def test_headline_of_one_sentence_or_nothing():
    assert split_headline("  Only one sentence.  ") == ("Only one sentence.", "")
    assert split_headline("") == ("", "")
    assert split_headline(None) == ("", "")


def test_high_risk_caveats_open_what_to_watch():
    assert has_high_risk(["Vendor B has a high-severity open risk: no ISO certificate."])
    assert has_high_risk(["Open risk (HIGH): late deliveries."])
    assert has_high_risk(["High risk of delay."])
    assert not has_high_risk(["Freight not quoted, so this total is before freight.", "Highest saving is on line 3."])
    assert not has_high_risk([])


def test_page_header_is_one_block_with_escaped_text():
    out = header_html("Ask", "Vendors <A> & B.")
    assert out.startswith("<h2 ") and "line-height: 1.2" in out and "<p " in out
    assert "&lt;A&gt; &amp; B." in out and "Next:" not in out


def test_severity_counts_high_first_and_skip_empty():
    items = [{"severity": "medium"}, {"severity": "high"}, {"severity": "medium"}]
    assert severity_counts_text(items) == "1 high · 2 medium"
    assert severity_counts_text([{"severity": "low"}]) == "1 low"


def test_no_open_items_says_nothing_to_clarify():
    assert severity_counts_text([]) == "Nothing to clarify"


def test_rfx_status_amber_until_every_check_passes():
    text, kind = rfx_status(3, 5)
    assert text == "Draft · 3 of 5 checks" and badge_color(kind) == "orange"
    text, kind = rfx_status(5, 5)
    assert text == "Ready to send" and badge_color(kind) == "green"
