"""Display helpers: one colour per status everywhere, and the workflow stepper."""

from aera.compare import COMPARABLE, FAIL, NOT_COMPARABLE, NOT_QUOTED, PASS, UNCLEAR, WITH_ASSUMPTION
from aera.ui import STEPS, badge_color, badge_md, sidebar_sections, status_text_md, stepper_html, tint
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
