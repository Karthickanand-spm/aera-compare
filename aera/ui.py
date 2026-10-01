"""Small display helpers shared by every page: header, workflow stepper, badges, cards, metrics.

Display only. Nothing here calculates a price, label or confidence; those come from
aera/compare.py and aera/award.py. One colour table below keeps badges the same everywhere.
"""

import html
import re
from collections.abc import Callable, Iterable

import streamlit as st

from aera.compare import COMPARABLE, FAIL, NOT_COMPARABLE, NOT_QUOTED, PASS, UNCLEAR, WITH_ASSUMPTION

# The buyer's path through the app. Names match the page titles in app.py.
STEPS = ("Create RFx", "Compare", "Ask", "Decide", "Clarify")

# One colour per status, used by every badge on every page (st.badge colour names, which
# Streamlit adjusts for light and dark mode). Risk severities are keyed in capitals.
BADGE_COLORS = {
    PASS: "green", FAIL: "red", UNCLEAR: "orange",
    "HIGH": "red", "MEDIUM": "orange", "LOW": "gray",
    COMPARABLE: "green", WITH_ASSUMPTION: "blue", NOT_COMPARABLE: "red", NOT_QUOTED: "gray",
}
DEFAULT_BADGE_COLOR = "gray"

# Soft background tints for table cells, in the same colour families as the badges. Semi-transparent,
# so the theme's own text colour stays on top and keeps its contrast in light and dark mode.
TINTS = {
    "green": "rgba(34, 197, 94, 0.16)",
    "red": "rgba(239, 68, 68, 0.16)",
    "orange": "rgba(245, 158, 11, 0.24)",
    "blue": "rgba(59, 130, 246, 0.14)",
    "gray": "rgba(128, 128, 128, 0.12)",
}

TEAL = "#0F766E"  # same as primaryColor in .streamlit/config.toml; white text on it passes contrast

_STYLES = f"""
<style>
/* Tighter page spacing than Streamlit's default. */
[data-testid="stMainBlockContainer"] {{ padding-top: 3.5rem; padding-bottom: 2rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
[data-testid="stSidebarUserContent"] hr {{ margin: 0.75rem 0; }}

/* Workflow stepper. Text inherits the theme colour, so it reads well in light and dark mode. */
.aera-stepper {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.5rem;
  list-style: none; margin: 0 0 0.25rem 0; padding: 0; font-size: 0.85rem; }}
.aera-step {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.2rem 0.7rem 0.2rem 0.3rem;
  border-radius: 999px; border: 1px solid rgba(128, 128, 128, 0.4); color: inherit; white-space: nowrap; }}
.aera-step .aera-dot {{ display: inline-flex; align-items: center; justify-content: center;
  width: 1.15rem; height: 1.15rem; border-radius: 50%; border: 1px solid currentColor;
  font-size: 0.7rem; font-weight: 600; }}
.aera-step.done .aera-dot {{ background: {TEAL}; border-color: {TEAL}; color: #fff; }}
.aera-step.current {{ background: {TEAL}; border-color: {TEAL}; color: #fff; font-weight: 600; }}
.aera-step.current .aera-dot {{ border-color: #fff; }}
.aera-step.current.done .aera-dot {{ background: #fff; color: {TEAL}; }}
.aera-sep {{ width: 1rem; height: 1px; background: rgba(128, 128, 128, 0.5); }}

/* "What to watch" expander under an answer: amber edge and tint, theme text colour on top. */
[class*="st-key-aera-watch"] details {{ border-color: rgba(245, 158, 11, 0.7);
  background: {TINTS["orange"]}; }}
</style>
"""


def apply_styles() -> None:
    """Page-wide CSS. Call once per run, before anything else is drawn."""
    st.html(_STYLES)


# ---------- Header and stepper ----------

def page_header(title: str, purpose: str, next_step: str | None = None) -> None:
    """Page title, one plain sentence on what the page is for, and a subtle hint of what comes next."""
    st.title(title, anchor=False)
    st.markdown(purpose)
    if next_step:
        st.caption(f"Next: {next_step}")


def stepper_html(current: str, completed: Iterable[str] = ()) -> str:
    """HTML for the stepper. `current` is highlighted; steps in `completed` get a tick."""
    done = set(completed)
    parts = []
    for i, step in enumerate(STEPS, start=1):
        classes = "aera-step" + (" done" if step in done else "") + (" current" if step == current else "")
        aria = ' aria-current="step"' if step == current else ""
        mark = "✓" if step in done else str(i)
        status = " (done)" if step in done else ""
        if i > 1:
            parts.append('<li class="aera-sep" aria-hidden="true"></li>')
        parts.append(f'<li class="{classes}"{aria}><span class="aera-dot" aria-hidden="true">{mark}</span>'
                     f'{html.escape(step)}<span style="position:absolute;left:-9999px">{status}</span></li>')
    return f'<ol class="aera-stepper" aria-label="Workflow">{"".join(parts)}</ol>'


def workflow_stepper(current: str, completed: Iterable[str] = ()) -> None:
    """Slim horizontal stepper: Create RFx → Compare → Ask → Decide → Clarify."""
    st.html(stepper_html(current, completed))


# ---------- Badges ----------

def badge_color(kind: str | None) -> str:
    """Colour for a status: PASS / FAIL / UNCLEAR, HIGH / MEDIUM / LOW risk, or a comparability label."""
    if not kind:
        return DEFAULT_BADGE_COLOR
    return BADGE_COLORS.get(kind) or BADGE_COLORS.get(kind.upper(), DEFAULT_BADGE_COLOR)


def status_badge(text: str, kind: str | None = None) -> None:
    """A pill badge. `kind` picks the colour (defaults to `text` itself, e.g. "PASS")."""
    st.badge(text, color=badge_color(kind or text))


def badge_md(text: str, kind: str | None = None) -> str:
    """The same badge as Markdown, to put inside a line of st.markdown text."""
    safe = text.replace("[", r"\[").replace("]", r"\]")
    return f":{badge_color(kind or text)}-badge[{safe}]"


def status_text_md(text: str, kind: str | None = None) -> str:
    """Bold coloured text in the badge colour. For widget labels (checkboxes), which can't show badges."""
    safe = text.replace("[", r"\[").replace("]", r"\]")
    return f":{badge_color(kind or text)}[**{safe}**]"


def tint(kind: str) -> str:
    """CSS background for a table cell, in the badge colour for `kind`."""
    return f"background-color: {TINTS[badge_color(kind)]}"


# ---------- Layout ----------

def card(key: str | None = None):
    """A bordered box. Use as `with card():`."""
    return st.container(border=True, key=key, gap="small")


def metric_row(items: list[tuple]) -> None:
    """3 to 5 headline numbers side by side. Each item is (label, value) or (label, value, help)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        col.metric(label, value, help=rest[0] if rest else None)


def empty_state(message: str, button_label: str, on_click: Callable[[], None], key: str = "empty_state") -> None:
    """Shown when a page has nothing to work on yet: a short message and one button that fixes it."""
    with card():
        st.markdown(message)
        if st.button(button_label, type="primary", key=key):
            on_click()
            st.rerun()  # redraw the whole page, sidebar included, with the result


def info_strip(message: str, button_label: str, key: str) -> bool:
    """A slim bordered strip: an info icon, one line of text and a button on the right.

    Returns True on the run the button is clicked."""
    with st.container(border=True, horizontal=True, vertical_alignment="center", key=key):
        st.markdown(f":material/info: {message}", width="stretch")
        return st.button(button_label, key=f"{key}_button", width="content")


# ---------- Answer cards ----------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|
")
_HIGH_RISK = re.compile(r"high[- ]?(severity|risk)|HIGH", re.IGNORECASE)


def split_headline(text: str) -> tuple[str, str]:
    """(first sentence, the rest). The first sentence stops at a full stop, ? or ! followed by
    a space, or at the first line break. Decimals like 1.5 don't end it."""
    text = (text or "").strip()
    end = _SENTENCE_END.search(text)
    if not end:
        return text, ""
    return text[:end.start()].strip(), text[end.end():].strip()


def has_high_risk(caveats: Iterable[str]) -> bool:
    """True if any caveat talks about a high-severity / HIGH risk."""
    return any(_HIGH_RISK.search(c or "") for c in caveats)


def watch_box(key: str, expanded: bool = False):
    """The amber "What to watch" expander for an answer's caveats. Use as `with watch_box(...):`."""
    return st.expander("What to watch", expanded=expanded, icon=":material/warning:", key=f"aera-watch-{key}")


def sidebar_section(title: str) -> None:
    """A labelled group in the sidebar."""
    st.subheader(title, anchor=False)


# ---------- Sidebar per page ----------

SIDEBAR_SECTIONS = ("Event", "Assumptions", "Session")
# Pages where the buyer is still writing the RFx: no event or FX rates to show yet.
SETUP_PAGES = ("Start here", "Create RFx")
SETUP_SIDEBAR_NOTE = "Vendor replies come in on the Compare page."


def sidebar_sections(page: str) -> tuple[str, ...]:
    """The sidebar sections that are relevant on `page` (a page title), in order."""
    return ("Session",) if page in SETUP_PAGES else SIDEBAR_SECTIONS


def render_sidebar(page: str, draw: dict[str, Callable[[], None]]) -> None:
    """Draw the sidebar for `page`. `draw` maps each section name to the function that fills it."""
    shown = sidebar_sections(page)
    with st.sidebar:
        if "Event" not in shown:
            st.caption(SETUP_SIDEBAR_NOTE)
            st.divider()
        for i, name in enumerate(shown):
            if i:
                st.divider()
            sidebar_section(name)
            draw[name]()
