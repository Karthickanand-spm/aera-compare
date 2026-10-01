"""Create RFx: a chat that turns what the buyer says into a draft RFx.

Claude asks follow-up questions and keeps a structured draft in the same shape as
data/sample/rfx.json. Code checks every number Claude puts in the draft against what
the buyer actually typed: a quantity, weight or defect rate the buyer never stated is
blanked, so nothing is invented or calculated by the model. Code also runs the
completeness checklist and builds the JSON the buyer downloads.
"""

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from aera.analyst import _ask_claude

TURN_MAX_TOKENS = 8000
MAX_FOLLOW_UPS = 3

LINE_FIELDS = ("description", "ply", "board_spec", "size", "print", "annual_qty", "uom", "nominal_weight_g")
LINE_NUMBER_FIELDS = {"annual_qty": "annual quantity", "nominal_weight_g": "nominal weight (g)"}
TERM_FIELDS = ("delivery", "prices", "validity", "payment")

GREETING = ("Tell me what you need to buy: the items, roughly how many a year, and where they "
            "should be delivered. I'll ask a few questions and build the RFx on the right as we go.")

Caller = Callable[..., tuple[dict, dict]]


# ---------- Structured output schema ----------

class DraftLine(BaseModel):
    description: str | None = Field(description="What the item is, e.g. 'Shipper carton, steam iron'.")
    ply: str | None = Field(description="e.g. '3-ply'. null if the buyer has not said.")
    board_spec: str | None = Field(description="Material / grade spec, e.g. '150/120/150 GSM, BF 18, "
                                               "B-flute'. null if the buyer has not said.")
    size: str | None = Field(description="Dimensions as the buyer gave them, e.g. '310 x 150 x 140 mm (ID)'.")
    print: str | None = Field(description="Print requirement, e.g. '1-colour flexo' or 'Plain'.")
    annual_qty: int | None = Field(description="Annual quantity ONLY if the buyer stated this exact number "
                                               "for this line. Never estimate, split or calculate. Else null.")
    uom: str | None = Field(description="Unit of measure, e.g. 'pcs'.")
    nominal_weight_g: float | None = Field(description="Nominal weight per piece in grams ONLY if the buyer "
                                                       "stated it. Never estimate. Else null.")


class DraftTerms(BaseModel):
    delivery: str | None = Field(description="Delivery location and basis, e.g. 'Delivered to Chakan plant "
                                             "(FOR destination)'. null until the buyer names a location.")
    prices: str | None = Field(description="How vendors must quote, e.g. 'INR per piece, ex-GST'.")
    validity: str | None = Field(description="Contract / price validity, e.g. '12 months'.")
    payment: str | None = Field(description="Payment terms, e.g. 'Vendor to state' or '60 days'.")


class DraftQualityBar(BaseModel):
    iso9001_required: bool | None = Field(description="true if the buyer requires a valid ISO 9001 certificate.")
    max_defect_rate_pct: float | None = Field(description="Maximum defect / rejection rate in percent, only "
                                                          "if the buyer stated it.")
    test_report_per_batch: bool | None = Field(description="true if the buyer wants a test report with "
                                                           "every batch.")
    other_requirements: list[str] = Field(description="Any other quality requirements the buyer stated.")


class RfxDraft(BaseModel):
    title: str | None = Field(description="Short RFx title, e.g. 'Annual rate contract: corrugated packaging'.")
    lines: list[DraftLine]
    questionnaire: list[str] = Field(description="Questions every vendor must answer with their quote.")
    terms: DraftTerms
    quality_bar: DraftQualityBar


class Question(BaseModel):
    id: str = Field(description="'Q1', 'Q2', 'Q3' in the order asked.")
    text: str = Field(description="One short question, e.g. 'How many kettle cartons a year?'")
    example: str = Field(description="A short example answer, e.g. '60,000'.")
    field: str = Field(description="The RFx field the answer fills, e.g. 'line 1 annual_qty', "
                                   "'line 2 board_spec' or 'terms delivery'.")
    allow_vendors_propose: bool = Field(description="true if 'vendors to propose' is a sensible answer "
                                                    "(spec, print, payment terms). Never for quantities.")


class TurnOutput(BaseModel):
    summary: str = Field(description="1 to 2 short sentences on what changed in the RFx this turn. Plain text.")
    questions: list[Question] = Field(description=f"At most {MAX_FOLLOW_UPS} follow-up questions, most "
                                                  "important first. Empty when nothing is missing.")
    rfx_draft: RfxDraft


SYSTEM = f"""You help a procurement buyer write an RFx (request for quotes) by chatting with them.

Every turn you return three things: a short summary of what changed, up to {MAX_FOLLOW_UPS} follow-up
questions, and the full updated RFx draft.

Rules for the draft:
- Start from the current draft you are given. It includes edits the buyer made by hand in the
  table; keep them unless the buyer asks for a change.
- Add or update only what the buyer has actually said. Leave a field null (or a list empty) when
  the buyer has not said it.
- NEVER invent, estimate, split or calculate numbers. annual_qty, nominal_weight_g and
  max_defect_rate_pct may only hold a number the buyer typed for that item. If the buyer gives a
  total for several items, ask how it splits instead of dividing it yourself.
- Never put prices or budgets in the draft.
- One line per distinct item the buyer needs.
- If the buyer answers "vendors to propose", write "Vendors to propose" in that text field.
- You may propose a standard vendor questionnaire (certificates, capacity, lead time, defect rate,
  payment terms) and standard quoting terms (e.g. 'INR per piece, ex-GST'); say in the summary that
  you proposed them so the buyer can change them.

Rules for the summary and questions:
- summary: 1 to 2 short plain sentences on what changed in the RFx this turn. Do not repeat the
  whole draft back. No markdown.
- Each question asks exactly ONE thing. Never combine two questions ("What size? And which ply?"
  is two questions: ask one now and the other later).
- If an answer covers only part of what was asked or is unclear (e.g. "yea mm" to a dimensions
  question), keep that field empty and ask again, saying what is missing (e.g. "I still need the
  length, width and height in mm for the kettle carton.").
- questions: NEVER more than {MAX_FOLLOW_UPS}. Ask the rest next turn. Order: items and quantities
  first, then size and spec (ply, board grade, print), then quality requirements and commercial
  terms, delivery last. One short question each, ids Q1, Q2, Q3, a short example answer, and the
  field it fills.
- Answers may come back tagged, e.g. "Q1 (line 1 annual_qty): 60,000 / Q2: skipped". Map each answer
  to its question. Ask skipped questions again in a later turn.
- If the buyer writes freely instead, map what they say to the open questions where you can."""


# ---------- Draft helpers (pure) ----------

def empty_draft() -> dict:
    return RfxDraft(title=None, lines=[], questionnaire=[], terms=DraftTerms(**dict.fromkeys(TERM_FIELDS)),
                    quality_bar=DraftQualityBar(iso9001_required=None, max_defect_rate_pct=None,
                                                test_report_per_batch=None, other_requirements=[])).model_dump()


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def rows_to_lines(rows: list[dict]) -> list[dict]:
    """Table rows (from st.data_editor) back into draft lines. Blank cells become None, never 0.

    Rows where every cell is blank (e.g. a freshly added empty row) are dropped."""
    lines = []
    for row in rows:
        line = {}
        for f in LINE_FIELDS:
            v = row.get(f)
            if _blank(v):
                line[f] = None
            elif f == "annual_qty":
                line[f] = int(round(float(v)))
            elif f == "nominal_weight_g":
                line[f] = float(v)
            else:
                line[f] = str(v).strip()
        if any(v is not None for v in line.values()):
            lines.append(line)
    return lines


_NUMBER = re.compile(r"(?<![\d.])(\d+(?:,\d+)*(?:\.\d+)?)(?:\s*(k|thousand|lakhs?|lacs?|million|mn|crores?|cr)\b)?",
                     re.IGNORECASE)
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
                "million": 1e6, "mn": 1e6, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def numbers_in_text(text: str) -> set[float]:
    """Every number written in `text`, with '50k', '1.1 lakh', '1,10,000' read as the buyer meant them."""
    found: set[float] = set()
    for m in _NUMBER.finditer(text or ""):
        value = float(m.group(1).replace(",", ""))
        found.add(value)
        if m.group(2):
            found.add(round(value * _MULTIPLIERS[m.group(2).lower()], 6))
    return found


def _stated(value: float, allowed: set[float]) -> bool:
    return any(math.isclose(value, a, rel_tol=1e-9, abs_tol=1e-9) for a in allowed)


def guard_numbers(new: dict, old: dict, buyer_texts: list[str]) -> tuple[dict, list[str]]:
    """Blank any number in Claude's draft that the buyer never typed and that wasn't already there.

    Returns (checked draft, plain-English notes on what was blanked)."""
    stated = set().union(*(numbers_in_text(t) for t in buyer_texts)) if buyer_texts else set()
    notes = []
    for f, label in LINE_NUMBER_FIELDS.items():
        allowed = stated | {ln[f] for ln in old["lines"] if ln.get(f) is not None}
        for i, ln in enumerate(new["lines"], start=1):
            v = ln.get(f)
            if v is not None and not _stated(float(v), allowed):
                ln[f] = None
                notes.append(f"Line {i} ({ln.get('description') or 'no description'}): Aera guide filled "
                             f"{label} {v:g}, but you haven't stated that number, so it is left blank.")
    qb_new, qb_old = new["quality_bar"], old["quality_bar"]
    v = qb_new.get("max_defect_rate_pct")
    old_v = qb_old.get("max_defect_rate_pct")
    if v is not None and not _stated(float(v), stated | ({old_v} if old_v is not None else set())):
        qb_new["max_defect_rate_pct"] = None
        notes.append(f"Aera guide filled a maximum defect rate of {v:g}%, but you haven't stated that number, "
                     "so it is left blank.")
    return new, notes


def quality_bar_set(qb: dict) -> bool:
    return (qb.get("iso9001_required") is True or qb.get("max_defect_rate_pct") is not None
            or qb.get("test_report_per_batch") is True or bool(qb.get("other_requirements")))


def checklist(draft: dict) -> list[tuple[str, bool, str]]:
    """(item, done, what is missing) for the completeness checklist."""
    lines = draft["lines"]

    def missing(field: str) -> list[int]:
        return [i for i, ln in enumerate(lines, start=1) if _blank(ln.get(field))]

    def line_item(label: str, field: str) -> tuple[str, bool, str]:
        gaps = missing(field)
        ok = bool(lines) and not gaps
        detail = "no lines yet" if not lines else ("missing on line " + ", ".join(map(str, gaps)) if gaps else "")
        return label, ok, detail

    return [
        line_item("Every line has a quantity", "annual_qty"),
        line_item("Every line has a spec", "board_spec"),
        line_item("Every line has a size", "size"),
        ("Quality bar set", quality_bar_set(draft["quality_bar"]), ""),
        ("Delivery location set", not _blank(draft["terms"].get("delivery")), ""),
    ]


def is_complete(draft: dict) -> bool:
    return all(ok for _, ok, _ in checklist(draft))


def to_rfx_json(draft: dict, today: date) -> dict:
    """The draft in the same shape as data/sample/rfx.json. Unknown values stay null."""
    qb = draft["quality_bar"]
    quality_bar = {
        "iso9001_valid_on": today.isoformat() if qb.get("iso9001_required") else None,
        "max_defect_rate_pct": qb.get("max_defect_rate_pct"),
        "test_report_per_batch": qb.get("test_report_per_batch"),
    }
    if qb.get("other_requirements"):
        quality_bar["other_requirements"] = list(qb["other_requirements"])
    return {
        "rfx_id": f"DRAFT-{today:%Y%m%d}",
        "buyer": None,
        "title": draft.get("title"),
        "issued": today.isoformat(),
        "due": None,
        "terms": {f: draft["terms"].get(f) for f in TERM_FIELDS},
        "quality_bar": quality_bar,
        "questionnaire": list(draft["questionnaire"]),
        "lines": [{"line_id": i, **{f: ln.get(f) for f in LINE_FIELDS}}
                  for i, ln in enumerate(draft["lines"], start=1)],
    }


def load_vendor_list(path: str | Path) -> list[dict]:
    """The short list of vendors the simulated send can pick from: [{"name", "email"}]."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- The Claude turn ----------

def build_messages(history: list[dict], draft: dict) -> list[dict]:
    """Chat history as Messages API turns; the current draft rides on the newest buyer message."""
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages[-1] = {"role": "user", "content": f"{messages[-1]['content']}\n\n<current_draft>\n"
                                              f"{json.dumps(draft, indent=1)}\n</current_draft>"}
    return messages


def one_question(text: str) -> str:
    """Only the first question when Claude combined several ('What size? And which ply?' -> 'What size?')."""
    text = (text or "").strip()
    first = text.find("?")
    return text[:first + 1] if first != -1 and "?" in text[first + 1:] else text


def cap_questions(questions: list[dict]) -> list[dict]:
    """At most MAX_FOLLOW_UPS questions, numbered Q1, Q2, Q3 in order, each asking one thing.

    Blank questions are dropped. A combined question keeps only its first part; Claude asks the rest later."""
    questions = [{**q, "text": one_question(q.get("text"))} for q in questions]
    kept = [q for q in questions if q["text"]][:MAX_FOLLOW_UPS]
    return [{**q, "id": f"Q{i}", "text": q["text"].strip(), "example": (q.get("example") or "").strip(),
             "field": (q.get("field") or "").strip()} for i, q in enumerate(kept, start=1)]


def parse_reply(parsed: dict) -> tuple[str, list[dict]]:
    """(summary, questions) from Claude's structured reply, with the question cap enforced in code."""
    questions = [Question.model_validate(q).model_dump() for q in parsed.get("questions") or []]
    return (parsed.get("summary") or "").strip(), cap_questions(questions)


def reply_text(summary: str, questions: list[dict]) -> str:
    """The assistant turn as plain text for the chat history Claude sees on the next turn."""
    return "\n".join([summary, *(f"{q['id']} ({q['field']}): {q['text']}" for q in questions)]).strip()


VENDORS_PROPOSE = "vendors to propose"


def tagged_answers(questions: list[dict], answers: dict[str, str], propose: dict[str, bool]) -> str:
    """The answer form as one message: 'Q1 (line 1 annual_qty): 60,000 / Q2: skipped'.

    `answers` and `propose` are keyed by question id. A blank answer with no tick is skipped."""
    parts = []
    for q in questions:
        text = (answers.get(q["id"]) or "").strip()
        if propose.get(q["id"]):
            text = VENDORS_PROPOSE + (f"; {text}" if text else "")
        parts.append(f"{q['id']} ({q['field']}): {text}" if text else f"{q['id']}: skipped")
    return " / ".join(parts)


_TAG = re.compile(r"\bQ\d+\s*(?:\([^)]*\))?\s*:")


def strip_tags(text: str) -> str:
    """Buyer text without the 'Q1 (line 1 annual_qty):' tags, so 'line 1' isn't read as a stated number."""
    return _TAG.sub(" ", text or "")


def next_turn(history: list[dict], draft: dict, call: Caller | None = None
              ) -> tuple[str, list[dict], dict, list[str], dict]:
    """One chat turn. `history` ends with the buyer's new message; `draft` includes their table edits.

    Returns (summary, questions, checked draft, notes on numbers code blanked, usage)."""
    parsed, usage = (call or _ask_claude)(SYSTEM, build_messages(history, draft), TurnOutput,
                                          TURN_MAX_TOKENS, "create rfx")
    summary, questions = parse_reply(parsed)
    new = RfxDraft.model_validate(parsed["rfx_draft"]).model_dump()
    new["lines"] = rows_to_lines(new["lines"])  # tidy blank strings into None
    buyer_texts = [strip_tags(h["content"]) for h in history if h["role"] == "user"]
    new, notes = guard_numbers(new, draft, buyer_texts)
    return summary, questions, new, notes, usage
