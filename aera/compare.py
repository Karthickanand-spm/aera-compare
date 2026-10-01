"""Build the side-by-side comparison from extracted vendor replies. No AI here.

Inputs: the RFx, last year's prices, cached extraction results (replies and
certificates), FX rates, and (optionally) the text of Excel / Word / email
replies so source snippets can be checked.

Outputs, both pandas DataFrames:
- comparison: one row per (RFx line, vendor) with the INR-per-piece price, a
  comparability label, visible assumptions, and a confidence built only from
  checks code can run.
- vendor summary: one row per vendor with commercial terms, quality status
  against rfx.quality_bar, and open risks.

All price maths goes through normalize.py. Missing is never zero.

Run as a script on the cached sample extractions (no API calls):
    python -m aera.compare
"""

import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from aera.config import FX_DATE, FX_RATES, LINE_TOTAL_TOLERANCE_PCT, PHOTO_RERUN_TOLERANCE_PCT
from aera.normalize import MissingFxRate, NormalizeError, currency_code, normalize_price, to_per_piece
from aera.rfx import RFx, RfxLine

COMPARABLE = "Comparable"
WITH_ASSUMPTION = "Comparable with assumption"
NOT_COMPARABLE = "Not comparable"
NOT_QUOTED = "Not quoted"
LABELS = (COMPARABLE, WITH_ASSUMPTION, NOT_COMPARABLE, NOT_QUOTED)
COUNTED_LABELS = (COMPARABLE, WITH_ASSUMPTION)  # counted in totals by default

PASS, FAIL, UNCLEAR = "PASS", "FAIL", "UNCLEAR"

HIGH, MEDIUM, LOW = "high", "medium", "low"
SEVERITY_ORDER = (HIGH, MEDIUM, LOW)

INJECTION_REASON = ("Document contains instructions aimed at automated processing; values were extracted "
                    "as written, so confirm them before they count")
INJECTION_RISK = ("Document contains instructions aimed at automated processing: '{text}'. "
                  "Values were extracted as written; please review")
# Code backstop for text documents: phrasing that addresses an AI or tries to override instructions.
# A manipulated model might not report these itself.
_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompt)\b",
    r"\b(note|message|instructions?)\s+(to|for)\s+(any|all|the)?\s*(ai|llm|assistant|model|automated|bot)\b",
    r"\b(ai|llm|language model|chatbot)\s+(system|assistant|model)s?\b",
    r"\bsystem prompt\b",
)]
_MAX_QUOTE = 200
# A certificate expiring within this many months of the RFx date is a high risk.
CERT_EXPIRY_HIGH_RISK_MONTHS = 6

# Questionnaire / commercial answers that are not in the quality bar but the buyer still
# needs. Missing ones become a single "Not provided: ..." risk for the clarification step.
# (section, field, plain name)
INFO_ANSWERS = (
    ("questionnaire", "capacity_t_month", "capacity"),
    ("questionnaire", "lead_time_days", "lead time"),
    ("commercial_terms", "payment_days", "payment terms"),
)

LAST_YEAR_ASSUMPTION = "Resolved from last-year contract; vendor said 'same as last year'"
AMBIGUOUS_ASSUMPTION = (
    "Ambiguous: vendor wording fits more than one price; using the higher one until confirmed"
)

COMPARISON_COLUMNS = [
    "rfx_line_id", "vendor", "display_name", "description", "raw_price_text", "price_inr_per_piece",
    "label", "included_in_totals", "assumptions", "confidence", "confidence_reasons",
    "source_file", "source_snippet", "page", "quoted_spec", "notes",
    "needs_review", "buyer_confirmed", "buyer_decision", "alternatives",
    "missing_fx_currency",  # e.g. "EUR" when the price can't count until the buyer enters that rate
]

SUMMARY_COLUMNS = [
    "vendor", "display_name", "source_file", "lines_quoted", "lines_comparable", "lines_needing_review",
    "freight", "payment_days", "discounts", "quality_status", "quality_reasons",
    "open_risks", "certificate_file",
]


# ---------- Confidence (code checks only) ----------

class _Confidence:
    """Starts high; each failed check can only lower it. Every check leaves a reason."""

    ORDER = ("high", "medium", "low")

    def __init__(self):
        self.level = "high"
        self.reasons: list[str] = []

    def note(self, reason: str) -> None:
        self.reasons.append(reason)

    def lower(self, to: str, reason: str) -> None:
        if self.ORDER.index(to) > self.ORDER.index(self.level):
            self.level = to
        self.reasons.append(reason)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def snippet_in_text(snippet: str | None, text: str | None) -> bool:
    """True if the snippet appears in the text, ignoring case and runs of whitespace."""
    if not snippet or not text:
        return False
    return _squash(snippet) in _squash(text)


def find_snippet_span(snippet: str | None, text: str | None) -> tuple[int, int] | None:
    """(start, end) of the snippet inside the text, ignoring case and runs of whitespace.

    Same matching rule as snippet_in_text, but returns where it is so the UI can highlight it.
    """
    if not snippet or not text or not snippet.strip():
        return None
    pattern = r"\s+".join(re.escape(word) for word in snippet.split())
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def _first_number(text: str | None) -> float | None:
    """'1,20,000 pcs' -> 120000.0. None if there is no number."""
    if not text:
        return None
    m = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def _within_pct(a: float, b: float, pct: float) -> bool:
    return abs(a - b) <= abs(b) * pct / 100


# ---------- Price candidates ----------

def _price_candidate(entry: dict, rfx_line: RfxLine, last_year_prices: dict[int, float],
                     fx_rates: dict[str, float], fx_date: str | dict[str, str] | None) -> dict:
    """One possible reading of a vendor's price for a line, converted to INR per piece.

    Returns {"entry", "price_inr_per_piece" (None if it can't be converted), "assumptions", "error"}.
    """
    basis = entry.get("unit_basis")

    if basis == "reference_last_year":
        last_year = last_year_prices.get(rfx_line.line_id)
        if last_year is None:
            return _candidate(entry, None, [], "Vendor said 'same as last year' but there is no "
                                               "last-year price on file for this line")
        return _candidate(entry, last_year, [f"{LAST_YEAR_ASSUMPTION} (INR {last_year:g} per piece)"])

    if entry.get("price") is None:
        return _candidate(entry, None, [], "Vendor gave no number for this line")
    if not entry.get("currency"):
        return _candidate(entry, None, [], "Currency not stated")

    # pack_size is only meaningful for per_pack. Passing it for per_100 etc. could divide twice.
    pack_size = entry.get("pack_size") if basis == "per_pack" else None
    weight_g = rfx_line.nominal_weight_g if basis == "per_kg" else None
    try:
        value, notes = normalize_price(entry["price"], entry["currency"], basis, pack_size,
                                       weight_g, fx_rates, fx_date)
    except MissingFxRate as e:
        return _candidate(entry, None, [], f"Not comparable: no FX rate for {e.currency}",
                          missing_fx=e.currency)
    except NormalizeError as e:
        return _candidate(entry, None, [], f"Could not convert: {e}")
    if basis == "per_kg":
        notes = [f"{n} (RFx nominal weight)" if "box weight" in n else n for n in notes]
    return _candidate(entry, value, notes)


def _candidate(entry, price, assumptions, error=None, missing_fx=None) -> dict:
    return {"entry": entry, "price_inr_per_piece": price, "assumptions": assumptions, "error": error,
            "missing_fx": missing_fx}


def _alternative(c: dict) -> dict:
    e = c["entry"]
    return {
        "raw_price_text": e.get("raw_price_text"),
        "unit_basis": e.get("unit_basis"),
        "price_inr_per_piece": c["price_inr_per_piece"],
        "assumptions": c["assumptions"],
        "source_snippet": e.get("source_snippet"),
    }


def currencies_without_rate(extractions: list[dict], fx_rates: dict[str, float]) -> dict[str, dict]:
    """Currencies vendors priced in that have no FX rate yet: {code: {"vendors": n, "lines": n}}.

    Only priced entries count ("same as last year" and blank prices need no rate)."""
    found: dict[str, dict[str, set]] = {}
    for ext in extractions:
        vendor = vendor_name(ext)
        for entry in ext["runs"][0].get("lines", []):
            code = currency_code(entry.get("currency"))
            if (entry.get("price") is None or entry.get("unit_basis") == "reference_last_year"
                    or not code or code == "INR" or code in fx_rates):
                continue
            seen = found.setdefault(code, {"vendors": set(), "lines": set()})
            seen["vendors"].add(vendor)
            seen["lines"].add((vendor, entry.get("rfx_line_id")))
    return {code: {"vendors": len(v["vendors"]), "lines": len(v["lines"])} for code, v in sorted(found.items())}


# ---------- Instructions hidden in a document (prompt injection) ----------

def suspicious_texts(ext: dict, doc_text: str | None = None) -> list[str]:
    """Text in a vendor document aimed at an AI or at changing how it is processed.

    From Claude's document check (any file type) plus a code scan of the extracted text
    (Excel / Word / email). Vendor documents are data: these are reported, never followed.
    """
    found = [t.strip() for t in (ext.get("quote_check") or {}).get("suspicious_instructions") or []
             if t and t.strip()]
    for line in (doc_text or "").splitlines():
        line = line.strip()
        if line and any(p.search(line) for p in _INJECTION_PATTERNS):
            if not any(_squash(line) in _squash(t) or _squash(t) in _squash(line) for t in found):
                found.append(line)
    return [t if len(t) <= _MAX_QUOTE else t[:_MAX_QUOTE - 1].rstrip() + "…" for t in found]


def _flag_injection(rows: list[dict]) -> None:
    """Every value from a document with hidden instructions needs the buyer's review."""
    for row in rows:
        row["confidence"] = LOW
        row["confidence_reasons"] = list(row.get("confidence_reasons") or []) + [INJECTION_REASON]
        row["needs_review"] = True


# ---------- Comparison rows ----------

def build_comparison(rfx: RFx, last_year_prices: dict[int, float], extractions: list[dict],
                     fx_rates: dict[str, float], fx_date: str | dict[str, str] | None = None,
                     document_texts: dict[str, str | None] | None = None) -> pd.DataFrame:
    """One row per (RFx line, vendor). `extractions` are reply results as cached by extract.py.

    `document_texts` maps source_file -> extracted text (Excel / Word / email), used to
    check that each source snippet really appears in the document.
    """
    texts = document_texts or {}
    rows: list[dict] = []
    for ext in extractions:
        rows.extend(_vendor_rows(rfx, last_year_prices, ext, fx_rates, fx_date,
                                 texts.get(ext.get("source_file"))))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def vendor_name(ext: dict) -> str:
    return ext["runs"][0].get("vendor") or ext.get("source_file") or "Unknown vendor"


# Words kept upper case when tidying an ALL-CAPS name.
_KEEP_UPPER = {"EOU", "LLP", "OPC", "ISO", "SEZ", "MSME"}


def display_name(name: str) -> str:
    """A readable vendor name: 'DECCAN CORRUPACK PVT LTD' -> 'Deccan Corrupack Pvt Ltd'.

    Names that already mix upper and lower case are kept as the vendor wrote them.
    In ALL-CAPS names, words in brackets like '(EOU)' and known acronyms stay upper case.
    """
    name = " ".join((name or "").split())
    if name != name.upper():
        return name
    words = []
    for word in name.split(" "):
        core = word.strip("().,&")
        if (word.startswith("(") and word.endswith(")")) or core in _KEEP_UPPER:
            words.append(word)
        else:
            words.append(word.title())
    return " ".join(words)


def _vendor_rows(rfx, last_year_prices, ext, fx_rates, fx_date, doc_text) -> list[dict]:
    first_run, other_runs = ext["runs"][0], ext["runs"][1:]
    vendor = vendor_name(ext)
    file_kind = ext.get("file_kind")

    entries_by_line: dict[int, list[dict]] = defaultdict(list)
    for entry in first_run.get("lines", []):
        entries_by_line[entry["rfx_line_id"]].append(entry)
    declined = {nq["rfx_line_id"]: nq for nq in first_run.get("not_quoted", [])}

    rows = []
    for rfx_line in rfx.lines:
        entries = entries_by_line.get(rfx_line.line_id, [])
        other_readings = [_entries_for(run, rfx_line.line_id) for run in other_runs]
        base = {
            "rfx_line_id": rfx_line.line_id,
            "vendor": vendor,
            "display_name": display_name(vendor),
            "description": rfx_line.description,
            "source_file": ext.get("source_file"),
            "buyer_confirmed": False,
            "buyer_decision": None,
        }
        if entries:
            row = _quoted_row(entries, rfx_line, last_year_prices, fx_rates, fx_date,
                              file_kind, doc_text, other_readings)
        else:
            row = _not_quoted_row(declined.get(rfx_line.line_id), other_readings)
        rows.append({**base, **row})
    if suspicious_texts(ext, doc_text):
        _flag_injection(rows)
    return rows


def _entries_for(run: dict, line_id: int) -> list[dict]:
    return [e for e in run.get("lines", []) if e["rfx_line_id"] == line_id]


def _not_quoted_row(declined: dict | None, other_readings: list[list[dict]]) -> dict:
    conf = _Confidence()
    if declined:
        reason = declined.get("reason")
        conf.note("Vendor explicitly declined this line" + (f": {reason}" if reason else ""))
    else:
        conf.lower("medium", "Line not mentioned anywhere in the reply")
    if any(other_readings):
        conf.lower("low", "Another reading of the photo found a price for this line")
    return {
        "raw_price_text": None,
        "price_inr_per_piece": None,  # missing is never zero
        "label": NOT_QUOTED,
        "included_in_totals": False,
        "assumptions": [],
        "confidence": conf.level,
        "confidence_reasons": conf.reasons,
        "source_snippet": declined.get("source_snippet") if declined else None,
        "page": None,
        "quoted_spec": None,
        "notes": None,
        "needs_review": conf.level == "low",
        "alternatives": [],
    }


def _quoted_row(entries, rfx_line, last_year_prices, fx_rates, fx_date,
                file_kind, doc_text, other_readings) -> dict:
    candidates = [_price_candidate(e, rfx_line, last_year_prices, fx_rates, fx_date) for e in entries]
    priced = [c for c in candidates if c["price_inr_per_piece"] is not None]
    # Several readings: default to the HIGHER price so savings are never overstated.
    chosen = max(priced, key=lambda c: c["price_inr_per_piece"]) if priced else candidates[0]
    alternatives = [_alternative(c) for c in candidates if c is not chosen]
    entry = chosen["entry"]
    price = chosen["price_inr_per_piece"]
    assumptions = list(chosen["assumptions"])
    conf = _Confidence()

    # Check 1: does the snippet really appear in the document?
    snippet = entry.get("source_snippet")
    if file_kind in ("pdf", "image"):
        conf.lower("medium", f"Read from a {'photo' if file_kind == 'image' else 'PDF'}; "
                             "snippet cannot be matched against extracted text")
    elif doc_text is None:
        conf.lower("medium", "Document text not available to check the snippet")
    elif snippet_in_text(snippet, doc_text):
        conf.note("Source snippet found in the document text")
    else:
        conf.lower("low", "Source snippet NOT found in the document text")

    # Check 2: a conversion was needed.
    if assumptions:
        conf.lower("medium", "Needed a conversion: " + "; ".join(assumptions))

    # Check 3: vendor's own line total = qty x unit price.
    _check_line_total(entry, rfx_line, conf)

    # Check 4: the reading itself is ambiguous.
    if len(candidates) > 1:
        assumptions.append(AMBIGUOUS_ASSUMPTION)
        conf.lower("low", f"Vendor wording fits {len(candidates)} different prices for this line")
    if entry.get("is_ambiguous"):
        conf.lower("low", "Extraction flagged as ambiguous"
                   + (f": {entry['ambiguity_reason']}" if entry.get("ambiguity_reason") else ""))

    # Check 5: photos are read twice; the readings must agree.
    for i, other in enumerate(other_readings, start=2):
        if _readings_agree(entries, other):
            conf.note(f"Photo reading 1 and reading {i} agree")
        else:
            conf.lower("low", f"Photo readings disagree: reading 1 {_describe(entries)}, "
                              f"reading {i} {_describe(other)}")

    if price is None:
        conf.lower("low", chosen["error"])

    spec = entry.get("quoted_spec_if_different")
    if price is None or spec:
        label = NOT_COMPARABLE
        if spec:
            conf.note(f"Vendor quoted a different spec: {spec}")
    elif assumptions:
        label = WITH_ASSUMPTION
    else:
        label = COMPARABLE

    return {
        "raw_price_text": entry.get("raw_price_text"),
        "price_inr_per_piece": price,
        "label": label,
        "included_in_totals": label in COUNTED_LABELS,
        "assumptions": assumptions,
        "confidence": conf.level,
        "confidence_reasons": conf.reasons,
        "source_snippet": snippet,
        "page": entry.get("page"),
        "quoted_spec": spec,
        "notes": entry.get("interpretation_note"),
        "needs_review": conf.level == "low",
        "alternatives": alternatives,
        "missing_fx_currency": chosen["missing_fx"] if price is None else None,
    }


def _check_line_total(entry: dict, rfx_line: RfxLine, conf: _Confidence) -> None:
    qty = _first_number(entry.get("vendor_qty_text"))
    total = _first_number(entry.get("vendor_line_total_text"))
    price = entry.get("price")
    basis = entry.get("unit_basis")
    if qty is None or total is None or price is None or basis in ("per_kg", "reference_last_year"):
        return
    try:
        pack = entry.get("pack_size") if basis == "per_pack" else None
        per_piece, _ = to_per_piece(price, basis, pack)  # in the vendor's own currency
    except NormalizeError:
        return
    expected = qty * per_piece
    if _within_pct(total, expected, LINE_TOTAL_TOLERANCE_PCT):
        conf.note(f"Vendor's line total {total:g} matches qty x price")
    else:
        conf.lower("low", f"Vendor's line total {total:g} does not match qty x price ({expected:g})")


def _signature(entries: list[dict]) -> list[tuple]:
    return sorted(
        (e.get("unit_basis") or "", (e.get("currency") or "").upper(),
         e.get("pack_size") if e.get("unit_basis") == "per_pack" else None, e.get("price"))
        for e in entries
    )


def _readings_agree(a: list[dict], b: list[dict]) -> bool:
    sa, sb = _signature(a), _signature(b)
    if len(sa) != len(sb):
        return False
    for (basis1, cur1, pack1, p1), (basis2, cur2, pack2, p2) in zip(sa, sb):
        if (basis1, cur1, pack1) != (basis2, cur2, pack2):
            return False
        if (p1 is None) != (p2 is None):
            return False
        if p1 is not None and not _within_pct(p1, p2, PHOTO_RERUN_TOLERANCE_PCT):
            return False
    return True


def _describe(entries: list[dict]) -> str:
    if not entries:
        return "found no price"
    return " / ".join(f"'{e.get('raw_price_text')}'" for e in entries)


# ---------- Buyer review decisions ----------

USE_EXTRACTED, USE_ALTERNATIVE, BUYER_EDIT = "extracted", "alternative", "edit"


def buyer_decision(choice: str, decided_at: str, alternative_index: int | None = None,
                   value_inr_per_piece: float | None = None) -> dict:
    """A buyer's decision on one comparison row. Stored in the UI, applied by apply_buyer_decisions."""
    if choice not in (USE_EXTRACTED, USE_ALTERNATIVE, BUYER_EDIT):
        raise ValueError(f"Unknown decision '{choice}'")
    if choice == USE_ALTERNATIVE and alternative_index is None:
        raise ValueError("Choosing an alternative needs its index")
    if choice == BUYER_EDIT and (value_inr_per_piece is None or value_inr_per_piece <= 0):
        raise ValueError("An edited price must be a positive INR-per-piece value")
    return {"choice": choice, "decided_at": decided_at, "alternative_index": alternative_index,
            "value_inr_per_piece": value_inr_per_piece}


def apply_buyer_decisions(comparison: pd.DataFrame,
                          decisions: dict[tuple[int, str], dict] | None) -> pd.DataFrame:
    """Return a copy of the comparison with the buyer's decisions applied.

    `decisions` maps (rfx_line_id, vendor) -> buyer_decision(...). Each decided row gets
    buyer_confirmed = True and a plain-English buyer_decision. Choosing an alternative or
    typing a price is recorded as a visible assumption. A decision that no longer fits
    its row (e.g. the alternative is gone after re-extraction) is ignored.
    """
    out = comparison.copy()
    for (line_id, vendor), d in (decisions or {}).items():
        match = out.index[(out["rfx_line_id"] == line_id) & (out["vendor"] == vendor)]
        if len(match) == 0:
            continue
        i = match[0]
        updated = _decided_row(out.loc[i].to_dict(), d)
        if updated is not None:
            for key, value in updated.items():
                out.at[i, key] = value
    return out


def _decided_row(row: dict, d: dict) -> dict | None:
    # pandas stores a blank text cell as NaN when other rows in the column have text,
    # and NaN is truthy. Read blanks back as None before checking anything.
    row = {k: None if _missing(v) else v for k, v in row.items()}
    when = d["decided_at"]
    old_price = row["price_inr_per_piece"]
    old_price_text = "no price" if _missing(old_price) else f"INR {old_price:.2f} per piece"
    changes = {"buyer_confirmed": True}

    if d["choice"] == USE_EXTRACTED:
        changes["buyer_decision"] = f"Buyer confirmed the extracted reading ({when})"
        changes["confidence_reasons"] = row["confidence_reasons"] + [changes["buyer_decision"]]
        return changes

    if d["choice"] == USE_ALTERNATIVE:
        alts = row["alternatives"] or []
        idx = d.get("alternative_index")
        if idx is None or not 0 <= idx < len(alts) or alts[idx]["price_inr_per_piece"] is None:
            return None
        alt = alts[idx]
        previous = {
            "raw_price_text": row["raw_price_text"], "unit_basis": None,
            "price_inr_per_piece": old_price,
            "assumptions": [a for a in row["assumptions"] if a != AMBIGUOUS_ASSUMPTION],
            "source_snippet": row["source_snippet"],
        }
        decision = (f"Buyer chose the alternative reading '{alt['raw_price_text']}' "
                    f"instead of '{row['raw_price_text']}' ({when})")
        changes.update({
            "raw_price_text": alt["raw_price_text"],
            "price_inr_per_piece": alt["price_inr_per_piece"],
            "source_snippet": alt["source_snippet"],
            "assumptions": list(alt["assumptions"]) + [decision],
            "alternatives": [a for j, a in enumerate(alts) if j != idx] + [previous],
        })
    else:  # BUYER_EDIT
        value = float(d["value_inr_per_piece"])
        decision = (f"Buyer entered INR {value:.2f} per piece by hand ({when}); "
                    f"extracted reading was {old_price_text}")
        changes.update({"price_inr_per_piece": value, "assumptions": [decision]})

    label = NOT_COMPARABLE if row["quoted_spec"] else WITH_ASSUMPTION
    changes.update({
        "label": label,
        "included_in_totals": label in COUNTED_LABELS,
        "buyer_decision": decision,
        "confidence_reasons": row["confidence_reasons"] + [decision],
    })
    return changes


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


# ---------- Quality ----------

_NAME_NOISE = {"pvt", "private", "ltd", "limited", "llp", "inc", "co", "company", "the",
               "and", "m", "s", "eou"}


def _name_tokens(name: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (name or "").lower())) - _NAME_NOISE


def match_certificate(vendor: str, certificates: list[dict]) -> dict | None:
    """The certificate whose holder name best matches the vendor name (word overlap), or None."""
    vendor_tokens = _name_tokens(vendor)
    best, best_score = None, 0.0
    for cert in certificates:
        holder_tokens = _name_tokens(cert["runs"][0].get("holder"))
        if not vendor_tokens or not holder_tokens:
            continue
        score = len(vendor_tokens & holder_tokens) / len(vendor_tokens | holder_tokens)
        if score > best_score:
            best, best_score = cert, score
    return best if best_score >= 0.5 else None


def parse_date(text: str | None) -> date | None:
    """Full dates only ('2026-12-31', '31-Aug-2027', '15 Jan 2028'). Vague ones ('Dec 26') -> None."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _claim_date(claim: dict | None) -> date | None:
    if not claim:
        return None
    return parse_date(claim.get("iso_date")) or parse_date(claim.get("value"))


def parse_months(text: str | None) -> int | None:
    """'12 months' -> 12, '1 year' -> 12. None if no length is stated."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*(month|year)", text.lower())
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2) == "year" else n


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    for day in (d.day, 30, 29, 28):  # clamp e.g. 31 Jan + 1 month to the month's last day
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise ValueError(f"Cannot add {months} months to {d}")


def whole_months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months - 1 if end.day < start.day else months


def _cert_expiry(cert: dict) -> tuple[date | None, str | None]:
    """(expiry date, problem text). Photos have two readings, which must agree."""
    dates = [_claim_date(run.get("valid_until")) for run in cert["runs"]]
    if dates[0] is None:
        return None, "certificate expiry date could not be read"
    if any(d != dates[0] for d in dates[1:]):
        return None, "two readings of the certificate give different expiry dates"
    return dates[0], None


def assess_quality(rfx: RFx, questionnaire: dict | None, certificate: dict | None) -> dict:
    """Check a vendor against rfx.quality_bar using questionnaire answers AND the certificate.

    The certificate beats the questionnaire. A missing answer is "not provided" (UNCLEAR),
    never a pass. Returns {"status", "reasons", "risks"}.
    """
    q = questionnaire or {}
    bar = rfx.quality_bar
    outcomes: list[str] = []
    reasons: list[str] = []
    risks: list[str] = []

    def record(outcome: str, reason: str) -> None:
        outcomes.append(outcome)
        reasons.append(f"{outcome}: {reason}")

    for key, required in bar.items():
        if key == "iso9001_valid_on":
            _check_iso(rfx, q, certificate, required, record, reasons, risks)
        elif key == "max_defect_rate_pct":
            rate = (q.get("defect_rate_pct") or {}).get("value")
            if rate is None:
                record(UNCLEAR, "defect rate not provided")
            elif rate > required:
                record(FAIL, f"defect rate {rate:g}% is above the {required:g}% limit")
            else:
                record(PASS, f"defect rate {rate:g}% is within the {required:g}% limit")
        elif key == "test_report_per_batch":
            if not required:
                continue
            answer = (q.get("test_report_per_batch") or {}).get("value")
            if answer == "yes":
                record(PASS, "test report supplied with every batch")
            elif answer == "no":
                record(FAIL, "vendor does not supply a test report with every batch")
            elif answer == "on_request":
                record(UNCLEAR, "test report only 'on request', not confirmed for every batch")
            else:
                record(UNCLEAR, "test report per batch not provided")
        else:
            record(UNCLEAR, f"requirement '{key}' is not checked automatically")

    if FAIL in outcomes:
        status = FAIL
    elif UNCLEAR in outcomes:
        status = UNCLEAR
    else:
        status = PASS
    return {"status": status, "reasons": reasons, "risks": risks}


def _check_iso(rfx, q, certificate, required_on, record, reasons, risks) -> None:
    required = parse_date(required_on) or parse_date(rfx.issued)
    claimed = (q.get("iso9001_claimed") or {}).get("value")
    claim_date_field = q.get("cert_valid_until_claimed") or {}
    claim_text = claim_date_field.get("value")

    if certificate is None:
        if claimed is True:
            when = f" (vendor says valid until '{claim_text}')" if claim_text else ""
            record(UNCLEAR, f"vendor claims ISO 9001{when} but no certificate was provided")
        elif claimed is False:
            record(FAIL, "vendor says it does not hold ISO 9001")
        else:
            record(UNCLEAR, "ISO 9001 not stated and no certificate provided")
        return

    cert_run = certificate["runs"][0]
    cert_file = certificate.get("source_file") or cert_run.get("source_file")
    standard = cert_run.get("standard") or ""
    if "9001" not in standard:
        record(UNCLEAR, f"certificate {cert_file} is not ISO 9001 (standard: '{standard}')")
        return
    expiry, problem = _cert_expiry(certificate)
    if expiry is None:
        record(UNCLEAR, f"{problem} ({cert_file})")
        return

    if expiry < required:
        claim_note = " even though the vendor claims ISO 9001" if claimed else ""
        record(FAIL, f"ISO 9001 certificate expired {expiry.isoformat()}, before the RFx date "
                     f"{required.isoformat()}{claim_note} ({cert_file})")
    else:
        record(PASS, f"ISO 9001 certificate valid until {expiry.isoformat()} ({cert_file})")
        months = parse_months(rfx.terms.get("validity"))
        issued = parse_date(rfx.issued)
        if months is None or issued is None:
            reasons.append("NOTE: contract length unknown; mid-contract expiry not checked")
        else:
            contract_end = add_months(issued, months)
            if expiry < contract_end:
                n = whole_months_between(issued, expiry)
                soon = expiry < add_months(issued, CERT_EXPIRY_HIGH_RISK_MONTHS)
                risks.append(risk(HIGH if soon else LOW,
                                  f"ISO certificate expires {expiry.isoformat()}, {n} months into "
                                  "the contract; ask for renewal evidence"))

    # Reconcile what the vendor wrote with the certificate. The certificate wins.
    claim_date = _claim_date(claim_date_field)
    if claim_date is not None and claim_date != expiry:
        reasons.append(f"NOTE: vendor wrote ISO valid until {claim_date.isoformat()}, certificate "
                       f"says {expiry.isoformat()}; certificate used")
        risks.append(risk(MEDIUM, f"Vendor's claimed ISO expiry ({claim_date.isoformat()}) "
                                  f"differs from the certificate ({expiry.isoformat()})"))
    elif claim_date is None and claim_text:
        reasons.append(f"NOTE: vendor wrote ISO valid till '{claim_text}'; certificate gives the "
                       f"full date {expiry.isoformat()}, used that")


# ---------- Vendor summary ----------

def risk(severity: str, text: str) -> dict:
    return {"severity": severity, "text": text}


def sort_risks(risks: list[dict]) -> list[dict]:
    """High first, then medium, then low. Keeps the original order within a severity."""
    return sorted(risks, key=lambda r: SEVERITY_ORDER.index(r["severity"]))


def _freight_risk(freight: dict | None) -> dict | None:
    value = (freight or {}).get("value")
    snippet = (freight or {}).get("source_snippet")
    said = f" (vendor wrote '{snippet}')" if snippet else ""
    if value == "extra":
        return risk(HIGH, f"Freight extra{said}: delivered cost unknown until freight is quoted")
    if value in ("unclear", None):
        return risk(MEDIUM, f"Freight terms unclear{said}: ask whether prices are delivered to site")
    return None


def _missing_answers_risk(run: dict) -> dict | None:
    """One medium risk listing the non-quality-bar answers the vendor left out."""
    missing = [
        label for section, field, label in INFO_ANSWERS
        if ((run.get(section) or {}).get(field) or {}).get("value") is None
    ]
    if not missing:
        return None
    return risk(MEDIUM, "Not provided: " + ", ".join(missing))


def _discount_text(d: dict) -> str:
    cond = f" (condition: {d['condition']})" if d.get("condition") else ""
    return f"{d.get('text')}{cond} [recorded, not applied]"


def build_vendor_summary(rfx: RFx, extractions: list[dict], certificates: list[dict],
                         comparison: pd.DataFrame,
                         document_texts: dict[str, str | None] | None = None) -> pd.DataFrame:
    texts = document_texts or {}
    rows = []
    for ext in extractions:
        vendor = vendor_name(ext)
        first_run = ext["runs"][0]
        terms = first_run.get("commercial_terms") or {}
        vendor_rows = comparison[comparison["vendor"] == vendor]
        certificate = match_certificate(vendor, certificates)
        quality = assess_quality(rfx, first_run.get("questionnaire"), certificate)

        open_risks = [r for r in (_freight_risk(terms.get("freight")),
                                  _missing_answers_risk(first_run)) if r]
        open_risks += [risk(HIGH, INJECTION_RISK.format(text=t))
                       for t in suspicious_texts(ext, texts.get(ext.get("source_file")))]
        open_risks = sort_risks(open_risks + quality["risks"])

        rows.append({
            "vendor": vendor,
            "display_name": display_name(vendor),
            "source_file": ext.get("source_file"),
            "lines_quoted": int((vendor_rows["label"] != NOT_QUOTED).sum()),
            "lines_comparable": int(vendor_rows["label"].isin(COUNTED_LABELS).sum()),
            "lines_needing_review": int((vendor_rows["needs_review"]
                                         & ~vendor_rows["buyer_confirmed"].astype(bool)).sum()),
            "freight": (terms.get("freight") or {}).get("value"),
            "payment_days": (terms.get("payment_days") or {}).get("value"),
            "discounts": [_discount_text(d) for d in terms.get("discounts") or []],
            "quality_status": quality["status"],
            "quality_reasons": quality["reasons"],
            "open_risks": open_risks,
            "certificate_file": certificate.get("source_file") if certificate else None,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compare(rfx: RFx, last_year_prices: dict[int, float], extractions: list[dict],
            certificates: list[dict], fx_rates: dict[str, float],
            fx_date: str | dict[str, str] | None = None,
            document_texts: dict[str, str | None] | None = None,
            decisions: dict[tuple[int, str], dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (comparison, vendor_summary). `decisions` are the buyer's review choices."""
    comparison = build_comparison(rfx, last_year_prices, extractions, fx_rates, fx_date, document_texts)
    comparison = apply_buyer_decisions(comparison, decisions)
    summary = build_vendor_summary(rfx, extractions, certificates, comparison, document_texts)
    return comparison, summary


# ---------- Script ----------

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def load_cached_sample(sample_dir: Path = SAMPLE_DIR) -> dict:
    """Load the RFx, last-year prices and CACHED extractions for every sample file. No API calls."""
    from aera.extract import _read_cache  # imported here: pulls in the anthropic SDK
    from aera.ingest import load_reply
    from aera.rfx import load_last_year_prices, load_rfx

    rfx = load_rfx(sample_dir / "rfx.json")
    last_year: dict[int, float] = {}
    for csv_path in sorted(sample_dir.glob("last_year*.csv")):
        last_year.update(load_last_year_prices(csv_path))

    extractions, certificates, texts, missing = [], [], {}, []
    for folder, kind, rfx_id, bucket in (
        ("replies", "reply", rfx.rfx_id, extractions),
        ("certificates", "certificate", None, certificates),
    ):
        for path in sorted((sample_dir / folder).iterdir()):
            if not path.is_file():
                continue
            payload = load_reply(path)
            cached = _read_cache(payload.sha256, kind, rfx_id)
            if cached is None:
                missing.append(path.name)
                continue
            bucket.append(cached)
            texts[payload.filename] = payload.text
    return {"rfx": rfx, "last_year": last_year, "extractions": extractions,
            "certificates": certificates, "texts": texts, "missing": missing}


def _fmt_price(v) -> str:
    return "None" if v is None or pd.isna(v) else f"{v:.2f}"


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    data = load_cached_sample()
    for name in data["missing"]:
        print(f"WARNING: no cached extraction for {name}; run python -m aera.extract on it first.")
    comparison, summary = compare(data["rfx"], data["last_year"], data["extractions"],
                                  data["certificates"], FX_RATES, FX_DATE, data["texts"])

    print("=" * 90)
    print("VENDOR SUMMARY")
    print("=" * 90)
    print(summary[["display_name", "lines_quoted", "lines_comparable", "lines_needing_review",
                   "freight", "payment_days", "quality_status"]].to_string(index=False))
    for _, s in summary.iterrows():
        print(f"\n{s['display_name']}  ({s['source_file']}; certificate: {s['certificate_file']})")
        print(f"  Quality: {s['quality_status']}")
        for r in s["quality_reasons"]:
            print(f"    - {r}")
        print("  Open risks:" + ("" if s["open_risks"] else " none"))
        for r in s["open_risks"]:
            print(f"    - [{r['severity'].upper()}] {r['text']}")
        if s["discounts"]:
            print("  Discounts:")
            for d in s["discounts"]:
                print(f"    - {d}")

    print("\n" + "=" * 90)
    print("LABEL COUNTS PER VENDOR")
    print("=" * 90)
    counts = pd.crosstab(comparison["display_name"], comparison["label"]).reindex(columns=list(LABELS),
                                                                             fill_value=0)
    print(counts.to_string())

    review = comparison[comparison["needs_review"]]
    print("\n" + "=" * 90)
    print(f"ROWS NEEDING REVIEW ({len(review)})")
    print("=" * 90)
    for _, r in review.iterrows():
        print(f"\n[{r['display_name']}] line {r['rfx_line_id']} {r['description']}")
        print(f"  vendor wrote: {r['raw_price_text']!r} -> INR/piece {_fmt_price(r['price_inr_per_piece'])}"
              f"  | {r['label']} | confidence {r['confidence']}")
        for a in r["assumptions"]:
            print(f"  assumption: {a}")
        for reason in r["confidence_reasons"]:
            print(f"  why: {reason}")
        for alt in r["alternatives"]:
            print(f"  alternative: {alt['raw_price_text']!r} -> INR/piece "
                  f"{_fmt_price(alt['price_inr_per_piece'])}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
