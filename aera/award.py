"""Build an award from the comparison. All the arithmetic for the Decide page. No AI, no Streamlit.

Line allocation reuses the analyst's award logic (analyses.award_by_line: cheapest counted
price per line among a set of vendors) and its freight sensitivity, so the Ask page and
the Decide page always agree on who wins a line.

Approaches:
- single vendor: one vendor takes every line it quoted (best vendor chosen, or the buyer's pick)
- cheapest per line: every allowed vendor competes on every line
- capped: the best combination of N allowed vendors, tried by brute force

"Best" means most lines covered first, then lowest annual total, so a vendor that quoted
fewer lines never wins just by looking cheaper. Missing is never zero: a line no allowed
vendor priced stays "not awarded" with no cost.
"""

import io
import re
from dataclasses import dataclass, field, replace
from itertools import combinations

import pandas as pd

from aera.analyses import AnalystData, SensitivityError, award_by_line, freight_sensitivity, isolated
from aera.money import describe_change, excel_header, format_inr
from aera.compare import (
    COUNTED_LABELS, FAIL, HIGH, LOW, MEDIUM, NOT_COMPARABLE, PASS, UNCLEAR, display_name, risk,
    sort_risks, vendor_name,
)
from aera.normalize import describe_rates

SINGLE = "Single vendor"
CHEAPEST = "Cheapest per line"
CAPPED = "Cheapest per line, capped at N vendors"
APPROACHES = (SINGLE, CHEAPEST, CAPPED)
MAX_CAP = 4

LAKH, CRORE = 100_000, 10_000_000
NOT_AWARDED = "Not awarded"


# ---------- Allowed vendors ----------

def vendor_choices(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per vendor: display_name, quality_status, reason (why not PASS), default_allowed.

    Only quality PASS vendors are allowed by default. The reason is the vendor's FAIL or
    UNCLEAR quality findings, so the buyer sees why a vendor starts unticked.
    """
    rows = []
    for s in summary.itertuples(index=False):
        status = s.quality_status
        prefix = f"{status}:" if status in (FAIL, UNCLEAR) else None
        found = [r.split(":", 1)[1].strip() for r in s.quality_reasons or [] if prefix and r.startswith(prefix)]
        rows.append({"display_name": s.display_name, "quality_status": status,
                     "reason": "; ".join(found) if found else ("" if status == PASS else "quality not passed"),
                     "default_allowed": status == PASS})
    return pd.DataFrame(rows, columns=["display_name", "quality_status", "reason", "default_allowed"])


# ---------- Discounts ----------

def vendor_discounts(extractions: list[dict]) -> dict[str, list[dict]]:
    """{display_name: [discount as extracted: text, condition, percent, source_snippet]}."""
    out: dict[str, list[dict]] = {}
    for ext in extractions:
        terms = ext["runs"][0].get("commercial_terms") or {}
        found = [d for d in terms.get("discounts") or [] if d]
        if found:
            out[display_name(vendor_name(ext))] = found
    return out


_ANNUAL = re.compile(r"\b(annual|annually|yearly|per\s+annum|p\.\s?a\.|per\s+year|a\s+year)\b", re.I)
_VALUE = re.compile(r"\b(order|orders|purchase|purchases|business|turnover|offtake|value|spend)\b", re.I)
_STRICT = re.compile(r"\b(exceed|exceeds|exceeding|above|over|more\s+than|greater\s+than|beyond)\b", re.I)
_INCLUSIVE = re.compile(r"\b(at\s+least|minimum|min\.?|or\s+more|and\s+above|not\s+less\s+than)\b|>=|≥", re.I)
_AMOUNT = re.compile(
    r"(?:(?P<cur>rs\.?|inr|₹|usd|\$|eur|€)\s*)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>crores?|cr\b|lakhs?|lacs?|lac\b|l\b|lakh\b|million|mn\b|k\b)?", re.I)
_UNIT_VALUE = {"crore": CRORE, "crores": CRORE, "cr": CRORE, "lakh": LAKH, "lakhs": LAKH,
               "lac": LAKH, "lacs": LAKH, "l": LAKH, "million": 1_000_000, "mn": 1_000_000, "k": 1_000}


def order_value_threshold(condition: str | None) -> dict | None:
    """Read 'annual order value over ₹X' from a discount condition. None if it isn't that kind.

    Returns {"amount_inr": float, "inclusive": bool}. Only INR amounts (Rs, INR, ₹ or no
    currency) are read; any other currency or condition can't be checked and returns None.
    """
    text = condition or ""
    if not (_ANNUAL.search(text) and _VALUE.search(text)):
        return None
    inclusive = bool(_INCLUSIVE.search(text))
    if not (inclusive or _STRICT.search(text)):
        return None
    amounts = []
    for m in _AMOUNT.finditer(text):
        cur = (m.group("cur") or "").lower().rstrip(".")
        if cur in ("usd", "$", "eur", "€"):
            return None
        value = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").lower()
        amounts.append(value * _UNIT_VALUE.get(unit, 1))
    if len(amounts) != 1:  # none, or several numbers we can't tell apart
        return None
    return {"amount_inr": amounts[0], "inclusive": inclusive}


def check_discount(discount: dict, order_value_inr: float, apply: bool) -> dict:
    """Is this vendor's discount condition met by the award? Applied only if `apply` and met.

    Returns {"text", "condition", "percent", "status" ('met' | 'not met' | 'cannot check'),
             "threshold_inr", "order_value_inr", "applied", "discount_inr", "explanation"}.
    """
    percent = discount.get("percent")
    condition = (discount.get("condition") or "").strip()
    out = {"text": discount.get("text"), "condition": condition, "percent": percent,
           "threshold_inr": None, "order_value_inr": order_value_inr, "applied": False,
           "discount_inr": 0.0, "status": "cannot check", "explanation": ""}
    if percent is None or not 0 < float(percent) < 100:
        out["explanation"] = "the discount has no usable percentage"
        return out
    if condition:
        threshold = order_value_threshold(condition)
        if threshold is None:
            out["explanation"] = f"condition '{condition}' can't be checked automatically"
            return out
        out["threshold_inr"] = threshold["amount_inr"]
        limit = format_inr(threshold["amount_inr"])
        value = format_inr(order_value_inr)
        met = (order_value_inr >= threshold["amount_inr"] if threshold["inclusive"]
               else order_value_inr > threshold["amount_inr"])
        out["status"] = "met" if met else "not met"
        word = "at least" if threshold["inclusive"] else "over"
        out["explanation"] = (f"annual order value in this award is {value}, "
                              f"{'which is' if met else 'not'} {word} {limit}")
    else:
        out["status"] = "met"
        out["explanation"] = "no condition stated"
    if out["status"] == "met" and apply:
        out["applied"] = True
        out["discount_inr"] = order_value_inr * float(percent) / 100
    return out


# ---------- The award ----------

@dataclass
class AwardSettings:
    approach: str = CHEAPEST
    allowed: list[str] = field(default_factory=list)  # display names
    cap: int = 2  # N for the capped approach
    single_vendor: str | None = None  # for SINGLE: the buyer's pick, or None for the best one
    include_not_comparable: bool = False
    apply_discounts: bool = False

    def describe(self) -> str:
        if self.approach == CAPPED:
            return f"Cheapest per line, capped at {self.cap} vendor{'s' if self.cap != 1 else ''}"
        return self.approach


@dataclass
class Award:
    settings: AwardSettings
    lines: pd.DataFrame  # one row per RFx line (see LINE_COLUMNS)
    vendor_subtotals: pd.DataFrame
    vendors_used: list[str]
    total_inr: float | None  # after any applied discounts; None if nothing was awarded
    saving_inr: float | None  # against last year, on lines that have a last-year price
    saving_pct: float | None
    saving_lines: list[int]  # lines in the saving
    lines_without_last_year: list[int]
    unawarded_lines: list[int]
    unconfirmed: pd.DataFrame  # awarded rows that need review and aren't confirmed
    discounts: list[dict]  # check_discount() results, with "vendor"
    sensitivity: list[dict]  # freight_sensitivity() results, with "vendor"
    risks: list[dict]  # {"severity", "vendor", "text"}, high first
    assumptions: list[str]
    combinations_tried: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.vendors_used


LINE_COLUMNS = [
    "rfx_line_id", "description", "annual_qty", "vendor", "display_name", "quoted_inr_per_piece",
    "discount_pct", "price_inr_per_piece", "annual_inr", "last_year_inr_per_piece", "label",
    "confidence", "unconfirmed", "assumptions", "raw_price_text", "source_file", "source_snippet",
    "page", "confidence_reasons",
]


def award_data(data: AnalystData, include_not_comparable: bool) -> AnalystData:
    """A copy of the analyst data whose included_in_totals follows the award's toggle.

    'Not comparable' rows with a price count only when the buyer switches them on.
    """
    df = isolated(data.df)
    counted = df["label"].isin(COUNTED_LABELS)
    if include_not_comparable:
        counted |= (df["label"] == NOT_COMPARABLE) & df["price_inr_per_piece"].notna()
    df["included_in_totals"] = counted
    return replace(data, df=df)


def _allocate(data: AnalystData, names: list[str]) -> pd.DataFrame:
    """Per line: the winning vendor and its price, via the analyst's award_by_line."""
    won = award_by_line(data, names)
    won["annual_inr"] = won["annual_qty"] * won["winner_price"]
    return won


def _apply_discounts(won: pd.DataFrame, discounts: dict[str, list[dict]], apply: bool) -> tuple[float, list[dict]]:
    """(total after applied discounts, discount checks for vendors in this allocation)."""
    checks = []
    total = float(won["annual_inr"].sum())
    for name, value in won.groupby("winner")["annual_inr"].sum().items():
        for d in discounts.get(name, []):
            c = {"vendor": name, **check_discount(d, float(value), apply)}
            checks.append(c)
            total -= c["discount_inr"]
    return total, checks


def _candidate_sets(settings: AwardSettings, allowed: list[str]) -> list[tuple[str, ...]]:
    if settings.approach == CHEAPEST:
        return [tuple(allowed)]
    if settings.approach == SINGLE:
        if settings.single_vendor:
            return [(settings.single_vendor,)] if settings.single_vendor in allowed else []
        return [(v,) for v in allowed]
    if settings.approach == CAPPED:
        n = max(1, min(int(settings.cap), MAX_CAP, len(allowed)))
        return list(combinations(allowed, n))
    raise ValueError(f"Unknown award approach '{settings.approach}'")


def best_allocation(data: AnalystData, settings: AwardSettings,
                    discounts: dict[str, list[dict]]) -> tuple[pd.DataFrame, list[dict], float, int]:
    """Try each vendor set the approach allows; keep the one covering most lines, then cheapest.

    Returns (allocation, discount checks, total, number of sets tried). `data` must already
    come from award_data(). Ties on lines and total go to the alphabetically first set.
    """
    allowed = sorted(set(settings.allowed) & set(data.df["display_name"]))
    best, best_key = None, None
    sets = _candidate_sets(settings, allowed)
    for names in sets:
        won = _allocate(data, list(names))
        if won.empty:
            continue
        total, checks = _apply_discounts(won, discounts, settings.apply_discounts)
        key = (-len(won), round(total, 2), names)
        if best_key is None or key < best_key:
            best, best_key = (won, checks, total), key
    if best is None:
        return _allocate(data, []), [], 0.0, len(sets)
    return (*best, len(sets))


def build_award(data: AnalystData, summary: pd.DataFrame, settings: AwardSettings,
                discounts: dict[str, list[dict]] | None = None) -> Award:
    """The award for these settings: per-line table, vendor subtotals, saving, risks, assumptions.

    `data` is the analyst data for the current comparison (analyst.analyst_data), `summary`
    the vendor summary, and `discounts` comes from vendor_discounts(event.replies).
    """
    discounts = discounts or {}
    adata = award_data(data, settings.include_not_comparable)
    won, checks, total, tried = best_allocation(adata, settings, discounts)

    applied_pct = {c["vendor"]: float(c["percent"]) for c in checks if c["applied"]}
    lines = _line_table(adata, won, applied_pct)
    awarded = lines[lines["display_name"].notna()]
    vendors_used = sorted(awarded["display_name"].unique())
    unawarded = [int(i) for i in lines.loc[lines["display_name"].isna(), "rfx_line_id"]]
    unconfirmed = awarded[awarded["unconfirmed"].astype(bool)]

    with_ly = awarded[awarded["last_year_inr_per_piece"].notna()]
    no_ly = [int(i) for i in awarded.loc[awarded["last_year_inr_per_piece"].isna(), "rfx_line_id"]]
    saving = saving_pct = None
    if not with_ly.empty:
        ly_total = float((with_ly["annual_qty"] * with_ly["last_year_inr_per_piece"]).sum())
        saving = ly_total - float(with_ly["annual_inr"].sum())
        saving_pct = saving / ly_total * 100 if ly_total else None

    award = Award(
        settings=settings, lines=lines, vendor_subtotals=_subtotals(awarded, checks),
        vendors_used=vendors_used, total_inr=float(awarded["annual_inr"].sum()) if vendors_used else None,
        saving_inr=saving, saving_pct=saving_pct,
        saving_lines=[int(i) for i in with_ly["rfx_line_id"]], lines_without_last_year=no_ly,
        unawarded_lines=unawarded, unconfirmed=unconfirmed, discounts=checks, sensitivity=[],
        risks=[], assumptions=[], combinations_tried=tried,
    )
    award.sensitivity = _sensitivities(adata, award, summary)
    award.risks = _risks(award, summary)
    award.assumptions = _assumptions(adata, award, summary)
    return award


def _line_table(data: AnalystData, won: pd.DataFrame, applied_pct: dict[str, float]) -> pd.DataFrame:
    df = data.df
    last_year = data.last_year.set_index("line_id")["price_inr_per_piece"]
    winners = won.set_index("line_id")
    rows = []
    for ln in data.rfx_lines.itertuples(index=False):
        line_id = int(ln.line_id)
        base = {"rfx_line_id": line_id, "description": ln.description, "annual_qty": int(ln.annual_qty),
                "last_year_inr_per_piece": float(last_year[line_id]) if line_id in last_year.index else None}
        if line_id not in winners.index:
            rows.append({**base, "vendor": None, "display_name": None, "quoted_inr_per_piece": None,
                         "discount_pct": None, "price_inr_per_piece": None, "annual_inr": None,
                         "label": NOT_AWARDED, "confidence": None, "unconfirmed": False,
                         "assumptions": [], "raw_price_text": None, "source_file": None,
                         "source_snippet": None, "page": None, "confidence_reasons": []})
            continue
        w = winners.loc[line_id]
        r = df[(df["rfx_line_id"] == line_id) & (df["display_name"] == w["winner"])
               & df["included_in_totals"].astype(bool)].iloc[0]
        pct = applied_pct.get(w["winner"])
        quoted = float(w["winner_price"])
        price = quoted * (1 - pct / 100) if pct else quoted
        rows.append({**base, "vendor": r["vendor"], "display_name": w["winner"],
                     "quoted_inr_per_piece": quoted, "discount_pct": pct, "price_inr_per_piece": price,
                     "annual_inr": price * int(ln.annual_qty), "label": r["label"],
                     "confidence": r["confidence"],
                     "unconfirmed": bool(r["needs_review"]) and not bool(r["buyer_confirmed"]),
                     "assumptions": list(r["assumptions"] or []), "raw_price_text": r["raw_price_text"],
                     "source_file": r["source_file"], "source_snippet": r["source_snippet"],
                     "page": None if pd.isna(r["page"]) else r["page"],
                     "confidence_reasons": list(r["confidence_reasons"] or [])})
    return pd.DataFrame(rows, columns=LINE_COLUMNS)


def _subtotals(awarded: pd.DataFrame, checks: list[dict]) -> pd.DataFrame:
    rows = []
    for name, g in awarded.groupby("display_name"):
        quoted = float((g["annual_qty"] * g["quoted_inr_per_piece"]).sum())
        discount = sum((c["discount_inr"] for c in checks if c["vendor"] == name), 0.0)
        rows.append({"display_name": name, "lines_won": len(g), "quoted_annual_inr": quoted,
                     "discount_inr": discount, "annual_inr": float(g["annual_inr"].sum()),
                     "unconfirmed_lines": int(g["unconfirmed"].sum())})
    return pd.DataFrame(rows, columns=["display_name", "lines_won", "quoted_annual_inr", "discount_inr",
                                       "annual_inr", "unconfirmed_lines"])


def _vendor_row(summary: pd.DataFrame, name: str):
    match = summary[summary["display_name"] == name]
    return None if match.empty else match.iloc[0]


def _sensitivities(data: AnalystData, award: Award, summary: pd.DataFrame) -> list[dict]:
    """Freight sensitivity (the analyst's) for each awarded vendor with freight extra.

    Lines can only move to vendors this approach allows: all allowed vendors for cheapest per
    line, the chosen set for capped, and nobody else for a single vendor.
    """
    if award.settings.approach == CHEAPEST:
        pool = sorted(set(award.settings.allowed) & set(data.df["display_name"]))
    else:
        pool = award.vendors_used
    out = []
    for name in award.vendors_used:
        row = _vendor_row(summary, name)
        if row is None or row["freight"] != "extra":
            continue
        try:
            out.append({"vendor": name, "pool": pool, **freight_sensitivity(data, name, pool)})
        except SensitivityError as e:
            out.append({"vendor": name, "pool": pool, "error": str(e)})
    return out


def _risks(award: Award, summary: pd.DataFrame) -> list[dict]:
    """Open risks for the vendors in the award only, high first."""
    risks = []
    for r in award.unconfirmed.itertuples(index=False):
        risks.append({**risk(HIGH, f"Line {r.rfx_line_id} ({r.description}): price "
                                   f"{format_inr(r.quoted_inr_per_piece, per_unit=True)}/piece needs "
                                   "your review on the Compare page and is not confirmed"),
                      "vendor": r.display_name})
    for line_id in award.unawarded_lines:
        risks.append({**risk(HIGH, f"Line {line_id}: no allowed vendor has a counted price, so it "
                                   "is not awarded and not in the total"), "vendor": None})

    sens = {s["vendor"]: s for s in award.sensitivity}
    for name in award.vendors_used:
        row = _vendor_row(summary, name)
        if row is None:
            continue
        for r in row["open_risks"] or []:
            text = r["text"]
            if name in sens and text.startswith("Freight extra"):
                text += "; " + _freight_summary(sens[name])
            risks.append({**risk(r["severity"], text), "vendor": name})

    for c in award.discounts:
        pct = f"{c['percent']:g}%" if c["percent"] is not None else "discount"
        if c["applied"]:
            text = (f"{pct} discount applied ({c['explanation']}); the saving depends on the vendor "
                    "honouring it and on actually ordering that much")
            sev = MEDIUM
        elif c["status"] == "met":
            text = f"{pct} discount available but not applied ({c['explanation']})"
            sev = LOW
        elif c["status"] == "not met":
            text = f"{pct} discount not applied: condition not met ({c['explanation']})"
            sev = LOW
        else:
            text = f"{pct} discount not applied: {c['explanation']}"
            sev = MEDIUM
        risks.append({**risk(sev, text), "vendor": c["vendor"]})
    return sort_risks(risks)


def _freight_summary(s: dict) -> str:
    if s.get("error"):
        return f"how freight changes the saving can't be worked out ({s['error']})"
    first = s["table"].iloc[0]
    before = describe_change(first["saving_inr"], first["saving_pct"])
    if s["zero_rate"] is not None:
        return (f"award {before} against last year before freight; that saving is gone at about "
                f"{format_inr(s['zero_rate'], per_unit=True)}/kg of freight (sensitivity table below)")
    return f"award {before} against last year before freight (sensitivity table below)"


def _assumptions(data: AnalystData, award: Award, summary: pd.DataFrame) -> list[str]:
    s = award.settings
    out = []
    out.append(f"FX rate {describe_rates(data.fx_rates, data.fx_date)}.")
    out.append(f"Award approach: {s.describe()}. Lines go to the cheapest allowed vendor; when vendor "
               "sets cover different lines, the set covering more lines wins, then the lower total.")
    if s.approach != CHEAPEST and award.combinations_tried > 1:
        out.append(f"{award.combinations_tried} vendor combinations were compared by annual total.")
    out.append("Allowed vendors: " + (", ".join(sorted(s.allowed)) or "none") + ".")
    out.append("'Not comparable' prices (different spec) are " +
               ("INCLUDED in the award by your choice." if s.include_not_comparable
                else "excluded from the award."))
    out.append("Annual cost = RFx annual quantity x INR per piece, ex-GST. Missing prices are never "
               "counted as zero.")
    if award.saving_lines:
        extra = (f" Lines {award.lines_without_last_year} have no last-year price and are left out of the "
                 "saving." if award.lines_without_last_year else "")
        out.append(f"Saving vs last year compares the {len(award.saving_lines)} awarded lines that have a "
                   f"last-year price, at the same quantities.{extra}")
    else:
        out.append("No awarded line has a last-year price, so no saving against last year is shown.")
    if not award.unconfirmed.empty:
        out.append(f"{len(award.unconfirmed)} awarded price(s) are counted as extracted but not yet "
                   "confirmed by you.")
    for name in award.vendors_used:
        row = _vendor_row(summary, name)
        freight = None if row is None else row["freight"]
        if freight == "extra":
            out.append(f"{name}: prices exclude freight (vendor quoted freight extra); totals are before freight.")
        elif freight in ("unclear", None):
            out.append(f"{name}: freight terms unclear; prices treated as quoted, freight not added.")
    for c in award.discounts:
        if c["applied"]:
            out.append(f"{c['vendor']}: {c['percent']:g}% discount applied to its ex-GST annual order value "
                       f"({c['explanation']}). Lines were allocated on undiscounted prices.")
        else:
            out.append(f"{c['vendor']}: discount '{c['text']}' recorded, not applied.")

    seen: dict[str, list[str]] = {}
    for r in award.lines[award.lines["display_name"].notna()].itertuples(index=False):
        for a in r.assumptions:
            seen.setdefault(a, []).append(f"{r.display_name} line {r.rfx_line_id}")
    out += [f"{a} ({', '.join(where)})" for a, where in seen.items()]
    return out


# ---------- Confirmation ----------

def confirm_blockers(award: Award) -> list[str]:
    """Why the award can't be confirmed yet: awarded lines whose review is still open."""
    return [f"line {r.rfx_line_id} ({r.display_name})" for r in award.unconfirmed.itertuples(index=False)]


def award_signature(award: Award) -> tuple:
    """What was awarded, so a later change to the award can be spotted."""
    return tuple((int(r.rfx_line_id), r.display_name,
                  None if r.price_inr_per_piece is None or pd.isna(r.price_inr_per_piece)
                  else round(float(r.price_inr_per_piece), 4))
                 for r in award.lines.itertuples(index=False))


def confirm_award(award: Award, note: str, confirmed_at: str) -> dict:
    """The buyer's confirmation record. Raises ValueError while review rows are open."""
    blockers = confirm_blockers(award)
    if blockers:
        raise ValueError("Confirm the reviewed prices first: " + ", ".join(blockers))
    if award.is_empty:
        raise ValueError("Nothing is awarded yet.")
    return {"approach": award.settings.describe(), "vendors": list(award.vendors_used),
            "total_inr": award.total_inr, "confirmed_at": confirmed_at, "note": (note or "").strip(),
            "signature": award_signature(award)}


# ---------- Excel export ----------

RUPEE_FORMAT = '"₹"#,##0.00'
PCT_FORMAT = '0.0"%"'


def award_to_excel(award: Award, rfx_id: str, fx_text: str, confirmation: dict | None,
                   exported_at: str) -> bytes:
    """Sheets: Award, Vendor subtotals, Assumptions, Provenance, Open risks, Freight sensitivity.

    Money stays numeric (rupees) with a ₹ number format. The Award sheet starts with a
    header block: RFx id, date, FX rate and its label, and the confirmation status.
    """
    lines = award.lines
    award_sheet = pd.DataFrame({
        "Line": lines["rfx_line_id"], "Description": lines["description"], "Annual qty": lines["annual_qty"],
        "Awarded vendor": lines["display_name"].fillna(NOT_AWARDED),
        "Quoted ₹/piece": lines["quoted_inr_per_piece"], "Discount %": lines["discount_pct"],
        "₹/piece": lines["price_inr_per_piece"], "Annual ₹": lines["annual_inr"],
        "Last year ₹/piece": lines["last_year_inr_per_piece"], "Label": lines["label"],
        "Confidence": lines["confidence"],
        "Unconfirmed review": lines["unconfirmed"].map(lambda u: "⚠ yes" if u else ""),
    })
    subtotals = award.vendor_subtotals.rename(columns={
        "display_name": "Vendor", "lines_won": "Lines won", "quoted_annual_inr": "Quoted annual ₹",
        "discount_inr": "Discount ₹", "annual_inr": "Annual ₹", "unconfirmed_lines": "Unconfirmed lines"})
    assumptions = pd.DataFrame({"Assumption": award.assumptions})
    awarded = lines[lines["display_name"].notna()]
    provenance = pd.DataFrame({
        "Vendor": awarded["display_name"], "Line": awarded["rfx_line_id"],
        "Raw price text": awarded["raw_price_text"], "File": awarded["source_file"],
        "Page": awarded["page"], "Source snippet": awarded["source_snippet"],
        "Confidence": awarded["confidence"],
        "Confidence reasons": awarded["confidence_reasons"].map(lambda rs: "; ".join(rs or [])),
    })
    risks = pd.DataFrame([{"Severity": r["severity"].upper(), "Vendor": r["vendor"] or "", "Risk": r["text"]}
                          for r in award.risks], columns=["Severity", "Vendor", "Risk"])
    freight = _freight_sheet(award)

    if confirmation:
        status = f"Confirmed by buyer at {confirmation['confirmed_at']}"
        if confirmation.get("note"):
            status += f". Note: {confirmation['note']}"
    else:
        status = "Not confirmed (draft recommendation)"
    header = [("RFx", rfx_id), ("Exported", exported_at), ("FX rate", fx_text),
              ("Approach", award.settings.describe()), ("Status", status)]

    money = {"Award": ["Quoted ₹/piece", "₹/piece", "Annual ₹", "Last year ₹/piece"],
             "Vendor subtotals": ["Quoted annual ₹", "Discount ₹", "Annual ₹"]}
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        start = len(header) + 2  # header block, a blank row, then the table
        award_sheet.to_excel(writer, sheet_name="Award", index=False, startrow=start)
        ws = writer.sheets["Award"]
        for i, (k, v) in enumerate(header, start=1):
            ws.cell(row=i, column=1, value=k)
            ws.cell(row=i, column=2, value=v)
        _format(ws, award_sheet, money["Award"], first_row=start + 2)
        total_row = start + len(award_sheet) + 2
        ws.cell(row=total_row, column=1, value="Total")
        total_cell = ws.cell(row=total_row, column=list(award_sheet.columns).index("Annual ₹") + 1,
                             value=award.total_inr)
        total_cell.number_format = RUPEE_FORMAT

        for name, frame in (("Vendor subtotals", subtotals), ("Assumptions", assumptions),
                            ("Provenance", provenance), ("Open risks", risks), ("Freight sensitivity", freight)):
            frame.to_excel(writer, sheet_name=name, index=False)
            _format(writer.sheets[name], frame, money.get(name, []), first_row=2)
        _format_freight(writer.sheets["Freight sensitivity"], freight)
    return buf.getvalue()


def _freight_sheet(award: Award) -> pd.DataFrame:
    frames = []
    for s in award.sensitivity:
        if s.get("error"):
            frames.append(pd.DataFrame([{"vendor": s["vendor"], "note": s["error"]}]))
            continue
        t = s["table"].copy()
        t.insert(0, "vendor", s["vendor"])
        frames.append(t)
    if not frames:
        return pd.DataFrame({"note": ["No vendor in this award quoted freight extra."]})
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns=excel_header)


def _format(ws, frame: pd.DataFrame, money_cols: list[str], first_row: int) -> None:
    cols = list(frame.columns)
    for name in money_cols:
        c = cols.index(name) + 1
        for r in range(first_row, first_row + len(frame)):
            ws.cell(row=r, column=c).number_format = RUPEE_FORMAT
    for c, name in enumerate(cols, start=1):
        width = max([len(str(name))] + [len(str(v)) for v in frame[name].head(200)])
        ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = min(max(10, width + 2), 60)


def _format_freight(ws, frame: pd.DataFrame) -> None:
    for c, name in enumerate(frame.columns, start=1):
        fmt = RUPEE_FORMAT if "₹" in str(name) else PCT_FORMAT if "pct" in str(name) else None
        if fmt:
            for r in range(2, len(frame) + 2):
                ws.cell(row=r, column=c).number_format = fmt
