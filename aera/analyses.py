"""The answers the app works out itself: one function per kind of question (intent).

Pure Python on the comparison. No AI, no Streamlit. Each function returns an Analysis:
the tables the buyer sees, and `facts`, plain strings with every ₹ figure already
formatted by format_inr. The facts are the ONLY thing the answer writer (Claude) is shown,
so it can't do arithmetic or pick up a figure from another kind of question.

Vendor scope: quality PASS vendors only, by default. FAIL and UNCLEAR vendors come in only
when the buyer explicitly asks (include_failed), and then the answer opens with a warning
about them. quality_risk is the exception: it is about quality itself, so it covers every
vendor (or the ones named).
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from aera.compare import AMBIGUOUS_ASSUMPTION, FAIL, HIGH, LOW, MEDIUM, NOT_QUOTED, PASS, SEVERITY_ORDER, UNCLEAR
from aera.money import (
    MISSING_TEXT, describe_change, display_table, format_inr, plain_number, pct_text,
)
from aera.normalize import fx_currency
from aera.rfx import RFx

# What the buyer is asking. The classifier picks one; analyst.ask() routes on it.
AWARD_SPLIT = "award_split"
AWARD_CAPPED = "award_capped"
VENDOR_TOTALS = "vendor_totals"
RECOMMENDATION = "recommendation"
LANDED_COST = "landed_cost"
LINE_LOOKUP = "line_lookup"
FX_SENSITIVITY = "fx_sensitivity"
QUALITY_RISK = "quality_risk"
OTHER_ANALYSIS = "other_analysis"  # Claude writes pandas in the sandbox
REFUSAL = "refusal"  # no code is run
INTENTS = (AWARD_SPLIT, AWARD_CAPPED, VENDOR_TOTALS, RECOMMENDATION, LANDED_COST, LINE_LOOKUP,
           FX_SENSITIVITY, QUALITY_RISK, OTHER_ANALYSIS, REFUSAL)

QUALITY_EXCLUSION_REASONS = {FAIL: "failed quality", UNCLEAR: "quality unclear"}
MAX_REVIEW_ROWS_NAMED = 10
MAX_LINES_IN_DETAIL = 8  # line_lookup words out each line up to this many
SUMMARY_RESULT_ROWS = 60  # rows of a table shown to the writer


# ---------- Data ----------

@dataclass
class AnalystData:
    df: pd.DataFrame  # comparison
    vendors: pd.DataFrame  # vendor summary
    rfx_lines: pd.DataFrame
    last_year: pd.DataFrame
    fx_rates: dict[str, float]
    fx_date: str | dict[str, str] | None  # one date for all rates, or a source text per currency

    def namespace(self) -> dict:
        """Fresh copies for one run of generated code. Nothing it does can reach the originals."""
        lines = isolated(self.rfx_lines)
        return {"df": isolated(self.df), "vendors": isolated(self.vendors),
                "rfx_lines": isolated(self.rfx_lines), "last_year": isolated(self.last_year),
                "annual_weight_kg": lambda line_ids: annual_weight_kg(lines, line_ids)}


def analyst_data(comparison: pd.DataFrame, summary: pd.DataFrame, rfx: RFx,
                 last_year_prices: dict[int, float], fx_rates: dict[str, float],
                 fx_date: str | dict[str, str] | None) -> AnalystData:
    rfx_lines = pd.DataFrame(
        [{"line_id": ln.line_id, "description": ln.description, "annual_qty": ln.annual_qty,
          "uom": ln.uom, "nominal_weight_g": ln.nominal_weight_g} for ln in rfx.lines],
        columns=["line_id", "description", "annual_qty", "uom", "nominal_weight_g"],
    )
    last_year = pd.DataFrame(sorted(last_year_prices.items()),
                             columns=["line_id", "price_inr_per_piece"])
    return AnalystData(comparison, summary, rfx_lines, last_year, dict(fx_rates), fx_date)


def isolated(frame: pd.DataFrame) -> pd.DataFrame:
    """A deep copy, including the lists and dicts inside cells (pandas' own copy shares those)."""
    out = frame.copy(deep=True)
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = pd.Series([copy.deepcopy(v) for v in out[col]], index=out.index, dtype=object)
    return out


def annual_weight_kg(rfx_lines: pd.DataFrame, line_ids) -> float:
    """Total kg a year for these RFx lines: sum of annual_qty x nominal_weight_g / 1000.

    Used for per-kg breakevens (e.g. freight). A line with no weight raises: never zero.
    """
    ids = [int(i) for i in pd.Series(list(line_ids)).dropna().unique()]
    if not ids:
        raise ValueError("annual_weight_kg needs at least one line id")
    lines = rfx_lines.set_index("line_id")
    unknown = [i for i in ids if i not in lines.index]
    if unknown:
        raise ValueError(f"annual_weight_kg: no RFx line {unknown}")
    chosen = lines.loc[ids]
    no_weight = chosen.index[chosen["nominal_weight_g"].isna()].tolist()
    if no_weight:
        raise ValueError(f"annual_weight_kg: no nominal weight for line(s) {no_weight}")
    return float((chosen["annual_qty"] * chosen["nominal_weight_g"]).sum() / 1000)


def result_frame(result) -> pd.DataFrame | None:
    """A DataFrame to show for `result`, or None if it is a single value."""
    if isinstance(result, pd.DataFrame):
        out = result.copy()
        if not isinstance(out.index, pd.RangeIndex):
            out = out.reset_index()
        return out
    if isinstance(result, pd.Series):
        return result.rename(result.name if result.name is not None else "value").reset_index()
    if isinstance(result, dict):
        return pd.Series(result, name="value").rename_axis("item").reset_index()
    if isinstance(result, list) and result and all(isinstance(r, dict) for r in result):
        return pd.DataFrame(result)
    return None


def formatted_rows(result, max_rows: int = SUMMARY_RESULT_ROWS) -> list[dict]:
    """`result` as rows of strings only: money via format_inr, savings in words, % as '3.2%'."""
    frame = result_frame(result)
    if frame is None:
        if isinstance(result, (int, float, np.number)) and not isinstance(result, bool):
            return [{"value": plain_number(result)}]
        return [{"value": str(result)}]
    rows = []
    for rec in display_table(frame.head(max_rows)).to_dict(orient="records"):
        out = {}
        for k, v in rec.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = "; ".join(map(str, v))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                out[k] = MISSING_TEXT
            elif isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                out[k] = pct_text(v) if str(k).endswith("_pct") else plain_number(v)
            else:
                out[k] = str(v)
        rows.append(out)
    return rows


# ---------- Vendor names and scope ----------

def join_names(parts: list[str]) -> str:
    parts = [str(p) for p in parts]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def short_name(name: str, vendors: list[str]) -> str | None:
    """First word of a vendor name if no other vendor starts with it ('Ganesh'), else None."""
    words = name.split()
    if not words or len(words[0]) < 3:
        return None
    first = words[0]
    clash = any(v != name and v.split()[:1] == [first] for v in vendors)
    return None if clash else first


def mentions(text: str, name: str, all_names: list[str]) -> bool:
    """Is the vendor named in this text? Full name or unique first word, exact case."""
    if name in text:
        return True
    short = short_name(name, all_names)
    return short is not None and re.search(rf"\b{re.escape(short)}\b", text) is not None


def named_in(text: str, name: str, all_names: list[str]) -> bool:
    """Like mentions(), in any case: for what the buyer typed."""
    folded = text.casefold()
    if name.casefold() in folded:
        return True
    short = short_name(name, all_names)
    return short is not None and re.search(rf"\b{re.escape(short.casefold())}\b", folded) is not None


def resolve_vendors(asked, data: AnalystData) -> tuple[list[str], list[str]]:
    """(display names, names not recognised) for the vendors the classifier passed.

    Accepts the display name or vendor key in any case, or a unique first word ('ganesh').
    """
    all_names = [str(n) for n in data.vendors["display_name"]]
    keys = {str(r.vendor).casefold(): str(r.display_name) for r in data.vendors.itertuples()}
    found, unknown = [], []
    for a in asked or []:
        text = str(a).strip()
        if not text:
            continue
        hit = keys.get(text.casefold()) or next((n for n in all_names if n.casefold() == text.casefold()), None)
        if hit is None:
            hits = [n for n in all_names if named_in(text, n, all_names)
                    or text.casefold() in n.casefold()]
            hit = hits[0] if len(hits) == 1 else None
        if hit is None:
            unknown.append(text)
        elif hit not in found:
            found.append(hit)
    return found, unknown


@dataclass
class Scope:
    names: list[str]  # display names the analysis uses
    excluded: list[dict] = field(default_factory=list)  # {"display_name", "reason"}: left out by the quality filter
    flagged: list[str] = field(default_factory=list)  # FAIL / UNCLEAR vendors included because the buyer asked
    unknown: list[str] = field(default_factory=list)  # names the buyer used that match no vendor


def vendor_scope(data: AnalystData, named=None, include_failed: bool = False) -> Scope:
    """The vendors an analysis may use. Quality PASS only, unless include_failed.

    `named` narrows it to the vendors the buyer named (all vendors if none). A named
    FAIL / UNCLEAR vendor still needs include_failed; otherwise it is listed as excluded.
    """
    statuses = data.vendors.set_index("display_name")["quality_status"]
    asked, unknown = resolve_vendors(named, data)
    pool = asked or [str(n) for n in data.vendors["display_name"]]
    scope = Scope(names=[], unknown=unknown)
    for n in pool:
        reason = QUALITY_EXCLUSION_REASONS.get(statuses.get(n))
        if reason is None:
            scope.names.append(n)
        elif include_failed:
            scope.names.append(n)
            scope.flagged.append(n)
        else:
            scope.excluded.append({"display_name": n, "reason": reason})
    return scope


_FILE_NOTE = re.compile(r"\s*\([^()]*\.(?:pdf|png|jpe?g|docx?|xlsx?)\)", re.IGNORECASE)


def quality_findings(name: str, data: AnalystData, status: str | None = None) -> list[str]:
    """The vendor's quality findings with this outcome (its own status by default), file names cut."""
    row = data.vendors[data.vendors["display_name"] == name].iloc[0]
    status = status or row["quality_status"]
    reasons = row.get("quality_reasons")
    prefix = f"{status}: "
    return [_FILE_NOTE.sub("", r[len(prefix):]).strip() for r in (reasons if isinstance(reasons, list) else [])
            if isinstance(r, str) and r.startswith(prefix)]


def quality_warning(names: list[str], data: AnalystData) -> str | None:
    """One sentence for FAIL / UNCLEAR vendors the buyer asked to include. The answer must open with it."""
    if not names:
        return None
    statuses = data.vendors.set_index("display_name")["quality_status"]
    parts = [f"{n} ({QUALITY_EXCLUSION_REASONS[statuses[n]]}: "
             f"{'; '.join(quality_findings(n, data)) or 'no reason recorded'})" for n in names]
    what = "not a valid award option" if len(names) == 1 else "not valid award options"
    return f"Includes {join_names(parts)}, {what} as things stand."


def vendor_label(name: str, data: AnalystData) -> str:
    """'Sahyadri Boxes & Cartons (failed quality)' for FAIL/UNCLEAR vendors, else the name."""
    status = data.vendors.set_index("display_name")["quality_status"].get(name)
    reason = QUALITY_EXCLUSION_REASONS.get(status)
    return f"{name} ({reason})" if reason else name


def _scope_facts(scope: Scope, data: AnalystData) -> dict:
    """Facts every scoped answer carries: the quality warning and who was left out."""
    facts = {}
    if scope.flagged:
        facts["quality_warning"] = quality_warning(scope.flagged, data)
    if scope.excluded:
        facts["excluded_vendors"] = [f"{e['display_name']} ({e['reason']})" for e in scope.excluded]
    return facts


def _scope_caveats(scope: Scope) -> list[str]:
    if not scope.unknown:
        return []
    return [f"No vendor in this comparison matches {join_names([repr(u) for u in scope.unknown])}; "
            "it was ignored."]


def review_caveats(data: AnalystData, names: list[str], line_ids=None) -> list[str]:
    """Code-checked caveat: prices these vendors quoted that the buyer still has to confirm."""
    df = data.df
    rows = df[df["display_name"].isin(names) & df["needs_review"].astype(bool)
              & ~df["buyer_confirmed"].astype(bool)]
    if line_ids is not None:
        rows = rows[rows["rfx_line_id"].isin(list(line_ids))]
    if rows.empty:
        return []
    named = [f"{r.display_name} line {r.rfx_line_id}" for r in rows.head(MAX_REVIEW_ROWS_NAMED).itertuples()]
    more = len(rows) - len(named)
    listed = "; ".join(named) + (f"; and {more} more" if more > 0 else "")
    return [f"Checked by code: {len(rows)} price(s) used here still need your review on the Compare "
            f"page and are not confirmed yet: {listed}."]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith((".", "?", "!")) else ".") if text else ""


# ---------- The result of one analysis ----------

@dataclass
class Analysis:
    intent: str
    facts: dict  # formatted strings only: the writer's sole input
    tag: str  # "vendor totals, like for like on 25 lines"
    func: Callable | None = None  # its source is shown under the answer
    call: str = ""  # how it was called, shown above the source
    explanation: str = ""
    table: pd.DataFrame | None = None
    answer_type: str = "table"  # "table" | "chart" | "text"
    chart_spec: dict | None = None
    extra_tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    sensitivity: list[dict] = field(default_factory=list)  # freight_sensitivity() results, with "vendor"
    excluded: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)  # price risks, for the Excel export
    values: dict = field(default_factory=dict)  # raw numbers behind the facts (tests, Excel)


# ---------- Award building blocks (shared with the Decide page) ----------

def award_by_line(data: AnalystData, names: list[str]) -> pd.DataFrame:
    """For each RFx line, the cheapest of these vendors (counted rows only) and the runner-up.

    Columns: line_id, annual_qty, winner, winner_price, runner, runner_price (runner may be None).
    """
    df = data.df
    d = df[df["display_name"].isin(names) & df["included_in_totals"].astype(bool)
           & df["price_inr_per_piece"].notna()]
    d = d.merge(data.rfx_lines[["line_id", "annual_qty"]], left_on="rfx_line_id", right_on="line_id")
    rows = []
    for line_id, g in d.sort_values(["rfx_line_id", "price_inr_per_piece", "display_name"]).groupby("line_id"):
        win, run = g.iloc[0], (g.iloc[1] if len(g) > 1 else None)
        rows.append({"line_id": int(line_id), "annual_qty": int(win["annual_qty"]),
                     "winner": win["display_name"], "winner_price": float(win["price_inr_per_piece"]),
                     "runner": None if run is None else run["display_name"],
                     "runner_price": None if run is None else float(run["price_inr_per_piece"])})
    return pd.DataFrame(rows, columns=["line_id", "annual_qty", "winner", "winner_price",
                                       "runner", "runner_price"])


FREIGHT_RATES = (0, 0.5, 1, 2, 3, 4)  # ₹ per kg shown in the sensitivity table
FREIGHT_STEP = 0.05  # search step for the rate where the saving is gone
FREIGHT_SEARCH_MAX = 50.0  # stop searching here (₹ per kg)


class SensitivityError(ValueError):
    """The sensitivity can't be worked out (e.g. a box weight or last-year price is missing)."""


def freight_sensitivity(data: AnalystData, vendor: str, allowed: list[str],
                        rates=FREIGHT_RATES) -> dict:
    """How the award's saving against last year changes if `vendor` adds freight.

    For each rate (₹ per kg) the vendor's per-piece price goes up by rate x nominal_weight_g / 1000,
    and the award is re-run: every line goes to the cheapest of the `allowed` vendors (counted
    rows only), so lines move away from `vendor` as its freight rises. Only lines with a
    last-year price are in the award, so the saving is like-for-like.

    Returns {"table": DataFrame(rate_inr_per_kg, total_inr, saving_inr, saving_pct, lines_won),
             "zero_rate": ₹/kg where the saving first reaches zero (₹0.05 steps) or None,
             "lines": lines in the award, "lines_without_last_year": lines left out for that reason}.
    """
    df = data.df
    rows = df[df["display_name"].isin(allowed) & df["included_in_totals"].astype(bool)
              & df["price_inr_per_piece"].notna()]
    if vendor not in set(rows["display_name"]):
        raise SensitivityError(f"{vendor} has no counted prices in this award")
    prices = rows.pivot_table(index="rfx_line_id", columns="display_name",
                              values="price_inr_per_piece", aggfunc="min").sort_index(axis=1)
    last_year = data.last_year.set_index("line_id")["price_inr_per_piece"]
    without_ly = [int(i) for i in prices.index if i not in last_year.index]
    prices = prices.drop(index=without_ly)
    if prices.empty:
        raise SensitivityError("no line in the award has a last-year price to compare against")

    lines = data.rfx_lines.set_index("line_id").loc[prices.index]
    vendor_col = list(prices.columns).index(vendor)
    needs_weight = prices.index[prices[vendor].notna() & lines["nominal_weight_g"].isna()].tolist()
    if needs_weight:
        raise SensitivityError(f"no box weight for line(s) {needs_weight}")

    price = prices.to_numpy(dtype=float, copy=True)  # writable: pandas 3 may hand back a read-only view
    price[np.isnan(price)] = np.inf  # a vendor that didn't quote a line can't win it
    qty = lines["annual_qty"].to_numpy(dtype=float)
    kg_per_piece = (lines["nominal_weight_g"].fillna(0) / 1000).to_numpy(dtype=float)
    ly_total = float((qty * last_year.loc[prices.index].to_numpy(dtype=float)).sum())

    def award(rate: float) -> tuple[float, int]:
        eff = price.copy()
        eff[:, vendor_col] += rate * kg_per_piece
        winner = eff.argmin(axis=1)  # ties go to the first vendor by name
        total = float((qty * eff[np.arange(len(qty)), winner]).sum())
        return total, int((winner == vendor_col).sum())

    table = []
    for rate in rates:
        total, won = award(float(rate))
        saving = ly_total - total
        table.append({"rate_inr_per_kg": float(rate), "total_inr": round(total, 2),
                      "saving_inr": round(saving, 2), "saving_pct": round(saving / ly_total * 100, 1),
                      "lines_won": won})

    zero_rate = None
    for i in range(int(round(FREIGHT_SEARCH_MAX / FREIGHT_STEP)) + 1):
        rate = round(i * FREIGHT_STEP, 2)
        if ly_total - award(rate)[0] <= 0.005:
            zero_rate = rate
            break

    end_total, end_won = award(FREIGHT_SEARCH_MAX)
    return {"table": pd.DataFrame(table), "zero_rate": zero_rate, "lines": len(prices),
            "lines_without_last_year": without_ly,
            # at the end of the search: what is left once freight has pushed lines elsewhere
            "saving_at_max_inr": round(ly_total - end_total, 2), "lines_won_at_max": end_won}


def freight_effect(data: AnalystData, name: str, pool: list[str], rates=FREIGHT_RATES,
                   terms: str = "extra") -> dict:
    """The award/freight sentence for one vendor, worded from freight_sensitivity().

    Returns {"vendor", "headline" ('Saves ₹X (p%) before freight') or None, "text",
             "must_include" (figures the answer has to keep), "sensitivity" or None}.
    """
    said = "hasn't quoted freight" if terms == "extra" else "hasn't said whether freight is included"
    out = {"vendor": name, "headline": None, "must_include": [], "sensitivity": None}
    try:
        s = freight_sensitivity(data, name, pool, sorted(set(rates) | {0}))
    except SensitivityError as e:
        out["text"] = f"{name} {said}; how freight changes the saving can't be worked out ({e})"
        return out

    out["sensitivity"] = {"vendor": name, **s}
    first = s["table"].iloc[0]  # rate 0: before freight
    n = int(first["lines_won"])
    before = describe_change(first["saving_inr"], first["saving_pct"])
    head = f"{name} wins {_plural(n, 'line')} but {said}"
    if first["saving_inr"] <= 0:
        tail = "any freight makes the award dearer still (table below)"
    elif s["zero_rate"] is not None:
        gone = format_inr(s["zero_rate"], per_unit=True) + "/kg"
        tail = f"the saving shrinks as freight rises and is gone at about {gone} (table below)"
        out["must_include"] = [gone]
    elif s["lines_won_at_max"] == 0:
        floor = format_inr(s["saving_at_max_inr"])
        tail = ("the saving shrinks as freight rises, but other vendors take its lines, so it never "
                f"falls below {floor} (table below)")
        out["must_include"] = [floor]
    else:
        tail = ("the saving shrinks as freight rises and is still there at "
                f"{format_inr(FREIGHT_SEARCH_MAX, per_unit=True)}/kg (table below)")
    out["headline"] = f"{before[0].upper()}{before[1:]} before freight"
    out["text"] = f"{head}; {tail}"
    return out


def _ambiguous_risks(data: AnalystData, won: pd.DataFrame) -> list[str]:
    """Winning prices that are the higher of two readings and not yet confirmed."""
    df = data.df
    out = []
    for name, g in won.groupby("winner", sort=True):
        mine = df[(df["display_name"] == name) & df["rfx_line_id"].isin(g["line_id"])
                  & ~df["buyer_confirmed"].astype(bool)]
        k = int(sum(AMBIGUOUS_ASSUMPTION in (a or []) for a in mine["assumptions"]))
        if k:
            out.append(f"{k} of the {len(g)} lines {name} wins {'is' if k == 1 else 'are'} "
                       "priced at the higher of two readings until you confirm")
    return out


def _last_year(data: AnalystData) -> pd.Series:
    return data.last_year.set_index("line_id")["price_inr_per_piece"]


def _saving(cost: pd.Series, last_year_cost: pd.Series) -> tuple[float | None, float | None, int]:
    """(saving, saving %, lines) against last year on the lines that have a last-year price."""
    has = last_year_cost.notna()
    before = float(last_year_cost[has].sum())
    if not has.any() or not before:
        return None, None, 0
    s = before - float(cost[has].sum())
    return round(s, 2), round(s / before * 100, 1), int(has.sum())


def _priced_award(data: AnalystData, won: pd.DataFrame) -> pd.DataFrame:
    """award_by_line() plus annual cost and last year's cost per line (None if no last-year price)."""
    won = won.copy()
    won["annual_cost_inr"] = won["annual_qty"] * won["winner_price"]
    won["last_year_cost_inr"] = won["annual_qty"] * won["line_id"].map(_last_year(data))
    return won


def _award_table(won: pd.DataFrame, total_label: str) -> pd.DataFrame:
    """One row per winning vendor (most lines first) and a total row."""
    rows = []
    for name, g in won.groupby("winner"):
        s, p, _ = _saving(g["annual_cost_inr"], g["last_year_cost_inr"])
        rows.append({"vendor": name, "lines_won": len(g), "annual_cost_inr": round(float(g["annual_cost_inr"].sum()), 2),
                     "saving_inr": s, "saving_pct": p})
    rows.sort(key=lambda r: (-r["lines_won"], r["vendor"]))
    s, p, _ = _saving(won["annual_cost_inr"], won["last_year_cost_inr"])
    rows.append({"vendor": total_label, "lines_won": len(won),
                 "annual_cost_inr": round(float(won["annual_cost_inr"].sum()), 2), "saving_inr": s, "saving_pct": p})
    return pd.DataFrame(rows, columns=["vendor", "lines_won", "annual_cost_inr", "saving_inr", "saving_pct"])


def _not_awarded(data: AnalystData, won: pd.DataFrame) -> list[int]:
    return [int(i) for i in data.rfx_lines["line_id"] if int(i) not in set(won["line_id"])]


def _wins_text(table: pd.DataFrame) -> list[str]:
    return [f"{r.vendor} wins {_plural(int(r.lines_won), 'line')}" for r in table.iloc[:-1].itertuples()]


def _saving_basis(lines: int) -> str:
    return f"against last year's prices on the {_plural(lines, 'awarded line')} that have one"


def _freight_terms(data: AnalystData, name: str):
    return data.vendors.set_index("display_name")["freight"].get(name)


# ---------- award_split ----------

def award_split(data: AnalystData, scope: Scope) -> Analysis:
    """Every line to the cheapest counted price among the vendors in scope, and the saving
    against last year. The only analysis (with landed_cost) whose facts carry the freight sentence:
    for each winner that quoted freight extra, the award is re-run with freight added."""
    won = _priced_award(data, award_by_line(data, scope.names))
    facts = {"vendors_considered": join_names(scope.names), **_scope_facts(scope, data)}
    a = Analysis(AWARD_SPLIT, facts, tag=f"award split, cheapest per line across {_plural(len(scope.names), 'vendor')}",
                 func=award_split, call="award_split(data, scope)", excluded=scope.excluded,
                 caveats=_scope_caveats(scope),
                 explanation="Gives every line to the vendor with the cheapest counted price among the vendors "
                             "considered, and compares the annual cost with last year's prices.")
    if won.empty:
        a.answer_type, a.table = "text", None
        facts["no_award"] = "No vendor considered has a counted price on any line, so there is nothing to award."
        return a

    table = _award_table(won, "Total split award")
    total = table.iloc[-1]
    s, p, n_ly = _saving(won["annual_cost_inr"], won["last_year_cost_inr"])
    facts.update({
        "lines_awarded": f"{len(won)} of {len(data.rfx_lines)} lines",
        "total_annual_cost": format_inr(total["annual_cost_inr"]),
        "saving_vs_last_year": describe_change(s, p) if s is not None else None,
        "saving_basis": _saving_basis(n_ly),
        "lines_won": _wins_text(table),
    })
    missing = _not_awarded(data, won)
    if missing:
        facts["lines_not_awarded"] = (f"{_plural(len(missing), 'line')} ({', '.join(map(str, missing))}) "
                                      "not awarded: no vendor considered has a counted price")

    freight = [freight_effect(data, n, scope.names) for n in sorted(set(won["winner"]))
               if _freight_terms(data, n) == "extra"]
    if freight:
        facts["headline_before_freight"] = next((f["headline"] for f in freight if f["headline"]), None)
        facts["freight_vendors"] = [f["vendor"] for f in freight]
        facts["freight_risks"] = [f["text"] for f in freight]
        facts["freight_must_include"] = [m for f in freight for m in f["must_include"]]
        facts["offer"] = ("Offer to draft a clarification asking "
                          f"{join_names(facts['freight_vendors'])} to quote freight.")
        a.missing_data = [f"{f['vendor']}'s freight charge in ₹ per kg" for f in freight]
        a.sensitivity = [f["sensitivity"] for f in freight if f["sensitivity"]]
    ambiguous = _ambiguous_risks(data, won)
    if ambiguous:
        facts["other_price_risks"] = ambiguous
    a.risks = facts.get("freight_risks", []) + ambiguous
    a.table = table
    a.caveats += review_caveats(data, scope.names, won["line_id"])
    a.values = {"total_inr": float(total["annual_cost_inr"]), "saving_inr": s, "saving_pct": p,
                "lines_won": {r.vendor: int(r.lines_won) for r in table.iloc[:-1].itertuples()}}
    return a


# ---------- award_capped ----------

def award_capped(data: AnalystData, scope: Scope, cap: int) -> Analysis:
    """The best set of at most `cap` vendors: every combination is tried (the Decide page's
    capped approach), keeping the one that covers most lines, then the lowest annual total."""
    from aera.award import CAPPED, MAX_CAP, AwardSettings, award_data, best_allocation  # award imports this module

    cap = max(1, min(int(cap or 2), MAX_CAP))
    facts = {"cap": _plural(cap, "vendor"), "vendors_considered": join_names(scope.names), **_scope_facts(scope, data)}
    a = Analysis(AWARD_CAPPED, facts, tag=f"award capped at {_plural(cap, 'vendor')}",
                 func=award_capped, call=f"award_capped(data, scope, cap={cap})", excluded=scope.excluded,
                 caveats=_scope_caveats(scope),
                 explanation=f"Tries every set of {cap} vendors considered, gives each line to the cheapest vendor "
                             "in the set, and keeps the set covering the most lines, then the lowest annual total.")
    settings = AwardSettings(approach=CAPPED, allowed=list(scope.names), cap=cap)
    alloc, _, _, tried = best_allocation(award_data(data, include_not_comparable=False), settings, {})
    if alloc.empty:
        a.answer_type, a.table = "text", None
        facts["no_award"] = "No vendor considered has a counted price on any line, so there is nothing to award."
        return a

    won = _priced_award(data, alloc[["line_id", "annual_qty", "winner", "winner_price", "runner", "runner_price"]])
    table = _award_table(won, f"Total, capped at {_plural(cap, 'vendor')}")
    total = float(won["annual_cost_inr"].sum())
    s, p, n_ly = _saving(won["annual_cost_inr"], won["last_year_cost_inr"])
    chosen = list(table["vendor"].iloc[:-1])
    split = _priced_award(data, award_by_line(data, scope.names))
    split_total = float(split["annual_cost_inr"].sum())
    facts.update({
        "chosen_vendors": chosen,
        "lines_awarded": f"{len(won)} of {len(data.rfx_lines)} lines",
        "total_annual_cost": format_inr(total),
        "saving_vs_last_year": describe_change(s, p) if s is not None else None,
        "saving_basis": _saving_basis(n_ly),
        "lines_won": _wins_text(table),
        "combinations_compared": str(tried),
    })
    if len(split) == len(won):  # same lines covered, so the two totals are like for like
        gap = total - split_total
        uncapped = (f"the uncapped split across {_plural(split['winner'].nunique(), 'vendor')} "
                    f"({format_inr(split_total)})")
        facts["against_uncapped_split"] = (
            f"costs the same as {uncapped}" if format_inr(abs(gap)) == "₹0" else
            f"costs {format_inr(abs(gap))} {'more' if gap > 0 else 'less'} than {uncapped}")
    if len(scope.names) <= cap:
        facts["cap_note"] = f"Only {_plural(len(scope.names), 'vendor')} considered, so the cap changes nothing."
    missing = _not_awarded(data, won)
    if missing:
        facts["lines_not_awarded"] = (f"{_plural(len(missing), 'line')} ({', '.join(map(str, missing))}) "
                                      "not awarded: no chosen vendor has a counted price")
    extra = [n for n in chosen if _freight_terms(data, n) == "extra"]
    a.caveats += [f"{n} hasn't quoted freight, so this total is before freight." for n in extra]
    a.missing_data = [f"{n}'s freight charge in ₹ per kg" for n in extra]
    a.table = table
    a.caveats += review_caveats(data, chosen, won["line_id"])
    a.values = {"chosen": chosen, "total_inr": total, "saving_inr": s, "split_total_inr": split_total}
    return a


# ---------- vendor_totals ----------

def counted_costs(data: AnalystData, names: list[str]) -> pd.DataFrame:
    """Counted rows only (included_in_totals, priced) for these vendors, with annual cost per line.

    Columns: display_name, rfx_line_id, annual_qty, price_inr_per_piece, annual_cost_inr.
    """
    df = data.df
    d = df[df["display_name"].isin(names) & df["included_in_totals"].astype(bool)
           & df["price_inr_per_piece"].notna()]
    d = d.merge(data.rfx_lines[["line_id", "annual_qty"]], left_on="rfx_line_id", right_on="line_id")
    d = d.assign(annual_cost_inr=d["annual_qty"] * d["price_inr_per_piece"])
    return d[["display_name", "rfx_line_id", "annual_qty", "price_inr_per_piece", "annual_cost_inr"]]


def common_lines(costs: pd.DataFrame, names: list[str]) -> list[int]:
    """RFx lines that EVERY one of these vendors priced comparably."""
    sets = [set(costs.loc[costs["display_name"] == n, "rfx_line_id"]) for n in names]
    return sorted(int(i) for i in set.intersection(*sets)) if sets else []


def totals_tables(data: AnalystData, names: list[str]) -> dict:
    """Totals per vendor that can't flatter a vendor for quoting fewer lines.

    - like_for_like: each vendor's total over the lines every compared vendor priced comparably.
    - full: each vendor's total over its own counted lines, with how many lines that is
      (these cover different lines and must not be compared).
    """
    costs = counted_costs(data, names)
    priced = [n for n in names if n in set(costs["display_name"])]
    common = common_lines(costs, priced)
    n_rfx = len(data.rfx_lines)
    main, full = [], []
    for n in priced:
        mine = costs[costs["display_name"] == n]
        shared = mine[mine["rfx_line_id"].isin(common)]
        main.append({"vendor": vendor_label(n, data),
                     "like_for_like_total_inr": round(float(shared["annual_cost_inr"].sum()), 2),
                     "lines_in_total": len(shared)})
        full.append({"vendor": vendor_label(n, data),
                     "total_on_own_lines_inr": round(float(mine["annual_cost_inr"].sum()), 2),
                     "lines_in_total": len(mine), "rfx_lines_not_in_total": n_rfx - len(mine)})
    main_cols = ["vendor", "like_for_like_total_inr", "lines_in_total"]
    like = (pd.DataFrame(main, columns=main_cols).sort_values(["like_for_like_total_inr", "vendor"])
            if common else pd.DataFrame(columns=main_cols))
    full_df = pd.DataFrame(full, columns=["vendor", "total_on_own_lines_inr", "lines_in_total",
                                          "rfx_lines_not_in_total"])
    return {"like_for_like": like.reset_index(drop=True),
            "full": full_df.sort_values(["lines_in_total", "vendor"], ascending=[False, True]).reset_index(drop=True),
            "common_lines": common, "rfx_lines": n_rfx,
            "without_counted_lines": [n for n in names if n not in priced]}


FULL_TOTALS_TITLE = "Full totals: each vendor's own counted lines (these cover different lines, so don't compare them)"


def vendor_totals(data: AnalystData, scope: Scope, chart: bool = False) -> Analysis:
    """Annual cost per vendor, like for like: the main table (or chart) covers only the lines
    EVERY vendor in scope quoted comparably; a second table has each vendor's full total and
    how many lines it covers."""
    t = totals_tables(data, scope.names)
    common = t["common_lines"]
    excluded = scope.excluded + [{"display_name": n, "reason": "no comparable lines"} for n in t["without_counted_lines"]]
    compared = len(scope.names) - len(t["without_counted_lines"])
    facts = {"vendors_compared": join_names([n for n in scope.names if n not in t["without_counted_lines"]]),
             **_scope_facts(scope, data)}
    if excluded:
        facts["excluded_vendors"] = [f"{e['display_name']} ({e['reason']})" for e in excluded]
    a = Analysis(VENDOR_TOTALS, facts, tag="", func=vendor_totals, call="vendor_totals(data, scope)",
                 excluded=excluded, caveats=_scope_caveats(scope),
                 explanation="Adds up annual cost (annual quantity x price per piece) per vendor on the lines every "
                             "compared vendor priced comparably, so no vendor looks cheaper for quoting fewer lines. "
                             "The second table is each vendor's total on its own lines.")
    if common:
        like = t["like_for_like"]
        a.tag = f"vendor totals, like for like on {_plural(len(common), 'line')}"
        a.table, a.extra_tables = like, [(FULL_TOTALS_TITLE, t["full"])]
        a.answer_type = "chart" if chart else "table"
        a.chart_spec = {"x": "vendor", "y": "like_for_like_total_inr", "kind": "bar"} if chart else None
        facts.update({
            "common_lines_count": str(len(common)),
            "like_for_like_lines": (f"{len(common)} of {t['rfx_lines']} lines, the ones all {compared} compared "
                                    "vendors quoted comparably") if compared > 1 else
                                   f"{len(common)} of {t['rfx_lines']} lines it quoted comparably",
            "like_for_like_ranking": [{"vendor": r.vendor, "total": format_inr(r.like_for_like_total_inr)}
                                      for r in like.itertuples()],
            "full_totals_not_like_for_like": [f"{r.vendor}: {format_inr(r.total_on_own_lines_inr)} on "
                                              f"{_plural(int(r.lines_in_total), 'line')}" for r in t["full"].itertuples()],
        })
    else:
        a.tag = "vendor totals, no line common to every vendor"
        a.table, a.answer_type = t["full"], "table"
        facts["no_common_lines"] = ("No line was priced comparably by every compared vendor, so there is no "
                                    "like-for-like total. The totals below cover different lines.")
        facts["full_totals_not_like_for_like"] = [
            f"{r.vendor}: {format_inr(r.total_on_own_lines_inr)} on {_plural(int(r.lines_in_total), 'line')}"
            for r in t["full"].itertuples()]
    a.caveats += review_caveats(data, scope.names, common or None)
    a.values = {"common_lines": common, "like_for_like": t["like_for_like"], "full": t["full"]}
    return a


# ---------- recommendation ----------

VIEW_COLUMNS = ["view", "vendors", "annual_cost_inr", "lines_covered", "saving_inr", "saving_pct",
                "main_caveat", "other_caveats"]
CLOSING_QUESTION = ("Tell me what matters most (lowest price, lowest risk or fewest vendors) "
                    "and I'll work it out.")


def recommendation(data: AnalystData, scope: Scope) -> Analysis:
    """'Which vendor is best?' as three views among the vendors in scope, each with its main
    caveat: cheapest single vendor, lowest-risk vendor, cheapest split. Saving is against
    last year on each view's own lines that have a last-year price."""
    facts = {"vendors_considered": join_names(scope.names), **_scope_facts(scope, data),
             "closing_question": CLOSING_QUESTION}
    a = Analysis(RECOMMENDATION, facts, tag="recommendation, 3 views", func=recommendation,
                 call="recommendation(data, scope)", excluded=scope.excluded, caveats=_scope_caveats(scope),
                 explanation="Among the vendors considered: the cheapest single vendor (compared on the lines they "
                             "all quoted), the vendor with the fewest open risks, and the cheapest line-by-line split.")
    costs = counted_costs(data, scope.names)
    priced = [n for n in scope.names if n in set(costs["display_name"])]
    if not priced:
        a.answer_type, a.table, a.tag = "text", None, "recommendation, no valid option"
        facts["no_valid_option"] = "No vendor considered has a comparable price yet, so there is no valid award option."
        return a

    info = data.vendors.set_index("display_name")
    n_rfx = len(data.rfx_lines)
    common = common_lines(costs, priced)
    ly = _last_year(data)

    def own(n: str) -> pd.DataFrame:
        return costs[costs["display_name"] == n]

    def on_common(n: str) -> float:
        mine = own(n)
        return float(mine.loc[mine["rfx_line_id"].isin(common), "annual_cost_inr"].sum())

    def risks(n: str, severity: str | None = None) -> list[str]:
        found = info.loc[n].get("open_risks")
        return [r["text"] for r in (found if isinstance(found, list) else [])
                if severity is None or r.get("severity") == severity]

    def review(n: str) -> int:
        return int(info.loc[n].get("lines_needing_review") or 0) if "lines_needing_review" in info.columns else 0

    def coverage(lines: int) -> str | None:
        gap = n_rfx - lines
        return f"covers {lines} of {n_rfx} lines, so {gap} need{'s' if gap == 1 else ''} another vendor" if gap else None

    def freight(n: str) -> str | None:
        return {"extra": f"{n} hasn't quoted freight, so its prices exclude it",
                "unclear": f"{n}'s freight terms are unclear"}.get(info.loc[n].get("freight"))

    def view(name: str, vendors: str, rows: pd.DataFrame, caveats: list) -> dict:
        found = [c for c in caveats if c]
        s, p, _ = _saving(rows["annual_cost_inr"], rows["annual_qty"] * rows["rfx_line_id"].map(ly))
        return {"view": name, "vendors": vendors, "annual_cost_inr": round(float(rows["annual_cost_inr"].sum()), 2),
                "lines_covered": len(rows), "saving_inr": s, "saving_pct": p,
                "main_caveat": _sentence(found[0]) if found else "None found.",
                "other_caveats": "; ".join(found[1:])}

    # Cheapest single vendor: on the lines every vendor priced; with no common line, the widest cover.
    cheapest = min(priced, key=(lambda n: (on_common(n), n)) if common
                   else (lambda n: (-len(own(n)), float(own(n)["annual_cost_inr"].sum()), n)))
    basis = (f"cheapest on the {len(common)} lines every vendor considered quoted" if common
             else "no line was quoted by every vendor considered; picked for the most lines covered")
    cheap_caveats = [freight(cheapest), coverage(len(own(cheapest))),
                     f"{review(cheapest)} of its prices still need your review" if review(cheapest) else None, basis]

    # Lowest risk: fewest high-severity risks, then fewest risks, then fewest prices to review.
    safest = min(priced, key=lambda n: (len(risks(n, HIGH)), len(risks(n)), review(n), -len(own(n)), on_common(n), n))
    gap = on_common(safest) - on_common(cheapest)
    safe_caveats = [("high-severity risk: " + "; ".join(risks(safest, HIGH))) if risks(safest, HIGH) else None,
                    f"costs {format_inr(gap)} more than {cheapest} on the {len(common)} common lines"
                    if common and gap > 0.005 else None,
                    coverage(len(own(safest))), freight(safest),
                    "also the cheapest single vendor" if safest == cheapest else None,
                    "no high-severity open risks" if not risks(safest, HIGH) else None]

    # Cheapest split: every line to the cheapest vendor that priced it.
    won = award_by_line(data, priced)
    split = won.assign(rfx_line_id=won["line_id"], annual_cost_inr=won["annual_qty"] * won["winner_price"])
    wins = split["winner"].value_counts()
    split_vendors = sorted(wins.index, key=lambda n: (-wins[n], n))
    df = data.df
    to_review = df[df["display_name"].isin(priced) & df["needs_review"].astype(bool) & ~df["buyer_confirmed"].astype(bool)]
    to_review = to_review.merge(split[["line_id", "winner"]], left_on=["rfx_line_id", "display_name"],
                                right_on=["line_id", "winner"])
    split_caveats = ([f"{n} wins {_plural(int(wins[n]), 'line')} but hasn't quoted freight, so the saving is before freight"
                      for n in split_vendors if info.loc[n].get("freight") == "extra"]
                     + [f"{len(to_review)} winning price{'s still need' if len(to_review) != 1 else ' still needs'} your review"
                        if len(to_review) else None,
                        f"{len(split_vendors)} vendors to manage" if len(split_vendors) > 1 else None,
                        coverage(len(split))])

    rows = [view("Cheapest single vendor", cheapest, own(cheapest), cheap_caveats),
            view("Lowest risk", safest, own(safest), safe_caveats),
            view("Cheapest split", ", ".join(f"{n} ({_plural(int(wins[n]), 'line')})" for n in split_vendors),
                 split, split_caveats)]
    a.table = pd.DataFrame(rows, columns=VIEW_COLUMNS)
    facts.update({
        "cheapest_single_vendor": cheapest, "lowest_risk_vendor": safest, "cheapest_split_vendors": split_vendors,
        "views": [{"view": r["view"], "vendors": r["vendors"], "annual_cost": format_inr(r["annual_cost_inr"]),
                   "lines_covered": str(r["lines_covered"]),
                   "against_last_year": describe_change(r["saving_inr"], r["saving_pct"]),
                   "main_caveat": r["main_caveat"]} for r in rows],
    })
    a.caveats += review_caveats(data, priced)
    a.values = {"cheapest": cheapest, "safest": safest, "split_vendors": split_vendors}
    return a


# ---------- landed_cost ----------

def landed_cost(data: AnalystData, scope: Scope, named=None, rate_inr_per_kg: float | None = None) -> Analysis:
    """Freight sensitivity of the cheapest-per-line award: for each winning vendor whose freight
    is extra or unclear (or the ones named), the award is re-run at each ₹/kg freight rate."""
    won = _priced_award(data, award_by_line(data, scope.names))
    asked, _ = resolve_vendors(named, data)
    terms = {n: _freight_terms(data, n) for n in scope.names}
    targets = [n for n in (asked or sorted(set(won["winner"]))) if n in scope.names and terms.get(n) in ("extra", "unclear")]
    rates = FREIGHT_RATES if rate_inr_per_kg is None else tuple(sorted(set(FREIGHT_RATES) | {float(rate_inr_per_kg)}))
    facts = {"vendors_considered": join_names(scope.names), **_scope_facts(scope, data)}
    a = Analysis(LANDED_COST, facts, tag="landed cost, freight sensitivity", func=landed_cost,
                 call=f"landed_cost(data, scope, named={asked!r}, rate_inr_per_kg={rate_inr_per_kg!r})",
                 excluded=scope.excluded, caveats=_scope_caveats(scope),
                 explanation="Adds freight at each ₹/kg rate (rate x box weight) to the vendor's prices, re-runs the "
                             "cheapest-per-line award so lines move when another vendor becomes cheaper, and shows "
                             "the saving against last year at each rate.")
    if won.empty:
        a.answer_type, a.table = "text", None
        facts["no_award"] = "No vendor considered has a counted price on any line, so there is nothing to award."
        return a
    s, p, n_ly = _saving(won["annual_cost_inr"], won["last_year_cost_inr"])
    facts["total_before_freight"] = format_inr(float(won["annual_cost_inr"].sum()))
    facts["saving_vs_last_year"] = describe_change(s, p) if s is not None else None
    facts["saving_basis"] = _saving_basis(n_ly)
    a.table = _award_table(won, "Total before freight")
    if not targets:
        a.tag = "landed cost, no freight to add"
        facts["no_freight_to_add"] = ("Every vendor that wins a line quotes freight included, so the landed cost is "
                                      "the quoted total.")
        return a

    effects = [freight_effect(data, n, scope.names, rates, terms=terms[n]) for n in targets]
    a.tag = f"landed cost, freight sensitivity for {join_names(targets)}"
    a.sensitivity = [e["sensitivity"] for e in effects if e["sensitivity"]]
    facts.update({
        "headline_before_freight": next((e["headline"] for e in effects if e["headline"]), None),
        "freight_vendors": targets,
        "freight_risks": [e["text"] for e in effects],
        "freight_must_include": [m for e in effects for m in e["must_include"]],
        "offer": f"Offer to draft a clarification asking {join_names(targets)} to quote freight.",
    })
    if rate_inr_per_kg is not None:
        at = []
        for e in effects:
            if e["sensitivity"]:
                row = e["sensitivity"]["table"].set_index("rate_inr_per_kg").loc[float(rate_inr_per_kg)]
                at.append(f"With {e['vendor']}'s freight at {format_inr(rate_inr_per_kg, per_unit=True)}/kg the award "
                          f"{describe_change(row['saving_inr'], row['saving_pct'])} against last year")
        facts["at_the_rate_asked"] = at
    a.missing_data = [f"{n}'s freight charge in ₹ per kg" for n in targets]
    a.risks = facts["freight_risks"]
    a.values = {"zero_rates": {e["vendor"]: (e["sensitivity"] or {}).get("zero_rate") for e in effects}}
    return a


# ---------- line_lookup ----------

LOOKUP_COLUMNS = ["line_id", "description", "vendor", "price_inr_per_piece", "label", "assumptions",
                  "confidence", "raw_price_text", "source_file", "source_snippet"]
# Said next to a price when the vendor's freight terms mean it is not a delivered price.
FREIGHT_NOTES = {"extra": "freight extra, not in this price", "unclear": "freight terms unclear"}


def line_lookup(data: AnalystData, scope: Scope, line_ids=None) -> Analysis:
    """What the vendors in scope quoted on these lines (every line if none given): price, label,
    confidence and the source snippet, plus the cheapest counted price per line and the gap to the next."""
    known = [int(i) for i in data.rfx_lines["line_id"]]
    asked = [int(i) for i in (line_ids or []) if str(i).strip().lstrip("-").isdigit()]
    unknown = [i for i in asked if i not in known]
    lines = [i for i in dict.fromkeys(asked) if i in known] or known
    which = (f"line{'s' if len(lines) > 1 else ''} {join_names([str(i) for i in lines])}"
             if asked and len(lines) <= MAX_LINES_IN_DETAIL else f"{len(lines)} lines")
    facts = {"vendors_shown": join_names(scope.names), **_scope_facts(scope, data)}
    a = Analysis(LINE_LOOKUP, facts, tag=f"line lookup, {which}", func=line_lookup,
                 call=f"line_lookup(data, scope, line_ids={lines if asked else None!r})", excluded=scope.excluded,
                 caveats=_scope_caveats(scope) + ([f"There is no RFx line {join_names([str(i) for i in unknown])}."]
                                                  if unknown else []),
                 explanation="Looks up what each vendor considered quoted on the lines asked about, with the label, "
                             "confidence and the source text, and finds the cheapest counted price per line.")
    df = data.df
    rows = df[df["rfx_line_id"].isin(lines) & df["display_name"].isin(scope.names)].copy()
    rows["line_id"] = rows["rfx_line_id"].astype(int)
    rows["vendor"] = rows["display_name"]
    a.table = rows.sort_values(["line_id", "price_inr_per_piece", "vendor"], na_position="last")[LOOKUP_COLUMNS].reset_index(drop=True)
    desc = data.rfx_lines.set_index("line_id")["description"]
    qty = data.rfx_lines.set_index("line_id")["annual_qty"]
    won = award_by_line(data, scope.names).set_index("line_id")

    price_notes: list[str] = []

    def quote_text(r) -> str:
        if pd.isna(r.price_inr_per_piece):
            return f"{r.vendor} {r.label.lower()}"
        notes = [str(x) for x in (r.assumptions if isinstance(r.assumptions, list) else []) if x]
        if freight := FREIGHT_NOTES.get(_freight_terms(data, r.vendor)):
            notes.append(freight)
        price_notes.extend(f"{r.vendor} line {r.line_id}: {n}" for n in notes)
        detail = f"{r.label}: {'; '.join(notes)}" if notes else r.label
        return f"{r.vendor} {format_inr(r.price_inr_per_piece, per_unit=True)} per piece ({detail})"

    def line_text(i: int) -> str:
        mine = a.table[a.table["line_id"] == i]
        quotes = [quote_text(r) for r in mine.itertuples()]
        text = f"Line {i} ({desc[i]}): " + ("; ".join(quotes) or "no vendor considered quoted it")
        if i in won.index:
            w = won.loc[i]
            text += f". Cheapest counted price: {w['winner']}"
            if w["runner"] is not None and pd.notna(w["runner_price"]):
                gap = float(w["runner_price"]) - float(w["winner_price"])
                text += (f", {format_inr(gap, per_unit=True)} per piece ({format_inr(gap * qty[i])} a year) "
                         f"below {w['runner']}")
        return text

    if len(lines) <= MAX_LINES_IN_DETAIL:
        facts["lines"] = [line_text(i) for i in lines]
        facts["line_ids_asked"] = [str(i) for i in lines] if asked else []
        if price_notes:
            facts["price_notes"] = price_notes
    else:
        counts = won.loc[won.index.isin(lines), "winner"].value_counts()
        facts["cheapest_counts"] = [f"{n} has the cheapest counted price on {_plural(int(k), 'line')}"
                                    for n, k in counts.items()]
        facts["table_note"] = f"The table below lists every vendor's price on all {len(lines)} lines with its source."
        skipped = a.table[a.table["label"] == NOT_QUOTED]
        facts["not_quoted"] = [f"{n} did not quote {_plural(len(g), 'line')}: {join_names([str(i) for i in g['line_id']])}"
                               for n, g in skipped.groupby("vendor", sort=False)] or \
                              [f"Every vendor shown quoted all {len(lines)} lines."]
        facts["not_quoted_line_ids"] = [str(i) for i in sorted(set(skipped["line_id"]))]
    a.caveats += review_caveats(data, scope.names, lines)
    a.values = {"lines": lines}
    return a


# ---------- fx_sensitivity ----------

def fx_sensitivity(data: AnalystData, scope: Scope, change_pct: float, currency: str = "USD") -> Analysis:
    """Re-run the cheapest-per-line award with the `currency` rate changed by change_pct %.

    Prices converted from that currency scale by the same factor (every conversion is linear
    in the rate). Lists the lines whose winner changes and how each repriced vendor moves.
    """
    code = currency.upper()
    factor = 1 + float(change_pct) / 100
    old_rate = data.fx_rates.get(code)
    sign = "+" if change_pct >= 0 else "-"
    pct = f"{plain_number(abs(change_pct))}%"
    move = f"a {pct} {'rise' if change_pct >= 0 else 'fall'} in the {code} rate"
    facts = {"change": move, "change_pct": plain_number(change_pct), "currency": code,
             "vendors_considered": join_names(scope.names), **_scope_facts(scope, data)}
    a = Analysis(FX_SENSITIVITY, facts, tag=f"FX sensitivity, {code} {sign}{pct}",
                 func=fx_sensitivity, call=f"fx_sensitivity(data, scope, change_pct={change_pct!r}, currency={code!r})",
                 excluded=scope.excluded, caveats=_scope_caveats(scope),
                 explanation=f"Multiplies every price converted from {code} by {factor:g} (the rate changed by "
                             f"{sign}{abs(change_pct):g}%), re-runs the cheapest-per-line award, and lists the lines "
                             "that change hands.")
    if old_rate is not None:
        facts["rate"] = (f"{format_inr(old_rate, per_unit=True)} per {code} becomes "
                         f"{format_inr(old_rate * factor, per_unit=True)}")

    shifted = isolated(data.df)
    in_code = shifted["assumptions"].map(fx_currency) == code
    shifted.loc[in_code, "price_inr_per_piece"] = shifted.loc[in_code, "price_inr_per_piece"] * factor
    moved = AnalystData(shifted, data.vendors, data.rfx_lines, data.last_year,
                        {**data.fx_rates, **({code: old_rate * factor} if old_rate else {})}, data.fx_date)

    currency_vendors = sorted(set(data.df.loc[in_code, "display_name"]))
    inside = [n for n in currency_vendors if n in scope.names]
    outside = [vendor_label(n, data) for n in currency_vendors if n not in scope.names]
    facts["vendors_priced_in_currency"] = join_names(inside) or "none"
    if outside:
        facts["currency_vendors_not_considered"] = outside

    before = award_by_line(data, scope.names).set_index("line_id")
    after = award_by_line(moved, scope.names).set_index("line_id")
    qty = data.rfx_lines.set_index("line_id")["annual_qty"]
    changes = []
    for i in before.index.intersection(after.index):
        b, f = before.loc[i], after.loc[i]
        if b["winner"] != f["winner"]:
            changes.append({"line_id": int(i), "winner_before": b["winner"], "price_before_inr_per_piece": b["winner_price"],
                            "winner_after": f["winner"], "price_after_inr_per_piece": f["winner_price"]})
    a.table = pd.DataFrame(changes, columns=["line_id", "winner_before", "price_before_inr_per_piece",
                                             "winner_after", "price_after_inr_per_piece"])
    total_before = float((before["annual_qty"] * before["winner_price"]).sum())
    total_after = float((after["annual_qty"] * after["winner_price"]).sum())
    facts["lines_changing_hands_count"] = str(len(changes))
    facts["lines_changing_hands"] = [
        f"Line {c['line_id']}: {c['winner_before']} ({format_inr(c['price_before_inr_per_piece'], per_unit=True)}) "
        f"to {c['winner_after']} ({format_inr(c['price_after_inr_per_piece'], per_unit=True)})" for c in changes]
    change = describe_change(total_before - total_after)
    facts["award_total"] = (f"{format_inr(total_before)} before, {format_inr(total_after)} after"
                            + ("" if change == "no change" else f": the award {change}")) if len(before) else "nothing to award"

    repriced = data.df[in_code & data.df["display_name"].isin(scope.names) & data.df["price_inr_per_piece"].notna()]
    prices = pd.DataFrame({"line_id": repriced["rfx_line_id"].astype(int), "vendor": repriced["display_name"],
                           "price_before_inr_per_piece": repriced["price_inr_per_piece"],
                           "price_after_inr_per_piece": shifted.loc[repriced.index, "price_inr_per_piece"]})
    prices = prices.reset_index(drop=True)
    # Main table: the lines that change hands; with none, the repriced prices; with neither, words only.
    if changes and not prices.empty:
        a.extra_tables = [(f"Prices converted from {code}, before and after", prices)]
    elif not changes:
        a.table = None if prices.empty else prices
        a.answer_type = "text" if prices.empty else "table"
    a.caveats += [f"Prices quoted in INR are unchanged; only prices converted from {code} move."]
    a.values = {"changes": changes, "prices": prices, "total_before": total_before, "total_after": total_after}
    return a


# ---------- quality_risk ----------

def quality_risk(data: AnalystData, named=None) -> Analysis:
    """Quality status per vendor, what each vendor claimed against its certificate, and the open
    risks by severity. Covers every vendor (or the ones named): this is about quality itself."""
    asked, unknown = resolve_vendors(named, data)
    vendors = data.vendors[data.vendors["display_name"].isin(asked)] if asked else data.vendors
    rows, risk_rows = [], []
    for r in vendors.itertuples():
        reasons = [x for x in (r.quality_reasons if isinstance(r.quality_reasons, list) else []) if isinstance(x, str)]
        cert = [_FILE_NOTE.sub("", x).strip() for x in reasons
                if "certificate" in x.lower() or "iso" in x.lower()]
        found = [x for x in (r.open_risks if isinstance(r.open_risks, list) else [])]
        rows.append({"vendor": r.display_name, "quality_status": r.quality_status,
                     "certificate_vs_claims": "; ".join(cert) or "no certificate finding",
                     "high_risks": sum(x["severity"] == HIGH for x in found),
                     "medium_risks": sum(x["severity"] == MEDIUM for x in found),
                     "low_risks": sum(x["severity"] == LOW for x in found)})
        risk_rows += [{"severity": x["severity"], "vendor": r.display_name, "risk": x["text"]} for x in found]
    risk_rows.sort(key=lambda x: (SEVERITY_ORDER.index(x["severity"]), x["vendor"]))
    table = pd.DataFrame(rows, columns=["vendor", "quality_status", "certificate_vs_claims", "high_risks",
                                        "medium_risks", "low_risks"])
    risks = pd.DataFrame(risk_rows, columns=["severity", "vendor", "risk"])
    facts = {
        "quality_status": [f"{r['vendor']}: {r['quality_status']}" + (
            f" ({'; '.join(quality_findings(r['vendor'], data))})" if r["quality_status"] in (FAIL, UNCLEAR) else "")
            for r in rows],
        "certificate_vs_claims": [f"{r['vendor']}: {r['certificate_vs_claims']}" for r in rows],
        "not_passed": [r["vendor"] for r in rows if r["quality_status"] != PASS],
    }
    for sev in SEVERITY_ORDER:
        facts[f"{sev}_risks"] = [f"{x['vendor']}: {x['risk']}" for x in risk_rows if x["severity"] == sev]
    facts["vendors_with_high_risks"] = sorted({x["vendor"] for x in risk_rows if x["severity"] == HIGH})
    facts["risk_counts"] = ", ".join(f"{len(facts[f'{sev}_risks'])} {sev}" for sev in SEVERITY_ORDER)
    a = Analysis(QUALITY_RISK, facts, tag="quality and risks", func=quality_risk,
                 call=f"quality_risk(data, named={asked or None!r})", table=table,
                 extra_tables=[("Open risks, high first", risks)] if not risks.empty else [],
                 caveats=_scope_caveats(Scope(names=[], unknown=unknown)),
                 explanation="Reads each vendor's quality checks (questionnaire answers checked against its "
                             "certificate; the certificate wins) and lists open risks, high severity first.")
    a.values = {"statuses": {r["vendor"]: r["quality_status"] for r in rows}}
    return a
