"""Answer a buyer's plain-English question about the comparison.

Claude reads the question and writes pandas code. Plain Python runs that code in a
locked-down namespace (copies of the data, no imports, no files, 5 second limit).
A second short Claude call words the answer using only the numbers in the code's
`result`, and code then checks that every number it wrote really is in `result`.
Claude never does arithmetic itself.

No Streamlit here except the API key lookup shared with extract.py.
"""

import ast
import builtins
import copy
import io
import json
import logging
import math
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Literal

import anthropic
import httpx2  # the HTTP library under anthropic 1.x; raises mid-stream network drops
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from aera.compare import AMBIGUOUS_ASSUMPTION, COMPARABLE, FAIL, NOT_COMPARABLE, NOT_QUOTED, PASS, UNCLEAR, WITH_ASSUMPTION
from aera.config import MODEL
from aera.extract import ExtractionError, _client, _usage_dict, cost_usd
from aera.normalize import describe_rates
from aera.rfx import RFx

log = logging.getLogger(__name__)

PLAN_MAX_TOKENS = 16000
SUMMARY_MAX_TOKENS = 1000
STREAM_DROP_RETRIES = 2
CODE_TIMEOUT_S = 5
SAMPLE_ROWS = 10
SAMPLE_TEXT_CHARS = 120  # long snippets in the sample rows are cut to this
SUMMARY_RESULT_ROWS = 60  # rows of `result` shown to the summary call
MAX_REVIEW_ROWS_NAMED = 10

FRIENDLY_FAIL = ("Sorry, I couldn't work that out from the comparison data. Try asking it a "
                 "different way, or about fewer vendors or lines at once. The technical details "
                 "are under \"Show the working\".")


class AnalystError(Exception):
    """A problem the user should see in plain words (bad key, network, refusal...)."""


class CodeError(Exception):
    """Generated code was rejected or failed. The message is sent back to Claude to fix."""


# ---------- Output schemas (what Claude must return) ----------

class ChartSpec(BaseModel):
    x: str = Field(description="Column of result for the categories (e.g. vendor display_name).")
    y: str = Field(description="Numeric column of result for the bar length.")
    kind: Literal["bar", "horizontal_bar"]


class ExcludedVendor(BaseModel):
    display_name: str
    reason: str = Field(description="Short plain words, e.g. 'failed quality', 'quality unclear', 'did not quote line 3'.")


class AnalystPlan(BaseModel):
    answer_type: Literal["text", "table", "chart"]
    pandas_code: str = Field(description="Python using only df, vendors, rfx_lines, last_year and pd. "
                                         "Must assign `result`. Empty string only if nothing useful can be computed.")
    chart_spec: ChartSpec | None = Field(description="Only when answer_type is chart; otherwise null.")
    explanation: str = Field(description="One or two plain sentences on what the code does.")
    caveats: list[str]
    data_sufficient: bool
    missing_data: list[str] = Field(description="What is missing, in plain words. Empty if data_sufficient.")
    excluded_vendors: list[ExcludedVendor] = Field(description="Every vendor your code's filters left out, with why. Empty if none.")


class Summary(BaseModel):
    answer: str = Field(description="2 to 4 plain sentences.")


# ---------- Prompts ----------

ANALYST_RULES = f"""You help a procurement buyer question a comparison of vendor quotes.
You never answer from memory and never do arithmetic yourself: you write pandas code, and plain Python runs it.

Data your code can use (all are copies):
- df: the comparison, one row per (RFx line, vendor). price_inr_per_piece is already converted to INR per piece.
- vendors: one row per vendor (quality, risks, freight, payment days, discounts).
- rfx_lines: line_id, description, annual_qty, uom, nominal_weight_g.
- last_year: line_id, price_inr_per_piece (last year's contract price; lines with no price on file are absent).
- annual_weight_kg(line_ids): total kg a year for those RFx lines (annual_qty x nominal_weight_g / 1000). Raises if a line has no weight.
- pd: pandas.

Code rules:
- Assign the final answer to a variable named result: a DataFrame, Series, number or short string. For a table or chart make it a DataFrame with clear column names.
- No import statements. No open, exec, eval, getattr, globals. No names or attributes starting with an underscore.
- No file or network access: no pd.read_*, no to_csv / to_excel / to_json. Do not use DataFrame.query or DataFrame.eval; filter with boolean masks. Do not use str.format; use f-strings.
- Join df to rfx_lines on df.rfx_line_id == rfx_lines.line_id, and to vendors on vendor.
- Money columns are named by what they hold, and the app formats them (₹, lakh, crore) from the name: yearly or total amounts end in _inr (e.g. annual_cost_inr, gap_inr), per-piece prices end in _inr_per_piece, per-kg rates end in _inr_per_kg. Keep them as plain numbers rounded with .round(2): never format, never divide into lakh or crore yourself.
- Put money in a DataFrame column even when the answer is a single value (a one-row DataFrame).
- Give percentages as 0-100 values rounded to 1 decimal, in columns whose names end in _pct.
- A saving or difference against a baseline (last year, another vendor) goes in a column ending in saving_inr (e.g. saving_inr, split_saving_inr), POSITIVE when it saves money and NEGATIVE when it costs more. Its percentage goes in the matching column ending in saving_pct (e.g. split_saving_pct), with the same sign. The app words these as "saves ₹X (p%)" or "costs ₹X (p%) more".
- For an award or split across vendors, include how many lines each vendor wins.
- Show vendors to the buyer by display_name.

Business rules:
- Annual cost of a line = annual_qty x price_inr_per_piece.
- Only rows with included_in_totals True count in totals (labels "{COMPARABLE}" and "{WITH_ASSUMPTION}"). Rows labelled "{NOT_QUOTED}" or "{NOT_COMPARABLE}" are excluded from totals unless the buyer explicitly asks to include them.
- Missing is never zero. Never fillna(0) a price, cost or total. A line a vendor did not quote has no cost, not a zero cost.
- When you total per vendor, also return how many lines each total covers (e.g. a lines_in_total column), so incomplete totals are visible. Totals over different sets of lines are not like-for-like: say so in caveats. When asked who is cheapest overall, compare on the lines every compared vendor quoted, and say which lines were left out.
- "Cleared quality" (passed quality, qualified) means vendors.quality_status == "{PASS}". Vendors with "{UNCLEAR}" are excluded: say so explicitly in caveats and name them by display_name. Vendors with "{FAIL}" are excluded too.
- Discounts in vendors.discounts are recorded, never applied. Apply one only if the buyer explicitly asks, and then state its condition in caveats.
- vendors.freight is "included", "extra", "unclear" or null. Prices do not include freight marked "extra". Say so in caveats when it affects a cost comparison.
- When a vendor has not quoted a cost (e.g. freight "extra" or "unclear"): set data_sufficient false and put the unquoted cost in missing_data, e.g. "<vendor>'s freight charge in ₹ per kg". For a direct comparison of two vendors (is X cheaper than Y), you may compute breakeven_freight_inr_per_kg = gap_inr / annual_weight_kg(line_ids of the lines both priced): above that rate Y is cheaper. For an award, split or saving across vendors, do NOT compute a freight breakeven or freight gap: the app re-runs the award at each freight rate itself and shows a freight sensitivity table.
- If any row your answer relies on has needs_review True and buyer_confirmed False, name those rows (display_name and line) in caveats: the buyer has not confirmed their price yet.

If the data cannot fully answer the question, set data_sufficient false and list what is missing in missing_data in plain words. Still write code for the closest useful facts if there are any (e.g. the cost gap that freight would have to exceed to change the answer, computed in code). Set pandas_code to an empty string only if nothing useful can be computed.

excluded_vendors: every vendor that a filter in your code left out (quality, freight, not quoted...), with the reason in a few words. Use display_name.

answer_type: "table" for a list or comparison; "chart" when a picture helps (chart_spec x and y must be columns of result, y numeric); "text" for a single fact. chart_spec is null unless answer_type is "chart".
explanation: one or two plain sentences for the buyer on what the code does.
caveats: short plain-English sentences. Empty list if there are none."""

FIX_PROMPT = """Running your pandas_code failed:

{error}

Return the full response again with corrected pandas_code. Follow the same rules."""

WRITER_RULES = """You write a short answer for a procurement buyer from a set of FACTS (JSON) worked out by code.
Write 2 to 4 plain sentences (a 5th only if needed to fit everything below). No headings, bullet points or markdown.

Every figure in the facts is already formatted. Copy figures exactly as written there. Never calculate, add, subtract, re-round or convert, and never write a number that is not in the facts.

The 5 rules:
1. Money: always the ₹ form given in the facts (e.g. ₹3.64 crore, ₹12.03 lakh, ₹35,780, ₹5.14 per piece). Never write raw digits for money.
2. Direction in words, not signs: "saves ₹12.03 lakh (3.2%) against last year" or "costs ₹2.10 lakh (0.6%) more". Never put a minus sign on money or a percentage. savings_in_words already has these phrases: copy them.
3. If a cost the vendor hasn't quoted affects the answer, use what the facts say about it (the saving before freight, how it shrinks, the ₹/kg where it is gone) and offer to draft a clarification asking the vendor for it. Freight always shrinks a saving from the first rupee: never say a saving shrinks only "above" some rate.
4. If headline_before_freight is given, the answer MUST open with it as its first sentence, e.g. "Saves ₹12.03 lakh (3.2%) before freight." Do not state the same saving again. Every item in price_risks MUST then appear, worded as given, in the next sentence, e.g. "<vendor> wins 21 lines but hasn't quoted freight; the saving shrinks as freight rises and is gone at about ₹3.50/kg (table below)."
5. Every vendor in excluded_vendors MUST be named with its reason, e.g. "<vendor> (failed quality) and <vendor> (quality unclear) were left out."

Name vendors as the facts do. If data_complete is false, say what is missing. Do not repeat other caveats; they are shown separately."""


# ---------- Data passed to the code ----------

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


# ---------- Context sent to Claude ----------

def build_context(data: AnalystData) -> str:
    df = data.df
    parts = [
        f"## Comparison table `df` ({len(df)} rows)",
        "Columns (dtype):",
        *[f"- {c} ({_dtype_text(df[c])})" for c in df.columns],
        "",
        "Distinct values:",
        *[f"- {c}: {json.dumps(sorted(map(str, df[c].dropna().unique())), ensure_ascii=False)}"
          for c in ("vendor", "display_name", "label", "confidence") if c in df.columns],
        "",
        f"{min(SAMPLE_ROWS, len(df))} sample rows (long text cut short):",
        _json(_sample_rows(df)),
        "",
        "## Vendors `vendors` (every vendor, full)",
        f"Columns: {', '.join(data.vendors.columns)}",
        _json(_records(data.vendors)),
        "",
        "## RFx lines `rfx_lines`",
        _json(_records(data.rfx_lines)),
        "",
        "## Last-year prices `last_year` (INR per piece)",
        _json(_records(data.last_year)),
        "",
        "## FX",
        "FX rates: " + describe_rates(data.fx_rates, data.fx_date) + ".",
    ]
    return "\n".join(parts)


def _dtype_text(col: pd.Series) -> str:
    sample = col.dropna().head(50)
    if any(isinstance(v, list) for v in sample):
        return "list"
    return str(col.dtype)


def _sample_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = df.sample(n=min(SAMPLE_ROWS, len(df)), random_state=0).sort_index()
    return [_cut(r) for r in _records(rows)]


def _cut(value):
    if isinstance(value, str) and len(value) > SAMPLE_TEXT_CHARS:
        return value[:SAMPLE_TEXT_CHARS] + "..."
    if isinstance(value, dict):
        return {k: _cut(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut(v) for v in value]
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [jsonable(r) for r in frame.to_dict(orient="records")]


def jsonable(value):
    """Plain JSON values: NaN / NA -> None, numpy numbers -> Python numbers."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# ---------- Restricted code runner ----------

# A plain dict of harmless builtins, so generated code cannot reach the real builtins module.
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "KeyError", "ValueError", "TypeError", "ZeroDivisionError",
)}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "input", "breakpoint", "help", "dir", "type", "object", "super", "memoryview",
    "BaseException", "SystemExit", "KeyboardInterrupt",
}
# Attributes that could reach files, the network, code evaluation or interpreter internals.
BLOCKED_ATTRS = {"eval", "query", "format", "format_map", "os", "sys", "io", "subprocess",
                 "builtins", "importlib", "mro"}
BLOCKED_ATTR_PREFIXES = ("_", "read_", "gi_", "cr_", "ag_", "tb_", "f_", "co_", "func_")
ALLOWED_TO_ATTRS = {"to_dict", "to_list", "to_frame", "to_numpy", "to_string", "to_numeric",
                    "to_datetime", "to_timedelta", "to_records", "to_period", "to_timestamp",
                    "to_flat_index"}
PD_ALLOWED = {
    "DataFrame", "Series", "Index", "MultiIndex", "concat", "merge", "to_numeric", "to_datetime",
    "to_timedelta", "isna", "isnull", "notna", "notnull", "NA", "NaT", "Timestamp", "Timedelta",
    "pivot_table", "pivot", "crosstab", "cut", "qcut", "melt", "unique", "Categorical",
    "CategoricalDtype", "IndexSlice", "NamedAgg", "date_range", "get_dummies",
}
BLOCKED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                 ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)


def check_code(code: str) -> ast.Module:
    """Parse the code and reject anything outside the allowed subset. Raises CodeError."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise CodeError(f"SyntaxError: {e.msg} (line {e.lineno})") from None

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise CodeError(f"Not allowed: {type(node).__name__} (line {node.lineno}).")
        if isinstance(node, ast.Name):
            _check_name(node, parents)
        elif isinstance(node, ast.Attribute):
            _check_attr(node)
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            _check_function(node)
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                raise CodeError(f"Not allowed: bare 'except:' (line {node.lineno}); "
                                "catch a specific error such as KeyError.")
    return tree


def _check_name(node: ast.Name, parents: dict) -> None:
    if node.id.startswith("_") or node.id in BLOCKED_NAMES:
        raise CodeError(f"Not allowed: the name '{node.id}' (line {node.lineno}).")
    if node.id == "pd":
        parent = parents.get(node)
        if not (isinstance(parent, ast.Attribute) and parent.value is node):
            raise CodeError(f"Use pandas only as pd.<function> (line {node.lineno}).")
        if parent.attr not in PD_ALLOWED:
            raise CodeError(f"Not allowed: pd.{parent.attr} (line {node.lineno}).")


def _check_attr(node: ast.Attribute) -> None:
    attr = node.attr
    blocked = (
        attr in BLOCKED_ATTRS
        or attr.startswith(BLOCKED_ATTR_PREFIXES)
        or (attr.startswith("to_") and attr not in ALLOWED_TO_ATTRS)
    )
    if blocked:
        raise CodeError(f"Not allowed: the attribute '.{attr}' (line {node.lineno}).")


def _check_function(node) -> None:
    if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
        raise CodeError(f"Not allowed: function name '{node.name}' (line {node.lineno}).")
    args = node.args
    for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
        if a is not None and a.arg.startswith("_"):
            raise CodeError(f"Not allowed: argument name '{a.arg}' (line {node.lineno}).")


class _Timeout(BaseException):
    """BaseException so `except Exception` in generated code cannot swallow it."""


def run_code(code: str, namespace: dict, timeout: float = CODE_TIMEOUT_S):
    """Run checked code with only `namespace` + pd + safe builtins. Returns its `result`.

    Runs in its own thread with a line tracer that stops Python-level loops after
    `timeout` seconds. Raises CodeError with a message Claude can use to fix the code.
    """
    tree = check_code(code)
    compiled = compile(tree, "<analyst code>", "exec")
    env = {"__builtins__": dict(SAFE_BUILTINS), "pd": pd, **namespace}
    outcome: dict = {}
    deadline = time.monotonic() + timeout

    def tracer(frame, event, arg):
        if time.monotonic() > deadline:
            raise _Timeout
        return tracer

    def target():
        sys.settrace(tracer)  # affects only this thread
        try:
            exec(compiled, env)  # noqa: S102 - checked above, restricted namespace
        except BaseException as e:  # noqa: BLE001 - reported back to Claude
            outcome["error"] = e
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=target, daemon=True, name="analyst-code")
    worker.start()
    worker.join(timeout + 1)
    if worker.is_alive() or isinstance(outcome.get("error"), _Timeout):
        raise CodeError(f"The code took longer than {timeout:g} seconds and was stopped.")
    if "error" in outcome:
        e = outcome["error"]
        raise CodeError(f"{type(e).__name__}: {e}")
    if "result" not in env:
        raise CodeError("The code did not assign a variable named result.")
    return env["result"]


# ---------- Money format (code, never the model) ----------

CRORE, LAKH = 10_000_000, 100_000
MISSING_TEXT = "—"  # missing is never zero
# Column-name suffix -> (kind, header unit). The analyst prompt asks Claude to use these names.
MONEY_SUFFIXES = (("_inr_per_piece", "piece", "₹/piece"), ("_inr_per_kg", "kg", "₹/kg"),
                  ("_inr", "amount", "₹"))


def _round_half_up(value: float, places: int) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def _indian_digits(n: int) -> str:
    """1234567 -> '12,34,567' (last three digits, then groups of two)."""
    s = str(n)
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups + [tail])


def _grouped(d: Decimal, places: int) -> str:
    whole = int(d)
    text = _indian_digits(whole)
    if places:
        text += "." + f"{d:.{places}f}".split(".")[1]
    return text


def format_inr(value, per_unit: bool = False) -> str:
    """INR in Indian units: '₹2.67 crore', '₹4.31 lakh', '₹35,780'.

    per_unit=True is for per-piece or per-kg prices: always 2 decimals, never lakh/crore
    ('₹5.14', '₹1,234.50'). Missing values give '—', never '₹0'.
    """
    if value is None or isinstance(value, bool):
        return MISSING_TEXT
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MISSING_TEXT
    if math.isnan(v) or math.isinf(v):
        return MISSING_TEXT
    sign, v = ("-" if v < 0 else ""), abs(v)

    if per_unit:
        text = _grouped(_round_half_up(v, 2), 2)
    elif v >= CRORE:
        text = _grouped(_round_half_up(v / CRORE, 2), 2) + " crore"
    elif _round_half_up(v / LAKH, 2) >= 100:  # e.g. 99,99,999 rounds up to 1 crore
        text = "1.00 crore"
    elif v >= LAKH:
        text = _grouped(_round_half_up(v / LAKH, 2), 2) + " lakh"
    elif _round_half_up(v, 0) >= LAKH:  # 99,999.6 rounds up to 1 lakh
        text = "1.00 lakh"
    else:
        text = _grouped(_round_half_up(v, 0), 0)
    if text.strip("0.,") == "":  # rounds to zero: no minus sign
        sign = ""
    return f"{sign}₹{text}"


def money_kind(column) -> str | None:
    """'amount', 'piece' or 'kg' from the column name's suffix; None if not money."""
    name = str(column)
    for suffix, kind, _ in MONEY_SUFFIXES:
        if name.endswith(suffix):
            return kind
    return None


def money_header(column) -> str:
    """'annual_cost_inr' -> 'annual_cost (₹)', 'price_inr_per_piece' -> 'price (₹/piece)'."""
    name = str(column)
    for suffix, _, unit in MONEY_SUFFIXES:
        if name.endswith(suffix):
            return f"{name[: -len(suffix)] or 'value'} ({unit})"
    return name


def format_money(value, kind: str) -> str:
    if kind == "amount":
        return format_inr(value)
    text = format_inr(value, per_unit=True)
    return text + "/kg" if kind == "kg" and text != MISSING_TEXT else text


_SAVING_COLUMN = re.compile(r"^(?P<stem>.*savings?)_inr$")


def saving_pct_column(column) -> str | None:
    """'split_saving_inr' -> 'split_saving_pct'. None if the column is not a saving."""
    m = _SAVING_COLUMN.match(str(column))
    return f"{m.group('stem')}_pct" if m else None


def describe_change(saving_inr, saving_pct=None) -> str:
    """Direction in words, never a minus sign: 'saves ₹12.03 lakh (3.2%)', 'costs ₹2.10 lakh (0.6%) more'.

    saving_inr is positive when money is saved. saving_pct (0-100, same sign) is optional.
    """
    if format_inr(saving_inr) == MISSING_TEXT:
        return MISSING_TEXT
    amount = format_inr(abs(float(saving_inr)))
    pct = None
    if saving_pct is not None and format_inr(saving_pct) != MISSING_TEXT:
        pct = _round_half_up(abs(float(saving_pct)), 1)
    if amount == "₹0" and not pct:
        return "no change"
    pct_text = f" ({pct}%)" if pct is not None else ""
    return f"saves {amount}{pct_text}" if float(saving_inr) > 0 else f"costs {amount}{pct_text} more"


def excel_header(column) -> str:
    """Excel keeps signed numbers, so saving headers say which way the sign goes."""
    if saving_pct_column(column):
        return f"{str(column)[:-len('_inr')]} (₹, positive = saves)"
    if str(column).endswith("saving_pct") or str(column).endswith("savings_pct"):
        return f"{column} (positive = saves)"
    return money_header(column)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    """The table as the buyer sees it: money in Indian units with ₹ headers, savings in words.

    A saving column and its matching _pct column become one column like 'saves ₹12.03 lakh (3.2%)'.
    """
    out = table.copy()
    for col in table.columns:
        pct_col = saving_pct_column(col)
        if pct_col and pd.api.types.is_numeric_dtype(table[col]):
            pcts = table[pct_col] if pct_col in table.columns else [None] * len(table)
            out[col] = pd.Series([describe_change(v, p) for v, p in zip(table[col], pcts)],
                                 index=table.index, dtype=object)
            if pct_col in out.columns:
                out = out.drop(columns=pct_col)
            continue
        kind = money_kind(col)
        if kind and pd.api.types.is_numeric_dtype(table[col]):
            out[col] = pd.Series([format_money(v, kind) for v in table[col]], index=table.index, dtype=object)
    return out.rename(columns=money_header)


# A minus sign on money or a percentage: "-₹4.31 lakh", "−2.1%", "- 3%".
_BARE_MINUS = re.compile(r"(?<![\w])[-−–]\s?(?:₹\s?\d|\d[\d,]*(?:\.\d+)?\s?%)")


def bare_minus_signs(text: str) -> list[str]:
    """Money or percentages written with a minus sign instead of words."""
    return [m.group() for m in _BARE_MINUS.finditer(text or "")]


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


# ---------- Turning `result` into a table and text ----------

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


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in_text(text: str) -> list[str]:
    return [m.group().rstrip(",") for m in _NUMBER.finditer(text or "")]


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def result_numbers(result) -> list[float]:
    """Every number in `result`: numeric cells, numbers inside text cells, and the row count."""
    values: list[float] = []

    def add(v) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float, np.number)):
            if not math.isnan(float(v)):
                values.append(float(v))
        elif isinstance(v, str):
            values.extend(f for f in map(_to_float, numbers_in_text(v)) if f is not None)
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)

    frame = result_frame(result)
    if frame is None:
        add(result if not isinstance(result, np.generic) else result.item())
        return values
    values.append(float(len(frame)))
    for col in frame.columns:
        add(str(col))
        for v in frame[col]:
            add(v.item() if isinstance(v, np.generic) else v)
    return values


def unsupported_numbers(answer: str, result, question: str = "") -> list[str]:
    """Numbers in the written answer that are not in `result` (or the question), allowing for rounding.

    Signs are ignored: "dearer by 414.50" correctly reports a result of -414.50.
    Money formatted by format_inr ('₹4.31 lakh') counts as in the result.
    """
    values = result_numbers(result)
    frame = result_frame(result)
    if frame is not None:
        values += result_numbers(display_table(frame))
    pool = [abs(v) for v in values]
    pool += [abs(f) for f in map(_to_float, numbers_in_text(question)) if f is not None]
    missing = []
    for text in numbers_in_text(answer):
        value = _to_float(text)
        if value is None:
            continue
        value = abs(value)
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(v - value) <= tolerance for v in pool) and text not in missing:
            missing.append(text)
    return missing


def chart_problem(table: pd.DataFrame | None, spec: dict | None) -> str | None:
    """Why this chart can't be drawn, or None if it can."""
    if table is None:
        return "the result is a single value"
    if not spec:
        return "no chart columns were given"
    for key in ("x", "y"):
        if spec.get(key) not in table.columns:
            return f"column '{spec.get(key)}' is not in the result"
    if not pd.api.types.is_numeric_dtype(table[spec["y"]]):
        return f"column '{spec['y']}' is not numeric"
    return None


def review_caveats(code: str, df: pd.DataFrame) -> list[str]:
    """Code-checked caveat: prices the buyer still has to confirm, if the code used prices."""
    if "price_inr_per_piece" not in code or df.empty:
        return []
    open_rows = df[df["needs_review"].astype(bool) & ~df["buyer_confirmed"].astype(bool)]
    if open_rows.empty:
        return []
    named = [f"{r.display_name} line {r.rfx_line_id}"
             for r in open_rows.head(MAX_REVIEW_ROWS_NAMED).itertuples()]
    more = len(open_rows) - len(named)
    names = "; ".join(named) + (f"; and {more} more" if more > 0 else "")
    return [f"Checked by code: {len(open_rows)} price(s) in the comparison still need your review "
            f"on the Compare page and are not confirmed yet: {names}."]


QUALITY_EXCLUSION_REASONS = {FAIL: "failed quality", UNCLEAR: "quality unclear"}


def vendors_in_result(result, vendors: pd.DataFrame) -> list[str]:
    """display_names of the vendors that `result` mentions (in cells or column names)."""
    frame = result_frame(result)
    text = str(result) if frame is None else frame.to_string() + " " + " ".join(map(str, frame.columns))
    return [r.display_name for r in vendors.itertuples()
            if str(r.display_name) in text or str(r.vendor) in text]


def excluded_vendors(plan_excluded: list[dict], code: str, result, vendors: pd.DataFrame) -> list[dict]:
    """Vendors left out by a filter, with why. Claude's list, checked against real vendor names,
    plus any vendor a quality filter removed that Claude did not list (reason from quality_status)."""
    names = {str(n).casefold(): str(n) for n in vendors["display_name"]}
    out: list[dict] = []
    for e in plan_excluded or []:
        name = names.get(str(e.get("display_name", "")).strip().casefold())
        if name and all(o["display_name"] != name for o in out):
            out.append({"display_name": name, "reason": (e.get("reason") or "left out").strip()})

    shown = vendors_in_result(result, vendors)
    if "quality_status" in code and shown:
        for r in vendors.itertuples():
            reason = QUALITY_EXCLUSION_REASONS.get(r.quality_status)
            if reason and r.display_name not in shown and all(o["display_name"] != r.display_name for o in out):
                out.append({"display_name": r.display_name, "reason": reason})
    return out


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


def price_risks(result, data: AnalystData) -> list[dict]:
    """HIGH risks that move the price, for vendors that WIN lines among the vendors in the result.

    - freight extra: the award is re-run with freight added (freight_sensitivity), giving the
      saving before freight, how it shrinks, and the ₹/kg where it is gone.
    - ambiguous lines defaulted to the higher reading and not confirmed, among the lines it wins.
    Each risk: {"vendor", "kind", "text", "must_include", "sensitivity"}; must_include are
    strings the written answer has to contain.
    """
    names = vendors_in_result(result, data.vendors)
    if not names:
        return []
    awards = award_by_line(data, names)
    df = data.df
    risks = []
    for name in names:
        won = awards[awards["winner"] == name]
        if won.empty:
            continue
        row = data.vendors[data.vendors["display_name"] == name].iloc[0]
        if row.get("freight") == "extra":
            risks.append(_freight_risk(name, names, data))
        mine = df[(df["display_name"] == name) & df["rfx_line_id"].isin(won["line_id"])
                  & ~df["buyer_confirmed"].astype(bool)]
        k = int(sum(AMBIGUOUS_ASSUMPTION in (a or []) for a in mine["assumptions"]))
        if k:
            risks.append({"vendor": name, "kind": "ambiguous", "must_include": [], "sensitivity": None,
                          "text": f"{k} of the {len(won)} lines {name} wins {'is' if k == 1 else 'are'} "
                                  "priced at the higher of two readings until you confirm"})
    return risks


def _freight_risk(name: str, allowed: list[str], data: AnalystData) -> dict:
    """Freight-extra risk worded from the sensitivity:
    'Saves ₹X (Y%) before freight. <vendor> wins N lines but hasn't quoted freight; the saving
    shrinks as freight rises and is gone at about ₹Z/kg (table below).'"""
    risk = {"vendor": name, "kind": "freight", "must_include": [], "sensitivity": None}
    try:
        s = freight_sensitivity(data, name, allowed)
    except SensitivityError as e:
        risk["text"] = f"{name} hasn't quoted freight; how freight changes the saving can't be worked out ({e})"
        return risk

    risk["sensitivity"] = {"vendor": name, **s}
    first = s["table"].iloc[0]
    n = int(first["lines_won"])
    before = describe_change(first["saving_inr"], first["saving_pct"])
    headline = f"{before[0].upper()}{before[1:]} before freight"
    head = f"{name} wins {n} line{'s' if n != 1 else ''} but hasn't quoted freight"
    if first["saving_inr"] <= 0:
        tail = "any freight makes the award dearer still (table below)"
    elif s["zero_rate"] is not None:
        gone = format_inr(s["zero_rate"], per_unit=True) + "/kg"
        tail = f"the saving shrinks as freight rises and is gone at about {gone} (table below)"
        risk["must_include"] = [gone]
    elif s["lines_won_at_max"] == 0:
        floor = format_inr(s["saving_at_max_inr"])
        tail = ("the saving shrinks as freight rises, but other vendors take its lines, so it never "
                f"falls below {floor} (table below)")
        risk["must_include"] = [floor]
    else:
        tail = ("the saving shrinks as freight rises and is still there at "
                f"{format_inr(FREIGHT_SEARCH_MAX, per_unit=True)}/kg (table below)")
    risk["headline"] = headline
    risk["must_include"].append("before freight")
    risk["text"] = f"{head}; {tail}"  # the headline is a separate fact the answer opens with
    return risk


def _plain_number(v) -> str:
    """Non-money numbers for the writer: Indian grouping, at most 2 decimals."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return MISSING_TEXT
    places = 0 if f == int(f) else 2
    d = _round_half_up(abs(f), places)
    return ("-" if f < 0 and d != 0 else "") + _grouped(d, places)


def _pct_text(v) -> str:
    f = float(v)
    return MISSING_TEXT if math.isnan(f) else f"{_round_half_up(f, 1)}%"


def formatted_rows(result, max_rows: int = SUMMARY_RESULT_ROWS) -> list[dict]:
    """`result` as rows of strings only: money via format_inr, savings in words, % as '3.2%'."""
    frame = result_frame(result)
    if frame is None:
        if isinstance(result, (int, float, np.number)) and not isinstance(result, bool):
            return [{"value": _plain_number(result)}]
        return [{"value": str(result)}]
    rows = []
    for rec in display_table(frame.head(max_rows)).to_dict(orient="records"):
        out = {}
        for k, v in rec.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = "; ".join(map(str, v))
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = MISSING_TEXT
            elif isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                out[k] = _pct_text(v) if str(k).endswith("_pct") else _plain_number(v)
            else:
                out[k] = str(v)
        rows.append(out)
    return rows


def build_facts(answer: "Answer", result) -> dict:
    """Everything the writer may say, as formatted strings. It never sees a raw number."""
    frame = result_frame(result)
    savings = []
    if frame is not None:
        label_col = next((c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])), None)
        for col in frame.columns:
            pct_col = saving_pct_column(col)
            if pct_col and pd.api.types.is_numeric_dtype(frame[col]):
                pcts = frame[pct_col] if pct_col in frame.columns else [None] * len(frame)
                for i, (v, p) in enumerate(zip(frame[col], pcts)):
                    who = f"{frame[label_col].iloc[i]}: " if label_col else ""
                    savings.append(f"{who}{describe_change(v, p)}")
    return {
        "question": answer.question,
        "what_the_calculation_did": answer.explanation,
        "result_rows": formatted_rows(result),
        "savings_in_words": savings,
        "excluded_vendors": [f"{e['display_name']} ({e['reason']})" for e in answer.excluded_vendors],
        "headline_before_freight": next((r["headline"] for r in answer.price_risks if r.get("headline")), None),
        "price_risks": [r["text"] for r in answer.price_risks],
        "freight_sensitivity": [
            {"vendor": sv["vendor"],
             "saving_gone_at": (format_inr(sv["zero_rate"], per_unit=True) + "/kg")
             if sv["zero_rate"] is not None else "not within the range searched",
             "table_shown_below_the_answer": formatted_rows(sv["table"])}
            for sv in answer.sensitivity],
        "data_complete": answer.data_sufficient,
        "missing_data": answer.missing_data,
    }


# ---------- Post-check of the written answer ----------

BIG_NUMBER_DIGITS = 5  # a bare number with more integer digits than this must be ₹-formatted


def _short_name(name: str, vendors: list[str]) -> str | None:
    """First word of a vendor name if no other vendor starts with it ('Ganesh'), else None."""
    words = name.split()
    if not words or len(words[0]) < 3:
        return None
    first = words[0]
    clash = any(v != name and v.split()[:1] == [first] for v in vendors)
    return None if clash else first


def _mentions(text: str, name: str, all_names: list[str]) -> bool:
    if name in text:
        return True
    short = _short_name(name, all_names)
    return short is not None and re.search(rf"\b{re.escape(short)}\b", text) is not None


def money_values(result) -> list[tuple[float, str]]:
    """(value, kind) for every money cell in `result`, to re-format bare numbers the writer typed."""
    frame = result_frame(result)
    if frame is None:
        return []
    out = []
    for col in frame.columns:
        kind = "amount" if saving_pct_column(col) else money_kind(col)
        if kind and pd.api.types.is_numeric_dtype(frame[col]):
            out += [(float(v), kind) for v in frame[col] if pd.notna(v)]
    return out


def _fix_big_numbers(text: str, money: list[tuple[float, str]]) -> tuple[str, list[str]]:
    notes = []

    def repl(m: re.Match) -> str:
        token = m.group()
        whole = token.replace(",", "").split(".")[0].lstrip("-")
        if len(whole) <= BIG_NUMBER_DIGITS or text[:m.start()].rstrip().endswith("₹"):
            return token
        value = abs(float(token.replace(",", "")))
        for v, kind in money:
            if abs(abs(v) - value) <= 0.5:
                fixed = format_money(abs(v), kind)
                notes.append(f"Code re-formatted {token} as {fixed}")
                return fixed
        notes.append(f"Unformatted number left in the answer: {token}")
        return token

    return _NUMBER.sub(repl, text), notes


def post_check(text: str, risks: list[dict], excluded: list[dict], money: list[tuple[float, str]],
               all_vendor_names: list[str]) -> tuple[str, list[str]]:
    """Checks the writer cannot skip. Returns (fixed text, notes on what code changed).

    1. Bare numbers over 5 digits are replaced with their ₹ form when they match a money value.
    2. Every excluded vendor must be named; if not, a 'Left out: ...' sentence is appended.
    3. Every price-risk vendor must be named in a sentence about that risk, with the ₹/kg
       breakeven when there is one; if not, the risk sentence from the facts is appended.
    """
    text, notes = _fix_big_numbers((text or "").strip(), money)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    added = []

    # Naming the vendor is not enough: the same sentence must say it was left out, or why.
    def said_left_out(e: dict) -> bool:
        words = (e["reason"].lower(), "left out", "excluded", "not included", "leaves out", "leave out")
        return any(_mentions(s, e["display_name"], all_vendor_names) and any(w in s.lower() for w in words)
                   for s in sentences)

    missing = [e for e in excluded if not said_left_out(e)]
    if missing:
        added.append("Left out: " + _join_names([f"{e['display_name']} ({e['reason']})" for e in missing]) + ".")

    risk_words = {"freight": ("freight",), "ambiguous": ("reading", "ambiguous", "confirm")}
    for r in risks:
        named = any(_mentions(s, r["vendor"], all_vendor_names)
                    and any(w in s.lower() for w in risk_words[r["kind"]]) for s in sentences)
        if not named or any(req not in text for req in r["must_include"]):
            sentence = r["text"][0].upper() + r["text"][1:] + "."
            if r.get("headline") and "before freight" not in text:
                sentence = f"{r['headline']}. {sentence}"
            added.append(sentence)
    if added:
        notes.append("Code added: " + " ".join(added))
        text = f"{text} {' '.join(added)}".strip()
    return text, notes


def _join_names(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------- Claude calls ----------

def _ask_claude(system: str, messages: list[dict], schema: type[BaseModel],
                max_tokens: int, purpose: str) -> tuple[dict, dict]:
    """One structured-output call. Returns (parsed JSON dict, usage dict with cost)."""
    try:
        client = _client()
    except ExtractionError as e:
        raise AnalystError(str(e)) from None
    try:
        response = _stream_with_retry(client, system, messages, schema, max_tokens)
    except anthropic.AuthenticationError:
        raise AnalystError("Authentication failed: check ANTHROPIC_API_KEY in .streamlit/secrets.toml.") from None
    except anthropic.RateLimitError:
        raise AnalystError("Rate limited by the Claude API. Wait a minute and try again.") from None
    except anthropic.BadRequestError as e:
        raise AnalystError(f"Claude rejected the request (400): {e.message}") from None
    except anthropic.APIStatusError as e:
        raise AnalystError(f"Claude API error ({e.status_code}): {e.message}") from None
    except (anthropic.APIConnectionError, httpx2.TransportError):
        raise AnalystError("Could not reach the Claude API. Check your internet connection.") from None

    usage = {**_usage_dict(response), "purpose": purpose}
    usage["cost_usd"] = cost_usd(usage)
    log.info("Claude call (%s, %s): %d in, %d out, %d cache write, %d cache read, ~$%.4f",
             purpose, usage["model"], usage["input_tokens"], usage["output_tokens"],
             usage["cache_write_tokens"], usage["cache_read_tokens"], usage["cost_usd"])

    if response.stop_reason == "refusal":
        raise AnalystError("Claude declined to answer this question.")
    if response.stop_reason == "max_tokens":
        raise AnalystError("Claude's answer was cut off (too long). Try a narrower question.")
    if response.parsed_output is None:
        raise AnalystError("Claude's answer did not match the expected format.")
    return response.parsed_output.model_dump(), usage


def _stream_with_retry(client, system, messages, schema, max_tokens):
    """Retry a connection that drops mid-answer (the SDK only retries failed connects)."""
    for attempt in range(1 + STREAM_DROP_RETRIES):
        try:
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=messages,
                output_format=schema,
            ) as stream:
                return stream.get_final_message()
        except httpx2.TransportError:
            if attempt == STREAM_DROP_RETRIES:
                raise
            time.sleep(2 * (attempt + 1))


# ---------- Public API ----------

@dataclass
class Answer:
    question: str
    asked_at: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    answer_type: str = "text"
    code: str = ""
    chart_spec: dict | None = None
    explanation: str = ""
    caveats: list[str] = field(default_factory=list)
    data_sufficient: bool = True
    missing_data: list[str] = field(default_factory=list)
    result: object = None
    table: pd.DataFrame | None = None
    text: str = ""  # the plain-English answer, worded from `result`
    error: str | None = None  # friendly message when the code failed twice
    code_errors: list[str] = field(default_factory=list)  # technical details, for "Show the working"
    unchecked_numbers: list[str] = field(default_factory=list)  # numbers in `text` not found in `result`
    excluded_vendors: list[dict] = field(default_factory=list)  # {"display_name", "reason"}
    price_risks: list[dict] = field(default_factory=list)  # {"vendor", "kind", "text"}
    wording_notes: list[str] = field(default_factory=list)  # what code added to or found in `text`
    facts: dict = field(default_factory=dict)  # the only input the answer writer saw
    sensitivity: list[dict] = field(default_factory=list)  # freight_sensitivity() per freight-extra winner
    usages: list[dict] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(u["cost_usd"] for u in self.usages)


Caller = Callable[..., tuple[dict, dict]]
_NO_CODE = object()


def ask(question: str, data: AnalystData, call: Caller | None = None) -> Answer:
    """Answer one question. Raises AnalystError only if the first Claude call fails
    (nothing was spent or nothing can be shown); later problems are kept on the Answer.

    `call` replaces the Claude call in tests.
    """
    call = call or _ask_claude
    answer = Answer(question=question.strip(), asked_at=_now())
    # The context is the same for every question on the same data, so it is cached on its
    # own; follow-up questions within a few minutes read it back at a fraction of the price.
    messages = [{"role": "user", "content": [
        {"type": "text", "text": build_context(data), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "## Buyer's question\n" + answer.question},
    ]}]

    plan = _plan(call, messages, answer)
    result, error = _try_run(plan, data)
    if error:
        answer.code_errors.append(f"Attempt 1: {error}")
        messages += [{"role": "assistant", "content": _json(plan)},
                     {"role": "user", "content": FIX_PROMPT.format(error=error)}]
        try:
            plan = _plan(call, messages, answer)
            result, error = _try_run(plan, data)
        except AnalystError as e:
            error = str(e)
        if error:
            answer.code_errors.append(f"Attempt 2: {error}")

    _apply_plan(answer, plan)
    if error:
        answer.error = FRIENDLY_FAIL
        return answer
    if result is _NO_CODE:
        answer.answer_type, answer.chart_spec = "text", None
        answer.text = answer.explanation or "The comparison data can't answer this question."
        return answer

    _set_result(answer, result)
    answer.caveats += review_caveats(answer.code, data.df)
    answer.excluded_vendors = excluded_vendors(plan.get("excluded_vendors"), answer.code, result, data.vendors)
    answer.price_risks = price_risks(result, data)
    answer.sensitivity = [r["sensitivity"] for r in answer.price_risks if r.get("sensitivity")]
    answer.facts = build_facts(answer, result)
    text = _summarise(call, answer)
    answer.text, notes = post_check(text, answer.price_risks, answer.excluded_vendors,
                                    money_values(result), list(data.vendors["display_name"]))
    answer.wording_notes += notes
    # Figures code worked out for the facts (e.g. the freight breakeven) count as checked.
    answer.unchecked_numbers = unsupported_numbers(answer.text, result,
                                                   answer.question + " " + _json(answer.facts))
    return answer


def _plan(call: Caller, messages: list[dict], answer: Answer) -> dict:
    parsed, usage = call(ANALYST_RULES, messages, AnalystPlan, PLAN_MAX_TOKENS, "analysis")
    answer.usages.append(usage)
    return parsed


def _try_run(plan: dict, data: AnalystData):
    """(result, None) on success, (_NO_CODE, None) if there was no code, (None, error) on failure."""
    code = (plan.get("pandas_code") or "").strip()
    if not code:
        return _NO_CODE, None
    try:
        return run_code(code, data.namespace()), None
    except CodeError as e:
        return None, str(e)


def _apply_plan(answer: Answer, plan: dict) -> None:
    answer.answer_type = plan.get("answer_type") or "text"
    answer.code = (plan.get("pandas_code") or "").strip()
    answer.chart_spec = plan.get("chart_spec")
    answer.explanation = plan.get("explanation") or ""
    answer.caveats = [c for c in plan.get("caveats") or [] if c and c.strip()]
    answer.data_sufficient = bool(plan.get("data_sufficient", True))
    answer.missing_data = [m for m in plan.get("missing_data") or [] if m and m.strip()]


def _set_result(answer: Answer, result) -> None:
    answer.result = result
    answer.table = result_frame(result)
    if answer.answer_type == "chart":
        problem = chart_problem(answer.table, answer.chart_spec)
        if problem:
            answer.caveats.append(f"Could not draw the chart ({problem}); showing the data instead.")
            answer.answer_type = "table"
    if answer.answer_type == "table" and answer.table is None:
        answer.answer_type = "text"
    if answer.answer_type != "chart":
        answer.chart_spec = None


def _summarise(call: Caller, answer: Answer) -> str:
    """The writer sees only the facts (formatted strings), never `result` or raw numbers."""
    content = "FACTS:\n" + json.dumps(answer.facts, ensure_ascii=False, indent=1)
    try:
        text = _writer_call(call, answer, content)
        minus = bare_minus_signs(text)
        if minus:  # one rewrite: direction in words, not signs
            text = _writer_call(call, answer, content + "\n\nYour previous answer was:\n" + text
                                + f"\n\nIt used minus signs ({', '.join(minus)}). Rewrite it with the "
                                  "direction in words ('saves ...' / 'costs ... more'), no minus signs.")
            minus = bare_minus_signs(text)
        if minus:
            answer.wording_notes.append("Minus sign still in the answer: " + ", ".join(minus))
    except AnalystError as e:
        answer.caveats.append(f"Could not write the summary sentences ({e}); showing the explanation instead.")
        return answer.explanation
    return text


def _writer_call(call: Caller, answer: Answer, content: str) -> str:
    parsed, usage = call(WRITER_RULES, [{"role": "user", "content": content}], Summary,
                         SUMMARY_MAX_TOKENS, "summary")
    answer.usages.append(usage)
    return parsed["answer"].strip()


def to_excel_bytes(answer: Answer) -> bytes:
    """The answer's table as an .xlsx, plus an 'About' sheet with the question, caveats and code.

    Money stays numeric (in rupees) so the buyer can work with it; headers carry the unit,
    e.g. 'annual_cost (₹)', 'price (₹/piece)'.
    """
    table = answer.table.copy() if answer.table is not None else pd.DataFrame()
    for col in table.columns:
        if table[col].dtype == object:
            table[col] = table[col].map(_cell_text)
    money_cols = [i for i, c in enumerate(table.columns, start=1) if money_kind(c)]
    table = table.rename(columns=excel_header)
    about = pd.DataFrame(
        [("Question", answer.question), ("Asked at", answer.asked_at), ("Answer", answer.text)]
        + [("Price risk", r["text"]) for r in answer.price_risks]
        + [("Left out", f"{e['display_name']} ({e['reason']})") for e in answer.excluded_vendors]
        + [("Caveat", c) for c in answer.caveats]
        + [("Missing data", m) for m in answer.missing_data]
        + [("Code", answer.code)],
        columns=["item", "value"],
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Answer", index=False)
        sheet = writer.sheets["Answer"]
        for col in money_cols:
            for row in range(2, len(table) + 2):
                sheet.cell(row=row, column=col).number_format = "#,##0.00"
        about.to_excel(writer, sheet_name="About", index=False)
        for sv in answer.sensitivity:
            name = re.sub(r"[\[\]:*?/\\]", "", f"Freight {sv['vendor']}")[:31]  # Excel sheet-name rules
            sv["table"].rename(columns=excel_header).to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def _cell_text(v):
    if isinstance(v, (list, tuple)):
        return "; ".join(_cell_text(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {x}" for k, x in v.items())
    return v


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
