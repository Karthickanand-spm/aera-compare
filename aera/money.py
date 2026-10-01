"""Money and number formatting for the buyer: ₹ in Indian units, savings in words.

Pure functions, no AI. Every ₹ figure a buyer reads is formatted here, never by the model.
"""

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

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


def plain_number(v) -> str:
    """Non-money numbers for the writer: Indian grouping, at most 2 decimals."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return MISSING_TEXT
    places = 0 if f == int(f) else 2
    d = _round_half_up(abs(f), places)
    return ("-" if f < 0 and d != 0 else "") + _grouped(d, places)


def pct_text(v) -> str:
    f = float(v)
    return MISSING_TEXT if math.isnan(f) else f"{_round_half_up(f, 1)}%"


# ---------- Reading numbers back out of text (for the answer checks) ----------

# A minus sign on money or a percentage: "-₹4.31 lakh", "−2.1%", "- 3%".
_BARE_MINUS = re.compile(r"(?<![\w])[-−–]\s?(?:₹\s?\d|\d[\d,]*(?:\.\d+)?\s?%)")
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def bare_minus_signs(text: str) -> list[str]:
    """Money or percentages written with a minus sign instead of words."""
    return [m.group() for m in _BARE_MINUS.finditer(text or "")]


def numbers_in_text(text: str) -> list[str]:
    return [m.group().rstrip(",") for m in NUMBER.finditer(text or "")]


def to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None
