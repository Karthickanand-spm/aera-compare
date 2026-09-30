"""Pure price maths: currency and unit conversion. No AI, no Streamlit.

Every function returns the converted value AND a plain-English assumption
string (or None when nothing was converted), so the UI can show its working.
"""

UNIT_BASES = ("per_piece", "per_100", "per_1000", "per_pack", "per_kg")


class NormalizeError(ValueError):
    """Raised when a price cannot be converted without guessing."""


def to_inr(value: float, currency: str, fx_rates: dict[str, float],
           fx_date: str | None = None) -> tuple[float, str | None]:
    """Convert value to INR. Returns (inr_value, assumption_text or None)."""
    code = (currency or "").strip().upper()
    if code == "INR":
        return value, None
    if code not in fx_rates:
        known = ", ".join(["INR", *sorted(fx_rates)])
        raise NormalizeError(
            f"No FX rate for currency '{currency}'. Known currencies: {known}."
        )
    rate = fx_rates[code]
    date_text = f" (rate dated {fx_date})" if fx_date else ""
    return value * rate, f"Converted {code} to INR at {rate:g} INR per {code}{date_text}"


def to_per_piece(value: float, unit_basis: str, pack_size: int | None = None,
                 weight_g: float | None = None) -> tuple[float, str | None]:
    """Convert a price on any unit basis to a per-piece price.

    Returns (value_per_piece, assumption_text or None).
    """
    if unit_basis == "per_piece":
        return value, None
    if unit_basis == "per_100":
        return value / 100, "Quoted per 100 pieces; divided by 100"
    if unit_basis == "per_1000":
        return value / 1000, "Quoted per 1000 pieces; divided by 1000"
    if unit_basis == "per_pack":
        if not pack_size or pack_size <= 0:
            raise NormalizeError("Price is per pack but the pack size is missing.")
        return value / pack_size, f"Quoted per pack of {pack_size}; divided by {pack_size}"
    if unit_basis == "per_kg":
        if not weight_g or weight_g <= 0:
            raise NormalizeError("Price is per kg but the box weight is missing.")
        return (value * weight_g / 1000,
                f"Quoted per kg; multiplied by box weight {weight_g:g} g")
    raise NormalizeError(
        f"Unknown unit basis '{unit_basis}'. Expected one of: {', '.join(UNIT_BASES)}."
    )


def normalize_price(value: float, currency: str, unit_basis: str,
                    pack_size: int | None, weight_g: float | None,
                    fx_rates: dict[str, float],
                    fx_date: str | None = None) -> tuple[float, list[str]]:
    """Convert any quoted price to INR per piece.

    Returns (inr_per_piece, assumptions). Assumptions is empty when the price
    was already INR per piece.
    """
    if value is None:
        raise NormalizeError("No price to normalize (missing is never zero).")
    inr_value, fx_note = to_inr(value, currency, fx_rates, fx_date)
    per_piece, unit_note = to_per_piece(inr_value, unit_basis, pack_size, weight_g)
    assumptions = [note for note in (fx_note, unit_note) if note]
    return per_piece, assumptions
