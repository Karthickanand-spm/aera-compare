import pytest

from aera.config import FX_RATES
from aera.normalize import (
    MissingFxRate, NormalizeError, describe_rates, normalize_price, to_inr, to_per_piece,
)

FX = {"USD": 94.50}


def test_per_100_pieces_inr():
    value, assumptions = normalize_price(514, "INR", "per_100", None, None, FX)
    assert value == pytest.approx(5.14)
    assert assumptions == ["Quoted per 100 pieces; divided by 100"]


def test_per_pack_of_25_in_usd():
    value, assumptions = normalize_price(0.86, "USD", "per_pack", 25, None, FX)
    assert round(value, 2) == 3.25
    assert len(assumptions) == 2
    assert "94.5" in assumptions[0]
    assert "pack of 25" in assumptions[1]


def test_per_kg_uses_box_weight():
    value, assumptions = normalize_price(42, "INR", "per_kg", None, 852, FX)
    assert round(value, 2) == 35.78
    assert "852 g" in assumptions[0]


def test_inr_per_piece_needs_no_assumption():
    value, assumptions = normalize_price(36.65, "INR", "per_piece", None, None, FX)
    assert value == 36.65
    assert assumptions == []


def test_per_1000():
    value, _ = to_per_piece(5140, "per_1000")
    assert value == pytest.approx(5.14)


def test_unknown_currency_raises():
    with pytest.raises(NormalizeError, match="EUR"):
        to_inr(10, "EUR", FX)


def test_missing_fx_rate_names_the_currency():
    with pytest.raises(MissingFxRate) as e:
        to_inr(395, " eur ", FX)
    assert e.value.currency == "EUR"
    assert str(e.value) == "No FX rate for EUR"


def test_eur_per_1000_at_buyer_entered_rate():
    """EUR 395 per 1000 at ₹101/EUR = ₹39.90 per piece."""
    rates = {**FX, "EUR": 101.0}
    sources = {"EUR": "buyer-entered on 2026-10-01"}
    value, assumptions = normalize_price(395, "EUR", "per_1000", None, None, rates, sources)
    assert value == pytest.approx(39.895)
    assert f"{value:.2f}" == "39.90"
    assert assumptions == ["EUR at ₹101 per EUR, buyer-entered on 2026-10-01",
                           "Quoted per 1000 pieces; divided by 1000"]


def test_one_date_for_all_rates_still_works():
    _, note = to_inr(1, "USD", {"USD": 94.5}, "2026-09-25")
    assert note == "USD at ₹94.5 per USD, rate dated 2026-09-25"


def test_describe_rates_lists_each_source():
    text = describe_rates({"USD": 94.5, "EUR": 101.0},
                          {"USD": "rate dated 2026-09-25", "EUR": "buyer-entered on 2026-10-01"})
    assert text == "1 USD = ₹94.5 (rate dated 2026-09-25); 1 EUR = ₹101 (buyer-entered on 2026-10-01)"


def test_missing_pack_size_raises():
    with pytest.raises(NormalizeError, match="pack size"):
        to_per_piece(21.5, "per_pack", pack_size=None)


def test_missing_weight_raises():
    with pytest.raises(NormalizeError, match="weight"):
        to_per_piece(42, "per_kg", weight_g=None)


def test_unknown_unit_basis_raises():
    with pytest.raises(NormalizeError, match="per_dozen"):
        to_per_piece(10, "per_dozen")


def test_missing_price_raises():
    with pytest.raises(NormalizeError):
        normalize_price(None, "INR", "per_piece", None, None, FX)


def test_config_usd_rate():
    assert FX_RATES["USD"] == 94.50
