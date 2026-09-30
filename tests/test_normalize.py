import pytest

from aera.config import FX_RATES
from aera.normalize import NormalizeError, normalize_price, to_inr, to_per_piece

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
