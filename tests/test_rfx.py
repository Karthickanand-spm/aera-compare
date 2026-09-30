from pathlib import Path

from aera.rfx import load_last_year_prices, load_rfx

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_load_rfx_lines():
    rfx = load_rfx(SAMPLE / "rfx.json")
    assert len(rfx.lines) > 0
    ids = [ln.line_id for ln in rfx.lines]
    assert len(ids) == len(set(ids))
    for ln in rfx.lines:
        assert ln.annual_qty > 0
        assert ln.nominal_weight_g > 0


def test_load_last_year_prices_matches_rfx_lines():
    rfx = load_rfx(SAMPLE / "rfx.json")
    prices = load_last_year_prices(SAMPLE / "last_year_contract_vendor_E.csv")
    assert set(prices) <= {ln.line_id for ln in rfx.lines}
    assert all(p > 0 for p in prices.values())
