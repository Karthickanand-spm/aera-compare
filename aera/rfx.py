"""Load the RFx (what the buyer asked for) and last year's contract prices."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RfxLine:
    line_id: int
    description: str
    ply: str
    board_spec: str
    size: str
    print: str
    annual_qty: int
    uom: str
    nominal_weight_g: float


@dataclass(frozen=True)
class RFx:
    rfx_id: str
    buyer: str
    title: str
    issued: str
    due: str
    terms: dict
    quality_bar: dict
    questionnaire: list[str]
    lines: list[RfxLine]

    def line(self, line_id: int) -> RfxLine:
        for ln in self.lines:
            if ln.line_id == line_id:
                return ln
        raise KeyError(f"RFx has no line {line_id}")


def load_rfx(path: str | Path) -> RFx:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = [
        RfxLine(
            line_id=int(ln["line_id"]),
            description=ln["description"],
            ply=ln["ply"],
            board_spec=ln["board_spec"],
            size=ln["size"],
            print=ln["print"],
            annual_qty=int(ln["annual_qty"]),
            uom=ln["uom"],
            nominal_weight_g=float(ln["nominal_weight_g"]),
        )
        for ln in data["lines"]
    ]
    return RFx(
        rfx_id=data["rfx_id"],
        buyer=data["buyer"],
        title=data["title"],
        issued=data["issued"],
        due=data["due"],
        terms=data.get("terms", {}),
        quality_bar=data.get("quality_bar", {}),
        questionnaire=data.get("questionnaire", []),
        lines=lines,
    )


def load_last_year_prices(path: str | Path) -> dict[int, float]:
    """Read a CSV with columns line_id, price_inr_per_piece into {line_id: price}.

    Blank prices are skipped (missing is never zero).
    """
    prices: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("price_inr_per_piece") or "").strip()
            if raw:
                prices[int(row["line_id"])] = float(raw)
    return prices
