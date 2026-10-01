"""Live evaluation of the analyst on the sample event: python -m aera.analyst_eval

Asks every question in tests/analyst_eval.yaml through the real pipeline (classify, compute,
write, validate), checks each answer's properties and prints a pass/fail table with the API cost.

This calls the Claude API and costs money, so it refuses to run under pytest.
  python -m aera.analyst_eval              all questions
  python -m aera.analyst_eval --only 6,7   just these ids
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

from aera.analyst import AnalystError, Answer, analyst_data, ask, mentions
from aera.compare import compare
from aera.config import FX_DATE, FX_RATES, FX_SOURCE
from aera.event import load_sample_event

EVAL_FILE = Path(__file__).resolve().parent.parent / "tests" / "analyst_eval.yaml"

MONEY = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)\s*(crore|lakh)?", re.IGNORECASE)
UNIT = {"crore": 1e7, "lakh": 1e5, "": 1.0}
LINE_RANGE = re.compile(r"\b(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\b")
FILE_PATH = re.compile(r"[\w-]+\.(?:py|json|csv|xlsx?|pdf|docx?|eml|jpe?g|png|toml|ya?ml|txt)\b|[A-Za-z]:\\|(?:^|\s)/\w+/",
                       re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹])")


def load_cases(path: Path = EVAL_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sample_data():
    """The sample event as the Ask page sees it: default FX rate, no buyer decisions yet."""
    event = load_sample_event()
    sources = {code: f"rate dated {FX_DATE}, {FX_SOURCE.lower()}" for code in FX_RATES}
    comparison, summary = compare(event.rfx, event.last_year, event.replies, event.certificates,
                                  FX_RATES, sources, event.texts, {})
    return analyst_data(comparison, summary, event.rfx, event.last_year, FX_RATES, sources)


# ---------- Checks: each returns a problem in plain words, or None ----------

def money_amounts(text: str) -> list[float]:
    """Every ₹ amount in the text, in rupees ('₹3.64 crore' -> 36400000)."""
    out = []
    for number, unit in MONEY.findall(text):
        out.append(float(number.replace(",", "")) * UNIT[(unit or "").lower()])
    return out


def lines_in(text: str) -> set[int]:
    """Line numbers written out, with ranges ('25 to 27') expanded."""
    found = {int(n) for n in re.findall(r"\b\d{1,3}\b", text)}
    for a, b in LINE_RANGE.findall(text):
        lo, hi = int(a), int(b)
        if lo < hi <= lo + 50:
            found.update(range(lo, hi + 1))
    return found


def first_sentence(text: str) -> str:
    return SENTENCE_END.split(text.strip(), maxsplit=1)[0]


def run_checks(checks: dict, answer: Answer, vendor_names: list[str]) -> list[str]:
    text = answer.text or ""
    low = text.lower()
    problems = []

    def fail(msg: str) -> None:
        problems.append(msg)

    if "intent" in checks:
        want = checks["intent"] if isinstance(checks["intent"], list) else [checks["intent"]]
        if answer.intent not in want:
            fail(f"intent {answer.intent}, wanted {'/'.join(want)}")
    if (t := checks.get("tag_contains")) and t.lower() not in answer.tag.lower():
        fail(f"tag '{answer.tag}' lacks '{t}'")
    for s in checks.get("contains_all", []):
        if s.lower() not in low:
            fail(f"missing '{s}'")
    if (any_of := checks.get("contains_any")) and not any(s.lower() in low for s in any_of):
        fail(f"none of {any_of}")
    for s in checks.get("not_contains", []):
        if s.lower() in low:
            fail(f"contains '{s}'")
    for r in checks.get("regex_all", []):
        if not re.search(r, text, re.IGNORECASE):
            fail(f"no match /{r}/")
    if (any_re := checks.get("regex_any")) and not any(re.search(r, text, re.IGNORECASE) for r in any_re):
        fail(f"no match for any of {any_re}")
    for r in checks.get("not_regex", []):
        if re.search(r, text, re.IGNORECASE):
            fail(f"matches /{r}/")
    if words := checks.get("first_sentence_contains"):
        first = first_sentence(text).lower()
        missing = [w for w in words if w.lower() not in first]
        if missing:
            fail(f"first sentence lacks {missing}")
    if m := checks.get("money_near"):
        if not any(abs(v - m["inr"]) <= m["tol_inr"] for v in money_amounts(text)):
            fail(f"no ₹ amount within ₹{m['tol_inr']:,.0f} of ₹{m['inr']:,.0f}")
    if want_lines := checks.get("lines_mentioned"):
        missing = [n for n in want_lines if n not in lines_in(text)]
        if missing:
            fail(f"lines not named: {missing}")
    if n := checks.get("min_vendors_named"):
        named = [v for v in vendor_names if mentions(text, v, vendor_names)]
        if len(named) < n:
            fail(f"names {len(named)} vendor(s), wanted at least {n}")
    if checks.get("has_chart") and not (answer.answer_type == "chart" and answer.chart_spec):
        fail("no chart")
    if checks.get("has_freight_table") and not answer.sensitivity:
        fail("no freight table")
    if checks.get("no_file_paths") and (hit := FILE_PATH.search(text)):
        fail(f"shows a file name or path ('{hit.group().strip()}')")
    if answer.error:
        fail(f"error: {answer.error}")
    return problems


# ---------- Runner ----------

def run_case(case: dict, data, vendor_names: list[str]) -> dict:
    try:
        answer = ask(case["question"], data)
    except AnalystError as e:
        return {"id": case["id"], "intent": "-", "passed": False, "cost": 0.0,
                "problems": [f"AnalystError: {e}"], "answer": None}
    problems = run_checks(case.get("checks") or {}, answer, vendor_names)
    return {"id": case["id"], "intent": answer.intent, "passed": not problems,
            "cost": answer.cost_usd, "problems": problems, "answer": answer}


def print_table(results: list[dict]) -> None:
    print()
    print(f"{'#':>3}  {'result':<6}  {'intent':<15}  {'cost':>8}  failed checks")
    print("-" * 100)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:>3}  {status:<6}  {r['intent']:<15}  ${r['cost']:>7.4f}  {'; '.join(r['problems'])}")
    print("-" * 100)
    passed = sum(r["passed"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    print(f"{passed}/{len(results)} passed. Total API cost: ${total_cost:.4f}")


def print_failures(results: list[dict], cases: dict) -> None:
    for r in results:
        if r["passed"] or r["answer"] is None:
            continue
        a: Answer = r["answer"]
        print(f"\n--- #{r['id']}: {cases[r['id']]['question']}")
        print(f"Answered as: {a.tag}   classification: {a.classification}")
        print(f"Answer: {a.text}")
        for note in a.wording_notes:
            print(f"Validator: {note}")


def main(argv: list[str] | None = None) -> int:
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        raise SystemExit("analyst_eval calls the Claude API and costs money; it never runs under pytest.")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated question ids, e.g. 6,7,15")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ on the Windows console

    cases = load_cases()
    if args.only:
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    data = sample_data()
    vendor_names = [str(n) for n in data.vendors["display_name"]]

    results = []
    for case in cases:
        print(f"Asking #{case['id']}: {case['question']}", flush=True)
        results.append(run_case(case, data, vendor_names))
    print_table(results)
    print_failures(results, {c["id"]: c for c in cases})
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
