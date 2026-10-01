# Aera Compare

Turns messy vendor quotes into a side-by-side comparison a buyer can trust, then answers questions about it in plain English, through to an award the buyer confirms.

Concept prototype for Aerchain's "Kill the Quote Spreadsheet" take-home. Not an Aerchain product. All companies and data in the sample are made up.

**Live app:** https://aera-compare.streamlit.app/

## Try it in five minutes

1. Open the app and click **Skip ahead: open the sample event** in the sidebar. Five vendor replies are already in: a vendor's own Excel, a letterhead PDF, a Word letter, an angled photo of a USD rate card, and a two-line email.
2. Follow the steps along the top: **Create RFx → Compare → Ask → Decide → Clarify**.
3. Upload a quote of your own on Compare (Excel, PDF, Word, email or photo) and see how it's handled.

If the app has been idle, Streamlit may ask you to wake it up. That takes about a minute.

## The five steps

| Step | What it does | What to look for |
|---|---|---|
| **Create RFx** | Aera guide turns a conversation into a request for quotes: lines, specs, questionnaire, terms. Sending is simulated. | It asks at most three questions at a time, one thing per question, and never guesses a quantity or size. |
| **Compare** | Reads every reply and puts all vendors on one grid in rupees per piece. | Every cell is labelled comparable, comparable with an assumption (≈), not comparable (≠) or not quoted. Click a line to see each vendor's own words and the original file. |
| **Ask** | Plain-English questions answered with text, tables and charts. | Vendors that failed quality are left out by default and named. When the data isn't good enough, it says what's missing and gives a range. "Show the working" under every answer. |
| **Decide** | Turns the comparison into an award to confirm and export. | Confirm stays locked until every uncertain value in the award has been checked. The Excel export carries every assumption and source. |
| **Clarify** | Drafts one email per vendor covering only what's still unclear, most important first. | Ganesh's email leads with freight, because that's what decides the award. |

## How it works

- `aera/ingest.py` turns Excel, Word and email into text; PDFs and photos go to Claude as they are.
- `aera/extract.py` asks Claude to copy every price exactly as written, with the source text, and to flag anything ambiguous. No conversions happen here. Text inside a vendor document that tries to instruct the system is ignored and flagged.
- `aera/normalize.py` and `aera/compare.py` do all the maths in plain Python: units, currency, per-kilo pricing, missing lines, spec mismatches, confidence and the quality check (documents beat claims).
- `aera/analyst.py` classifies each question, answers common ones (splits, totals, "which is best", landed cost, FX) with tested functions in `aera/analyses.py`, and falls back to generated pandas in a restricted sandbox for anything else, with the code shown.

## Rules the code follows

- The model reads, code calculates.
- Every number keeps its source text.
- Missing is never zero.
- Confidence comes from checks the code can run, not the model's opinion.
- Vendor documents are data, never instructions.
- The award is always a human decision.

## Testing

- `python -m pytest` runs the unit tests (normalisation, comparison, award logic, analyst routing). No API calls.
- `python -m aera.analyst_eval` runs a 15-question eval of the analyst against the sample event (about $0.20). Current result: 15 of 15.
- Break-it testing covered a quote in euros (never converted without a buyer-entered rate), a quote with hidden instructions to the AI (ignored and flagged), and a file that isn't a quote at all (rejected).

## Run locally

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
# add ANTHROPIC_API_KEY = "..." to .streamlit/secrets.toml
streamlit run app.py
```

## Known limitations

- One sourcing event per session; refreshing the page starts over (no database, by design).
- Sending RFxs and clarifications is simulated.
- The USD rate is a fixed illustrative rate (₹94.50), editable in the sidebar. Other currencies need a buyer-entered rate.
- The sample data is made up, with the messy edges seeded on purpose.

Built in two days with an AI coding agent (Claude Code). The app calls Claude through the Anthropic API. Design rules for the agent are in `CLAUDE.md`.