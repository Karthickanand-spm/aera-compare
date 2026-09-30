# CLAUDE.md: Aera Compare

## What this is

Aera Compare turns messy vendor quotes into a trustworthy side-by-side comparison. A buyer sends a request for quotes; vendors reply in any format (Excel, PDF, Word, phone photo, plain email). The app reads every reply, makes the numbers comparable (or clearly marks them as not), lets the buyer ask questions in plain English, and ends on a Decide screen where a person confirms the award.

Concept prototype, not an Aerchain product. Show this line in the app footer: "Concept prototype. Not an Aerchain product."

The hard problem is not reading quotes. It's knowing whether two numbers can truly be compared, and asking the vendor when they can't.

## Hard rules (never break these)

1. **AI reads, code calculates.** Claude extracts values and proposes interpretations ("this price is per 100 pieces"). Plain Python does ALL arithmetic: unit conversion, currency, totals, savings, award splits. No money maths in a prompt.
2. **Every number has a receipt.** Every extracted value stores vendor, source file, and the exact source snippet (plus page if known). The UI shows it on click.
3. **Missing is never zero.** A line a vendor didn't quote is `None` with status "Not quoted". Never fill with 0, never average it in.
4. **Show doubt.** Confidence comes from checks code can run, not the model's self-rating. Show the reasons, not just a score.
5. **Assumptions are visible.** FX rate and its date, every unit conversion, box weight used for per-kg prices, how "freight extra" was treated.
6. **Humans own the award.** The app recommends. The buyer confirms. Low-confidence values need buyer confirmation before they count.
7. **No hardcoded answers.** Never special-case sample file names, vendor names, or demo questions. Extraction and analysis must work on a file or question nobody has seen.
8. **Secrets stay secret.** Read the key only from `st.secrets["ANTHROPIC_API_KEY"]`. Never print it, log it, or put it in code. Never edit `.gitignore` entries for `.streamlit/secrets.toml` or `.venv/`.

## Comparability labels

Every cell in the comparison gets exactly one:

| Label | When |
|---|---|
| Comparable | Same spec, same unit basis, INR |
| Comparable with assumption | Needed a conversion (unit, currency, per-kg via box weight). Store the assumption text. |
| Not comparable | Vendor quoted a different spec (e.g. 150 GSM vs 180 GSM). Show it, exclude from totals by default. |
| Not quoted | Vendor skipped the line |

## Confidence checks (code, not model)

- Source snippet actually appears in the extracted document text (not possible for photos, so photo values start lower)
- Vendor's own line total equals qty x unit price, where a total exists
- A unit or currency conversion was needed
- For photos: extract twice; if the two runs disagree, flag for review

Output per value: `confidence` (high / medium / low) and `confidence_reasons` (list of plain-English strings).

## Stack

- Python 3.12, Streamlit, anthropic SDK, pandas, openpyxl, python-docx, pytest
- Excel and Word are converted to text in code first. PDFs and images go to Claude directly.
- Use structured outputs (JSON schema) for extraction so responses are validated JSON.
- Model name lives in ONE place: `aera/config.py` -> `MODEL = "claude-sonnet-5-5"`. Same file holds `FX_USD_INR`, `FX_DATE`, `FX_SOURCE`.
- Claude API rules for this model: do NOT set `temperature`, `top_p`, or `top_k` (non-default values return a 400). Do NOT send `thinking: {"type": "disabled"}` (400); omit `thinking` instead. No assistant-message prefill. No forced `tool_choice` (`any`/`tool`); use structured outputs for JSON.
- Hosted on Streamlit Community Cloud. Dev machine is Windows; use Command Prompt commands in instructions.

## Suggested layout

```
app.py                  # Streamlit entry, page navigation
aera/
  config.py             # MODEL, FX rate + date, thresholds
  rfx.py                # RFx schema: line items, spec, qty, unit, nominal box weight, last-year price
  ingest.py             # file -> text or image payload
  extract.py            # Claude calls, returns validated JSON per vendor
  normalize.py          # PURE functions: units, FX, per-kg -> per-piece. No AI.
  compare.py            # builds comparison table, labels, confidence
  analyst.py            # plain-English questions -> pandas code -> answer/table/chart
  clarify.py            # drafts one clarification email per vendor
data/sample/            # rfx.json, last_year_prices.csv, replies/, certificates/
data/cache/             # cached extraction JSON keyed by file hash
tests/                  # pytest, especially normalize.py and compare.py
```

## Extraction output (per vendor, per RFx line)

`rfx_line_id, vendor, raw_price_text, price, currency, unit_basis (per_piece | per_100 | per_box_of_N | per_kg | other), quoted_spec, source_file, source_snippet, page, notes`

Code then adds: `price_inr_per_piece, label, assumptions[], confidence, confidence_reasons[]`.

## Analyst rules

- Claude writes pandas code over a read-only copy of the comparison DataFrame. The UI shows the code under every answer.
- Run generated code in a restricted namespace: only `df`, `pd`, and a chart helper. No imports, no file or network access. Catch errors and show them.
- If the data can't support an answer, say so and say what's missing (e.g. "freight not quoted, E stays cheapest unless freight is above X"). Offer to draft a clarification.

## Caching

Cache extraction results by file hash in `data/cache/`. The UI has a visible "Re-extract" button. Uploading a new file always runs extraction live.

## How to work with me

I'm new to building live apps. So:
- Make one small change at a time. Tell me in plain words what you changed and why.
- After each change, tell me exactly what to run to see it (e.g. `streamlit run app.py`) and what I should see.
- Write pytest tests for anything in `normalize.py` and `compare.py`, and run them.
- Ask before big refactors, new dependencies, or deleting files. If you add a dependency, update `requirements.txt`.
- Suggest a short commit message when a change works.