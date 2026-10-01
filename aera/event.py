"""One RFx event: the RFx, last year's prices, and every extracted reply and certificate.

Extraction goes through extract.py, which reads the cache unless force=True.
One file failing never stops the others; its problem is kept in plain words.
The original file bytes are kept so the UI can show the source of every number.
No price maths here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from aera.compare import vendor_name
from aera.extract import ExtractionError, extract_certificate, extract_reply
from aera.ingest import load_reply_bytes
from aera.rfx import RFx, load_last_year_prices, load_rfx

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
REPLY, CERTIFICATE = "reply", "certificate"

Progress = Callable[[str], None]


class NotAQuote(Exception):
    """An uploaded file that doesn't look like a quote for this RFx. Nothing was added.

    The message is the plain-English reason. Upload it with allow_non_quote=True to add it anyway."""


@dataclass
class SourceFile:
    name: str
    kind: str  # excel | word | email | pdf | image
    role: str  # reply | certificate
    sha256: str
    data: bytes
    text: str | None  # extracted text for excel / word / email


@dataclass
class Event:
    rfx: RFx
    last_year: dict[int, float]
    replies: list[dict] = field(default_factory=list)
    certificates: list[dict] = field(default_factory=list)
    files: dict[str, SourceFile] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def texts(self) -> dict[str, str | None]:
        return {name: f.text for name, f in self.files.items()}

    @property
    def extraction_cost_usd(self) -> float:
        """What extracting every file cost last time: a guide to what re-extracting will cost."""
        return sum(e.get("cost_usd") or 0.0 for e in self.replies + self.certificates)


def load_sample_event(sample_dir: Path = SAMPLE_DIR, force: bool = False,
                      progress: Progress | None = None) -> Event:
    """Load rfx.json, last_year*.csv, replies/ and certificates/ from a folder.

    With force=False, files with a cached extraction make no API calls.
    """
    sample_dir = Path(sample_dir)
    event = Event(rfx=load_rfx(sample_dir / "rfx.json"), last_year={})
    for csv_path in sorted(sample_dir.glob("last_year*.csv")):
        event.last_year.update(load_last_year_prices(csv_path))

    for folder, role in (("replies", REPLY), ("certificates", CERTIFICATE)):
        folder_path = sample_dir / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(p for p in folder_path.iterdir() if p.is_file()):
            _load_into(event, path.name, path.read_bytes(), role, force, progress)
    return event


def reextract_all(event: Event, progress: Progress | None = None) -> Event:
    """Call Claude again for every file in the event, ignoring the cache.

    A file that fails keeps its earlier extraction, so a network problem never empties the event.
    """
    previous = {e.get("source_file"): e for e in event.replies + event.certificates}
    fresh = Event(rfx=event.rfx, last_year=dict(event.last_year))
    for f in event.files.values():
        _load_into(fresh, f.name, f.data, f.role, True, progress, fallback=previous.get(f.name))
    return fresh


def _has_answer(claim: dict | None, empty=(None, "", "unclear", "not_stated")) -> bool:
    return isinstance(claim, dict) and claim.get("value") not in empty


def quote_problem(ext: dict, rfx: RFx) -> str | None:
    """Why this extraction doesn't look like a quote for the RFx, or None if it does.

    Rejected when Claude's quote check says it isn't a quote, or when code finds nothing a
    quote would have: no RFx line priced or declined, no questionnaire answer, no commercial term.
    Older cached extractions have no quote_check; only the code check applies to them.
    """
    check = ext.get("quote_check") or {}
    if check.get("is_quote_for_rfx") is False:
        reason = (check.get("not_a_quote_reason") or "").strip().rstrip(".")
        return reason or "it doesn't offer prices or terms for the items in this RFx"

    runs = ext.get("runs") or [{}]

    line_ids = {ln.line_id for ln in rfx.lines}
    for run in runs:
        matched = any(q.get("rfx_line_id") in line_ids for q in run.get("lines") or [])
        declined = any(n.get("rfx_line_id") in line_ids and (n.get("source_snippet") or "").strip()
                       for n in run.get("not_quoted") or [])
        answers = any(_has_answer(c) for c in (run.get("questionnaire") or {}).values())
        terms = run.get("commercial_terms") or {}
        has_terms = (any(_has_answer(terms.get(k)) for k in ("freight", "payment_days", "validity"))
                     or bool(terms.get("discounts")) or bool(terms.get("other_conditions")))
        if matched or declined or answers or has_terms:
            return None
    return "no RFx lines, questionnaire answers or commercial terms were found in it"


def add_reply(event: Event, filename: str, data: bytes, force: bool = True,
              allow_non_quote: bool = False) -> tuple[bool, str | None]:
    """Extract one uploaded reply and add it to the event.

    Returns (added, note), where note is a plain-English message or None. A reply from a
    vendor already in the event replaces that vendor's earlier reply. Raises
    ExtractionError / ValueError with a readable message if the file can't be used, and
    NotAQuote (event unchanged) if it doesn't look like a quote, unless allow_non_quote=True.
    """
    payload = load_reply_bytes(filename, data)
    for ext in event.replies:
        if ext.get("sha256") == payload.sha256:
            return False, f"{filename} is already in the comparison ({vendor_name(ext)})."
    existing = event.files.get(filename)
    if existing is not None:
        raise ValueError(f"A different file named {filename} is already loaded. "
                         "Rename the file and upload it again.")

    ext = extract_reply(payload, event.rfx, force=force)
    problem = quote_problem(ext, event.rfx)
    if problem and not allow_non_quote:
        raise NotAQuote(problem)
    new_vendor = vendor_name(ext)
    replaced = [e for e in event.replies if vendor_name(e) == new_vendor]
    event.replies = [e for e in event.replies if vendor_name(e) != new_vendor] + [ext]
    event.files[filename] = SourceFile(filename, payload.kind, REPLY, payload.sha256, data, payload.text)
    for old in replaced:
        event.files.pop(old.get("source_file"), None)
    if replaced:
        old_files = ", ".join(e.get("source_file") or "?" for e in replaced)
        return True, f"Replaced the earlier reply from {new_vendor} ({old_files}) with {filename}."
    return True, None


def _load_into(event: Event, name: str, data: bytes, role: str, force: bool,
               progress: Progress | None, fallback: dict | None = None) -> None:
    """Read and extract one file into the event. On failure, use `fallback` if given."""
    if progress:
        progress(f"Reading {name}")
    bucket = event.replies if role == REPLY else event.certificates
    try:
        payload = load_reply_bytes(name, data)
        event.files[name] = SourceFile(name, payload.kind, role, payload.sha256, data, payload.text)
        if role == REPLY:
            bucket.append(extract_reply(payload, event.rfx, force=force))
        else:
            bucket.append(extract_certificate(payload, force=force))
        return
    except (ExtractionError, ValueError) as e:
        problem = f"{name}: {e}"
    except Exception as e:  # a bad file must not stop the rest of the event
        problem = f"{name}: could not be read ({type(e).__name__}: {e})"
    if fallback is not None:
        bucket.append(fallback)
        problem += " Kept the earlier extraction."
    event.errors.append(problem)
