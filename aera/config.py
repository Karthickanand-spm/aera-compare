"""All settings in one place: model name, FX rates, confidence thresholds."""

# Claude model used for extraction and the analyst. Change it only here.
MODEL = "claude-sonnet-5-5"

# FX: how many INR one unit of each currency buys. INR itself is implied (1.0).
FX_RATES = {"USD": 94.50}
FX_DATE = "2026-09-25"
FX_SOURCE = "Illustrative rate"

# Confidence thresholds (used by code checks in compare.py, not by the model).
# A vendor's own line total "matches" qty x unit price if within this tolerance.
LINE_TOTAL_TOLERANCE_PCT = 0.5
# Two photo extraction runs "agree" if their prices differ by at most this much.
PHOTO_RERUN_TOLERANCE_PCT = 0.5
# Confidence labels, from best to worst.
CONFIDENCE_LEVELS = ("high", "medium", "low")
