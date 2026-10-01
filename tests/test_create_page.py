"""Create RFx page: chat helpers that don't need the API."""

from ui.create_page import OPEN_QUESTIONS_HINT, claim_submission, open_questions, placeholder
from aera.rfx_builder import empty_draft


def test_submission_id_is_processed_only_once():
    processed = set()
    assert claim_submission(processed, "turn-2") is True
    assert claim_submission(processed, "turn-2") is False
    assert claim_submission(processed, "turn-4") is True


def test_open_questions_only_until_the_buyer_replies():
    q = {"id": "Q1", "text": "How many?", "example": "60,000", "field": "line 1 annual_qty",
         "allow_vendors_propose": False}
    history = [{"role": "user", "content": "cartons"}, {"role": "assistant", "content": "x", "questions": [q]}]
    assert open_questions(history) == [q]
    assert open_questions(history + [{"role": "user", "content": "Q1 (line 1 annual_qty): 60,000"}]) == []
    assert placeholder(empty_draft(), questions_open=True) == OPEN_QUESTIONS_HINT
