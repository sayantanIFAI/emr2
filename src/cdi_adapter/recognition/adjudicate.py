"""Qwen adjudication of an engine disagreement (cascade L8) - ADVISORY ONLY.

Runs only after two independent readings of a line have disagreed. Qwen is then
shown the crop and BOTH recorded readings and asked which one the ink matches exactly.
This call is deliberately anchored (it sees the candidates), so it is never evidence of
agreement and it never changes the line's state: a disagreement still goes to review.
Its only effect is ordering - the reviewer (and S4) see the preferred reading first -
plus a recorded observation explaining why.

Position bias: the options are asked in both orders (A/B then B/A). A preference counts
only when both orders pick the same reading; otherwise the verdict is ``inconsistent``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..logging import get_logger

log = get_logger(__name__)

ADJUDICATION_PROMPT = (
    "This image crop is one line from a medical prescription. Two transcriptions were "
    "produced independently:\n"
    "A: {a}\n"
    "B: {b}\n"
    "Look only at the ink. Which transcription matches the image EXACTLY, including every "
    "number, unit and dosing notation? Answer with exactly one word: A, B, or NEITHER."
)
ENGINE = "qwen2.5-vl-adjudicator"
PREFERS, INCONSISTENT, NEITHER, FAILED = "prefers", "inconsistent", "neither", "failed"


@dataclass
class Adjudication:
    verdict: str                          # prefers | inconsistent | neither | failed
    preferred_index: int | None           # index into the disagreeing readings
    preferred_text: str | None
    answers: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "preferred_index": self.preferred_index,
                "preferred_text": self.preferred_text, "answers": self.answers,
                "advisory": True, "error": self.error}


def prompt_hash() -> str:
    return hashlib.sha256(ADJUDICATION_PROMPT.encode()).hexdigest()[:16]


def _parse(answer: str) -> str:
    m = re.search(r"\b(A|B|NEITHER)\b", (answer or "").upper())
    return m.group(1) if m else "UNPARSED"


def adjudicate(crop_png: bytes, texts: list[str], client: Any | None = None) -> Adjudication:
    """``texts`` = the two disagreeing readings, in engine order."""
    if len(texts) != 2:
        return Adjudication(FAILED, None, None, error="needs exactly two readings")
    if client is None:
        from ..ml.client import get_client

        client = get_client()
    answers: list[str] = []
    picks: list[int | None] = []
    try:
        for order in ((0, 1), (1, 0)):
            q = ADJUDICATION_PROMPT.format(a=texts[order[0]], b=texts[order[1]])
            ans = _parse(client.vlm_generate(crop_png, q,
                                             max_tokens=settings.qwen_adjudication_max_tokens))
            answers.append(ans)
            picks.append(order[0] if ans == "A" else order[1] if ans == "B"
                         else -1 if ans == "NEITHER" else None)
    except Exception as exc:  # noqa: BLE001 - advisory step: failure changes nothing
        log.warning("adjudication_failed", error=str(exc)[:200])
        return Adjudication(FAILED, None, None, answers, error=str(exc)[:200])
    if picks[0] == -1 and picks[1] == -1:
        return Adjudication(NEITHER, None, None, answers)
    if picks[0] is not None and picks[0] >= 0 and picks[0] == picks[1]:
        return Adjudication(PREFERS, picks[0], texts[picks[0]], answers)
    return Adjudication(INCONSISTENT, None, None, answers)
