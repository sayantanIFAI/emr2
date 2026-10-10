"""Disagreement engine (E2-S11): reader vs reader, OCR vs terminology, candidate vs grounding,
cross-field. Any MATERIAL disagreement forces review regardless of model confidence.

"Material" = after numeric-context normalisation (O->0 inside numbers) the numbers differ,
OR the text similarity is below ``engine_agree_similarity``.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .engines import Reading
from .grammar import normalize_numeric_context, numbers_in

AGREE, DISAGREE, SINGLE, NONE = "agree", "disagree", "single_engine", "no_reading"


def _canon(text: str) -> str:
    t = normalize_numeric_context(text)
    t = re.sub(r"[^\w½%/.\- ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    ca, cb = _canon(a), _canon(b)
    if not ca and not cb:
        return 1.0
    return difflib.SequenceMatcher(None, ca, cb).ratio()


def compare_texts(a: str, b: str, threshold: float | None = None) -> tuple[bool, dict[str, Any]]:
    thr = settings.engine_agree_similarity if threshold is None else threshold
    na, nb = numbers_in(a), numbers_in(b)
    sim = similarity(a, b)
    numbers_equal = na == nb
    agree = numbers_equal and sim >= thr
    return agree, {"similarity": round(sim, 3), "numbers_a": na, "numbers_b": nb,
                   "numbers_equal": numbers_equal}


@dataclass
class EngineVerdict:
    state: str                                   # agree | disagree | single_engine | no_reading
    display_text: str                            # what S4 sees in ocr_block.text
    detail: dict[str, Any] = field(default_factory=dict)


def compare_engines(readings: list[Reading]) -> EngineVerdict:
    """Compare independent handwriting readings of ONE crop.

    ``display_text`` is only the text handed to extraction; governance uses ``state``.
    With disagreement both readings are kept side by side so the reviewer (and S4) sees both -
    neither is silently chosen."""
    ok = [r for r in readings if r.ok and r.text.strip()]
    if not ok:
        return EngineVerdict(NONE, "", {"errors": [r.error for r in readings if r.error]})
    if len(ok) == 1:
        r = ok[0]
        return EngineVerdict(SINGLE, r.text, {"engine": r.engine,
                             "missing": [x.engine for x in readings if x is not r]})
    a, b = ok[0], ok[1]
    agree, det = compare_texts(a.text, b.text)
    det.update({"engines": [a.engine, b.engine], "texts": [a.text, b.text]})
    if agree:
        return EngineVerdict(AGREE, a.text, det)
    return EngineVerdict(DISAGREE, f"{a.text} ⟂ {b.text}", det)


def self_consistency(verdict: EngineVerdict, first: Reading, second: Reading) -> EngineVerdict:
    """RD-S3 self-consistency: the same engine read the same crop twice with different padding.

    Two readings that differ materially mean the reading is unstable, which counts as another
    disagreement: an ``agree`` verdict becomes ``disagree`` (a person looks), and the detail says
    why. Anything else is left as it was, a failed second read changes nothing, and neither reading
    is ever edited."""
    if not (first.ok and second.ok and first.text.strip() and second.text.strip()):
        return verdict
    same, det = compare_texts(first.text, second.text)
    info = {"consistent": same, "texts": [first.text, second.text], **{k: det[k] for k in ("similarity", "numbers_equal")}}
    detail = {**verdict.detail, "self_consistency": info}
    if same or verdict.state != AGREE:
        return EngineVerdict(verdict.state, verdict.display_text, detail)
    return EngineVerdict(DISAGREE, verdict.display_text, detail)


def ocr_vs_terminology(reading: str, concept_display: str | None,
                       aliases: list[str] | None = None) -> tuple[bool, float]:
    """Does the bound concept plausibly match what was read? (number-exact, text-similar)"""
    if not concept_display:
        return True, 1.0
    cands = [concept_display] + list(aliases or [])
    best = max(similarity(reading, c) for c in cands)
    nums_ok = any(numbers_in(reading) == numbers_in(c) or not numbers_in(c) for c in cands)
    return (best >= 0.6 and nums_ok), round(best, 3)


def strength_fits_product(strength: float | None, attrs: dict[str, Any] | None) -> bool | None:
    """Cross-field: is this strength one the matched product is made in? None = unknown."""
    if strength is None or not attrs:
        return None
    avail = attrs.get("strengths_available")
    if not avail:
        return None
    return any(abs(float(strength) - float(s)) < 1e-6 for s in avail)
