"""A liver-enzyme pair written with a slash ("SGPT/SGOT") where the line reader could read only one of the two ("?APT/SGOT").

The unreadable half is never filled in by a rule. The picture of the line is shown to the model several times, at several sizes and in several
orders, and asked which of a SHORT closed list is written before the slash: the pair's other half and the standard tests closest to what was
read, or none of these. The pair's other half is listed only when what was read is at least half alike it. It is accepted only when most of the
readings pick it; it is then listed as a test the page's own text corroborates (the neighbour it is paired with), and stays "to check".
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)

_MATES = {"sgot": "SGPT", "sgpt": "SGOT", "ast": "ALT", "alt": "AST", "got": "GPT", "gpt": "GOT"}
_LEFT = re.compile(r"(?<![A-Za-z])([?A-Za-z]{2,8})\s*/\s*(sgot|sgpt|ast|alt|got|gpt)(?![A-Za-z])", re.I)
_RIGHT = re.compile(r"(?<![A-Za-z])(sgot|sgpt|ast|alt|got|gpt)\s*/\s*([?A-Za-z]{2,8})(?![A-Za-z])", re.I)
MIN_ALIKE = 0.5
SCALES = (1.6, 2.4, 3.0)
SHUFFLES = 3
MIN_VOTES = 5            # of 9 readings


def slips(blocks: list[dict[str, Any]] | None, have: set[str]) -> list[tuple[dict[str, Any], str, str, str]]:
    """``(block, as read, the written partner, the test the pair's other half would be)`` for every pair whose second half the reader could not
    place and that is at least half alike the pair's other half. ``have`` is the names already listed (letters and digits only, lower case)."""
    from .test_cluster import placed_text

    out: list[tuple[dict[str, Any], str, str, str]] = []
    for b in blocks or []:
        text = str(b.get("text") or "")
        for m in _LEFT.finditer(text):
            out.append((b, m.group(1), m.group(2), _MATES[m.group(2).casefold()]))
        for m in _RIGHT.finditer(text):
            out.append((b, m.group(2), m.group(1), _MATES[m.group(1).casefold()]))
    keep = []
    seen: set[str] = set()
    for b, read, partner, mate in out:
        letters = re.sub(r"[^a-z]", "", read.casefold())
        if len(letters) < 2 or mate.casefold() in have or mate in seen:
            continue
        if read.casefold() in _MATES or (placed_text(read) and "?" not in read):
            continue                                     # a test the lists place (also the partner itself): nothing to read
        if difflib.SequenceMatcher(None, letters, mate.casefold()).ratio() < MIN_ALIKE:
            continue
        seen.add(mate)
        keep.append((b, read, partner, mate))
    return keep


def _crop(image: bytes, box: Any, scale: float) -> bytes | None:
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or not box or len(box) < 4:
        return None
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    px, py = max(20, int(0.02 * w)), max(12, int(0.01 * h))
    c = arr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
    if c.size == 0:
        return None
    c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", c)
    return png.tobytes() if ok else None


def resolve(client: Any, image: bytes, blocks: list[dict[str, Any]] | None, have: set[str]) -> dict[str, str]:
    """``{test: note}`` for each pair half the picture shows clearly enough. {} when nothing is pending or nothing wins the vote."""
    from . import lab_resolve
    from .resolve_llm import _choice_votes

    out: dict[str, str] = {}
    for b, read, partner, mate in slips(blocks, have):
        options = [mate]
        for c in lab_resolve.suggest(re.sub(r"[^A-Za-z]", "", read), k=3):
            if c.casefold() != mate.casefold() and c not in options and len(options) < 4:
                options.append(c)
        crops = [c for c in (_crop(image, b.get("bbox"), s) for s in SCALES) if c]
        if len(options) < 2 or not crops:
            continue

        def ask(opts: list[str], read: str = read, partner: str = partner) -> str:
            allo = [*opts, "none of these"]
            return ("This is a list of lab tests ordered on a handwritten prescription. Two tests are written together with a slash and "
                    f"the second one is {partner.upper()}. Which test is written just BEFORE the slash that comes before {partner.upper()} "
                    f"(it may look like '{read}')? Which of these is exactly what is written, letter by letter? "
                    + " ".join(f"{i + 1}) {o}" for i, o in enumerate(allo)) + '. Answer with the number only as JSON {"choice": n}.')

        votes = _choice_votes(client, crops, options, ask, SHUFFLES)
        total = sum(votes.values())
        won = votes.get(mate, 0)
        log.info("pair_mate", read=read, partner=partner, votes=votes)
        if won >= MIN_VOTES and won * 2 > total:
            out[mate] = (f"written as '{read}/{partner}' where the line reader could read only {partner}: "
                         f"{won} of {len(crops) * SHUFFLES} readings of the line's picture chose {mate} from a short list")
    return out
