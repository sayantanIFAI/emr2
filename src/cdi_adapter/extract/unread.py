"""A handwritten line nobody could read, where the doctor usually writes the tests, is raised to a person instead of vanishing.

MEASURED on a real veterinary prescription: the line "CBC/KFT/LFT" at the foot of the page was read as garbage by both readers and by the
full-page answer, so three ordered tests never reached anyone and the page said nothing about it. The line is flagged (``unreadable_order_line``)
when it is a handwritten line with no usable reading AND it sits next to the words that introduce an order (advice, investigation, review,
follow up ...) or is one of the last two handwritten lines of the page, where a review / test line is written. At most three lines are flagged."""
from __future__ import annotations

import re
from typing import Any

_CUE = re.compile(r"(?i)\b(?:adv(?:ice|ised)?|inv(?:estigations?)?|ix|r[eo]v[il]?ew|rev|follow\s*up|f/?u|pls|please|next|to\s+do)\b")
_FLAG = "unreadable_order_line"
MAX_FLAGS = 3


def _box(b: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in b["bbox"][:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _state(b: dict[str, Any]) -> str:
    rec = b.get("recognition")
    return str(rec.get("state") or "") if isinstance(rec, dict) else ""


def _unusable(text: str, state: str) -> bool:
    if state == "no_reading":
        return True
    t = text or ""
    if re.search(r"\d{1,2}\s*[/.\-]\s*\d{1,2}", t):
        return False                                     # a date, however partly read, is a reading
    letters = sum(c.isalpha() for c in t)
    return bool(t) and (t.count("?") / max(1, len(t)) >= 0.4 or letters < 0.4 * len(re.sub(r"\s", "", t)))


def unreadable_order_lines(blocks: list[dict[str, Any]] | None) -> list[tuple[int, str]]:
    """``[(block index, what was read)]``: unreadable handwritten lines near an order cue or at the foot of the handwriting."""
    boxed = [(i, b, _box(b)) for i, b in enumerate(blocks or []) if _box(b)]
    if not boxed:
        return []
    page_w = max(bb[2] for _i, _b, bb in boxed)
    hand = sorted((t for t in boxed if _state(t[1]) != "printed"), key=lambda t: (t[2][1], t[2][0]))
    last_two = {t[0] for t in hand[-2:]}
    ordered = sorted(boxed, key=lambda t: (t[2][1], t[2][0]))
    pos = {t[0]: k for k, t in enumerate(ordered)}
    out: list[tuple[int, str]] = []
    for i, b, bb in hand:
        text = str(b.get("text") or "")
        if bb[2] - bb[0] < 0.06 * page_w or not _unusable(text, _state(b)):
            continue
        k = pos[i]
        near = ordered[max(0, k - 2):k + 3]
        if i in last_two or any(_CUE.search(str(nb.get("text") or "")) for _j, nb, _bb in near if _j != i):
            out.append((i, text))
    return out[:MAX_FLAGS]


def flags(blocks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The result's ``flags`` entries for those lines."""
    return [{"code": _FLAG, "line": i, "text": text[:80]} for i, text in unreadable_order_lines(blocks)]
