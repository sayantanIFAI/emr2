"""Two clean-ups of the investigations a page lists, both made so that a test is never lost and a wrong big statement is never left beside right ones.

``drop_covered``  an entry that holds nothing but tests ALREADY listed on their own, with at most a garbled heading word ("Br Av TSH FT4" beside TSH and
                  FT4 listed separately) is not a test; MEASURED on a real page, it was shown as a third "test" that no list could place.
``reread``        an investigation written by hand that nobody could read ("Lang?erseid?" for "Laryngoscopy": the model itself reads the word as "Langueschy",
                  "Langnesby" at other sizes) is looked at again as a CHOICE. The readings at three sizes bring up the few names of the lab lists and the
                  investigation names that are most alike in letters; the line is then shown with those names (and "none of these") several times, in several
                  orders and sizes. The name is taken only when most of the answers pick it, and it is kept "to check" with that said. A name no answer
                  settles on is left as it was. MEASURED on the real crop: "Laryngoscopy" was picked 9 times of 9; with only wrong names offered the picks
                  were scattered (2 and 3 of 9) and nothing was taken."""
from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from ..config import settings
from ..logging import get_logger
from .test_cluster import placed_text
from .test_names import looks_like_medicine

log = get_logger(__name__)

SCALES = (1.0, 1.6, 2.4)
SHUFFLES = 3
MIN_VOTES = 5                    # of 9 answers
MIN_ALIKE = 0.5                  # the readings must be at least this alike a name (letters only, same first letter) for it to be offered
MAX_OPTIONS = 4
MAX_DEPARTMENT = 4                # of the department's usual names added to the offer
PLAIN = "Transcribe exactly the handwriting in this image, letter by letter, on one line."
_HEADING = re.compile(r"(?i)^(?:adv(?:ice|ised)?|inv(?:estigations?)?|ix|rx|test(?:s)?)$")


def _dept_typical(department: str | None) -> list[str]:
    from . import department as _dept

    return _dept.typical(department)


def _key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def _text_of(it: Any) -> str:
    return str(it.get("text") if isinstance(it, dict) else it or "")


def drop_covered(investigations: list[Any]) -> tuple[list[Any], list[str]]:
    """``(the entries without the covered ones, what was dropped)``. An entry is covered when it has at least two test names the lab lists place, each
    of them also listed as an entry of its own, and its other words are only short pieces (a heading read badly: "Br Av", "Adv")."""
    own = {_key(_text_of(it)) for it in investigations}
    kept: list[Any] = []
    dropped: list[str] = []
    for it in investigations:
        text = _text_of(it)
        words = re.findall(r"[A-Za-z0-9+]+", text)
        tests = [w for w in words if placed_text(w) is not None]
        rest = [w for w in words if placed_text(w) is None]
        if (len(tests) >= 2 and len(words) > len(tests) and all(_key(w) in own for w in tests)
                and all(len(w) <= 3 or _HEADING.match(w) for w in rest) and _key(text) not in {_key(w) for w in tests}):
            dropped.append(text)
            continue
        kept.append(it)
    return kept, dropped


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").casefold())


def candidates(reads: list[str], department: str | None = None) -> list[str]:
    """The names (investigations and tests, long ones) most alike what the readings say, best first, then the usual investigations of the doctor's
    department (the front desk named it, or the printed header shows it) whatever their letters: an ENT doctor's unreadable word is offered
    "Laryngoscopy" even when no letter of it fits. They are only OFFERED; the line must still be chosen by looking."""
    from . import department as _dept, lab_mapping

    best: dict[str, float] = {}
    ks = [k for k in (_letters(r) for r in reads) if len(k) >= 4]
    for key, m in lab_mapping._load().items():                  # noqa: SLF001 - the table's own keys
        if len(key) < 6 or not key.isalpha():
            continue
        for k in ks:
            r = difflib.SequenceMatcher(None, k, key).ratio()
            # the same first letter, or a reading that is almost all of the name (a tick mark joined to the first letter: "VSGT whole Abdomen" for "USG whole abdomen")
            if (k[0] == key[0] and r >= MIN_ALIKE or r >= 0.8) and r > best.get(m.canonical, 0.0):
                best[m.canonical] = r
    out = [c for c, _r in sorted(best.items(), key=lambda kv: -kv[1])][:MAX_OPTIONS]
    for name in _dept.typical(department):
        if name not in out:
            out.append(name)
    return out[:MAX_OPTIONS + MAX_DEPARTMENT]


def _block_for(text: str, blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    k = _key(text)
    best, best_r = None, 0.0
    for b in blocks:
        bk = _key(str(b.get("text") or ""))
        r = 1.0 if bk == k else difflib.SequenceMatcher(None, k, bk).ratio()
        if r > best_r:
            best, best_r = b, r
    return best if best_r >= 0.85 else None


def _crops(image: bytes, bbox: list[Any]) -> list[bytes]:
    arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return []
    if arr is None or x1 <= x0 or y1 <= y0:
        return []
    h, w = arr.shape[:2]
    px, py = max(20, int(0.15 * (x1 - x0))), max(12, int(0.3 * (y1 - y0)))
    c = arr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
    if c.size == 0:
        return []
    base = max(1.0, 500.0 / c.shape[1])
    out = []
    for f in SCALES:
        ok, png = cv2.imencode(".png", cv2.resize(c, None, fx=base * f, fy=base * f, interpolation=cv2.INTER_CUBIC))
        if ok:
            out.append(png.tobytes())
    return out


def _qualifies(text: str, not_lab: dict[str, Any]) -> bool:
    letters = _letters(text)
    return (5 <= len(letters) <= 28 and len(text.split()) <= 3 and not re.search(r"\d", text) and placed_text(text) is None
            and not looks_like_medicine(text) and text not in not_lab and not _HEADING.match(text.strip()))


def reread(client: Any, image: bytes, blocks: list[dict[str, Any]], payload: dict[str, Any], department: str | None = None) -> dict[str, str]:
    """``{as written: the name chosen}``; the entries are changed in ``payload["investigations"]``."""
    if not settings.unplaced_choice:
        return {}
    from .resolve_llm import _choice_votes

    not_lab = payload.get("_not_lab") or {}
    chosen: dict[str, str] = {}
    inv = list(payload.get("investigations") or [])
    for i, it in enumerate(inv):
        text = _text_of(it).strip()
        if not _qualifies(text, not_lab):
            continue
        b = _block_for(text, blocks)
        crops = _crops(image, b["bbox"]) if b else []
        if not crops:
            continue

        def plain(png: bytes) -> str:
            try:
                t, _ = client.vlm_generate_ex(png, PLAIN, max_tokens=40)
                return str(t or "").strip()
            except Exception as exc:  # noqa: BLE001 - an extra look must never cost the document
                log.warning("unplaced_read_failed", error=str(exc)[:200])
                return ""

        with ThreadPoolExecutor(max_workers=len(crops)) as pool:
            reads = [r for r in pool.map(plain, crops) if r]
        options = candidates([text, *reads], department)
        if len(options) < 1:
            continue

        def ask(opts: list[str]) -> str:
            allo = [*opts, "none of these"]
            return ("This is one line from a doctor's handwritten prescription: an investigation (a test or a procedure) to be done. Which of these is exactly "
                    "what is written, letter by letter? " + " ".join(f"{i + 1}) {o}" for i, o in enumerate(allo)) + '. Answer with the number only as JSON {"choice": n}.')

        votes = _choice_votes(client, crops, options if len(options) >= 2 else [*options, "(no other name)"], ask, SHUFFLES)
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        total = len(crops) * SHUFFLES
        if ranked and ranked[0][1] >= MIN_VOTES and ranked[0][0] in options and (len(ranked) == 1 or ranked[0][1] > 2 * ranked[1][1]):
            name, n = ranked[0]
            chosen[text] = name
            inv[i] = {**it, "text": name, "source": "choice_vote"} if isinstance(it, dict) else {"text": name, "evidence": [], "source": "choice_vote"}
            payload.setdefault("_text_scan", {})[name] = (f"written as '{text}' (read as {', '.join(repr(r) for r in reads[:3])}); chosen from a short list by looking at the line"
                                                          f"{' (a usual investigation of ' + department + ')' if department and name in _dept_typical(department) else ''}: "
                                                          f"{n} of {total} answers")
            payload["_corroborated"] = sorted({*payload.get("_corroborated", []), name})
        log.info("unplaced_choice", written=text, options=options, votes=votes)
    payload["investigations"] = inv
    return chosen
