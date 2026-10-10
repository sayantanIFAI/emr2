"""A BOX OF TESTS the doctor wrote by hand ("TSH/FT4, CBC, ESR / IgE, RBS, CRP"), read again as ONE enlarged picture.

MEASURED on a real page: the line reader cut the box into pieces and read them as "TSU/RTU", "CBC, ER", "TgE", "yRB"; a name-closest-in-the-list step then
turned "TgE" into TBG and "yRB" into RBC, two tests nobody ordered, while TSH, FT4, ESR, IgE and RBS were lost. The same box cropped and shown whole, at
three sizes, was read "TSH, FT4, CBC, ESR / IgE, RBS, CRP" every time.

The box is the place the whole-page answer pointed at (the lines it cited for the tests it listed). It is cut out and asked about two ways at each size:
a plain transcription (no hint at all) and a request to list the tests written there, with the usual abbreviations named to help read bad writing.
A name asked for with hints could be an echo of the hint, so a test is accepted only when BOTH

* the lab lists place it exactly (``test_cluster.placed_text``), and at least two of the three sizes list it, and
* the plain transcription, which had no hint, has a piece that is close to it in letters (it is written there, however badly).

Everything accepted is still "to check" for a person. A misreading of the same box by the earlier steps (a name that is not a test, or one letter from an
accepted test) is dropped; a test the box does not cover is left alone. A page whose box cannot be read this way is left exactly as it was."""
from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from ..config import settings
from ..logging import get_logger
from . import fields as F
from .test_cluster import near_miss, placed_text
from .test_names import split_tests

log = get_logger(__name__)

SCALES = (1.0, 1.6, 2.4)
MIN_SIZES = 2                    # of the three sizes must list the test
PLAIN = "Transcribe exactly the handwriting in this image, letter by letter, line by line."
LIST = ("This is a box from a doctor's handwritten prescription listing laboratory tests to be done (separated by commas or slashes). Write each test exactly "
        "as it is written, one per line. Write only what is there. A tick mark before a line is not a letter. Common ones: CBC, ESR, CRP, TSH, FT4, FT3, IgE, RBS, FBS, "
        "PPBS, HbA1c, LFT, KFT, Urine RE, Ca2+ (calcium), Vit D3, Vit B12, Uric acid, USG, X-ray.")


def _key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def _entry_blocks(it: Any, labels: dict[str, dict[str, Any]], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The page lines an investigation entry stands on: the lines the whole-page answer cited, else the lines whose text is mostly contained in the entry's
    letters ("?it D3" and "un?c ac?d" for "B/F -> +, Vit D3, Uric acid"; "?SGT[whole]" and "Abdocmem" for "VSGT (whole Abdomen)"). The answer often cites
    nothing for a line of tests, and the line is then found by what it says."""
    cited = [labels[str(lab)] for lab in ((it.get("evidence") if isinstance(it, dict) else None) or []) if str(lab) in labels]
    if cited:
        return cited
    ek = _key(it.get("text") if isinstance(it, dict) else str(it))
    if len(ek) < 4:
        return []
    out = []
    for b in blocks:
        bk = _key(str(b.get("text") or ""))
        if len(bk) < 4 or not _bbox(b):
            continue
        m = difflib.SequenceMatcher(None, ek, bk)
        hit = sum(x.size for x in m.get_matching_blocks())
        if hit / len(bk) >= 0.75:
            out.append(b)
    return out


def _boxes(blocks: list[dict[str, Any]], payload: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """The boxes of handwriting the tests listed by the whole-page answer stand on (``_entry_blocks``), the lines in and right around them included
    (a superscript read as a stray line: "Ca2+" as "Ca??"), grouped by vertical closeness."""
    labels = F.blocks_by_label(blocks)
    cited = []
    for it in payload.get("investigations") or []:
        for b in _entry_blocks(it, labels, blocks):
            bx = _bbox(b)
            if bx:
                cited.append(bx)
    if not cited:
        return []
    cited.sort(key=lambda r: r[1])
    mean_h = float(np.mean([r[3] - r[1] for r in cited]))
    groups: list[list[tuple[int, int, int, int]]] = []
    for r in cited:
        if groups and r[1] - max(g[3] for g in groups[-1]) <= 1.2 * mean_h:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = []
    for g in groups[:3]:
        box = (min(r[0] for r in g), min(r[1] for r in g), max(r[2] for r in g), max(r[3] for r in g))
        pad = int(0.8 * mean_h)
        grown = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        near = [bx for bx in (_bbox(b) for b in blocks) if bx and bx[0] < grown[2] and bx[2] > grown[0] and bx[1] < grown[3] and bx[3] > grown[1]
                and (bx[3] - bx[1]) <= 3 * mean_h]
        if near:
            box = (min(box[0], *(r[0] for r in near)), min(box[1], *(r[1] for r in near)), max(box[2], *(r[2] for r in near)), max(box[3], *(r[3] for r in near)))
        out.append(box)
    merged = True
    while merged and len(out) > 1:                    # two boxes that overlap once their neighbouring lines are in are one box
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                p, q = out[i], out[j]
                if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                    out[i] = (min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]))
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return out


def _bbox(b: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in b["bbox"][:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _crops(image: bytes, box: tuple[int, int, int, int]) -> list[bytes]:
    arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    px, py = max(20, int(0.04 * (x1 - x0))), max(14, int(0.6 * (y1 - y0) / 3))
    c = arr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
    if c.size == 0:
        return []
    out = []
    for f in SCALES:
        # the crop is made large enough to read: a small box is enlarged to about 900 px wide first
        base = max(1.0, 900.0 / c.shape[1])
        im = cv2.resize(c, None, fx=base * f, fy=base * f, interpolation=cv2.INTER_CUBIC)
        ok, png = cv2.imencode(".png", im)
        if ok:
            out.append(png.tobytes())
    return out


def _names(text: str) -> set[str]:
    """The tests the lab lists place in a reading of a box ("TSH, FT4, CBC, ESR\\nIgE, RBS, CRP"); a name one confusable letter from one counts."""
    found: set[str] = set()
    for piece in split_tests(re.sub(r"[\r\n]+", ", ", text or "")):
        piece = piece.strip(" .,;:-")
        got = placed_text(piece) if len(re.findall(r"[A-Za-z]", piece)) >= 2 else None
        got = got or near_miss(piece)
        if got:
            found.add(got)
    return found


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in re.findall(r"[A-Za-z0-9]{2,}", text or "")]


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _supported(name: str, tokens: list[str]) -> bool:
    """Some piece of the unhinted plain reading (a word, or up to four words run together: "usg whole abdomen") is within a third of the test's letters of
    it (rounded up): "FTU" for FT4, "TSU" for TSH, "TgE" for IgE, "VSGT whole Abdomen" for "USG whole abdomen"."""
    k = _key(name)
    allowed = -(-len(k) // 3)
    pieces = list(tokens) + ["".join(tokens[i:i + n]) for n in (2, 3, 4) for i in range(len(tokens) - n + 1)]
    return any(_distance(k, t) <= allowed for t in pieces)


def reread(client: Any, image: bytes, blocks: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any] | None:
    """Read the box(es) of tests the whole-page answer pointed at, once more as enlarged pictures. Changes ``payload["investigations"]`` and returns what
    it did, or None when there is no box or nothing could be accepted."""
    if not settings.test_box_reread:
        return None
    boxes = _boxes(blocks, payload)
    if not boxes:
        return None
    accepted: dict[str, tuple[int, str]] = {}
    for box in boxes:
        crops = _crops(image, box)
        if not crops:
            continue
        jobs = [(c, p) for c in crops for p in (PLAIN, LIST)]

        def ask(job: tuple[bytes, str]) -> str:
            try:
                txt, _ = client.vlm_generate_ex(job[0], job[1], max_tokens=100)
                return str(txt or "")
            except Exception as exc:  # noqa: BLE001 - an extra look must never cost the document
                log.warning("test_box_read_failed", error=str(exc)[:200])
                return ""

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            outs = list(pool.map(ask, jobs))
        plain = [o for (_c, p), o in zip(jobs, outs) if p == PLAIN]
        listed = [o for (_c, p), o in zip(jobs, outs) if p == LIST]
        tokens = [t for o in plain for t in _tokens(o)]
        votes: dict[str, int] = {}
        for o in listed:
            for nm in _names(o):
                votes[nm] = votes.get(nm, 0) + 1
        for nm, v in votes.items():
            if v >= MIN_SIZES and _supported(nm, tokens):
                old = accepted.get(_key(nm))
                if not old or v > old[0]:
                    accepted[_key(nm)] = (v, nm)
    if not accepted:
        return None
    names = {k: nm for k, (_v, nm) in accepted.items()}
    inv = payload.get("investigations") or []
    dropped: list[str] = []
    kept: list[Any] = []
    labels = F.blocks_by_label(blocks)

    def in_box(it: Any) -> bool:
        for blk in _entry_blocks(it, labels, blocks):
            bx = _bbox(blk)
            if bx:
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                    return True
        return False

    for it in inv:
        text = it.get("text") if isinstance(it, dict) else str(it)
        parts = split_tests(text) or [text]
        inside = in_box(it)
        keep = []
        for p in parts:
            k = _key(p)
            near = [a for a in names if a != k and difflib.SequenceMatcher(None, k, a).ratio() >= 0.6]
            exact = placed_text(p) is not None
            if k in names:
                keep.append(p)
            elif inside and not exact:
                dropped.append(p)                  # a line reading of the box that no list places: the whole box was read again, this is a piece of it misread
            elif near and (not exact or any(difflib.SequenceMatcher(None, k, a).ratio() >= 0.66 for a in near)):
                dropped.append(p)                  # one letter from an accepted test: a misreading of it (RBC for RBS)
            else:
                keep.append(p)
        if len(keep) == len(parts):
            kept.append(it)
        elif keep:
            kept.append({**it, "text": ", ".join(keep)} if isinstance(it, dict) else ", ".join(keep))
    have = {_key(p) for it in kept for p in (split_tests(it.get("text") if isinstance(it, dict) else str(it)) or [])}
    scan = payload.setdefault("_text_scan", {})
    corro = set(payload.get("_corroborated") or [])
    for k, (v, nm) in accepted.items():
        scan[nm] = f"read again from the enlarged box of tests: {v} of {len(SCALES)} sizes list it and the plain reading has letters like it"
        corro.add(nm)
        if k not in have:
            kept.append({"text": nm, "evidence": [], "source": "test_box"})
            have.add(k)
    payload["investigations"] = kept
    payload["_corroborated"] = sorted(corro)
    info = {"boxes": len(boxes), "accepted": sorted(nm for _v, nm in accepted.values()), "dropped": sorted(set(dropped))}
    payload["_test_box"] = info
    return info
