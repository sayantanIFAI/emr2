"""A PRINTED CHECKLIST of tests, ordered by marking names: the pad prints a menu ("CBC -", "Bilirubin / ALT / AST -", "EEG -", "MRI Scan of Brain -" ...)
and the doctor strikes, ticks, circles or scratches the names wanted. Any pen mark on a name means that name is ordered; an unmarked printed name is
only the menu and is never a test.

This module is only run on a page that LOOKS like such a menu (``rows`` finds at least ``MIN_ROWS`` aligned printed rows ending in a dash), so an
ordinary prescription never pays for it. It needs no model and no colour: the print and the pen are both black on many scans, so marks are found by
SHAPE, inside each row's own box as the printed-text reader drew it:

* a piece of ink that STICKS OUT of the row's box (a strike that runs past the name, a tick, cross, circle or dot beside it, a stroke joined to the
  letters) is a mark: printed letters never leave the box the reader drew round them;
* a long straight run of ink INSIDE the box (a strike that stays within the name) is a mark: no printed letter has a straight stroke that long.

The form's own ruled frame is taken away first. The test is deliberately generous: a faint mark counts as a mark (the result is always "to check" for
a person), but a speck of dust below ``MIN_MARK_AREA`` times the text height squared never does.

``rows``         the menu rows of a page: where each is, its text and the test names in it;
``mark_rows``    which rows carry a mark, with the kind of mark and where along the row it lies;
``marked_names`` the test names ordered by those marks: the name the mark touches in a row of several ("Bilirubin / ALT / AST"), else every name of
                 the row (a tick in the margin or after the dash).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

MIN_ROWS = 8                    # aligned printed rows ending in a dash before a page counts as a checklist
MIN_MARK_AREA = 0.09            # ink area, in text-height squared, below which a piece is a speck, not a mark
INK_BELOW_PAPER = 30            # grey levels darker than the paper that count as ink
STICK_OUT = 0.25                # a piece reaching this far (in text heights) beyond the row's box is not printed letters
STRIKE_RUN = 1.3                # a straight run of ink this long (in text heights) inside the box is a strike
_DASH_END = re.compile(r"[-–—_.]\s*$")
_ALIGN = 0.015                  # columns: left edges within this share of the page width are one column
_STRIP = " -–—_.:•·*"


@dataclass
class Row:
    index: int                                  # the OCR block's place in the page's block list
    box: tuple[int, int, int, int]
    text: str
    names: list[tuple[str, float, float]] = field(default_factory=list)       # (name, start, end) as shares of the row's width


@dataclass
class RowMark:
    row: Row
    kind: str                                   # strike | tick | circle | dot | mark
    x0: float                                   # where along the row it lies, as shares of the row's width (can be < 0 or > 1: beside the name)
    x1: float


def _box(b: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in b["bbox"][:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _placed(name: str) -> bool:
    from .test_cluster import placed_text

    return placed_text(name) is not None


def names_in(text: str, placed: Callable[[str], bool] | None = None) -> list[tuple[str, float, float]]:
    """The test names of a menu row with where each stands along the row. A slash separates two tests only when EVERY part is a test the lab lists
    place ("Bilirubin / ALT / AST", "FT4 / TSH"); otherwise the whole row is one test ("Urine C/S", "X-Ray Chest PA/ AP View", "Dengue IgG / IgM")."""
    placed = placed or _placed
    clean = text.strip().rstrip(_STRIP).lstrip(_STRIP)
    n = max(len(text), 1)
    if len(re.findall(r"[A-Za-z]", clean)) < 2:
        return []
    parts = [p.strip(_STRIP) for p in re.split(r"\s*/\s*", clean)]
    if len(parts) > 1 and all(p and placed(p) for p in parts):
        out, pos = [], 0
        for p in parts:
            at = clean.find(p, pos)
            at = pos if at < 0 else at
            out.append((p, at / n, (at + len(p)) / n))
            pos = at + len(p)
        return out
    return [(clean, 0.0, len(clean) / n)]


def rows(blocks: list[dict[str, Any]] | None, placed: Callable[[str], bool] | None = None) -> list[Row]:
    """The menu rows of a page, or [] when the page is not a printed checklist. A column of aligned lines most of which end in a dash is a menu column;
    every line of the column between its first dash line and one line past its last is a row (a struck name loses its dash in the reading: "EEG." for
    "EEG -", and the menu's last name often has none: "X-Ray Nasopharynx")."""
    items = [(i, _box(b), str(b.get("text") or "").strip()) for i, b in enumerate(blocks or [])]
    items = [(i, bx, t) for i, bx, t in items if bx and t]
    if len(items) < MIN_ROWS:
        return []
    page_w = max(bx[2] for _i, bx, _t in items)
    cols: list[list[tuple[int, tuple[int, int, int, int], str]]] = []
    for it in sorted(items, key=lambda x: x[1][0]):
        for c in cols:
            if abs(c[0][1][0] - it[1][0]) <= _ALIGN * page_w:
                c.append(it)
                break
        else:
            cols.append([it])
    out: list[Row] = []
    for c in cols:
        c.sort(key=lambda x: x[1][1])
        dashed = [k for k, (_i, _bx, t) in enumerate(c) if _DASH_END.search(t)]
        if len(dashed) < 3:
            continue
        lo, hi = dashed[0], dashed[-1]
        gaps = [c[k + 1][1][1] - c[k][1][1] for k in range(lo, hi)]
        pitch = float(np.median(gaps)) if gaps else 0.0
        if hi + 1 < len(c) and pitch and (c[hi + 1][1][1] - c[hi][1][1]) <= 1.35 * pitch:
            hi += 1
        for _i, bx, t in c[lo:hi + 1]:
            names = names_in(t, placed)
            if names:
                out.append(Row(_i, bx, t, names))
    return out if len(out) >= MIN_ROWS else []


def _ink(gray: np.ndarray) -> np.ndarray:
    paper = float(np.percentile(gray, 85))
    return (gray < paper - INK_BELOW_PAPER).astype(np.uint8)


def _frame(ink: np.ndarray, pitch: float) -> np.ndarray:
    """The form's own ruled lines: very long thin horizontals (a good share of the page wide) and verticals (several rows high). A strike through a name
    is shorter than a column; a tick is less than two rows high."""
    _h, w = ink.shape
    hk = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(25, int(0.4 * w))), np.uint8))
    vk = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(25, int(3.0 * pitch)), 1), np.uint8))
    return cv2.dilate(np.maximum(hk, vk), np.ones((5, 5), np.uint8))


def _long_run(crop: np.ndarray, length: int) -> tuple[int, int] | None:
    """The horizontal extent of a straight run of ink at least ``length`` long in ``crop``, tried at slopes up to 20 degrees either way (a hand-drawn
    strike is rarely level); None when there is none."""
    if crop.size == 0 or crop.shape[1] < length:
        return None
    pad = max(crop.shape) // 2 + 2
    big = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = big.shape[1] / 2, big.shape[0] / 2
    kernel = np.ones((1, length), np.uint8)
    for ang in (0, -2.5, 2.5, -5, 5, -7.5, 7.5, -10, 10, -12.5, 12.5, -15, 15, -17.5, 17.5, -20, 20):
        m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot = cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
        hit = cv2.morphologyEx(rot, cv2.MORPH_OPEN, kernel)
        if hit.any():
            cols = np.where(hit.any(axis=0))[0]
            back = cv2.invertAffineTransform(m)
            pts = np.array([[cols.min(), cy], [cols.max(), cy]], np.float32)
            xs = (back[0, 0] * pts[:, 0] + back[0, 1] * pts[:, 1] + back[0, 2]) - pad
            return int(xs.min()), int(xs.max())
    return None


def mark_rows(img_bgr: np.ndarray, menu: list[Row]) -> list[RowMark]:
    """The marks on the menu rows, one entry per mark (a row can carry two)."""
    if not menu:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    text_h = float(np.median([r.box[3] - r.box[1] for r in menu]))
    ys = sorted(r.box[1] for r in menu)
    steps = [b - a for a, b in zip(ys, ys[1:]) if b - a > text_h * 0.5]
    pitch = float(np.median(steps)) if steps else text_h * 1.4
    ink = _ink(gray)
    ink[_frame(ink, pitch) > 0] = 0
    boxes = [r.box for r in menu]
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink, 8)
    out: list[RowMark] = []

    def dist(r: Row, ccx: float, ccy: float) -> float:
        bx0, by0, bx1, by1 = r.box
        return float(np.hypot(max(bx0 - ccx, 0, ccx - bx1), max(by0 - ccy, 0, ccy - by1)))

    for row in menu:
        x0, y0, x1, y1 = row.box
        h = max(y1 - y0, 8)
        width = max(x1 - x0, 1)
        tol = STICK_OUT * h
        left = max([b[2] for b in boxes if b is not row.box and b[2] <= x0 and abs((b[1] + b[3]) / 2 - (y0 + y1) / 2) < 0.8 * pitch] or [0]) + 2
        right = min([b[0] for b in boxes if b is not row.box and b[0] >= x1 and abs((b[1] + b[3]) / 2 - (y0 + y1) / 2) < 0.8 * pitch] or [W]) - 2
        zx0, zx1 = max(left, x0 - int(2.5 * h)), min(right, x1 + int(2.5 * h))
        zy0, zy1 = y0 - int(0.6 * h), y1 + int(0.6 * h)
        found = False
        for k in range(1, n):
            cx, cy, cw, ch, area = (int(v) for v in stats[k, :5])
            if cx + cw < zx0 or cx > zx1 or cy + ch < zy0 or cy > zy1 or area < MIN_MARK_AREA * h * h:
                continue
            if dist(row, cx + cw / 2, cy + ch / 2) > min(dist(o, cx + cw / 2, cy + ch / 2) for o in menu):
                continue                                          # a piece belongs to the ONE row whose box it is nearest to
            if not (cx < x0 - tol or cx + cw > x1 + tol or cy < y0 - tol or cy + ch > y1 + tol):
                continue                                          # inside the box: printed letters (a strike inside is looked for below)
            band = min(cy + ch, y1 + 0.45 * h) - max(cy, y0 - 0.45 * h)
            if band < 0.6 * ch:
                continue                                          # mostly above / below the row: handwriting from another place (the weight written above the menu)
            if cx + cw / 2 > x1 or cx + cw / 2 < x0 or cy + ch / 2 > y1 or cy + ch / 2 < y0:
                kind = "dot" if cw <= 0.7 * h and ch <= 0.7 * h else ("circle" if cw > 1.2 * h and ch > 0.9 * h and area < 0.45 * cw * ch else
                                                                      ("strike" if cw >= 1.6 * h and cw >= 1.8 * ch else "tick"))
            else:
                kind = "strike" if cw > (x1 - x0) else "mark"
            out.append(RowMark(row, kind, (cx - x0) / width, (cx + cw - x0) / width))
            found = True
        if found:
            continue
        run = _long_run(ink[max(0, y0 - int(0.2 * h)):min(H, y1 + int(0.2 * h)), max(0, x0):min(W, x1)], max(18, int(STRIKE_RUN * h)))
        if run is not None:
            out.append(RowMark(row, "strike", run[0] / width, run[1] / width))
    return out


def marked_names(menu: list[Row], marks: list[RowMark]) -> list[tuple[str, str, str]]:
    """``(test name, as printed, mark kind)`` for every name a mark orders. A mark over the stretch of one name of a row of several orders that name; a
    mark beside the row, or over none of the names, orders every name of the row."""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in menu:
        mine = [m for m in marks if m.row is row]
        if not mine or not row.names:
            continue
        chosen: list[tuple[str, str]] = []
        for m in mine:
            hit = [(nm, m.kind) for nm, a, b in row.names if m.x0 < b and m.x1 > a]
            chosen += hit if hit and len(row.names) > 1 else [(nm, m.kind) for nm, _a, _b in row.names]
        for nm, kind in chosen:
            if (row.index, nm) not in seen:
                seen.add((row.index, nm))
                out.append((nm, row.text, kind))
    return out


# ---------------------------------------------------------------------------------------------------------------- applying it to an extraction
def _key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def _text_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("name") or "")
    return str(item or "")


def _menu_name(raw: str) -> str:
    """The printed name of a marked row, tidied. A struck row is read badly by the printed-text reader ("MRrScanofBrain-" for "MRI Scan of Brain -"):
    when ONE name in the mapping table is at least 88% alike it (letters only, same length within a letter) that name is used; else what was read."""
    import difflib

    from . import lab_mapping

    name = raw.strip().strip(_STRIP).strip()
    k = _key(name)
    if len(k) < 4:
        return name
    best: dict[str, float] = {}
    for key, m in lab_mapping._load().items():                      # noqa: SLF001 - the table's own keys
        if abs(len(key) - len(k)) > 1:
            continue
        r = 1.0 if key == k else difflib.SequenceMatcher(None, k, key).ratio()
        if r >= 0.88:
            best[m.alias] = max(best.get(m.alias, 0.0), r)
    if not best or any(r == 1.0 for r in best.values()):
        return name
    top = sorted(best.items(), key=lambda kv: -kv[1])
    return top[0][0] if len(top) == 1 or top[0][1] - top[1][1] >= 0.05 else name


def apply(payload: dict[str, Any], blocks: list[dict[str, Any]] | None, image_png: bytes | None,
          placed: Callable[[str], bool] | None = None) -> dict[str, Any] | None:
    """On a page that is a printed checklist: the tests the pen marks order are listed (marked as such, to be checked), and every other name of the menu
    is taken out of the tests (a printed menu is not an order) unless the page also writes it by hand somewhere else. Returns what was found, or None for
    an ordinary page, which costs nothing but the look for a menu in the text lines."""
    from .test_names import split_tests

    menu = rows(blocks, placed)
    if not menu or not image_png:
        return None
    img = cv2.imdecode(np.frombuffer(image_png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    if max(r.box[2] for r in menu) > 1.05 * iw or max(r.box[3] for r in menu) > 1.05 * ih:
        return None                                              # the boxes are not in this picture's frame: say nothing rather than look in the wrong place
    marks = mark_rows(img, menu)
    ordered = marked_names(menu, marks)
    menu_idx = {r.index for r in menu}
    menu_keys = {_key(nm) for r in menu for nm, _a, _b in r.names}
    ordered_keys = {_key(nm) for nm, _t, _k in ordered}
    outside = [_key(str(b.get("text") or "")) for i, b in enumerate(blocks or []) if i not in menu_idx]

    def written_elsewhere(k: str) -> bool:
        return len(k) >= 3 and any(k in o for o in outside)

    drop = {k for k in menu_keys - ordered_keys if not written_elsewhere(k)}
    dropped: list[str] = []
    kept: list[Any] = []
    for it in payload.get("investigations") or []:
        parts = split_tests(_text_of(it)) or [_text_of(it)]
        keep = [p for p in parts if _key(p) not in drop]
        if len(keep) == len(parts):
            kept.append(it)
            continue
        dropped += [p for p in parts if _key(p) in drop]
        if keep:
            kept.append({**it, "text": ", ".join(keep)} if isinstance(it, dict) else ", ".join(keep))
    payload["investigations"] = kept
    have = {_key(p) for it in kept for p in (split_tests(_text_of(it)) or [_text_of(it)])}
    scan = payload.setdefault("_text_scan", {})
    corro = set(payload.get("_corroborated") or [])
    names: list[str] = []
    for nm, text, kind in ordered:
        name = _menu_name(nm)
        names.append(name)
        scan[name] = f"marked on the printed checklist ({kind}): the pen mark is on '{text.strip(_STRIP)}'"
        corro.add(name)
        if _key(name) not in have and _key(nm) not in have:
            kept.append({"text": name, "evidence": [], "source": "printed_checklist"})
            have.add(_key(name))
    payload["_corroborated"] = sorted(corro)
    info = {"rows": len(menu), "marked": names, "marks": [{"row": m.row.text.strip(_STRIP), "kind": m.kind} for m in marks], "dropped": sorted(set(dropped))}
    payload["_checklist"] = info
    return info
