"""Tests are written together: find them in the page's own text by where they sit, with no model involved.

A doctor writes the tests as a list, in one place (a box, a margin, the bottom of the page), often with no heading at all. So a
name's NEIGHBOURS are evidence about it:

- a STRONG test name (CBC, FBS, LFT, TSH ...) the lab lists place exactly is a test wherever it is written;
- an AMBIGUOUS name (vitamin D, calcium, iron ...) is also a supplement: it is a test only beside other test evidence, or when
  "25(OH)" is written with it. On its own, or among medicine lines, it is not taken;
- a ONE-LETTER slip of a strong abbreviation ("fas" for FBS: handwriting makes b look like a) is taken only inside a group of
  lines that also holds other test evidence, and is always marked "read as ...".

Position decides what is "beside": lines whose boxes are close (a line or so apart, and near in width) form one group. Every
name still has to pass the lab-test gate afterwards, and nothing found here is ever accepted without a person."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger
from . import lab_mapping, lab_resolve
from .test_names import _STRONG, is_known_test, looks_like_medicine, split_tests

log = get_logger(__name__)

# a line with the marks of a medicine order is not looked at (and is never part of a group)
_MEDICINE_LINE = re.compile(r"(?i)\b(?:tabs?|tablets?|caps?|capsules?|syp|syr|inj|drops?|oint|cream|gel|susp)\b\.?"
                            r"|\d\s*(?:mg|mcg|ml|gm|iu)\b|\b\d+\s*tabs?\b")
# words of the form itself (vital-sign labels, field names) and generic words: the national lab list holds some of them ("Height" is a
# LOINC term, "Blood" is "Blood [Presence] in Urine"), but written on a prescription they are labels. MEASURED on a real page once the
# national list was loaded: "Height" and "Blood" came out as lab tests.
_FORM_LABELS = frozenset("height weight pulse bp temp temperature spo2 pr rr age sex date history vitals vital signs name complaint complaints "
                         "diagnosis blood urine serum plasma stool fluid sample specimen ph tel phone mob mobile fax email".split())
# a word that also names a supplement: the name is a test only beside other test evidence
_WEAK_WORDS = frozenset("vitamin vit iron calcium zinc magnesium folic potassium sodium pt".split())      # "pt" is also "patient"
# "25(OH)", "25 OH", "2OH" (the 5 lost): the vitamin D test is written, whatever else the line says
_MARKER = re.compile(r"(?i)(?<![0-9])(?:25|2)\s*[\(\[]?\s*oh\b")

# letters handwriting makes look alike (one can be read as another); both directions
_PAIRS = ("ab", "ao", "ae", "bh", "b6", "ce", "co", "eo", "go", "gq", "hn", "il", "lt", "mn", "nu", "nr", "ps", "pb", "rs", "s5", "uv", "vy", "tf")
_CONFUSABLE: set[tuple[str, str]] = {p for a, b in _PAIRS for p in ((a, b), (b, a))}


@dataclass(frozen=True)
class Found:
    test: str          # the name to list (what the lab lists place)
    as_read: str       # what the readers wrote
    why: str           # "exact" | "beside" | "near" | "marker" | "marked"

    @property
    def note(self) -> str:
        """Where it came from, in words for the screen."""
        return {"exact": "found in the page's text",
                "beside": f"found in the page's text beside other tests (read as '{self.as_read}')",
                "near": f"read as '{self.as_read}' (one letter from {self.test}), written beside other tests",
                "marker": f"'{self.as_read}' is written: the vitamin D test",
                "marked": f"printed on the pad and marked by hand ('{self.as_read}')",
                "printed_near": f"a SUGGESTION, not a reading: the printed entry '{self.as_read}' is cut off at the edge of the photo and the closest standard name is {self.test}"}[self.why]


def placed_text(gram: str) -> str | None:
    """The name to list when the lab lists place these words EXACTLY (the mapping table, the national list, the gazetteer), or when
    a reading with ``?`` for unreadable letters fits exactly one standard test; otherwise None. A fuzzy match never counts."""
    if sum(ch.isalpha() for ch in gram) < 2:
        return None
    if "?" in gram:
        hit = lab_mapping.fit(gram)
        if hit:
            return hit.alias.capitalize()
        # a "?" at the edge of a word is most often the tick-box or bullet before the name misread ("?Lipid Profile", "?HBsAg"), not a letter
        edge = " ".join(w.strip("?") for w in gram.split())
        return placed_text(edge) if edge != gram and "?" not in edge else None
    rz = lab_resolve.resolve(gram)
    return gram if rz is not None and not getattr(rz, "fuzzy", False) else None


def _abbreviations() -> frozenset[str]:
    """The strong abbreviations a slip is compared with: 3 to 6 letters, a test and nothing else."""
    out = {w for w in _STRONG if 3 <= len(w) <= 6 and w.isalpha()}
    for key in lab_mapping._load():                                   # noqa: SLF001 - the table's own keys
        if " " not in key and 3 <= len(key) <= 6 and key.isalpha() and key not in _WEAK_WORDS:
            out.add(key)
    return frozenset(out)


def near_miss(token: str) -> str | None:
    """The ONE strong abbreviation this word is a single handwriting-confusable letter away from (``fas`` -> ``FBS``), else None.
    Words that already are a test, and words with two or more differing letters, are not slips."""
    w = token.casefold()
    if not (3 <= len(w) <= 6 and w.isalpha()):
        return None
    abbr = _abbreviations()
    if w in abbr:
        return None
    hits = []
    for a in abbr:
        if len(a) == len(w):
            diff = [(x, y) for x, y in zip(w, a) if x != y]
            if len(diff) == 1 and diff[0] in _CONFUSABLE:
                hits.append(a)
        elif len(a) == len(w) + 1 and any(a[:k] + a[k + 1:] == w and a[k] == a[k - 1] for k in range(1, len(a))):
            hits.append(a)                                         # a doubled letter written once ("apt" for APTT)
    return hits[0].upper() if len(hits) == 1 else None             # two ways to read it: it is not taken


@dataclass
class _Hit:
    test: str
    as_read: str
    kind: str          # "strong" | "weak" | "near" | "marker"


# a heading that says what follows is ordered: "Adv" (advice), "Inv" / "Investigations", "Ix". The words after it, in the same line, are a
# declared test region: a long line is no reason to doubt a test name there. MEASURED on a real page: the reader merged several lines into
# "4 Difficulty in ? 5 Gren or Strain Adv Digital OP? 2 FBS, BJS CT", and FBS was dropped as a test name in a long mixed line.
_HEADING = re.compile(r"(?i)\b(?:adv(?:ice|ised)?|inv(?:estigations?)?|ix)\b\s*[:\-\u2013\u2014]*")


def _has_value(line: str) -> bool:
    from .results_written import _spans

    return bool(_spans(line))


def _line_hits(line: str, long_rule: bool = True) -> list[_Hit]:
    """Every test-like thing in one line: the longest run of up to three words the lists place wins, and the line goes on."""
    from .results_written import clean_entry
    line = clean_entry(line)[0] if _has_value(line) else line          # a value written next to a name makes it a result, not a test to be done
    heading = _HEADING.search(line) if long_rule else None
    if heading and line[heading.end():].strip():
        before = _line_hits(line[:heading.start()], long_rule) if line[:heading.start()].strip() else []
        return before + _line_hits(line[heading.end():], long_rule=False)       # the region after the heading: every word is checked
    out: list[_Hit] = []
    sentence = len(re.findall(r"[A-Za-z0-9?]+", line)) > 4
    for piece in split_tests(line):
        words = re.findall(r"[A-Za-z0-9?]+", piece)
        i = 0
        while i < len(words):
            for n in (3, 2, 1):
                gram = " ".join(words[i:i + n])
                got = placed_text(gram) if i + n <= len(words) else None
                if got and all(w.casefold() in _FORM_LABELS for w in gram.split()):
                    got = None                                  # a printed label of the form, not a test
                if got:
                    weak = any(w.casefold() in _WEAK_WORDS for w in (*gram.split(), *got.split()))
                    if weak and sentence:
                        i += n                                  # an ambiguous name inside a sentence ("calcium rich diet ...") is not a list entry
                        break
                    out.append(_Hit(got, gram, "weak" if weak else "strong"))
                    i += n
                    break
            else:
                slip = near_miss(words[i])
                if slip:
                    out.append(_Hit(slip, words[i], "near"))
                i += 1
    # the same slip written twice in a list ("fas, fas") is two entries: split_tests listed it once, so the repeats are added back
    for slip_hit in [x for x in out if x.kind == "near"]:
        written = len(re.findall(r"(?i)(?<![a-z])" + re.escape(slip_hit.as_read) + r"(?![a-z])", line))
        for _ in range(written - 1):
            out.append(_Hit(slip_hit.test, slip_hit.as_read, "near"))
    total = len(re.findall(r"[A-Za-z0-9?]+", line))
    if long_rule and total >= 6 and sum(h.kind == "strong" for h in out) / total < 0.25:
        # a long line that is mostly other words (the clinic's printed list of services, a sentence): a test name inside it is weak
        # evidence, like an ambiguous name (MEASURED: "ECG" in a printed footer "Endoscopy Ultrasonography Echocardiography ...")
        out = [_Hit(h.test, h.as_read, "weak") if h.kind == "strong" else h for h in out]
    m = _MARKER.search(line)
    if m:
        out.append(_Hit("Vitamin D", m.group(0), "marker"))               # beside or not, "25(OH)" is the vitamin D test
    return out


def _box(b: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in b["bbox"][:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _groups(items: list[tuple[int, dict[str, Any]]]) -> list[list[int]]:
    """Groups of lines (indexes into ``items``) that sit close together. Close = centres within 1.6 line heights vertically and the
    boxes within 3 line heights sideways; a box more than 3 line heights tall (a loose box around a whole paragraph) joins nothing.
    Lines without a box are grouped with the line before and after them in reading order."""
    boxes = [_box(b) for _, b in items]
    heights = [bb[3] - bb[1] for bb in boxes if bb]
    lh = min(60.0, max(12.0, statistics.median(heights))) if heights else 30.0
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = boxes[i], boxes[j]
            if a is None or b is None:
                if j == i + 1:
                    join(i, j)                                    # reading order is all there is
                continue
            if (a[3] - a[1]) > 3 * lh or (b[3] - b[1]) > 3 * lh:
                continue
            dy = abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
            gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            if dy <= 1.6 * lh and gap <= 3 * lh:
                join(i, j)
    out: dict[int, list[int]] = {}
    for i in range(len(items)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _by_pen_marks(found: list[tuple[int, Found]], blocks: list[dict[str, Any]] | None, colour: Any) -> list[tuple[int, Found]]:
    """Among the tests found in PRINTED lines (the pad's own checklist), keep and mark the ones a pen mark touches, and drop the rest, but
    only on a page where at least one printed name is marked. Anything else (no colour page, no printed line, no mark anywhere) is
    returned unchanged."""
    if colour is None or not blocks or not found:
        return found
    try:
        from .marks import Marks

        mk = Marks(colour, list(blocks))
    except Exception as exc:  # noqa: BLE001 - the marks are an extra: never cost the page
        log.warning("marks_failed", error=str(exc)[:200])
        return found
    verdict: dict[int, bool] = {}
    for k, (idx, f) in enumerate(found):
        b = blocks[idx]
        if f.why == "exact" and mk.is_printed(b):
            verdict[k] = mk.marked(b, str(b.get("text") or ""), f.as_read)
    if not any(verdict.values()):
        return found
    out: list[tuple[int, Found]] = []
    for k, (idx, f) in enumerate(found):
        if k not in verdict:
            out.append((idx, f))
        elif verdict[k]:
            out.append((idx, Found(f.test, f.as_read, "marked")))
    return out


_REPORT_HEAD = re.compile(r"(?i)^\W*(?:report|results?)\W*$")
_STOP_HEAD = re.compile(r"(?i)\b(?:r[eo]v[il]?ew|adv(?:ice)?|advised|inv(?:estigations?)?|follow\s*up|rx)\b")
_SEP = re.compile(r"[,;/+&|]")


def _state(b: dict[str, Any]) -> str:
    rec = b.get("recognition")
    return str(rec.get("state") or "") if isinstance(rec, dict) else ""


def result_label_indexes(blocks: list[dict[str, Any]] | None) -> set[int]:
    """The printed RESULT fields of a report pad: under a "Report" heading the form prints one test name per line, each waiting for a value
    (Sonoscan: HDL, LDL, TG, SGPT, Hb% ...). They are labels, not orders. A stack of four or more such single-name lines directly under a
    "Report" heading, ending where the pad's "Review after" / "Advice" starts, is left out of the scan. MEASURED on a real pad: HDL, TG, SGPT
    and haemoglobin were listed as ordered tests. Comma lists (the pad's "Review after: FPG, creatinine, lipid profile") are not touched."""
    boxed = [(i, _box(b)) for i, b in enumerate(blocks or []) if _box(b)]
    order = [i for i, bb in sorted(boxed, key=lambda kv: (kv[1][1], kv[1][0]))]
    out: set[int] = set()
    for pos, i in enumerate(order):
        if not _REPORT_HEAD.search(str((blocks or [])[i].get("text") or "")):
            continue
        stack: list[int] = []
        misses = 0
        hb = _box((blocks or [])[i])
        page_w = max(bb[2] for _k, bb in boxed)
        for j in order[pos + 1:pos + 30]:
            bj = _box((blocks or [])[j])
            if hb is None or bj is None or abs(bj[0] - hb[0]) > 0.12 * page_w:
                continue                                          # another column of the page (the doctor's handwriting beside the pad)
            t = str((blocks or [])[j].get("text") or "").strip()
            if _STOP_HEAD.search(t):
                break
            words = re.findall(r"[A-Za-z0-9?%.]+", t)
            if 1 <= len(words) <= 3 and not _SEP.search(t) and (any(h.kind == "strong" for h in _line_hits(t)) or re.search(r"[-:]\s*$", t)):
                stack.append(j)
            else:
                misses += 1
                if misses > 14:
                    break
        if len(stack) >= 4:
            out.update(stack)
    return out


def _printed_near(piece: str) -> tuple[str, str] | None:
    """A printed list entry the photo's edge cut a letter or two from ("IRINE RE" for "URINE RE", "RIC ACID" for "URIC ACID"): the ONE mapping-table
    name that is at least 82% alike and has the same length within one letter. Printed text only (handwriting is never matched this loosely)."""
    import difflib

    k = re.sub(r"[^a-z]", "", piece.casefold())
    if len(k) < 5:
        return None
    best: list[tuple[float, str]] = []
    for key, m in lab_mapping._load().items():                       # noqa: SLF001
        kk = re.sub(r"[^a-z]", "", key)
        if len(kk) < 5 or abs(len(kk) - len(k)) > 1:
            continue
        r = difflib.SequenceMatcher(None, k, kk).ratio()
        if r >= 0.82:
            best.append((r, m.canonical))
    names = {c for _r, c in best}
    return (next(iter(names)), piece) if len(names) == 1 else None


def scan(blocks: list[dict[str, Any]] | None, colour: Any = None) -> list[Found]:
    """The tests the page's text holds, by position (see the module text). Medicine lines are never looked at. Candidates only.
    ``colour`` is the colour page (an array the size of the picture the text blocks were found in): with it, a test name in a PRE-PRINTED
    list that has a pen mark by it is marked ("marked"), and when at least one printed name on the page is marked the printed names with
    no mark are not orders and are dropped (``marks.py``). A page with no marked printed name is left exactly as it was."""
    labels = result_label_indexes(blocks)                              # printed result fields of a report pad are not orders
    items = [(i, b) for i, b in enumerate(blocks or []) if str(b.get("text") or "").strip() and i not in labels
             and not _MEDICINE_LINE.search(str(b["text"])) and not looks_like_medicine(str(b["text"]))]
    hits = [_line_hits(str(b["text"])) for _, b in items]
    found: list[tuple[int, Found]] = []
    for (idx, b), hh in zip(items, hits):                              # a printed list with an exact test: its cut-off entries are read too
        if _state(b) == "printed" and _SEP.search(str(b["text"])) and any(h.kind == "strong" for h in hh):
            have = {re.sub(r"[^a-z0-9]", "", h.as_read.casefold()) for h in hh}
            for piece in split_tests(str(b["text"])):
                if re.sub(r"[^a-z0-9]", "", piece.casefold()) in have or placed_text(piece):
                    continue
                near = _printed_near(piece)
                if near:
                    found.append((idx, Found(near[0], near[1], "printed_near")))
    for group in _groups(items):
        allh = [h for g in group for h in hits[g]]
        strong = sum(h.kind == "strong" for h in allh)
        marker = sum(h.kind == "marker" for h in allh)
        weak = sum(h.kind == "weak" for h in allh)
        near = sum(h.kind == "near" for h in allh)
        for g in group:
            for h in hits[g]:
                if h.kind == "strong":
                    f = Found(h.test, h.as_read, "exact")
                elif h.kind == "marker":
                    f = Found(h.test, h.as_read, "marker")
                elif h.kind == "weak" and strong + near + marker >= 1:
                    f = Found(h.test, h.as_read, "beside")
                elif h.kind == "near" and strong + marker + weak + (near - 1) >= 1:
                    f = Found(h.test, h.as_read, "near")
                else:
                    continue
                found.append((items[g][0], f))
    found = _by_pen_marks(found, blocks, colour)
    found.sort(key=lambda kv: kv[0])                                   # the page's reading order
    seen: set[str] = set()
    out: list[Found] = []
    for _, f in found:
        mapped = lab_mapping.lookup(f.test)                              # "vit D" and "Vitamin D" are one test
        k = (mapped.canonical if mapped else re.sub(r"[^a-z0-9]", "", f.test.casefold())).casefold()
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the model's own test entries, put right by the same rules. MEASURED on a real prescription: "Chest ECG" was read "Chest ECO" and
# "Na+ & K+ Level Test" was read "Nat & Kit Level Test"; neither was placed by the lab lists, so both showed as "outside the lab list".
# Na+ & K+ as handwriting gets read: the + comes out as t ("Nat"), the & as "-2" / "+ q" / q, the N as NP; "Kit" for "K+".
# MEASURED on a real page: "NAT & Kit Level Test", "NAT-2 Kit Level Th", "NAT + Q Kit Level Test", "NPT & K+ Level F".
_ELECTROLYTE_CORE = r"(?:s[.\s]*)?n[ap]\s*[+t]?\s*(?:(?:&|and|-?\s*2|\+\s*q|[+,/q])\s*)?k\s*[+it]{0,2}\b"
_ELECTROLYTE = re.compile(r"(?i)^\W*" + _ELECTROLYTE_CORE)
_ELECTROLYTE_IN = re.compile(r"(?i)\b" + _ELECTROLYTE_CORE)


def answer_tests(text: str) -> list[str]:
    """The tests a model's answer holds, WORD by word, however garbled the rest of the string is: ``"CBC w. NPT & K+ Level F"`` holds CBC
    and sodium-and-potassium. MEASURED on a real page: the enlarged views read CBC inside strings that differed every time
    ("CBC w. diff.", "CBC + WBC", "CBC w. NPT & K+ ..."), so comparing whole strings never found it read in two views."""
    out: list[str] = []
    for h in _line_hits(text, long_rule=False):
        if h.kind != "strong":
            continue
        mapped = lab_mapping.lookup(h.test)
        name = "Na+ & K+" if mapped and mapped.canonical.startswith("Sodium and potassium") else h.test      # "Na K" and "Na+ & K+" are one test
        if name not in out:
            out.append(name)
    if _ELECTROLYTE_IN.search(text) and "Na+ & K+" not in out:
        out.append("Na+ & K+")
    return out


def repair_piece(piece: str, evidence: bool) -> tuple[str, str] | None:
    """A test the lists do not place, put right: sodium and potassium written "Na+ & K+" (any reading of the + signs), or a word that is a
    one-letter handwriting slip of a strong abbreviation ("ECO" -> ECG) when ``evidence`` says other tests are listed beside it. The
    corrected name must be PLACED by the lists, and the note says what was read. ``None`` when nothing applies."""
    if placed_text(piece):
        return None
    if _ELECTROLYTE.match(piece):
        return "Na+ & K+", f"read as '{piece}' (sodium and potassium)"
    words = piece.split()
    if len(words) >= 2 and (len(words[-1].strip(".")) <= 3 or words[-1].casefold().strip(".") in ("test", "tests", "level")):
        trimmed = " ".join(words[:-1]).strip(" .")
        if trimmed and placed_text(trimmed):
            return trimmed, f"read as '{piece}' (the doctor's \"Test\" abbreviation dropped)"      # "Lipid Profile Tr" -> Lipid Profile
    if not evidence:
        return None
    fixed: list[str] = []
    slips: list[str] = []
    for w in piece.split():
        slip = near_miss(re.sub(r"[^A-Za-z]", "", w))
        if slip and not placed_text(w):
            fixed.append(slip)
            slips.append(slip)
        else:
            fixed.append(w)
    name = " ".join(fixed)
    if slips and placed_text(name):
        return name, f"read as '{piece}' (one letter from {slips[0]})"
    return None


def repair_investigations(payload: dict[str, Any]) -> dict[str, str]:
    """Apply ``repair_piece`` to every test the model listed (in place). Returns ``{corrected name: note}`` so the result can say what was
    read. Evidence for a slip: at least one other listed test the lists place exactly."""
    inv = payload.get("investigations")
    if not isinstance(inv, list):
        return {}

    def text_of(x: Any) -> str:
        return str(x.get("text") or "") if isinstance(x, dict) else (x if isinstance(x, str) else "")

    entries = [(i, [seg for seg in re.split(r"\s*[,;]\s*", text_of(x)) if seg.strip()]) for i, x in enumerate(inv)]     # "Nat & Kit" is one test: not cut at the &
    placed = sum(1 for _, segs in entries for seg in segs for p in split_tests(seg) if placed_text(p) or is_known_test(p))     # "CBC Test" counts: CBC is a test word
    notes: dict[str, str] = {}
    for i, segs in entries:
        new: list[str] = []
        changed = False
        for seg in segs:
            whole = repair_piece(seg, evidence=placed >= 1) if _ELECTROLYTE.match(seg) else None
            if whole:
                new.append(whole[0])
                notes[whole[0]] = whole[1]
                changed = True
                continue
            for p in split_tests(seg):
                r = repair_piece(p, evidence=placed >= 1)
                if r:
                    new.append(r[0])
                    notes[r[0]] = r[1]
                    changed = True
                else:
                    new.append(p)
        if changed:
            text = ", ".join(new)
            inv[i] = {**inv[i], "text": text} if isinstance(inv[i], dict) else text
    return notes


# ---------------------------------------------------------------------------------------------------------------------
# a LIST in which one entry is a test is a list of tests. MEASURED on a real page: the model wrote "Digital OPG, FBS, BJS CT" into the follow-up
# text and left the investigations empty ("No lab test is written on this page"). FBS is a test the lists place, so the other entries beside
# it (OPG, CT, and "BJS", which nothing places) are tests too: the placed ones by their standard name, the others as entries to check.
# capital-letter abbreviations that are not tests (dose timings, forms, titles): never entries of a test list
_NON_TEST_ABBR = frozenset("od bd tds qid sos hs pc ac stat prn tab cap inj syp mg ml iv im sc po rx dr mr mrs ms no yes ok wt ht fu rt lt rs".split())


def list_entries(text: str | None) -> list[tuple[str, str, bool]]:
    """``(name, as read, placed)`` for each entry of a list that holds at least one test the lists place EXACTLY; ``[]`` when none does. The
    words after a heading that orders tests (Adv, Inv ...) are the list. An entry the lists do not place is kept only when it is a short
    capital-letter abbreviation as read ("BJS": the shape of a test abbreviation); ordinary words (rest, diet, avoid) are not entries."""
    if not text or not text.strip():
        return []
    heading = list(_HEADING.finditer(text))
    region = text[heading[-1].end():] if heading else text
    entries: list[tuple[str, str, bool]] = []
    for piece in split_tests(region):
        words = re.findall(r"[A-Za-z0-9?]+", piece)
        i = 0
        while i < len(words):
            for n in (3, 2, 1):
                gram = " ".join(words[i:i + n])
                got = placed_text(gram) if i + n <= len(words) else None
                if got and not all(w.casefold() in _FORM_LABELS for w in gram.split()):
                    entries.append((got, gram, True))
                    i += n
                    break
            else:
                w = words[i]
                if w.isalpha() and w.isupper() and 2 <= len(w) <= 5 and w.casefold() not in _NON_TEST_ABBR and w.casefold() not in _FORM_LABELS:
                    entries.append((w, w, False))
                i += 1
    if not any(placed for _n, _a, placed in entries):
        return []
    out: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for name, as_read, placed in entries:
        key = re.sub(r"[^a-z0-9]", "", name.casefold())
        if key not in seen:
            seen.add(key)
            out.append((name, as_read, placed))
    return out


def drop_composites(investigations: list[Any], entries: list[tuple[str, str, bool]]) -> list[Any]:
    """The model's own entry that is only the words of entries already listed one by one ("BJS CT" beside BJS and CT) is dropped: it says
    nothing the entries do not say better. An entry the lists place is never dropped."""
    covered = {as_read.casefold() for _n, as_read, _p in entries} | {n.casefold() for n, _a, _p in entries}
    out = []
    for item in investigations:
        text = str(item.get("text") or "") if isinstance(item, dict) else str(item or "")
        words = re.findall(r"[A-Za-z0-9?]+", text)
        if len(words) >= 2 and not placed_text(text) and all(w.casefold() in covered for w in words):
            continue
        out.append(item)
    return out
