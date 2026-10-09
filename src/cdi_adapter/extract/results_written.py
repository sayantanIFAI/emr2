r"""A value written next to a test name means the test is ALREADY DONE: it is a result, not a test to be done.

The owner's thumb rule: if a digit, a number, +ve / -ve, positive, negative, normal or the like is written with a test, it is a result and is
never captured as a test to be done, from any prescription. MEASURED on a real page: "Hb-11.9", "Platelet-2.25 lac", "TLC-6900", "CRP-0.02 (<0.8)",
"Creat-0.35", "SGPT-28/22", "LDL 94", "TG-133" and a bone-density line "DXA L1-L4 -0.8 Femur -1.4" were all listed as tests; the only tests the
doctor ordered ("Review after 3 months: CBC, CRP, Creatinine, SGPT, SGOT") were written plainly at the foot of the page.

* ``clean_entry`` takes the results out of one written entry ("CBC, CRP, Hb-11.9" -> "CBC, CRP"; "Hb-11.9" -> nothing);
* ``page_result_reason`` says why a name is a result when EVERY place it is written on the page has a value after it ("CRP" written once as
  "CRP-0.02" and once in the ordered list is still a test: the plain one counts).
A digit that is part of a test's own name (B12, T3, FT4, HbA1c, 25 OH vitamin D) is not a value, and a number that is a preparation or a duration
(FBS 12 hrs, review after 2 weeks) is not a result."""
from __future__ import annotations

import re
from typing import Any

# words after which a bare number (no hyphen) is read as a value: the analytes doctors copy results of
_ANALYTES = frozenset("hb hgb haemoglobin hemoglobin tlc wbc rbc dlc plt platelet platelets esr crp ldl hdl tg tgl chol cholesterol triglyceride sgpt sgot alt ast "
                      "alp ggt creat creatinine urea bun uric tsh ft3 ft4 t3 t4 fbs ppbs rbs ppg fpg hba1c hbaic hba1 na k sodium potassium ca calcium "
                      "bilirubin albumin protein glucose sugar inr pt aptt ferritin iron vitamin b12 psa ilc fms ish dxa dexa femur spine bmd".split())
_NUM = r"[<>~≤≥]?\s*\d+(?:[.,·]\d+)?"
_UNIT = (r"(?:\s*(?:%|lacs?|lakhs?|mg/dl|g/dl|gm/dl|u/l|iu/l|mmol/l|meq/l|ng/ml|pg/ml|ug/dl|µg/dl|fl|pg|/cumm|/cu\s*mm|mm/hr|cells?)\b)?"
         r"(?:\s*\([^)]{0,14}\))?")
# NAME - 11.9 / NAME : 0.02 / NAME = 9100 / NAME(-28/22): a hyphen, colon or equals sign and then a number
_HYPHEN = re.compile(r"(?P<w>[A-Za-z][A-Za-z0-9+.]{0,15})\s*\(?\s*[-–:=]\s*\(?\s*(?:" + _NUM + r")(?:\s*[/\-]\s*\d+(?:[.,·]\d+)?)?" + _UNIT, re.I)
# a known analyte then a bare number: LDL 94 / LDL94 / Hb 11.9 (not "FBS 12 hrs": a duration or a preparation)
_BARE = re.compile(r"(?P<w>ft3|ft4|t3|t4|b12|hba1c|hbaic|hba1|[A-Za-z]+)" + r"\s*(?P<n>" + _NUM + r")(?![\d.,\u00b7])" + _UNIT +
                   r"(?!\s*(?:hrs?|hours?|days?|d\b|wks?|weeks?|months?|mo\b|min|times|x\b))", re.I)
# NAME +ve / -ve / positive / negative / normal / WNL / reactive / nil
_WORD = re.compile(r"(?P<w>[A-Za-z][A-Za-z0-9+./]{0,15})\s*[-–:=]?\s*\(?\s*(?:\+\s*ve|[-–]\s*ve|positive|negative|reactive|non[- ]?reactive|normal|wnl|nad|"
                   r"not\s+detected|detected|nil)\b", re.I)
_TRIMS = " \t,;:|/·."


def _analyte_like(w: str) -> bool:
    """A word that can be the name of something measured: a known analyte or test word, a short abbreviation (Hb, TG, Fg), or a capitalised one of up
    to six letters (SHOT, ISH, FMS, ILC). Ordinary words ("smoking - 2 weeks", "diet - 1600") are not."""
    from .test_names import is_known_test

    lw = w.lower().strip(".")
    return lw in _ANALYTES or is_known_test(w) or len(lw) <= 3 or (w.isupper() and len(w) <= 6)


def _spans(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, name word)`` of every value written with a name in ``text``."""
    out: list[tuple[int, int, str]] = []
    for m in _HYPHEN.finditer(text):
        w = m.group("w")
        if re.search(r"[A-Za-z]", w) and not re.fullmatch(r"[A-Za-z]\d?", w) and _analyte_like(w) or w.lower() in _ANALYTES:
            out.append((m.start(), m.end(), w))
    for m in _BARE.finditer(text):
        w = m.group("w").lower()
        if w in _ANALYTES and not any(s <= m.start() < e for s, e, _ in out):
            out.append((m.start(), m.end(), m.group("w")))
    for m in _WORD.finditer(text):
        if not any(s <= m.start() < e for s, e, _ in out) and _analyte_like(m.group("w")):
            out.append((m.start(), m.end(), m.group("w")))
    return sorted(out)


def clean_entry(text: str | None) -> tuple[str, list[str]]:
    """``(the entry without its results, the results taken out)``. An entry that is nothing but results comes back empty. A single-piece entry that
    held a result and little else ("DXA L1-L4 -0.8 Femur -1.4") is a result line as a whole and comes back empty."""
    t = str(text or "")
    spans = _spans(t)
    if not spans:
        return t, []
    removed = [t[s:e].strip() for s, e, _ in spans]
    out, last = [], 0
    for s, e, _ in spans:
        out.append(t[last:s])
        last = e
    out.append(t[last:])
    rest = re.sub(r"\s+", " ", " ".join(out)).strip(_TRIMS + " ")
    rest = re.sub(r"\s*([,;|])\s*(?:[,;|]\s*)+", r"\1 ", rest).strip(_TRIMS + " ")
    words = re.findall(r"[A-Za-z]{2,}", rest)
    if not words or (len(words) <= 2 and not re.search(r"[,;|]", t)):
        return "", removed
    return rest, removed


def is_result_entry(text: str | None) -> bool:
    """True when the whole written entry is a result (nothing is left once the values are taken out)."""
    clean, removed = clean_entry(text)
    return bool(removed) and not clean


def _after_has_value(after: str, next_piece: str = "") -> bool:
    """Does what follows a test name (in its line, or the piece written right after it) start with a value?"""
    a = after.lstrip(" \t")
    probes = [a, a[1:].lstrip() if a[:1] in "-–:=(" else a, next_piece.lstrip()]
    pat = re.compile(r"^\(?\s*(?:" + _NUM + r"|\+\s*ve|[-–]\s*ve|positive|negative|reactive|non[- ]?reactive|normal|wnl|nad|detected|not\s+detected|nil)\b", re.I)
    for p in probes:
        if pat.match(p) and not re.match(r"^\(?\s*\d+(?:[.,]\d+)?\s*(?:hrs?|hours?|days?|d\b|wks?|weeks?|months?|mo\b|min)\b", p, re.I):
            return True
    return False


def page_result_reason(key: str, blocks: list[dict[str, Any]] | None) -> str | None:
    """The reason when EVERY occurrence of the name on the page has a value after it (a result already written), else None. A name that is written
    once without a value (the doctor's ordered list) is a test whatever else the page says. A name found nowhere in the page text is left alone."""
    k = re.sub(r"\s+", " ", (key or "").strip())
    if len(re.sub(r"[^A-Za-z0-9]", "", k)) < 2 or not blocks:
        return None
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(k).replace(r"\ ", r"[\s.\-]*") + r"(?![A-Za-z])", re.I)
    seen, valued, example = 0, 0, ""
    for i, b in enumerate(blocks):
        text = str(b.get("text") or "")
        for m in pat.finditer(text):
            seen += 1
            nxt = str(blocks[i + 1].get("text") or "") if i + 1 < len(blocks) and not text[m.end():].strip() else ""
            if _after_has_value(text[m.end():], nxt):
                valued += 1
                example = example or text[m.start():min(len(text), m.end() + 14)].strip()
    if seen and valued == seen:
        return f"a result is already written next to it ('{example}'): not a test to be done"
    return None
