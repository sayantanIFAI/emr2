"""The doctor's name and the clinic's name from the printed header, when the model left them empty.

MEASURED on a real page: "Dr. Kumar Sourav" and "KS HEALTHCARE" were clearly printed at the top and the page text held them, but the
model answered with no prescriber at all, so the screen said "Nothing readable on the page". The text the readers produced is the
evidence here: the first "Dr ..." line in the top of the first page, and the first line in the top with exactly ONE organisation
word (clinic, hospital, healthcare ...). Only an empty field is filled; a value the model gave is never replaced."""
from __future__ import annotations

import re
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)

_HEADER_FRACTION = 0.35                                  # the top of the page: where a letterhead is
_DR = re.compile(r"(Prof\.?\s*)?\(?\s*Dr\s*\.?\s*\)?\s*(?=[A-Z])")
_NAME_WORD = re.compile(r"[A-Z][a-z]{1,}\.?|[A-Z]\.")      # Kumar, Sourav, K.  (not MBBS, not MDPAT)
_ORG_WORD = re.compile(r"(?i)hospitals?|clinics?|healthcare|health\s*care|polyclinic|diagnostics?|nursing\s*home|medical\s*cent(?:re|er)|"
                       r"centre|center|institute|laborator(?:y|ies)|pharmacy")
_NOT_A_NAME = frozenset("Mob Ph Phone Reg Regd Contact Contacts Tel Consultant Senior Clinic Hospital Mbbs Timing Timings".split())
_CUT = re.compile(r"(?i)\b(?:mob|ph|phone|contacts?|tel|reg|regd)\b\.?:?.*$|[+\d].*$")      # the phone / registration part of a line


def doctor_name(line: str) -> str | None:
    """``"Dr.Kumar Sourav MBBS ..."`` -> ``"Dr. Kumar Sourav"``: the words that look like a name after "Dr", up to four, stopping at the
    first qualification (an all-capital word)."""
    t = (line or "").replace("（", "(").replace("）", ")")
    m = _DR.search(t)
    if not m:
        return None
    words: list[str] = []
    for w in t[m.end():].split():
        w = w.strip(",;:|")
        if not _NAME_WORD.fullmatch(w) or w.rstrip(".") in _NOT_A_NAME:
            break
        words.append(w)
        if len(words) == 4:
            break
    if not words:
        return None
    return ("Prof. Dr. " if m.group(1) else "Dr. ") + " ".join(words)


def clinic_name(line: str) -> str | None:
    """The clinic's name when the line holds exactly ONE organisation word: ``"Core Clinic Mob.98302..."`` -> ``"Core Clinic"``,
    ``"KSHEALTHCARE"`` -> ``"KS HEALTHCARE"``. A list of clinics (``|``), a line with several organisation words, and a line with
    none give None."""
    if "|" in (line or ""):
        return None
    t = _CUT.sub("", line or "").strip()
    found = list(_ORG_WORD.finditer(t))
    if len(found) == 2 and found[1].start() - found[0].end() <= 12:
        found = found[1:]                                   # "LIFE CENTRE POLYCLINIC": two organisation words that belong to one name
    if len(found) != 1:
        return None
    m = found[0]
    head = t[:m.end()]
    head = head[:m.start()] + (" " if m.start() > 0 and head[m.start() - 1].isalpha() else "") + head[m.start():]    # "KSHEALTHCARE" -> "KS HEALTHCARE"
    words = head.split()
    # a clinic's name is a few plain words ending at the organisation word; a long garbled line that happens to contain one
    # ("Dr.Kumar Sourav MBBSPATMDPATalMine ConsuPyin/Nerndogist KSHEALTHCARE /") is not a name (MEASURED on a real page)
    if not 2 <= len(words) <= 4 or not all(re.fullmatch(r"[A-Za-z&.'\-]+", w) for w in words):
        return None
    name = " ".join(words)
    return name if sum(c.isalpha() for c in name) >= 5 else None


def fill(payload: dict[str, Any], blocks: list[dict[str, Any]]) -> list[str]:
    """Fill ``prescriber.name`` and ``prescriber.clinic.name`` from the first page's header text when they are empty. Returns what was
    filled (for the log). ``blocks`` are the first page's text blocks (with ``bbox``)."""
    pres = payload.get("prescriber")
    if pres is not None and not isinstance(pres, dict):
        return []
    pres = dict(pres or {})
    clinic = pres.get("clinic") if isinstance(pres.get("clinic"), dict) else {}
    boxes = []
    for b in blocks or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in b["bbox"][:4])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        boxes.append((y0, y1, str(b.get("text") or "")))
    if not boxes:
        return []
    height = max(y1 for _, y1, _ in boxes)
    header = [t for y0, _y1, t in boxes if y0 <= _HEADER_FRACTION * height and t.strip()]
    filled: list[str] = []
    if not (pres.get("name") or "").strip():
        name = next((n for n in (doctor_name(t) for t in header) if n), None)
        if name:
            pres["name"] = name
            filled.append("doctor")
    if not (clinic.get("name") or "").strip():
        org = next((n for n in (clinic_name(t) for t in header) if n), None)
        if org:
            clinic = {**clinic, "name": org}
            pres["clinic"] = clinic
            filled.append("clinic")
    if filled:
        payload["prescriber"] = pres
        log.info("header_filled", filled=filled)
    return filled


# ---------------------------------------------------------------------------------------------------------------------
# the PATIENT in the header: some clinics (Apollo Sugar Clinics) print the patient's name with the age and sex in the header and
# leave the "Patient Name" space on the form empty. A name followed by "(40 Y / MALE)" is the patient, not the doctor.
_AGE_SEX = re.compile(r"(?P<name>[A-Za-z][A-Za-z .'\-]{2,40}?)\s*\(\s*(?P<age>\d{1,3})\s*(?:Y|Yr|Yrs|Years?)?\s*/\s*(?P<sex>MALE|FEMALE|M|F)\s*\)", re.I)
_PHONE = re.compile(r"(?i)\b(?:M|Mob|Mobile|Ph|Phone)\b\s*[:.\-]?\s*(?:\+?91[\s\-]?)?([6-9]\d{9})\b")
_TITLE = re.compile(r"^\s*(?:mr|mrs|ms|miss|master|baby|smt|shri|sri)\b\.?\s*", re.I)
_EMPTY = frozenset(("", "null", "none", "n/a", "na"))


def _empty(v: Any) -> bool:
    return v is None or str(v).strip().casefold() in _EMPTY


def patient_in_header(blocks: list[dict[str, Any]]) -> dict[str, str] | None:
    """The patient's name, age, sex (and the mobile number printed beside them) from the top of the first page, or None. The line must
    hold a name followed by ``(40 Y / MALE)``; a name that starts with Dr or holds an organisation word is not a patient. The phone is
    taken only from the same line or a line right beside it, never from the doctor's own header lines further away."""
    rows: list[tuple[float, float, float, str]] = []
    for b in blocks or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in b["bbox"][:4])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        rows.append((y0, y1, x0, str(b.get("text") or "")))
    if not rows:
        return None
    top = max(y1 for _, y1, _, _ in rows) * _HEADER_FRACTION
    for y0, y1, _x0, text in rows:
        if y0 > top:
            continue
        m = _AGE_SEX.search(text)
        if not m:
            continue
        name = _TITLE.sub("", m.group("name")).strip(" .-'")
        if not name or re.match(r"(?i)dr\b", name) or _ORG_WORD.search(name) or len(name.split()) > 4:
            continue
        if name.isupper():
            name = name.title()
        out = {"name": name, "age_text": f"{int(m.group('age'))} Y", "sex": "F" if m.group("sex").upper().startswith("F") else "M"}
        near = max(y1 - y0, 20.0) * 2.5
        for ny0, ny1, _nx0, ntext in rows:
            if abs((ny0 + ny1) / 2 - (y0 + y1) / 2) <= near:
                p = _PHONE.search(ntext)
                if p:
                    out["phone"] = p.group(1)
                    break
        return out
    return None


def fill_patient(payload: dict[str, Any], blocks: list[dict[str, Any]]) -> list[str]:
    """Fill the patient's empty name, age and sex from the header (``patient_in_header``). A value the model gave is never replaced;
    "null" written as text counts as empty. Returns what was filled. The PHONE is never taken from the page: the patient's mobile number is the
    one typed at upload (MEASURED: a letterhead's WhatsApp number was shown as the patient's phone)."""
    found = patient_in_header(blocks)
    if not found:
        return []
    pat = payload.get("patient")
    if pat is not None and not isinstance(pat, dict):
        return []
    pat = dict(pat or {})
    filled = []
    for key in ("name", "age_text", "sex"):
        if key in found and _empty(pat.get(key)):
            pat[key] = found[key]
            filled.append(key)
    if filled:
        payload["patient"] = pat
        log.info("header_patient_filled", filled=filled)
    return filled
