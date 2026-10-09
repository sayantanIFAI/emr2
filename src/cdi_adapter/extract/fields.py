"""Deterministic checks on what the model read for a prescription (EX-S1, S2, S4, S5; follow-up).

The model fills the form; this module never changes a value, it judges each one with plain rules
and says WHY: ``checked`` (the format is right and the value is on the page as written),
``needs_check`` (a person must look), ``absent`` (not written: normal, no review) or
``not_gated`` (a visual judgement nothing here can verify). It also drops what is not on the page
at all (a preparation note the model made up) and keeps those drops visible in ``retracted``.

Everything here is a pure function of the extraction payload and the OCR block texts, so the
same document gives the same answer every time, and none of it needs a database or a GPU.

The rules that matter for patient safety:

* a number is judged on the LITERAL digits on the page: no look-alike repair (a "1Z hrs" that the
  model read as 12 is a number nobody can vouch for, so it goes to review);
* nothing is added: a preparation or follow-up that is not on the page is retracted, and the system
  never contributes a "standard" fasting time of its own;
* sex and age are never inferred: a value not literally written goes to review;
* a context link between a test and a diagnosis says what it rests on (same line, or only the same
  page), never "ordered because".
"""
from __future__ import annotations

import difflib
import re
from datetime import date
from typing import Any

from ..config import settings

CHECKED, NEEDS_CHECK, ABSENT, NOT_GATED = "checked", "needs_check", "absent", "not_gated"
PREP_TYPES = ("fasting", "timing", "diet", "medicine_hold", "sample_collection", "bring_documents", "other")
_NOT_ON_PAGE = "not found on the page as written"


def check(value: Any, status: str, reason: str | None = None) -> dict[str, Any]:
    return {"value": value, "status": status, "reason": reason}


def absent() -> dict[str, Any]:
    return check(None, ABSENT)


# ------------------------------------------------------------------ the page text


def blocks_by_label(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """'b1' -> block, in the order the extraction prompt numbered them."""
    return {f"b{i}": b for i, b in enumerate(blocks, start=1)}


def page_text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(str(b.get("text") or "") for b in blocks)


def _compact(s: str) -> str:
    return re.sub(r"[\s\-.()/,:]", "", s).casefold()


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.casefold()) if len(w) >= 3 or w.isdigit()]


def _on_page(value: str, page: str, *, ratio: float = 0.82) -> bool:
    """Every word of ``value`` is on the page as written (allowing small OCR noise in a word)."""
    page_words = set(re.findall(r"[A-Za-z0-9]+", page.casefold()))
    for w in _words(value):
        if w in page_words:
            continue
        if w.isdigit() or not any(difflib.SequenceMatcher(None, w, p).ratio() >= ratio for p in page_words):
            return False
    return True


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# ------------------------------------------------------------------ patient (EX-S1)


def check_text(value: Any, page: str) -> dict[str, Any]:
    """A name / address / clinic / designation as written: it must be on the page."""
    v = value.strip() if isinstance(value, str) else ""
    if not v:
        return absent()
    return check(v, CHECKED) if _on_page(v, page) else check(v, NEEDS_CHECK, _NOT_ON_PAGE)


def check_phone(raw: Any, page: str, *, kind: str = "mobile") -> dict[str, Any]:
    v = raw.strip() if isinstance(raw, str) else ""
    if not v:
        return absent()
    d = _digits(v)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    if kind == "mobile" and not (len(d) == 10 and d[0] in "6789"):
        return check(v, NEEDS_CHECK, f"a mobile number has 10 digits starting 6-9; this has {len(d)}")
    if kind == "any" and not 8 <= len(d) <= 12:
        return check(v, NEEDS_CHECK, f"a phone number has 8-12 digits; this has {len(d)}")
    if d not in _compact(page):
        return check(v, NEEDS_CHECK, "the digits are not clearly readable on the page")
    return check(v, CHECKED)


def check_abha(raw: Any, page: str) -> dict[str, Any]:
    v = raw.strip() if isinstance(raw, str) else ""
    if not v:
        return absent()
    if re.fullmatch(r"[A-Za-z0-9._]{3,}@(abdm|sbx)", v):                       # an ABHA address
        return check(v, CHECKED) if v.casefold() in page.casefold() else check(v, NEEDS_CHECK, _NOT_ON_PAGE)
    d = _digits(v)
    if len(d) != 14:
        return check(v, NEEDS_CHECK, f"an ABHA number has 14 digits; this has {len(d)}")
    if d not in _compact(page):
        return check(v, NEEDS_CHECK, "the digits are not clearly readable on the page")
    return check(f"{d[:2]}-{d[2:6]}-{d[6:10]}-{d[10:]}", CHECKED)


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_date_text(s: str) -> date | None:
    """DD-MM-YYYY, DD/MM/YY, DD-Mon-YYYY, YYYY-MM-DD, as written; ``None`` if it is not a date."""
    s = s.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    parts: tuple[int, int, int] | None = None
    if m:
        parts = (int(m[1]), int(m[2]), int(m[3]))
    else:
        m = re.fullmatch(r"(\d{1,2})[-/. ]([A-Za-z]{3})[a-z]*[-/. ](\d{2,4})", s)
        if m and m[2].casefold()[:3] in _MONTHS:
            y = int(m[3])
            parts = (y + 2000 if y < 100 else y, _MONTHS[m[2].casefold()[:3]], int(m[1]))
        else:
            m = re.fullmatch(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})", s)
            if m:
                y = int(m[3])
                parts = (y + 2000 if y < 100 else y, int(m[2]), int(m[1]))
    if parts is None:
        return None
    try:
        return date(*parts)
    except ValueError:
        return None


_AGE_SLASH = re.compile(r"^\s*(\d{1,3})\s*(y|yr|yrs|years?)?\s*[/|\\]\s*([A-Za-z0-9|]{1,6})?\s*$", re.IGNORECASE)


def normalise_age(text: Any) -> tuple[Any, str | None]:
    """``"73 yrs / Female"`` read as ``"74/1"`` (MEASURED on a real page: the bar between age and sex was taken for a slash and the sex for a digit)
    is an age of 74 years: the number before the slash, with its unit, and the sex only when the tail is a clear M / F / Male / Female. A text
    that is not of this shape is returned unchanged."""
    m = _AGE_SLASH.match(str(text or ""))
    if not m:
        return text, None
    tail = (m.group(3) or "").strip().casefold()
    sex = "F" if tail in ("f", "female") else "M" if tail in ("m", "male") else None
    return f"{int(m.group(1))} {m.group(2) or 'yrs'}", sex


def _age_years(text: Any) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,3})\s*(?:y|yr|yrs|year|years)?\s*", str(text or ""), re.IGNORECASE)
    if m:
        return int(m[1])
    if re.search(r"\d+\s*(m|mo|month|months|d|day|days|w|wk|week|weeks)\b", str(text or ""), re.IGNORECASE):
        return 0
    return None


def check_age_and_dob(age_text: Any, dob_raw: Any, page: str, today: date) -> tuple[dict, dict]:
    """Age as written and date of birth: each plausible, each on the page, and not contradicting
    each other (more than a year apart goes to review for BOTH)."""
    age_c = check_text(age_text, page) if age_text else absent()
    dob_c = absent()
    dob = None
    if isinstance(dob_raw, str) and dob_raw.strip():
        dob = parse_date_text(dob_raw)
        nums = re.findall(r"\d+", dob_raw)
        if dob is None:
            dob_c = check(dob_raw, NEEDS_CHECK, "not a readable date")
        elif dob > today or (today.year - dob.year) > 120:
            dob_c = check(dob_raw, NEEDS_CHECK, "this date of birth is not plausible")
        elif any(n not in re.findall(r"\d+", page) for n in nums if len(n) >= 2):
            dob_c = check(dob_raw, NEEDS_CHECK, "the digits are not clearly readable on the page")
        else:
            dob_c = check(dob.isoformat(), CHECKED)
    yrs = _age_years(age_text) if age_text else None
    if age_text and yrs is None:
        age_c = check(age_c["value"], NEEDS_CHECK, "the age is not a number of years, months or days")
    elif yrs is not None and yrs > 120:
        age_c = check(age_c["value"], NEEDS_CHECK, "this age is not plausible")
    if dob is not None and yrs is not None and dob_c["status"] == CHECKED:
        derived = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if abs(derived - yrs) > 1:
            why = f"age {yrs} does not match a date of birth in {dob.year} (that is age {derived})"
            age_c, dob_c = check(age_c["value"], NEEDS_CHECK, why), check(dob_c["value"], NEEDS_CHECK, why)
    return age_c, dob_c


def check_sex(value: Any, page: str) -> dict[str, Any]:
    """Sex is copied, never inferred: it must be literally written (M / F / Male / Female / Other)."""
    v = value.strip() if isinstance(value, str) else ""
    if not v:
        return absent()
    word = {"m": r"(m|male)", "male": r"(m|male)", "f": r"(f|female)", "female": r"(f|female)",
            "o": r"(o|other)", "other": r"(o|other)"}.get(v.casefold())
    if word and re.search(rf"(?<![A-Za-z]){word}(?![A-Za-z])", page, re.IGNORECASE):
        return check(v, CHECKED)
    return check(v, NEEDS_CHECK, "sex is not written as such on the page (never inferred from a name)")


def check_patient(patient: dict[str, Any], page: str, today: date) -> dict[str, dict[str, Any]]:
    age_c, dob_c = check_age_and_dob(patient.get("age_text"), patient.get("dob"), page, today)
    return {
        "name": check_text(patient.get("name"), page),
        "mrn": check_text(patient.get("mrn"), page),
        "age_text": age_c, "dob": dob_c,
        "sex": check_sex(patient.get("sex"), page),
        "phone": check_phone(patient.get("phone"), page),
        "address": check_text(patient.get("address"), page),
        "abha_id": check_abha(patient.get("abha_id"), page) if settings.abha_enabled else absent(),
    }


# ------------------------------------------------------------------ doctor (EX-S2)


def check_registration(raw: Any, page: str) -> dict[str, Any]:
    v = raw.strip() if isinstance(raw, str) else ""
    if not v:
        return absent()
    return check(v, CHECKED) if _compact(v) in _compact(page) else check(v, NEEDS_CHECK, _NOT_ON_PAGE)


def _visual(v: Any) -> dict[str, Any]:
    if v is None:
        return absent()
    return check(bool(v), NOT_GATED, "a visual judgement by the model; nothing here can verify it")


def check_doctor(prescriber: dict[str, Any], page: str) -> dict[str, Any]:
    clinic = _dict(prescriber.get("clinic"))
    return {
        "name": check_text(prescriber.get("name"), page),
        "reg_no": check_registration(prescriber.get("reg_no"), page),
        "department": check_text(prescriber.get("department"), page),
        "designation": check_text(prescriber.get("designation"), page),
        "qualification": check_text(prescriber.get("qualification"), page),
        "clinic": {"name": check_text(clinic.get("name"), page),
                   "address": check_text(clinic.get("address"), page),
                   "phone": check_phone(clinic.get("phone"), page, kind="any")},
        "stamp_present": _visual(prescriber.get("stamp_present")),
        "signature_present": _visual(prescriber.get("signature_present")),
    }


# ------------------------------------------------------------------ lab preparation (EX-S4)

_FASTING_HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", re.IGNORECASE)
_TIMING_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|min(?:ute)?s?)\s*(?:after|post)", re.IGNORECASE)
_ALL_TESTS = re.compile(r"\b(all|every|each)\b[^,;]{0,20}\b(tests?|investigations?)\b|\ball tests\b", re.IGNORECASE)
_RESULT_LOOK = re.compile(r"\b(fbs|ppbs|rbs|fbg|sugar|glucose|hba1c|fasting\s+sugar)\b[^0-9]{0,15}\d+(?:\.\d+)?\s*"
                          r"(?:mg/dl|mg|mmol|%)", re.IGNORECASE)


def _best_match(text: str, blocks: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, float]:
    """The line of the page the words came from, the stretch of that line they match, and how well.
    A note often shares a line with other words ("HbA1c, FBS - fasting 12 hrs"), so the words are
    compared with every run of about the same number of words, not with the whole line."""
    toks_t = text.split()
    t = " ".join(toks_t).casefold()
    best: tuple[dict[str, Any] | None, str, float] = (None, "", 0.0)
    if not t:
        return best
    n = len(toks_t)
    for b in blocks:
        raw = str(b.get("text") or "")
        low = " ".join(raw.split()).casefold()
        if not low:
            continue
        if t in low:
            i = low.index(t)
            return b, " ".join(raw.split())[i:i + len(t)], 1.0
        toks = raw.split()
        for size in sorted({max(1, n - 1), n, n + 1}):
            for i in range(max(1, len(toks) - size + 1)):
                window = " ".join(toks[i:i + size])
                s = difflib.SequenceMatcher(None, t, window.casefold()).ratio()
                if s > best[2]:
                    best = (b, window, s)
    return best


def _literal_numbers(s: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


def parse_preparation_words(text: str) -> tuple[str | None, float | None, str | None]:
    """(type, value, unit) read from the words themselves: never from the model's say-so."""
    low = text.casefold()
    m = _TIMING_AFTER.search(text)
    if m and re.search(r"meal|food|lunch|dinner|breakfast|eating", low):
        return "timing", float(m[1]), "h" if m[2].lower().startswith("h") else "min"
    if re.search(r"\bfasting\b|empty stomach|overnight|nil by mouth|npo|no food", low):
        m = _FASTING_HOURS.search(text)
        return "fasting", (float(m[1]) if m else None), ("h" if m else None)
    if re.search(r"\b(stop|hold|avoid|withhold|do not take|skip)\b.*\b(biotin|metformin|medicine|tablet|insulin|aspirin|drug)", low):
        return "medicine_hold", None, None
    if re.search(r"first.?morning|early morning|midstream|clean catch|24.?h(?:ou)?r? urine|collect", low):
        return "sample_collection", None, None
    if re.search(r"bring|carry|previous report|old report|earlier report", low):
        return "bring_documents", None, None
    if re.search(r"\b(morning|before breakfast|evening|afternoon)\b", low):
        return "timing", None, None
    if re.search(r"\b(diet|avoid fatty|low salt|no alcohol|alcohol)\b", low):
        return "diet", None, None
    return None, None, None


def _resolve_targets(applies: Any, text: str, tests: list[str]) -> tuple[list[str] | str, str | None]:
    """The tests a note applies to: ``["all"]``, canonical test texts, or ``"unclear"``."""
    raw = [a for a in applies if isinstance(a, str) and a.strip()] if isinstance(applies, list) else []
    if _ALL_TESTS.search(text) or any(a.casefold() in ("all", "all tests") for a in raw):
        return ["all"], None
    found: list[str] = []
    for a in raw:
        hit = max(tests, key=lambda t: difflib.SequenceMatcher(None, a.casefold(), t.casefold()).ratio(), default=None)
        if hit and difflib.SequenceMatcher(None, a.casefold(), hit.casefold()).ratio() >= 0.8 and hit not in found:
            found.append(hit)
    if not found:
        mentioned = [t for t in tests if t and re.search(rf"\b{re.escape(t.casefold())}\b", text.casefold())]
        found = mentioned
    if not found:
        return "unclear", "it is not clear which test this note belongs to"
    return found, None


def check_preparation(items: Any, tests: list[str], blocks: list[dict[str, Any]]
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(kept, retracted)``. Only what is written is kept; a model-made-up note is retracted."""
    kept: list[dict[str, Any]] = []
    gone: list[dict[str, Any]] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        words = str(it.get("text") or "").strip()
        if not words:
            continue
        block, window, score = _best_match(words, blocks)
        if block is None or score < 0.8:
            gone.append({"text": words, "reason": "retracted: not on the page, nothing is added that is not written"})
            continue
        line = str(block.get("text") or "")
        if _RESULT_LOOK.search(line):
            gone.append({"text": words, "reason": "retracted: this line is a result value, not a preparation instruction"})
            continue
        ptype, value, unit = parse_preparation_words(words)
        problems: list[str] = []
        if ptype is None:
            ptype = it.get("type") if it.get("type") in PREP_TYPES else None
            problems.append("the kind of preparation is not recognised")
        model_value = it.get("value")
        if value is None and isinstance(model_value, (int, float)):
            value, unit = float(model_value), it.get("unit")
        if value is not None:
            lit = _literal_numbers(window)
            if f"{value:g}" not in lit:                        # judged on the digits actually written
                problems.append("the number is not clearly readable on the page")
        if isinstance(model_value, (int, float)) and value is not None and abs(float(model_value) - value) > 1e-9:
            problems.append("the value differs from the written words")
        targets, why = _resolve_targets(it.get("applies_to"), words, tests)
        if why:
            problems.append(why)
        kept.append({"type": ptype, "value": value, "unit": unit, "text": words, "applies_to": targets,
                     "status": NEEDS_CHECK if problems else CHECKED, "reason": "; ".join(problems) or None,
                     "evidence": it.get("evidence") or []})
    return kept, gone


# ------------------------------------------------------------------ follow-up (the doctor's "come back after ...")

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"(day|days|week|weeks|wk|wks|month|months|mo|year|years)"
_FU_RANGE = re.compile(rf"{_NUM}\s*(?:-|to)\s*{_NUM}\s*{_UNIT}\b", re.IGNORECASE)
_FU_AFTER = re.compile(rf"(?:after|in|within|next|following)\s+{_NUM}\s*{_UNIT}\b|{_NUM}\s*{_UNIT}\s*(?:later|after)", re.IGNORECASE)
_FU_SOS = re.compile(r"\b(sos|if needed|when required|if symptoms|if pain persists|as needed)\b", re.IGNORECASE)
_FU_DATE = re.compile(r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/. ][A-Za-z]{3}[a-z]*[-/. ]\d{2,4})")
_UNIT_NORM = {"day": "days", "days": "days", "week": "weeks", "weeks": "weeks", "wk": "weeks", "wks": "weeks",
              "month": "months", "months": "months", "mo": "months", "year": "years", "years": "years"}


# a "next date" written on the page: "(Next dose: April 2026)", "Next visit 12/06/26", "Next appointment - June 2026"
_NEXT = re.compile(r"\bnext\s+(?:date|dose|visit|appointment|appt|review|follow[\s-]?up|due)\b[\s:\-]*([^\n]{3,60})", re.I)
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
_NEXT_MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s'\u2019,?]*((?:19|20)\d{2})\b", re.I)


def find_next_date(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The "next date / dose / visit / appointment" the doctor wrote, from the lines the readers produced: ``{text, iso}`` where ``iso``
    is ``YYYY-MM-DD`` (a day written) or ``YYYY-MM`` (a month and year written), or None when the words give no date. Nothing is
    worked out: a line with no readable date is not a next date."""
    for b in blocks or []:
        line = str(b.get("text") or "")
        m = _NEXT.search(line)
        if not m:
            continue
        tail = m.group(1)
        iso = None
        md = _FU_DATE.search(tail)
        d = parse_date_text(md[1]) if md else None
        if d:
            iso = d.isoformat()
        else:
            mm = _NEXT_MONTH.search(tail)
            if mm:
                iso = f"{int(mm[2]):04d}-{_MONTHS[mm[1].lower()[:3]]:02d}"
        if iso is None:
            continue
        text = re.sub(r"\s+", " ", re.sub(r"[?()\[\]]+", " ", m.group(0))).strip(" :-")
        return {"text": text, "iso": iso}
    return None


# printed form text that is not the doctor's instruction (a letterhead footer, a booking line)
_FU_PRINTED = re.compile(r"\bbring\b.{0,30}\bprescription\b|\bfor\s+appointment\b.{0,12}\bcall\b|\bappointment\s+call\b", re.I)


def parse_follow_up(text: Any, page: str) -> dict[str, Any]:
    """The follow-up as written plus a structured reading of it. Nothing is guessed: no interval is
    produced unless the words give one, and the words must be on the page."""
    t = text.strip() if isinstance(text, str) else ""
    base: dict[str, Any] = {"text": t or None, "kind": None, "interval_value": None, "interval_unit": None,
                            "interval_value_max": None, "date": None}
    if t and _FU_PRINTED.search(t):
        return {**absent(), "detail": {**base, "text": None}}          # printed form text: nothing written for a visit
    if not t:
        return {**absent(), "detail": base}
    if not _on_page(t, page):
        return {**check(t, NEEDS_CHECK, _NOT_ON_PAGE), "detail": base}
    m = _FU_RANGE.search(t)
    if m:
        base.update(kind="interval", interval_value=float(m[1]), interval_value_max=float(m[2]),
                    interval_unit=_UNIT_NORM[m[3].lower()])
    else:
        m = _FU_AFTER.search(t)
        if m:
            g = [x for x in m.groups() if x]
            base.update(kind="interval", interval_value=float(g[0]), interval_unit=_UNIT_NORM[g[1].lower()])
        else:
            md = _FU_DATE.search(t)
            d = parse_date_text(md[1]) if md else None
            if d:
                base.update(kind="date", date=d.isoformat())
            elif _FU_SOS.search(t):
                base.update(kind="as_needed")
    for key in ("interval_value", "interval_value_max"):
        v = base[key]
        if v is not None and f"{v:g}" not in _literal_numbers(t):
            return {**check(t, NEEDS_CHECK, "the number is not clearly readable on the page"), "detail": base}
    return {**check(t, CHECKED, None if base["kind"] else "no interval or date recognised in the words"),
            "detail": base}


# ------------------------------------------------------------------ context for a test (EX-S5)


def link_context(investigations: list[dict[str, Any]], reasons: list[dict[str, Any]],
                 labels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For each ordered test: the diagnoses / complaints written on the same prescription and what
    each link rests on. ``same_line`` = they share a line of text (the quote is shown);
    ``same_page`` = only the same page, which is shown as exactly that and never as a reason."""
    out = []
    for inv in investigations:
        ev = [e for e in (inv.get("evidence") or []) if e in labels]
        links = []
        for r in reasons:
            rev = [e for e in (r.get("evidence") or []) if e in labels]
            shared = [e for e in ev if e in rev]
            if shared:
                links.append({"text": r["text"], "kind": r["kind"], "relation": "same_line",
                              "quote": labels[shared[0]].get("text")})
            elif ev and rev and {labels[e].get("page_id") for e in ev} & {labels[e].get("page_id") for e in rev}:
                links.append({"text": r["text"], "kind": r["kind"], "relation": "same_page", "quote": None})
        out.append({"test": inv["text"], "context": links})
    return out


def _texts(items: Any, kind: str) -> list[dict[str, Any]]:
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, str):
            it = {"text": it}
        if isinstance(it, dict):
            t = it.get("text") or it.get("code") or it.get("display") or it.get("name")
            if isinstance(t, str) and t.strip():
                out.append({"text": t.strip(), "evidence": it.get("evidence") or [], "kind": kind})
    return out


# ------------------------------------------------------------------ text on the page that talks to the model

_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?)"
    r"|disregard\s+(?:the\s+|all\s+|your\s+)?(?:previous|above|prior|instructions?)"
    r"|forget\s+(?:everything|all)\s+(?:above|previous)|system\s+prompt|you\s+are\s+now\b"
    r"|reveal\s+(?:the\s+|your\s+)?(?:prompt|instructions?)|list\s+all\s+patients|new\s+instructions\s*:",
    re.IGNORECASE)


def injection_suspects(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lines on the page that read like an instruction to the system. The model is told to treat
    page text as data; this makes the attempt visible and sends the document to a person."""
    return [{"code": "prompt_injection_suspected", "line": i, "text": str(b.get("text") or "")[:160]}
            for i, b in enumerate(blocks, start=1) if _INJECTION.search(str(b.get("text") or ""))]


# ------------------------------------------------------------------ everything for one document


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def build_checks(payload: dict[str, Any], blocks: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """``today`` is passed in (the document's own date), never read from the clock: the same
    document must give the same answer whenever it is looked at."""
    page = page_text(blocks)
    labels = blocks_by_label(blocks)
    tests = _texts(payload.get("investigations"), "test")
    prep, retracted = check_preparation(payload.get("investigation_preparation"), [t["text"] for t in tests], blocks)
    reasons = _texts(payload.get("diagnoses"), "diagnosis") + _texts(payload.get("chief_complaints"), "complaint")
    checks: dict[str, Any] = {
        "patient": check_patient(_dict(payload.get("patient")), page, today),
        "doctor": check_doctor(_dict(payload.get("prescriber")), page),
        "preparation": prep, "retracted": retracted,
        "follow_up": parse_follow_up(payload.get("follow_up"), page),
        "context": link_context(tests, reasons, labels),
        "flags": injection_suspects(blocks),
    }
    checks["review"] = review_items(checks)
    return checks


def review_items(checks: dict[str, Any]) -> list[dict[str, str]]:
    """Everything a person must look at, as ``{field, reason}``."""
    out: list[dict[str, str]] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict) and "status" in node:
            if node["status"] == NEEDS_CHECK:
                out.append({"field": prefix, "reason": node.get("reason") or "needs a check"})
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}", v)

    for section in ("patient", "doctor"):
        for k, v in checks[section].items():
            walk(f"{section}.{k}", v)
    for i, p in enumerate(checks["preparation"]):
        if p["status"] == NEEDS_CHECK:
            out.append({"field": f"lab_preparation[{i}]", "reason": p["reason"] or "needs a check"})
    for f in checks.get("flags", []):
        out.append({"field": "page", "reason": "text on the page looks like an instruction to the system "
                    f"(line {f['line']}); it was treated as ordinary text"})
    if checks["follow_up"]["status"] == NEEDS_CHECK:
        out.append({"field": "follow_up", "reason": checks["follow_up"]["reason"] or "needs a check"})
    return out
