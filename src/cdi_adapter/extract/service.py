from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .. import repo, storage
from ..config import settings
from ..db import session_scope
from ..logging import get_logger
from ..ml.client import MLError, get_client
from .prompt import block_id_map, build_extraction_prompt, load_schema, max_tokens_for, slim_active
from . import indian_codes, lab_resolve, medicine_resolve, resolve_llm
from .medicine_lexicon import medicine_match
from .test_names import is_known_test, is_test_list, looks_like_medicine, split_tests

log = get_logger(__name__)

_MODEL_STACK = {
    "classifier": "mlserve/qwen2.5-vl",
    "ocr": "rapidocr+vlm",
    "extractor": "mlserve/qwen2.5-vl",
    "terminology": "seed-v1",
}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _parse_date(s: str | None) -> tuple[datetime | None, str]:
    if not s:
        return None, "day"
    s = s.strip()
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        prec = "year" if not mo else "month" if not d else "day"
        return datetime(int(y), int(mo or 1), int(d or 1), tzinfo=timezone.utc), prec
    m = re.match(r"^(\d{1,2})[-/ ]([A-Za-z]{3})[a-z]*[-/ ](\d{2,4})$", s)
    if m:
        d, mon, y = int(m.group(1)), m.group(2).lower()[:3], int(m.group(3))
        if y < 100:
            y += 2000
        if mon in _MONTHS:
            return datetime(y, _MONTHS[mon], d, tzinfo=timezone.utc), "day"
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc), "day"
        except ValueError:
            return None, "day"
    return None, "day"


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, (dict, list, bool)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


def _asdict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


_UNIT_RE = re.compile(r"-?\d+(?:\.\d+)?\s*([A-Za-z%/µ][A-Za-z%/0-9µ.\[\]*^-]*)")


def _qty(x: Any) -> tuple[float | None, str | None, list[str] | None]:
    """Read a value/unit/evidence out of a quantity-ish field that the model may
    have emitted as an object, a bare number, or a string like '500 mg'."""
    if isinstance(x, dict):
        return (_num(x.get("value")),
                x.get("unit_ucum") or x.get("unit_text") or x.get("unit"),
                x.get("evidence"))
    if isinstance(x, (int, float)):
        return float(x), None, None
    if isinstance(x, str):
        m = _UNIT_RE.search(x)
        return _num(x), (m.group(1) if m else None), None
    return None, None, None


@dataclass
class ExtractResult:
    document_id: str
    schema: str
    n_facts: int
    skipped: bool = False
    patient_id: str | None = None
    mpi_id: str | None = None
    identity: dict[str, Any] | None = None   # raw candidate read from THIS doc


class _Ctx:
    def __init__(self, sess, doc_id, patient_id, encounter_id, extraction_id,
                 page_id, blkmap, blocks_by_id, run_ids):
        self.sess = sess
        self.doc_id = doc_id
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.extraction_id = extraction_id
        self.page_id = page_id
        self.blkmap = blkmap
        self.blocks_by_id = blocks_by_id
        self.run_ids = run_ids
        self.n = 0

    def add(self, *, fact_type: str, local_text: str, evidence: list[str] | None = None,
            extracted_text: str | None = None, **cols: Any) -> UUID:
        cols.setdefault("confidence_extract", self._conf)
        cols.setdefault("confidence_overall", round(0.5 * (self._conf + self._ocr_conf(evidence)), 3))
        cols.setdefault("confidence_ocr", self._ocr_conf(evidence))
        fid = repo.insert_clinical_fact(
            self.sess, patient_id=self.patient_id, encounter_id=self.encounter_id,
            fact_type=fact_type, local_text=(local_text or "")[:2000],
            extraction_id=self.extraction_id, source_doc_ids=[self.doc_id], **cols,
        )
        ids = [self.blkmap[e] for e in (evidence or []) if e in self.blkmap]
        bbox = self._bbox_union(ids)
        repo.insert_fact_provenance(
            self.sess, fact_id=fid, source_doc_id=self.doc_id, page_id=self.page_id,
            ocr_block_ids=ids, bbox_union=bbox,
            extracted_text=extracted_text or local_text or "",
            pipeline_run_ids=self.run_ids, model_stack=_MODEL_STACK,
        )
        self.n += 1
        return fid

    _conf = 0.8

    def _ocr_conf(self, evidence: list[str] | None) -> float:
        ids = [self.blkmap[e] for e in (evidence or []) if e in self.blkmap]
        vals = [float(self.blocks_by_id[i]["ocr_conf"]) for i in ids if i in self.blocks_by_id]
        return round(sum(vals) / len(vals), 3) if vals else 0.5

    def _bbox_union(self, ids: list[str]) -> list[int] | None:
        boxes = [self.blocks_by_id[i]["bbox"] for i in ids if i in self.blocks_by_id]
        if not boxes:
            return None
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]


# --------------------------------------------------------------------------- #
# per-doc-type payload -> facts
# --------------------------------------------------------------------------- #
def _coded_text(x: Any) -> tuple[str, list[str] | None]:
    if isinstance(x, dict):
        return (x.get("text") or x.get("code") or x.get("display") or x.get("name") or "",
                x.get("evidence"))
    if isinstance(x, str):
        return x, None
    return "", None


_PREFIX = re.compile(r"^\s*(?:inj|in|tab|t|cap|c|syp|sy|syr|susp|oint|cream|gel|drops?|rx)\b[\s.:\-]*", re.I)


def _first_word(text: str) -> str:
    m = re.search(r"[A-Za-z]{3,}", _PREFIX.sub("", text or "", count=1))
    return m.group(0).casefold() if m else ""


def _has_digit(x: Any) -> bool:
    if isinstance(x, dict):
        x = x.get("value", x.get("text"))
    return x is not None and any(ch.isdigit() for ch in str(x))


def _advice_listed_as_medicine(m: Any) -> bool:
    """A medication entry that is really advice: advice words, and nothing that says medicine (no reference-list hit,
    no strength or dose with a number). Frequency, duration, route and form are NOT evidence: the model fills them
    for anything it lists (MEASURED: it wrote 'qds' on every entry of one real prescription, advice lines included)."""
    if isinstance(m, str):
        return medicine_resolve.advice_like(m)
    if not isinstance(m, dict):
        return False
    name = m.get("drug_text") or m.get("text") or m.get("name") or ""
    return medicine_resolve.advice_like(name, has_dose=_has_digit(m.get("strength")) or _has_digit(m.get("dose")))


def _misfiled_medicine(text: str) -> bool:
    """A medicine written among the tests: the Indian drug list / medicine marks say so, and no test list knows it."""
    if is_known_test(text) or lab_resolve.resolve(text) is not None:
        return False
    return bool(indian_codes.drug_lookup(text) or medicine_match(text) or looks_like_medicine(text))


def _facts_prescription(c: _Ctx, p: dict[str, Any]) -> None:
    for dx in p.get("diagnoses") or []:
        t, ev = _coded_text(dx)
        if t:
            c.add(fact_type="condition", local_text=t, value_code_display=t, evidence=ev,
                  clinical_status="active", verification="confirmed")
    c.med_resolved = p.get("_med_resolved") or {}              # {as read: reference name the model chose}
    for m in p.get("medications") or []:
        if _advice_listed_as_medicine(m):
            # "steam inhalation", "gargle with warm water", "plenty of fluids": advice, not a medicine
            t, ev = _coded_text(m if isinstance(m, dict) else {"text": m})
            t = t or (m.get("drug_text") if isinstance(m, dict) else "") or ""
            if t:
                c.add(fact_type="advice", local_text=t, value_text=t, evidence=(m.get("evidence") if isinstance(m, dict) else ev))
            continue
        _add_medication(c, m, intent="order",
                        status=m.get("status") if isinstance(m, dict) else None)
    seen_tests: set[str] = set()                                # a test is listed once, however many views / lines found it
    for a in p.get("advice") or []:
        t, ev = _coded_text(a)
        if not t:
            continue
        if _misfiled_medicine(t):
            continue                       # a medicine in the advice list: medicines are not extracted in this profile
        if lab_resolve.resolve(t) is not None or is_known_test(t):
            # a test written among the advice ("S. Lipase", "S. Fructosamine"): it belongs with the tests
            listed = {w.casefold() for io in p.get("investigations") or [] if isinstance(io, dict) and io.get("source") == "list_context"
                      for w in re.findall(r"[A-Za-z0-9?]+", str(io.get("text") or ""))}      # entries of a test list, already listed one by one
            for one in split_tests(t):
                words = re.findall(r"[A-Za-z0-9?]+", one)
                if len(words) >= 2 and all(w.casefold() in listed for w in words) and not lab_resolve.resolve(one):
                    continue                                       # "BJS CT" beside BJS and CT says nothing more
                if _test_key(one) in seen_tests:
                    continue
                seen_tests.add(_test_key(one))
                c.add(fact_type="investigation_order", local_text=one, value_code_display=one, value_text=one, evidence=ev)
            continue
        c.add(fact_type="advice", local_text=t, value_text=t, evidence=ev)
    resolved = p.get("_test_resolved") or {}                  # {as read: reference name the model chose}
    on_page = {_first_word(m.get("drug_text") or m.get("text") or "") for m in p.get("medications") or []
               if isinstance(m, dict)}
    for io in p.get("investigations") or []:
        t, ev = _coded_text(io)
        for one in split_tests(t) if t else []:          # one fact per test, however the doctor wrote the list
            if _misfiled_medicine(one):
                # a drug listed among the tests: it belongs with the medicines (once), never shown as a test
                if _first_word(one) not in on_page:
                    _add_medication(c, {"drug_text": one, "evidence": ev}, intent="order")
                    on_page.add(_first_word(one))
                continue
            if _test_key(one) in seen_tests:
                continue
            if _test_key(one) == "bloodsugar" and any(k.startswith("bloodsugar") for k in seen_tests):
                continue                                     # "Blood sugar" with no letters is already said by "Blood sugar F" / "PP"
            seen_tests.add(_test_key(one))
            c.add(fact_type="investigation_order", local_text=one, value_code_display=resolved.get(one) or one,
                  value_text=one, evidence=ev)
    _add_vitals(c, p.get("vitals") or [])


def _test_key(name: str) -> str:
    """Two spellings of the same written test ("S.Lipase", "S. Lipase", "Lipase") share a key."""
    k = re.sub(r"^\s*(?:s|serum)[\s.]+", "", (name or "").casefold())
    return re.sub(r"[^a-z0-9]", "", k)


_facts_opd_note = _facts_prescription


def _facts_lab(c: _Ctx, p: dict[str, Any]) -> None:
    coll, prec = _parse_date(p.get("reported_at") or p.get("collected_at"))
    for r in p.get("results") or []:
        if not isinstance(r, dict):
            continue
        val, unit, _ev = _qty(r.get("value"))
        c.add(
            fact_type="lab_result",
            local_text=r.get("analyte_text") or r.get("text") or "",
            value_code_display=r.get("analyte_text") or r.get("text"),
            value_kind="quantity" if val is not None else "string",
            value_num=val,
            value_unit_ucum=unit or r.get("unit"),
            value_text=r.get("value_text"),
            ref_range_low=_num(r.get("ref_low")), ref_range_high=_num(r.get("ref_high")),
            ref_range_text=r.get("ref_text"),
            abnormal_flag=r.get("flag"),
            effective_time=coll, effective_precision=prec,
            evidence=r.get("evidence"),
        )


def _facts_vitals(c: _Ctx, p: dict[str, Any]) -> None:
    _add_vitals(c, p.get("vitals") or [])
    for s in p.get("symptoms") or []:
        t, ev = _coded_text(s)
        if t:
            c.add(fact_type="symptom", local_text=t, value_code_display=t, evidence=ev)
    for a in p.get("allergies") or []:
        if isinstance(a, str):
            a = {"substance_text": a}
        elif not isinstance(a, dict):
            continue
        sub = a.get("substance_text") or a.get("text") or a.get("name") or ""
        if sub:
            c.add(fact_type="allergy", local_text=sub, value_code_display=sub,
                  value_text=a.get("reaction_text"), evidence=a.get("evidence"),
                  clinical_status="active", verification="unconfirmed")


def _facts_discharge(c: _Ctx, p: dict[str, Any]) -> None:
    adm, _ = _parse_date(p.get("admission_date"))
    for dx in p.get("diagnoses") or []:
        t, ev = _coded_text(dx)
        if t:
            c.add(fact_type="condition", local_text=t, value_code_display=t, evidence=ev,
                  clinical_status="active", verification="confirmed", onset=adm)
    for pr in p.get("procedures") or []:
        if isinstance(pr, str):
            pr = {"name": pr}
        elif not isinstance(pr, dict):
            continue
        nm = pr.get("name") or pr.get("text") or ""
        pdt, pprec = _parse_date(pr.get("date"))
        if nm:
            c.add(fact_type="procedure", local_text=nm, value_code_display=nm,
                  evidence=pr.get("evidence"), effective_time=pdt,
                  effective_precision=pprec, value_text=pr.get("findings"),
                  clinical_status="completed")
    for m in p.get("medications_on_discharge") or []:
        _add_medication(c, m, intent="order")


def _facts_radiology(c: _Ctx, p: dict[str, Any]) -> None:
    sd, prec = _parse_date(p.get("study_date"))
    title = (p.get("study_name") or p.get("procedure_name") or p.get("modality")
             or "Imaging study")
    findings = p.get("findings")
    impression = p.get("impression")
    concl = " | ".join(str(x) for x in (impression, findings) if isinstance(x, str) and x.strip())

    # the report itself
    c.add(fact_type="diagnostic_report", local_text=title,
          value_text=concl or (str(findings) if findings else None),
          effective_time=sd, effective_precision=prec)

    # the procedure, if named
    if p.get("procedure_name"):
        c.add(fact_type="procedure", local_text=p["procedure_name"],
              value_code_display=p["procedure_name"], effective_time=sd,
              effective_precision=prec, clinical_status="completed",
              value_text=findings if isinstance(findings, str) else None)

    # individual findings (both the structured list and the impression concepts)
    for concept in (p.get("findings_list") or []) + (p.get("impression_concepts") or []):
        t, ev = _coded_text(concept)
        if t:
            c.add(fact_type="finding", local_text=t, value_code_display=t,
                  evidence=ev, effective_time=sd)
    # if the model only gave prose, split it into findings by line/semicolon
    if not (p.get("findings_list") or p.get("impression_concepts")) and isinstance(findings, str):
        for line in re.split(r"[\n;]+", findings):
            line = line.strip(" -•\t")
            if len(line) > 3:
                c.add(fact_type="finding", local_text=line[:200], value_code_display=line[:200],
                      effective_time=sd)

    # diagnoses
    for dx in p.get("diagnoses") or []:
        t, ev = _coded_text(dx)
        if t:
            c.add(fact_type="condition", local_text=t, value_code_display=t, evidence=ev,
                  clinical_status="active", verification="provisional", onset=sd)
    if isinstance(impression, str) and "diagnos" in impression.lower() and not (p.get("diagnoses")):
        m = re.search(r"diagnos\w*[:\-]\s*(.+)", impression, re.I)
        if m:
            c.add(fact_type="condition", local_text=m.group(1).strip()[:200],
                  value_code_display=m.group(1).strip()[:200], verification="provisional")

    if isinstance(p.get("advice"), str) and p["advice"].strip():
        c.add(fact_type="advice", local_text=p["advice"][:300], value_text=p["advice"])


_VITAL_ALIAS = {
    "pulse": "heart_rate", "hr": "heart_rate", "heart_rate": "heart_rate",
    "resp": "resp_rate", "rr": "resp_rate", "respiratory_rate": "resp_rate",
    "temp": "temperature", "temperature": "temperature",
    "spo2": "spo2", "o2_sat": "spo2", "oxygen_saturation": "spo2",
    "weight": "weight", "wt": "weight", "height": "height", "ht": "height",
    "bmi": "bmi", "pain_score": "pain_score", "pain": "pain_score",
    "systolic_bp": "systolic_bp", "diastolic_bp": "diastolic_bp",
}


def _add_vitals(c: _Ctx, vitals: list[Any]) -> None:
    seen: set[str] = set()

    def put(name: str, val: float | None, unit: str | None, ev: list[str] | None) -> None:
        if val is None or name in seen:
            return
        seen.add(name)
        c.add(fact_type="vital_sign", local_text=name.replace("_", " "),
              value_code_display=name, value_kind="quantity",
              value_num=val, value_unit_ucum=unit, evidence=ev)

    for v in vitals:
        if not isinstance(v, dict):
            continue
        raw = (v.get("name") or "").strip().lower().replace(" ", "_")
        name = _VITAL_ALIAS.get(raw, raw)
        ev = v.get("evidence")
        if v.get("systolic") is not None or v.get("diastolic") is not None:
            put("systolic_bp", _num(v.get("systolic")), "mm[Hg]", ev)
            put("diastolic_bp", _num(v.get("diastolic")), "mm[Hg]", ev)
            continue
        val, unit, _ev = _qty(v.get("value"))
        vstr = v.get("value") if isinstance(v.get("value"), str) else None
        if vstr and (m := re.match(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", vstr)):
            put("systolic_bp", float(m.group(1)), "mm[Hg]", ev)
            put("diastolic_bp", float(m.group(2)), "mm[Hg]", ev)
            continue
        put(name or "vital", val, unit, ev)


_STRENGTH_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|gm|g|ml|iu|units?|u/ml|iu/ml|%)\b", re.I)
_UCUM_MAP = {"mcg": "ug", "µg": "ug", "ug": "ug", "mg": "mg", "gm": "g", "g": "g",
             "ml": "mL", "iu": "[IU]", "u": "[IU]", "unit": "[IU]", "units": "[IU]",
             "iu/ml": "[IU]/mL", "u/ml": "[IU]/mL", "%": "%"}
_FREQ_TOKENS = {
    "od": 1, "qd": 1, "hs": 1, "once daily": 1, "once a day": 1, "at night": 1,
    "morning": 1, "before breakfast": 1, "after breakfast": 1, "after food daily": 1,
    "before food daily": 1, "after dinner daily": 1, "daily": 1,
    "bd": 2, "bid": 2, "twice daily": 2, "twice a day": 2,
    "tds": 3, "tid": 3, "thrice daily": 3,
    "qid": 4, "qds": 4,
    "sos": 0, "prn": 0, "stat": 0,
    "weekly": 0.143, "once a week": 0.143,
    "once in a year": 0.003, "twice in a year": 0.006, "once in 6 months": 0.006,
    "once in 3 months": 0.011,
}


def _parse_strength_from_name(drug: str) -> tuple[str, float | None, str | None]:
    """'METPURE XL 50 MG' -> ('Metpure XL', 50, 'mg').  'RYZODEG PENFILL 100IU/ML 3ML INJ' -> (..., 100, '[IU]/mL')."""
    matches = list(_STRENGTH_RE.finditer(drug or ""))
    if not matches:
        return drug, None, None
    # a pen/vial often prints its fill volume ("3ML") next to the real strength
    # ("100IU/ML") - prefer a dose-like unit over a bare volume when both appear
    m = next((x for x in matches
              if _UCUM_MAP.get(x.group(2).lower(), x.group(2).lower()) not in ("mL", "g")),
             matches[0])
    num = float(m.group(1))
    unit = _UCUM_MAP.get(m.group(2).lower(), m.group(2).lower())
    clean = (drug[:m.start()] + drug[m.end():]).strip(" -,/")
    clean = re.sub(r"\s{2,}", " ", clean) or drug
    return clean, num, unit


def _freq_per_day(text_: str | None) -> float | None:
    if not text_:
        return None
    t = str(text_).strip().lower()
    # dose-timing pattern "1-0-1" / "1-0-0-1" -> sum of the slots
    m = re.search(r"(?<![\d.])(\d)\s*-\s*(\d)\s*-\s*(\d)(?:\s*-\s*(\d))?(?![\d.])", t)
    if m:
        return float(sum(int(x) for x in m.groups() if x))
    for k, v in sorted(_FREQ_TOKENS.items(), key=lambda kv: -len(kv[0])):
        if k in t:
            return float(v)
    # "3 times a day", "2x/day", "twice per day" - digits must precede times/x
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:times|x)\s*(?:/|per|a|an|each)?\s*day\b", t)
    if m:
        return float(m.group(1))
    m = re.search(r"\bq\s*(\d+)\s*h(?:rs?|ours?)?\b", t)   # q8h -> 3/day
    if m and float(m.group(1)) > 0:
        return round(24.0 / float(m.group(1)), 2)
    return None


def _duration_days(text_: str | None, dur: Any) -> int | None:
    if isinstance(dur, (int, float)):
        return int(dur)
    n = _num(dur)
    if n:
        return int(n)
    if text_:
        m = re.search(r"x?\s*(\d+)\s*days?", str(text_).lower())
        if m:
            return int(m.group(1))
        if re.search(r"\bmonth\b", str(text_).lower()):
            mm = re.search(r"(\d+)\s*month", str(text_).lower())
            return int(mm.group(1)) * 30 if mm else 30
    return None


def _add_medication(c: _Ctx, m: Any, *, intent: str, status: str | None = None) -> None:
    if not isinstance(m, dict):
        if isinstance(m, str):
            m = {"drug_text": m, "evidence": []}
        else:
            return
    raw_drug = m.get("drug_text") or m.get("text") or m.get("name") or ""
    if is_test_list(raw_drug) and not any(m.get(k) for k in (
            "strength", "dose", "frequency_text", "frequency", "duration_days", "timing", "dosage", "sig",
            "schedule", "dose_pattern")):
        # a list of test abbreviations with no dose or duration is an order for tests, not a medicine
        for t in split_tests(raw_drug):
            c.add(fact_type="investigation_order", local_text=t, value_code_display=t, value_text=t,
                  evidence=m.get("evidence"))
        return
    drug, name_str, name_unit = _parse_strength_from_name(raw_drug)
    s_val, s_unit, _ = _qty(m.get("strength"))
    d_val, d_unit, _ = _qty(m.get("dose"))
    if s_val is None:
        s_val, s_unit = name_str, name_unit
    # keep every frequency-bearing field the model may have used (dedup, order-preserving)
    _fbits = [str(x).strip() for x in (m.get("frequency_text"), m.get("frequency"),
                                       m.get("timing"), m.get("dosage"), m.get("sig"),
                                       m.get("schedule"), m.get("dose_pattern"))
              if isinstance(x, str) and x.strip()]
    freq_text = " ".join(dict.fromkeys(_fbits)) or None
    instr = " ".join(str(x) for x in (m.get("instructions"), m.get("notes"), m.get("composition"))
                     if isinstance(x, str) and x.strip()) or None
    _combined = " ".join(x for x in (freq_text, instr) if x) or None
    fpd = _freq_per_day(freq_text)
    if fpd is None:
        fpd = _freq_per_day(_combined)
    dur = _duration_days(_combined, m.get("duration_days"))
    route = m.get("route") if isinstance(m.get("route"), str) else None
    if not route and re.search(r"\binj|injection|penfill|s/?c\b|subcut", raw_drug.lower()):
        route = "subcutaneous" if "s/c" in raw_drug.lower() or "subcut" in raw_drug.lower() else "injection"
    cs = {"stopped": "stopped", "changed": "active"}.get(status or "", "active")
    chosen = (getattr(c, "med_resolved", None) or {}).get(raw_drug)       # the model's pick among reference names, if any
    fid = c.add(
        fact_type="medication", local_text=drug or raw_drug, value_code_display=chosen or drug or raw_drug,
        evidence=m.get("evidence"), clinical_status=cs, verification="confirmed",
    )
    repo.insert_medication_detail(
        c.sess, fid,
        drug_text=drug or raw_drug,
        form=m.get("form") if isinstance(m.get("form"), str) else None,
        strength_num=s_val, strength_unit=s_unit,
        dose_num=d_val, dose_unit_ucum=d_unit,
        route=route,
        frequency_code=str(freq_text) if freq_text else None,
        frequency_per_day=fpd,
        duration_days=dur,
        prn=m.get("prn") if isinstance(m.get("prn"), bool) else None,
        instructions=instr,
        intent=intent,
    )


_HANDLERS = {
    "cdi:prescription.v3": _facts_prescription,
    "cdi:opd_note.v3": _facts_opd_note,
    "cdi:lab_report.v3": _facts_lab,
    "cdi:vitals.v3": _facts_vitals,
    "cdi:discharge_summary.v3": _facts_discharge,
    "cdi:radiology.v3": _facts_radiology,
}


# --------------------------------------------------------------------------- #
def _check_the_name(client: Any, image: bytes, blocks: list[dict[str, Any]], payload: dict[str, Any],
                    images: list[bytes] | None = None) -> None:
    """Read the patient's name again from its own line at several sizes and compare with the first reading. When most readings
    agree on a name that is not the first reading, that name is used. ``payload['_name_reads']`` keeps every reading so the
    screen can show them; the name stays "to confirm" whatever happens (a person confirms it)."""
    from ..names import alike, consensus
    from . import resolve_llm

    from ..names import org_like

    first = payload["patient"].get("name")
    first = first if isinstance(first, str) and first.strip() and first.strip().casefold() not in ("null", "none") else None
    if first and org_like(first):                     # the clinic's name on the letterhead was taken for the patient: it is not a name
        first, payload["patient"]["name"] = None, None
    plain_reads, json_reads = resolve_llm.name_reads_split(client, image, blocks, first)
    # the PLAIN transcriptions are the readings (they decide); the JSON "Indian personal name" readings are only a cross-check and only offered as
    # other readings of the name (that prompt snaps to the commonest name and drops words: "Sumita Gupta Gangopadhyay" came out "Sunita Gupta")
    reads = [r for r in (plain_reads or json_reads) if not org_like(r)]
    cross = [r for r in json_reads if plain_reads and not org_like(r) and not any(alike(r, p, 0.85) for p in plain_reads)]
    payload["_name_reads"] = ([first, *reads] if first else list(reads)) + cross
    if not reads:
        return
    chosen, agree, total = consensus([first or "", *reads])
    payload["_name_agreement"] = [agree, total]
    if chosen and ((first is None) or (agree >= 3 and not alike(first, chosen, 0.85))):
        payload["patient"]["name"] = chosen                 # no usable first reading, or most readings agree on a different spelling
    # the readings from the name line's own enlarged crops decide over the whole-page reading: when the same spelling (exactly, title
    # ignored) is read by MORE THAN HALF of them it is the shown name, even when it is close to the first reading. MEASURED on a real
    # page: "Shibaji Sen" was read 8 of 12 times from the crops while the first reading "Shibay Sen" stayed in the field because the
    # two are 86% alike and were counted as one group.
    from collections import Counter

    from ..names import name_key

    modal = Counter(name_key(r) for r in reads if name_key(r)).most_common(1)
    if modal and modal[0][1] >= 2 and modal[0][1] * 2 > len(reads) and modal[0][0] != name_key(payload["patient"].get("name") or ""):
        payload["patient"]["name"] = next(r for r in reads if name_key(r) == modal[0][0])
    # the PLAIN transcriptions of the name line decide over the whole-page answer: when at least two of them agree with each other (85 % alike) and
    # the shown name is not alike them, their agreed reading is the shown name. MEASURED on a real page: the whole-page answer said "Mr. Amit Gupta
    # Gangadhay" while the plain readings of the name line said "Sumita Gupta Gangopadhyay" (twice).
    plain_ok = [r for r in plain_reads if not org_like(r)]
    if len(plain_ok) >= 2:
        agreed, how_many, _total = consensus(plain_ok)
        shown_now = payload["patient"].get("name")
        if agreed and how_many >= 2 and agreed != shown_now:
            payload["patient"]["name"] = agreed
            payload["_name_note"] = f"the name line was read the same way {how_many} times as '{agreed}'; the whole-page answer said '{shown_now}'"
    from ..names import prefer_complete

    now = payload["patient"].get("name")
    fuller = prefer_complete(now, [n for n in payload["_name_reads"] if isinstance(n, str) and not org_like(n)]) if now else now
    if fuller and fuller != now:
        payload["patient"]["name"] = fuller                 # a reading that goes on where the shown one stops (a long name cut short)
    shown = payload["patient"].get("name")
    if first and isinstance(shown, str) and len(first.split()) >= 2 and len(shown.split()) == 1:
        payload["patient"]["name"] = first                  # the re-reads of a small crop gave one word where the page gave a full name
        payload["_name_note"] = f"the re-reads gave only '{shown}'; the first reading '{first}' is kept"
    _suggest_joined_name(payload)
    if settings.name_choice_votes:
        _suggest_first_names(client, images or [image], blocks, payload)
    if settings.name_surname_votes:
        _suggest_surnames(client, images or [image], blocks, payload)


_TITLE_WORDS = frozenset(("mr", "mrs", "ms", "miss", "master", "smt", "shri", "sri", "dr"))


def _name_tokens(name: str) -> list[str]:
    """The words of a person's name without a leading title ("Mr. Onkar Chowdhury" -> Onkar, Chowdhury)."""
    toks = name.split()
    return toks[1:] if toks and toks[0].rstrip(".").lower() in _TITLE_WORDS else toks


def _suggest_joined_name(payload: dict[str, Any]) -> None:
    """A word the writer left a gap in ("Sayanta ni Sarkar" for Sayantani Sarkar, MEASURED on a real page: the first reading kept "ni",
    three re-reads dropped it). When one reading has a short word (up to 3 letters) between the first name and the surname and
    another reading has no such word, the commonest first word + that fragment is offered as one more reading. Never the shown
    name: a short middle word can also be a real part of a name."""
    from collections import Counter

    reads = [n for n in payload.get("_name_reads") or [] if isinstance(n, str)]
    toks = [_name_tokens(n) for n in reads]
    mids = {t[1] for t in toks if len(t) == 3 and len(t[1]) <= 3 and t[1].isalpha()}
    if not mids or not any(len(t) == 2 for t in toks):
        return
    shown = _name_tokens(payload["patient"].get("name") or "")
    if len(shown) >= 3 and shown[1] in mids:
        shown = [shown[0], *shown[2:]]                       # the shown reading may be the one that kept the fragment
    surname = " ".join(shown[1:])
    firsts = Counter(t[0] for t in toks if len(t) >= 2 and len(t[0]) >= 4).most_common(1)
    have = {n.casefold() for n in reads}
    for first, _n in firsts:
        for mid in sorted(mids):
            full = f"{first}{mid.lower()} {surname}".strip()
            if surname and full.casefold() not in have:
                payload["_name_reads"].append(full)
                have.add(full.casefold())


def _suggest_surnames(client: Any, images: list[bytes], blocks: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    """The surname put to the model as a choice among the readings and the closest common surnames (``names.surname_options``; a
    ``?`` for an unread letter is a wildcard), plus "none of these". Up to five (the ones picked, then the closest)
    are added to the offered readings as <first name> <surname>. Suggestions only: the shown name does not change here. Skipped
    when every reading agrees on a surname with no ``?`` in it."""
    from ..names import surname_options
    from . import resolve_llm

    shown = _name_tokens(payload["patient"].get("name") or "")
    if len(shown) < 2:
        return
    lasts = [t[-1] for n in payload.get("_name_reads") or [] if isinstance(n, str) for t in [_name_tokens(n)] if len(t) >= 2]
    if len({x.casefold() for x in lasts}) <= 1 and not any("?" in x for x in lasts):
        return
    options = surname_options(lasts)
    crops = _name_line_crops(images, blocks, payload["patient"].get("name") or "")
    votes = resolve_llm.surname_votes(client, crops, options)
    payload["_surname_votes"] = votes
    if not votes:
        return                                              # the model gave no answer: nothing is suggested
    # the spellings the model picked, most picked first, then the closest list names it did not pick: one run's votes can miss the
    # right surname (MEASURED: Sarkar got 1 vote in one run and none in the next) and it must still be on the screen to pick
    ranked = [w for w, _n in sorted(votes.items(), key=lambda kv: -kv[1])] + [o for o in options if o not in votes]
    have = {n.casefold() for n in payload.get("_name_reads") or [] if isinstance(n, str)}
    added = 0
    for word in ranked:
        full = f"{shown[0]} {word}"
        if full.casefold() in have or word.casefold() == shown[-1].casefold():
            continue
        payload.setdefault("_name_reads", []).append(full)
        have.add(full.casefold())
        added += 1
        if added >= 5:
            break


def _name_line_crops(images: list[bytes], blocks: list[dict[str, Any]], shown: str) -> list[bytes]:
    """The name line, cut out and enlarged, from each picture of the page (two sizes of each)."""
    from . import resolve_llm

    crops: list[bytes] = []
    for img in images:
        try:
            crops += resolve_llm.name_crops(img, blocks, shown)[:2]
        except Exception as exc:  # noqa: BLE001
            log.warning("name_crop_failed", error=str(exc)[:200])
    return crops


def _suggest_first_names(client: Any, images: list[bytes], blocks: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    """Put the first name to the model as a choice among the readings and their one-letter confusions, and add the spellings it
    picks most often to the readings the screen offers. Suggestions only: the shown name does not change here."""
    from . import resolve_llm

    shown = payload["patient"].get("name") or ""
    surname = " ".join(_name_tokens(shown)[1:])
    firsts = [t[0] for n in payload.get("_name_reads") or [] if isinstance(n, str) for t in [_name_tokens(n)] if t]
    options = resolve_llm.first_name_options(firsts)
    crops = _name_line_crops(images, blocks, shown)
    votes = resolve_llm.first_name_votes(client, crops, options)
    payload["_name_votes"] = votes
    have = {n.casefold() for n in payload.get("_name_reads") or [] if isinstance(n, str)}
    for word, _n in sorted(votes.items(), key=lambda kv: -kv[1])[:3]:
        full = f"{word} {surname}".strip()
        if full.casefold() not in have:
            payload.setdefault("_name_reads", []).append(full)
            have.add(full.casefold())


def _page_image(pg: dict[str, Any]) -> bytes:
    """The picture of a page the vision model reads (``CDI_MODEL_IMAGE``): the colour source page, or the normalized copy."""
    uri = None
    if settings.model_image == "source":
        pre = pg.get("preproc")
        uri = pre.get("src_uri") if isinstance(pre, dict) else None
    return storage.get_bytes(storage.key_from_uri(uri or pg["image_uri"]))


def _colour_page(pg: dict[str, Any], model_image: bytes) -> Any:
    """The colour source page as an array when it is the same size as the picture the readers saw (so the text blocks' boxes fit it); else
    None. The picture the model reads may be a grey, contrast-normalised copy with no colour in it."""
    try:
        pre = pg.get("preproc")
        uri = pre.get("src_uri") if isinstance(pre, dict) else None
        if not uri:
            return None
        import cv2
        import numpy as np

        src = cv2.imdecode(np.frombuffer(storage.get_bytes(storage.key_from_uri(uri)), np.uint8), cv2.IMREAD_COLOR)
        ref = cv2.imdecode(np.frombuffer(model_image, np.uint8), cv2.IMREAD_COLOR)
        return src if src is not None and ref is not None and src.shape[:2] == ref.shape[:2] else None
    except Exception:  # noqa: BLE001
        return None


def _both_pictures(pg: dict[str, Any], one: bytes) -> list[bytes]:
    """The page as the colour source AND as the normalized grey copy (the name is looked at in both), or just ``one``."""
    out = [one]
    for key in ("image_uri",):
        try:
            other = storage.get_bytes(storage.key_from_uri(pg[key]))
            if other != one:
                out.append(other)
        except Exception:  # noqa: BLE001
            pass
    pre = pg.get("preproc")
    if isinstance(pre, dict) and pre.get("src_uri"):
        try:
            src = storage.get_bytes(storage.key_from_uri(pre["src_uri"]))
            if src not in out:
                out.append(src)
        except Exception:  # noqa: BLE001
            pass
    return out[:2]


def _collapse_repeats(x: Any) -> Any:
    """A list in which the same entry is written again and again (a model loop) keeps its first copy: the same text, ignoring the
    evidence it points at and its case, is one entry."""
    if isinstance(x, dict):
        return {k: _collapse_repeats(v) for k, v in x.items()}
    if not isinstance(x, list):
        return x
    seen: set[str] = set()
    out: list[Any] = []
    for item in x:
        key = (item.get("text") or item.get("name") or "") if isinstance(item, dict) else item
        key = " ".join(str(key).casefold().split()) if isinstance(key, str) and key.strip() else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(_collapse_repeats(item))
    return out


def _unescape(x: Any) -> Any:
    """The model sometimes writes an ampersand as ``&amp;`` (and a quote as ``&quot;``): prescriptions never contain HTML,
    so every string in the answer is turned back into the plain characters."""
    import html

    if isinstance(x, str):
        return html.unescape(x) if "&" in x else x
    if isinstance(x, list):
        return [_unescape(i) for i in x]
    if isinstance(x, dict):
        return {k: _unescape(v) for k, v in x.items()}
    return x


def _read_each_page(client: Any, pages: list[dict[str, Any]], blocks: list[dict[str, Any]], doc_type: str,
                    schema: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Read every page with its own image and its own OCR lines (the [bN] numbers stay the document's), at the same time,
    then merge. A later page that cannot be read is left out (logged); the first page failing fails the document."""
    from concurrent.futures import ThreadPoolExecutor

    from . import visits as V

    def one(pg: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        img = _page_image(pg)
        pr = build_extraction_prompt(doc_type, blocks, only_page=pg["id"])
        return client.vlm_json_ex(img, pr, schema, max_tokens=max_tokens_for(doc_type), retries=settings.extract_retries)

    def safe(pair: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], str | None] | None:
        i, pg = pair
        try:
            return one(pg)
        except Exception as exc:  # noqa: BLE001
            if i == 0:
                raise
            log.warning("page_read_failed", page=i + 1, error=str(exc)[:200])
            return None

    with ThreadPoolExecutor(max_workers=min(4, len(pages))) as pool:
        got = list(pool.map(safe, list(enumerate(pages))))
    done = [g for g in got if g is not None]
    return V.merge([g[0] if g else None for g in got]), done[0][1]


_PREFETCH: dict[str, Any] = {}
_prefetch_pool = None


class _Steps(dict):
    """Seconds spent in each step of one extraction (kept in the stored answer as ``_timings``)."""

    def run(self, name: str, fn, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            self[name] = round(self.get(name, 0.0) + time.perf_counter() - t0, 2)


def prefetch_main_call(document_id: str) -> bool:
    """Start the main page call NOW, with the text the printed-text reader already found, so it runs while the handwritten lines are being read
    (the call takes about 5 s and used to wait for those readings). The answer is kept for ``extract_document``, which uses it only when the page
    is one page and the document type is the one it was asked for; anything else is thrown away and the normal call is made. No evidence ids are
    needed from the model (the compact answer has none), so the later block numbers do not matter. Returns True when a call was started."""
    global _prefetch_pool
    if not settings.prefetch_main_call or not settings.extract_compact_answer:
        return False
    with session_scope() as sess:
        cls = repo.get_doc_classification(sess, document_id)
        pages = repo.list_document_pages(sess, document_id)
        blocks = repo.list_ocr_blocks(sess, document_id)
    if not cls or len(pages) != 1 or not slim_active(cls["doc_type"]):
        return False
    loaded = load_schema(cls["doc_type"])
    if not loaded:
        return False
    _schema_id, schema = loaded
    doc_type = cls["doc_type"]
    prompt = build_extraction_prompt(doc_type, blocks)
    image = _page_image(pages[0])
    client = get_client()
    if _prefetch_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _prefetch_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cdi-prefetch")
    _PREFETCH[document_id] = (doc_type, _prefetch_pool.submit(
        client.vlm_json_ex, image, prompt, schema, max_tokens=max_tokens_for(doc_type), retries=settings.extract_retries))
    return True


def extract_document(document_id: str, *, patient_id: str | None = None,
                     encounter_id: str | None = None,
                     abha_hint: str | None = None) -> ExtractResult:
    client = get_client()
    with session_scope() as sess:
        doc = repo.get_document(sess, document_id)
        if not doc:
            raise ValueError(f"document {document_id} not found")
        cls = repo.get_doc_classification(sess, document_id)
        if not cls:
            raise ValueError(f"document {document_id} not classified")
        pages = repo.list_document_pages(sess, document_id)
        blocks = repo.list_ocr_blocks(sess, document_id)
        run_id = repo.start_pipeline_run(
            sess, document_id=document_id, stage="extract",
            model_name="mlserve/vlm", params={"doc_type": cls["doc_type"]},
        )
        run_ids = [str(r["id"]) for r in sess.execute(
            __import__("sqlalchemy").text(
                "SELECT id FROM pipeline_run WHERE document_id=:d"), {"d": document_id}
        ).mappings().all()]

    rerouted_why = ""
    if settings.recall_reroute:
        from . import recall
        new_type, rerouted_why = recall.reroute(cls["doc_type"], blocks)
        if new_type:
            # the page text holds ordered tests beside the marks of a prescription: it is handled (and recorded) as one
            with session_scope() as sess:
                repo.insert_doc_classification(
                    sess, document_id=document_id, doc_type=new_type, specialty=cls.get("specialty"),
                    is_handwritten=bool(cls.get("is_handwritten")), languages=list(cls.get("language") or ["en"]),
                    confidence=min(float(cls.get("confidence") or 0.8), 0.8), page_spans=cls.get("page_spans") or [])
            log.info("doc_type_rerouted", document_id=document_id, was=cls["doc_type"], now=new_type, why=rerouted_why)
            cls = {**cls, "doc_type": new_type}

    loaded = load_schema(cls["doc_type"])
    if not loaded:
        with session_scope() as sess:
            repo.finish_pipeline_run(sess, run_id, status="skipped",
                                     metrics={"reason": f"no schema for {cls['doc_type']}"})
            repo.set_document_status(sess, document_id, "normalized")
        log.info("extract_no_schema", document_id=document_id, doc_type=cls["doc_type"])
        return ExtractResult(document_id, cls["doc_type"], 0, skipped=True)

    steps = _Steps()
    t_start = time.perf_counter()
    schema_id, schema = loaded
    prompt = build_extraction_prompt(cls["doc_type"], blocks)
    blkmap = block_id_map(blocks)
    image = _page_image(pages[0])

    from . import visits as V
    from .prompt import slim_active

    slim = slim_active(cls["doc_type"])
    try:
        # served_model = the model the gateway reports it used (the OOM fallback is recorded as such)
        if slim and settings.extract_per_page and len(pages) > 1:
            # a paper of several pages (front, back, a continuation sheet): each page is read on its own and the
            # latest dated visit's tests and booking are picked (extract/visits.py)
            payload, served_model = _read_each_page(client, pages, blocks, cls["doc_type"], schema)
        else:
            pre = _PREFETCH.pop(document_id, None)
            got = None
            if pre is not None and pre[0] == cls["doc_type"] and len(pages) == 1:
                try:
                    got = pre[1].result()                  # started right after classification: usually finished by now
                except Exception as exc:  # noqa: BLE001 - the normal call below is the fallback
                    log.warning("prefetch_failed", document_id=document_id, error=str(exc)[:200])
            if got is not None:
                payload, served_model = got
            else:
                payload, served_model = steps.run("main_answer", client.vlm_json_ex,
                    image, prompt, schema, max_tokens=max_tokens_for(cls["doc_type"]),
                    retries=settings.extract_retries)
            if slim and isinstance(payload, dict):
                payload = V.merge([payload])           # older dated entries below a ruled line: the latest one is used
    except MLError as exc:
        with session_scope() as sess:
            repo.finish_pipeline_run(sess, run_id, status="failed", error_detail=str(exc)[:400])
            repo.set_document_status(sess, document_id, "error", error_detail=f"extract: {exc}")
        raise

    if isinstance(payload, dict):
        payload = _collapse_repeats(_unescape(payload))
    if isinstance(payload, dict):
        # a value written next to a test (Hb-11.9, CRP-0.02, TLC 6900, +ve / -ve) is a RESULT already done, never a test to be done (owner's rule)
        from . import results_written
        written: list[str] = []
        for _key in ("investigations", "advice"):
            if not isinstance(payload.get(_key), list):
                continue
            kept = []
            for it in payload[_key]:
                txt = it.get("text") if isinstance(it, dict) else it
                clean, removed = results_written.clean_entry(txt if isinstance(txt, str) else "")
                written += removed
                if removed and not clean:
                    continue
                kept.append(({**it, "text": clean} if isinstance(it, dict) else clean) if removed else it)
            payload[_key] = kept
        if written:
            payload["_results_written"] = written
    if isinstance(payload, dict) and slim and settings.extract_compact_answer:
        from . import evidence as _evidence
        _evidence.attach(payload, blocks)                   # the links to the OCR blocks, found here instead of written by the model
    if isinstance(payload, dict) and isinstance(payload.get("patient"), dict):
        # the patient's mobile number is what was typed at upload, never what the page prints (a letterhead's WhatsApp / clinic number)
        payload["patient"]["phone"] = (doc.get("phone") or "").strip() or None
    if isinstance(payload, dict) and isinstance(payload.get("patient"), dict):
        from .fields import normalise_age
        fixed_age, sex_hint = normalise_age(payload["patient"].get("age_text"))
        if fixed_age != payload["patient"].get("age_text"):
            payload["patient"]["age_text"] = fixed_age         # "74/1" is an age of 74 years, not "74 over 1"
            if sex_hint and not payload["patient"].get("sex"):
                payload["patient"]["sex"] = sex_hint
    if not settings.abha_enabled and isinstance(payload, dict) and isinstance(payload.get("patient"), dict):
        payload["patient"]["abha_id"] = None        # never used, whatever the model wrote: not for matching, not stored

    focus_blocks = blocks
    latest_no = int(payload.get("_latest_page") or 1) if isinstance(payload, dict) else 1
    page1_image, page1_blocks = image, [b for b in blocks if str(b.get("page_id")) == str(pages[0]["id"])] or blocks
    if len(pages) > 1 and 1 <= latest_no <= len(pages) and latest_no != 1:
        # the second look and the choose-from-list step read the page the latest visit is on
        image = _page_image(pages[latest_no - 1])
    if len(pages) > 1 and 1 <= latest_no <= len(pages):
        pid_latest = str(pages[latest_no - 1]["id"])
        focus_blocks = [b for b in blocks if str(b.get("page_id")) == pid_latest] or blocks

    if isinstance(payload, dict) and cls["doc_type"] in ("prescription", "opd_note", "referral"):
        from . import header
        header.fill_patient(payload, page1_blocks)  # a patient printed in the header with age and sex (Apollo Sugar Clinics), when the model found none
    if isinstance(payload, dict) and isinstance(payload.get("patient"), dict) and cls["doc_type"] in ("prescription", "opd_note", "referral"):
        steps.run("name", _check_the_name, client, page1_image, page1_blocks, payload, _both_pictures(pages[0], page1_image))     # the name is on the first page
        if settings.age_sex_reread:
            from . import age_sex
            try:
                steps.run("age_sex", age_sex.apply, client, page1_image, page1_blocks, payload)          # "74/F" read from its own crop at three sizes
            except Exception as exc:  # noqa: BLE001 - an extra: never cost the document
                log.warning("age_sex_failed", document_id=document_id, error=str(exc)[:160])

    if isinstance(payload, dict) and cls["doc_type"] in ("prescription", "opd_note", "referral"):
        from . import header
        header.fill(payload, page1_blocks)         # the printed header's doctor and clinic, when the model left them empty

    if isinstance(payload, dict) and cls["doc_type"] in ("prescription", "opd_note", "referral"):
        # the hybrid step: a test the gate cannot place but that is close to reference names is put to the model
        # as a CHOICE among those names (never free text); what it picks is kept apart from what was written
        from . import test_cluster
        payload["_text_scan"] = test_cluster.repair_investigations(payload)      # "Chest ECO" -> ECG, "Nat & Kit" -> Na+ & K+, with a note of what was read
        names = [one for io in payload.get("investigations") or [] for one in split_tests(_coded_text(io)[0])]
        fu = payload.get("follow_up")
        fu_text = fu if isinstance(fu, str) else (fu.get("text") if isinstance(fu, dict) else None)
        # a list with one test the lists place is a list of tests (the follow-up text and the advice lines): the model wrote "Digital OPG, FBS, BJS CT" as the
        # booking text and listed no test. The placed names go in by their standard name, the entries nothing places go in to be checked.
        have_names = {re.sub(r"[^a-z0-9]", "", n.casefold()) for n in names}
        listed_entries: list[tuple[str, str, bool]] = []
        for src in [fu_text, *[_coded_text(a)[0] for a in payload.get("advice") or []]]:
            if src and looks_like_medicine(src):
                continue                       # a medicine order ("Tab Zincovit 1 tab ...") is not a list of tests
            for name, as_read, placed in test_cluster.list_entries(src):
                key = re.sub(r"[^a-z0-9]", "", name.casefold())
                if key in have_names:
                    continue
                have_names.add(key)
                payload.setdefault("investigations", []).append({"text": name, "evidence": [], "source": "list_context"})
                payload["_text_scan"][name] = (f"listed together with a test the lists place ('{as_read}')" if placed else
                                               f"written in a list with a test, not recognised (read as '{as_read}')")
                names.append(name)
                listed_entries.append((name, as_read, placed))
        colour = None if not settings.marks_enabled else _colour_page(pages[latest_no - 1] if len(pages) > 1 and 1 <= latest_no <= len(pages) else pages[0], image)     # for the pen marks
        corroborated: set[str] = set()
        from . import results_written as _rw
        extra = steps.run("second_look", resolve_llm.followup_tests, client, image, fu_text, names, focus_blocks, colour=colour, corroborated=corroborated)    # looks even when no follow-up was found
        extra = [e for e in extra if not _rw.is_result_entry(e)]      # the second look copies a result as readily as an order
        payload["_corroborated"] = sorted(corroborated)           # tests of the main answer that the enlarged second look read again in enough views
        if extra:                          # tests written with the follow-up line, found by the focused second look
            payload.setdefault("investigations", []).extend({"text": t, "evidence": [], "source": "second_look"} for t in extra)
            payload["_second_look"] = list(extra)          # kept so the result can say where these tests came from
            payload["_text_scan"].update({f.test: f.note for f in test_cluster.scan(focus_blocks, colour) if f.test in extra})
            names += extra
        if listed_entries:                  # after the second look has added its own: "BJS CT" beside BJS and CT says nothing more
            payload["investigations"] = test_cluster.drop_composites(payload.get("investigations") or [], listed_entries)
        # imaging / ECG / physiotherapy, a clinic's printed list of services and the words of a medicine line are not laboratory tests:
        # they stay in the result as rejected with the reason, and are not put to the choose-from-list step
        from . import department, not_lab
        payload["_department"] = department.detect(department.header_texts(page1_blocks))    # from the printed header only; None when it shows none
        payload["_not_lab"] = not_lab.classify(names, blocks, lab_only=settings.lab_tests_only)
        if rerouted_why:
            payload["_rerouted"] = rerouted_why
        names = [n for n in names if n not in payload["_not_lab"]]
        if settings.verify_tests_enabled and names:
            # the direct question is only asked of a test no line reader saw (the others already have the page's own text behind them): about
            # 10 fewer model calls per page (MEASURED: the extra checks had made a page take 18-21 s in the extract step)
            from . import verify
            from .test_names import is_grounded
            page_text = " ".join(str(b.get("text") or "") for b in blocks)
            unseen = [n for n in names if not is_grounded(n, page_text)]
            payload["_verify"] = steps.run("verify", verify.verify_tests, [image], unseen[:12]) if unseen else {}      # {name: the model's probability that it is written here as an order}
        payload["_test_resolved"] = steps.run("choose_from_list", resolve_llm.resolve_tests, client, image, names)
        # the same for medicines: a name close to reference medicine names is a CHOICE among them, never free text
        meds = [(_coded_text(m)[0] if not isinstance(m, dict) else (m.get("drug_text") or m.get("text") or m.get("name") or ""))
                for m in payload.get("medications") or []]
        payload["_med_resolved"] = resolve_llm.resolve_medicines(client, image, [x for x in meds if x])

    blocks_by_id = {str(b["id"]): b for b in blocks}
    handler = _HANDLERS.get(schema_id)

    from ..mpi.service import (candidate_from_payload, merge_identity_evidence,
                               record_alias, resolve_identity)

    with session_scope() as sess:
        # re-extracting a document supersedes the previous attempt - clear the
        # facts / extraction / encounter / aliases it wrote so they are not doubled
        repo.purge_document_facts(sess, document_id)
        # ---- identity: read name / sex / age from THIS document ----
        cand = candidate_from_payload(payload, abha_hint=abha_hint, source_doc_id=document_id)
        mpi_id = None
        if patient_id:
            pid = patient_id
            record_alias(sess, pid, cand)
            row = sess.execute(
                __import__("sqlalchemy").text(
                    "SELECT mpi_id FROM patient_identity WHERE id = :i"), {"i": pid}
            ).first()
            mpi_id = row[0] if row else None
        else:
            res = resolve_identity(sess, cand)
            pid, mpi_id = res.patient_id, res.mpi_id
            record_alias(sess, pid, cand)
        identity_out = {
            "name": cand.name_full, "sex": cand.sex, "age_years": cand.age_years,
            "birth_date": cand.birth_date.isoformat() if cand.birth_date else None,
            "source_document_id": document_id,
        }
        # ---- practitioner link (E18): prescriber field + header lines vs doctor master ----
        prac_ref = None
        try:
            from ..recognition.practitioner import link_document

            pres = _asdict(payload.get("prescriber"))
            p1 = str(pages[0]["id"])
            header = [b["text"] for b in blocks if str(b["page_id"]) == p1][:12]
            header += [b["text"] for b in blocks if re.search(r"\breg", b["text"] or "", re.I)]
            with sess.begin_nested():     # savepoint: a failure here cannot sink S4
                pm = link_document(sess, document_id, name=pres.get("name"),
                                   reg_no=pres.get("reg_no"), header_texts=header)
            # linked -> the master's name; otherwise what was read (never a guess)
            prac_ref = pm.db_name if pm.linked else (pres.get("name") or None)
        except Exception as exc:  # noqa: BLE001 - a missing doctor master must not block S4
            log.warning("practitioner_link_skipped", document_id=document_id, error=str(exc)[:200])
        eid = encounter_id
        if not eid:
            edate, eprec = _parse_date(
                payload.get("encounter_date") or payload.get("reported_at")
                or payload.get("study_date") or payload.get("discharge_date")
                or payload.get("recorded_at")
            )
            enc_class = "IMP" if cls["doc_type"] in ("discharge_summary", "operative_note") else "AMB"
            eid = str(repo.create_encounter(
                sess, patient_id=pid, enc_class=enc_class,
                period_start=edate or doc.get("captured_at") or doc["ingested_at"],
                period_end=None, period_precision=eprec, specialty=cls.get("specialty"),
                practitioner_ref=prac_ref,
                derived_from=[document_id], confidence=float(cls["confidence"]),
            ))

        if served_model:
            repo.set_run_model_version(sess, run_id, served_model)
        from ..ml.client import last_raw_answer
        from ..provenance import engine_versions, prompt_version

        if isinstance(payload, dict):
            steps["total_extract"] = round(time.perf_counter() - t_start, 2)
            payload["_timings"] = dict(steps)                  # where the seconds of this extraction went
        ext_id = repo.insert_extraction(
            sess, document_id=document_id, schema_name=schema_id,
            schema_version="v3", payload=payload,
            evidence_map={"block_ids": blkmap}, model_run_id=run_id,
            prompt_version=prompt_version(), engine_versions=engine_versions(served_model),
            raw_answer=last_raw_answer(),
        )

        ctx = _Ctx(sess, document_id, pid, eid, ext_id, pages[0]["id"],
                   blkmap, blocks_by_id, run_ids)
        ctx._conf = float(payload.get("extracted_at_confidence") or 0.75)
        if handler:
            handler(ctx, payload)

        repo.finish_pipeline_run(sess, run_id, status="ok",
                                 metrics={"facts": ctx.n, "schema": schema_id})
        repo.set_document_status(sess, document_id, "extracted")
        repo.write_audit(sess, actor="extract-svc", action="create", entity="clinical_fact",
                         entity_id=document_id, patient_id=pid,
                         detail={"facts": ctx.n, "doc_type": cls["doc_type"]})

    log.info("extracted", document_id=document_id, doc_type=cls["doc_type"], facts=ctx.n,
             patient=identity_out.get("name"), mpi_id=mpi_id)
    return ExtractResult(document_id, cls["doc_type"], ctx.n, patient_id=pid,
                         mpi_id=mpi_id, identity=identity_out)
