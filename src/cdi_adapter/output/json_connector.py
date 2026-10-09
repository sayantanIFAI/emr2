"""Result JSON (UP-S3 / OUT-S2): ``result.v1``.

A *connector* turns one finished document into whatever a downstream system wants. The real
downstream (HIS / EMR) is not chosen yet, so the format is this repo's own, versioned and STRICT:
``schemas/result.v1.json`` (no field the schema does not list, ``null`` for unknown, fixed lists for
every category) and every result is validated against it before anyone sees it. When the real
downstream is chosen its field names change under a new version (``result.v2``); the interface does not.

Scope (MLP1): patient, doctor, lab tests (with preparation and context), advice and follow-up.

Rules the JSON keeps (the point of a placeholder is that these hold from day one):

* every value is an object ``{"value", "status", "reason", "confidence"}``: a value is never shown
  without its status;
* fact-based items (lab tests, advice, medications ...) are ``accepted`` (the gate or a person
  accepted it), ``needs_check`` or ``rejected``; values read from the page that are judged by the
  rules of ``extract/fields.py`` are ``checked`` (right format and literally on the page),
  ``needs_check``, ``absent`` (not written: normal) or ``not_gated`` (a visual judgement);
* a doubtful value is never presented as final: the document ``status`` is ``needs_check`` while
  any value is;
* unknown is ``null``, never a guess; what the system does not extract yet is listed in
  ``not_extracted``; a preparation note the model made up is dropped and listed in
  ``retracted_preparation``;
* the same document always gives the same bytes (no timestamps, fixed key order).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import text

from .. import repo
from ..config import settings
from ..db import session_scope
from ..extract import fields as F
from ..extract import indian_codes, lab_resolve
from ..extract import not_lab as _not_lab
from ..extract.indian_codes import norm as _norm_name
from ..extract.medicine_lexicon import medicine_match
from ..terminology.service import licensed_only
from ..extract.test_names import SECOND_LOOK, UNCONFIRMED, page_support, UNRECOGNISED, is_grounded, is_known_test, looks_like_medicine

SCHEMA_VERSION = "result.v1"
NOTICE = ("Read by a machine. Values marked needs a check must be verified by a person. "
          "Not for diagnosis.")        # PLACEHOLDER wording: the owner and a clinician approve the real text

ACCEPTED = {"auto_accepted", "clinician_confirmed", "corrected"}
# what the extraction does not produce yet (docs/upload-screen.md); kept visible, not hidden
NOT_EXTRACTED = ["patient.guardian", "doctor.registration_council", "lab_tests.specimen",
                 "lab_tests.urgency", "lab_tests.does_not_fit_check"]
_FINISHED = {"validated", "normalized"}


def value(v: Any, status: str, reason: str | None = None, confidence: float | None = None) -> dict:
    return {"value": v, "status": status,
            "reason": reason, "confidence": None if confidence is None else round(float(confidence), 3)}


def _num(x: Any) -> float | int | None:
    if x is None:
        return None
    f = float(x)
    return int(f) if f == int(f) else round(f, 6)


def _fact_status(fact: dict[str, Any]) -> tuple[str, str | None]:
    state = fact.get("review_state")
    if state in ACCEPTED:
        return "accepted", None
    if state == "rejected":
        return "rejected", fact.get("review_note")
    return "needs_check", fact.get("review_note") or "waiting for a person to check it"


def _item(fact: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    status, reason = _fact_status(fact)
    return {"fact_id": str(fact["id"]), "text": fact.get("local_text"), **fields,
            "status": status, "reason": reason,
            "confidence": None if fact.get("confidence_overall") is None
            else round(float(fact["confidence_overall"]), 3)}


def _medication(f: dict[str, Any]) -> dict[str, Any]:
    d = f.get("medication") or {}
    return _item(f, {
        "drug": d.get("drug_text") or f.get("local_text"),
        "strength": _num(d.get("strength_num")), "strength_unit": d.get("strength_unit"),
        "dose": _num(d.get("dose_num")), "dose_unit": d.get("dose_unit_ucum"),
        "route": d.get("route"), "frequency": d.get("frequency_code"),
        "duration_days": d.get("duration_days"), "instructions": d.get("instructions"),
        # the reference name the MODEL chose among the list when the handwriting was close to it (None = none chosen)
        "reference_name": (f.get("value_code_display")
                           if f.get("value_code_display") and f.get("value_code_display") != (d.get("drug_text") or f.get("local_text"))
                           else None)})


def _lab_result(f: dict[str, Any]) -> dict[str, Any]:
    return _item(f, {
        "name": f.get("local_text"), "value": _num(f.get("value_num")),
        "value_text": f.get("value_text"), "unit": f.get("value_unit_ucum"),
        "ref_low": _num(f.get("ref_range_low")), "ref_high": _num(f.get("ref_range_high")),
        "ref_text": f.get("ref_range_text"), "flag": f.get("abnormal_flag"),
        "code": f.get("code"), "code_system": f.get("code_system")})


def _lab_order(f: dict[str, Any]) -> dict[str, Any]:
    """A test ORDERED on a prescription: as written, plus the standard name / code if one matched
    (``code_status`` says how sure: unmapped / candidate / bound / local_only)."""
    return _item(f, {
        "as_written": f.get("local_text"), "code": f.get("code"), "code_system": f.get("code_system"),
        "code_display": f.get("code_display"), "code_status": f.get("code_status"),
        "standard_name": None, "standard_source": None, "page_support": None, "gate_recognised": False,
        "context": [], "preparation": []})


def _vital(f: dict[str, Any]) -> dict[str, Any]:
    return _item(f, {"name": f.get("local_text"), "value": _num(f.get("value_num")),
                     "value_text": f.get("value_text"), "unit": f.get("value_unit_ucum")})


def _plain(f: dict[str, Any]) -> dict[str, Any]:
    return _item(f, {})


_BUCKETS = {"medication": ("medications", _medication), "investigation_order": ("lab_tests", _lab_order),
            "lab_result": ("lab_results", _lab_result), "vital_sign": ("vitals", _vital),
            "condition": ("diagnoses", _plain), "advice": ("advice", _plain)}


@dataclass
class ResultInputs:
    """Everything the JSON is built from (read once, so the build itself is a pure function)."""

    document: dict[str, Any]
    classification: dict[str, Any] | None = None
    pages: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)       # latest extraction payload
    facts: list[dict[str, Any]] = field(default_factory=list)   # current facts, medication detail merged
    blocks: list[dict[str, Any]] = field(default_factory=list)  # OCR blocks, in the order the prompt numbered them
    source: dict[str, Any] = field(default_factory=dict)        # where a dropped file came from (listener_file)
    corrections: list[dict[str, Any]] = field(default_factory=list)  # earlier readings a person replaced
    extraction: dict[str, Any] = field(default_factory=dict)         # which models / prompt / schema produced it


def _quality(pages: list[dict[str, Any]], doc: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    codes: list[str] = []
    warnings: list[str] = []
    seen = False
    for p in pages:
        q = (p.get("preproc") or {}).get("quality")
        if not q:
            continue
        seen = True
        reasons += [f"page {p['page_no']}: {r}" for r in q.get("reasons") or []]
        codes += list(q.get("reason_codes") or [])
        warnings += [f"page {p['page_no']}: {w}" for w in q.get("warnings") or []]
    if doc.get("status") == "quality_hold" and not reasons and doc.get("error_detail"):
        reasons = [str(doc["error_detail"])]
    return {"checked": seen, "passed": not reasons if (seen or reasons) else None,
            "reasons": reasons, "reason_codes": codes, "warnings": warnings}


def _capture_flagged(pages: list[dict[str, Any]]) -> bool:
    """True if a page was read despite a capture problem that is not worth a retake (the page's edges
    could not be found, so it was cropped to its writing or read whole): the result needs a check."""
    return any("page_edges_not_found" in ((p.get("preproc") or {}).get("quality") or {}).get("warning_codes", [])
               for p in pages)


def _v(c: dict[str, Any]) -> dict[str, Any]:
    """A checked field as a value object."""
    return value(c["value"], c["status"], c["reason"])


def _doc_date(doc: dict[str, Any]) -> date:
    d = doc.get("captured_at") or doc.get("ingested_at")
    if isinstance(d, datetime):
        return d.date()
    return d if isinstance(d, date) else date(2000, 1, 1)       # a fixed day, never "today": same bytes every time


def _applies(prep: dict[str, Any], test_text: str) -> bool:
    t = prep["applies_to"]
    return t == ["all"] or (isinstance(t, list) and test_text in t)


ENGINE_KEYS = ("vlm_configured", "vlm_served", "vlm_revision", "trocr", "trocr_revision", "trocr_device", "printed_ocr",
               "pdf_renderer", "image_library", "qwen_line_mode", "recognition_v2")


def _provenance(ext: dict[str, Any]) -> dict[str, Any]:
    """Which models, prompt and schema produced the values (saved with the extraction, never recomputed from
    today's settings): the answer to "what read this?" for an audit or a rollback."""
    eng = ext.get("engine_versions") or {}
    return {"schema_version": ext.get("schema_version"), "prompt_version": ext.get("prompt_version"),
            "engines": {k: eng.get(k) for k in ENGINE_KEYS}}


def _not_extracted(doc: dict[str, Any]) -> list[str]:
    """What this result does not carry: the standing list, plus what the slim extraction profile never asks for."""
    from ..extract import prompt as P

    return NOT_EXTRACTED + (P.SLIM_NOT_EXTRACTED if P.slim_active("prescription") else [])


NAME_TO_CONFIRM = "read from handwriting: a person must confirm the name"


def _patient_name(doc: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    """The patient's name is never final on its own: a read name is 'needs_check' until someone confirms or corrects it."""
    if doc.get("name_confirmed_at") and (doc.get("patient_name") or "").strip():
        return value(doc["patient_name"], "checked", f"confirmed by {doc.get('name_confirmed_by') or 'the front desk'}")
    v = _v(c)
    if v["value"] not in (None, ""):
        v["status"] = "needs_check"
        v["reason"] = NAME_TO_CONFIRM + (f" ({v['reason']})" if v.get("reason") else "")
    return v


def _confirmed_names_for(doc: dict[str, Any], shown: str | None) -> list[str]:
    """Names a person CONFIRMED on other prescriptions of the same mobile number that are at least 75 % alike the name read here ("Smita Gupta
    Gangopadhyay" read from a blurred photo of a patient confirmed earlier as "Sumita Gupta Gangopadhyay"). Only offered as a choice: the name
    shown is never replaced, and a person confirms it. [] without a mobile number or a name."""
    phone, did = (doc.get("phone") or "").strip(), str(doc.get("id") or "")
    if not phone or not shown:
        return []
    from ..names import alike
    try:
        with session_scope() as sess:
            rows = sess.execute(text("SELECT DISTINCT patient_name FROM source_document WHERE phone = :p AND name_confirmed_at IS NOT NULL "
                                     "AND patient_name IS NOT NULL AND id <> CAST(:d AS uuid)"), {"p": phone, "d": did or "00000000-0000-0000-0000-000000000000"}).all()
    except Exception:  # noqa: BLE001 - an extra: never cost the result
        return []
    return [r[0].strip() for r in rows if r[0] and r[0].strip() and alike(shown, r[0], 0.75)][:3]


def _intake(doc: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """What the front desk typed (token, mobile), the name as shown, as read, whether a person confirmed it, and the other
    readings of the name (so the screen can offer them)."""
    shown = (doc.get("patient_name") or "").strip() or None
    cands: list[str] = []
    from ..names import name_key, org_like

    seen = {name_key(shown)}                       # "MR. Debabrata Sanwar" and "Debabrata Sanwar" are one offer
    for n in _confirmed_names_for(doc, shown):      # a name a person already confirmed for this mobile number, close to the one read: offered FIRST
        if name_key(n) not in seen:
            seen.add(name_key(n))
            cands.append(n)
    for n in payload.get("_name_reads") or []:
        if isinstance(n, str) and n.strip() and name_key(n) not in seen and not org_like(n):
            seen.add(name_key(n))
            cands.append(n.strip())
    return {"token_no": doc.get("token_no"), "phone": doc.get("phone"), "patient_name": shown,
            "name_read": doc.get("name_read") or shown, "name_confirmed": bool(doc.get("name_confirmed_at")),
            "name_confirmed_by": doc.get("name_confirmed_by") if doc.get("name_confirmed_at") else None,
            "name_candidates": cands[:10]}


def _next_date(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """The "next date" written on the page (``Next dose: April 2026``), beside the follow-up: its words and, when it is a date, ISO."""
    got = F.find_next_date(blocks)
    return {"next_date": got["text"], "next_date_iso": got["iso"]} if got else {"next_date": None, "next_date_iso": None}


def _visits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The dated visits found on the pages (newest first), as stored by the page merge (extract/visits.py)."""
    out: list[dict[str, Any]] = []
    for v in payload.get("visits") or []:
        if not isinstance(v, dict):
            continue
        out.append({"page": v.get("page") if isinstance(v.get("page"), int) else None, "where": str(v.get("where") or ""),
                    "date": v.get("date") if isinstance(v.get("date"), str) else None,
                    "date_text": v.get("date_text") if isinstance(v.get("date_text"), str) else None,
                    "is_latest": bool(v.get("is_latest")),
                    "lab_tests": [str(x) for x in (v.get("lab_tests") or []) if isinstance(x, str)],
                    "follow_up": v.get("follow_up") if isinstance(v.get("follow_up"), str) else None})
    return out


def build_result(inp: ResultInputs) -> dict[str, Any]:
    doc, payload = inp.document, inp.payload or {}
    checks = F.build_checks(payload, inp.blocks, _doc_date(doc))
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _BUCKETS.values()}
    other: list[dict[str, Any]] = []
    for f in inp.facts:
        name, build = _BUCKETS.get(f.get("fact_type") or "", ("other", _plain))
        (buckets.get(name) if name != "other" else other).append(build(f))      # type: ignore[union-attr]

    context = {c["test"]: c["context"] for c in checks["context"]}
    page_text = " ".join(str(b.get("text") or "") for b in inp.blocks)       # what the page readers saw
    facts_by_id = {str(f.get("id")): f for f in inp.facts}
    second_look = {_norm_name(x) for x in payload.get("_second_look") or [] if isinstance(x, str)}
    text_scan = {_norm_name(k): v for k, v in (payload.get("_text_scan") or {}).items() if isinstance(k, str) and isinstance(v, str)}
    not_lab = {_norm_name(k): v for k, v in (payload.get("_not_lab") or {}).items() if isinstance(k, str) and isinstance(v, str)}
    from ..extract import department as _department
    dept = payload.get("_department") if isinstance(payload.get("_department"), str) else None
    verified = {_norm_name(k): float(v) for k, v in (payload.get("_verify") or {}).items() if isinstance(k, str) and isinstance(v, (int, float))}
    for t in buckets["lab_tests"]:
        key = t["as_written"]
        why_not = not_lab.get(_norm_name(key)) or _not_lab.entry_reason(key, lab_only=settings.lab_tests_only) or _not_lab.line_reason(
            key, inp.blocks, lab_only=settings.lab_tests_only)          # also the tests the model filed under advice or that a second look added
        pv = verified.get(_norm_name(key))
        if pv is not None and t.get("status") != "rejected":
            t["confidence"] = round(pv, 3)                 # the model's own probability that this test is written here as an order; it never accepts a test
            if pv < settings.verify_low_p:
                t["status"] = "needs_check"
                t["reason"] = (f"the model doubts this is written on the page (probability {pv:.2f}): check it first"
                               + (f"; {t['reason']}" if t.get("reason") else ""))
        dn = _department.note(t.get("standard_name") or key, dept) if t.get("status") != "rejected" else None
        if dn is not None:
            if dn[0] == "unusual":
                t["status"] = "needs_check"
                t["reason"] = dn[1] + (f"; {t['reason']}" if t.get("reason") else "")
            else:
                t["reason"] = (t["reason"] + "; " if t.get("reason") else "") + dn[1]      # supporting information only
        if why_not and t.get("status") != "rejected":
            t["status"] = "rejected"                       # kept, not shown as a test: imaging / ECG / physiotherapy / a service list / a medicine line
            t["reason"] = "not a laboratory test: " + why_not
            t["gate_recognised"] = False
            t["context"], t["preparation"] = [], []
            continue
        t["context"] = context.get(key, [])
        t["preparation"] = [p["text"] for p in checks["preparation"] if _applies(p, key)]
        fact = facts_by_id.get(str(t.get("fact_id")), {})
        alt = (fact.get("value_code_display") or "").strip()                 # a reference name the model CHOSE, never what was written
        alt = alt if alt and _norm_name(alt) != _norm_name(key) else ""
        rz = lab_resolve.resolve(key) or (lab_resolve.resolve(alt) if alt else None)     # the lab-test gate (Indian list + table)
        known = rz is not None or is_known_test(key) or (bool(alt) and is_known_test(alt))
        drug = None if known else indian_codes.drug_lookup(key)                # Common Drug Codes for India
        med = None if known else (drug.matched if drug else medicine_match(key))
        placed = rz is not None and not getattr(rz, "fuzzy", False)                # the lab LISTS place this name (mapping table, national list, table)
        t["gate_recognised"] = placed                                              # (a loose word match such as "load" in "Tab X 1 tab OD x load" is NOT this)
        t["page_support"] = page_support(key, page_text) if page_text else None   # a similarity score against the page's text, never a reason to reject
        if t.get("status") != "rejected" and not placed and (med or looks_like_medicine(key)):
            # a drug line the reader filed under the tests (a crowded handwritten page): never shown as a test
            t["status"] = "rejected"
            t["reason"] = f"looks like a medicine ('{med}'), not a test" if med else "looks like a medicine, not a test"
        elif t.get("status") != "rejected" and not known and not t.get("code"):
            # no test name in it and no standard code: a misreading, a procedure or a diagnosis, not a test the
            # screen can stand behind. Kept (nothing is lost) but marked, so it is not shown as a test.
            t["status"] = "needs_check"
            t["reason"] = UNRECOGNISED + (f": {t['reason']}" if t.get("reason") else "")
        elif (t.get("status") != "rejected" and page_text
              and not (is_grounded(key, page_text) or (alt and is_grounded(alt, page_text)))):
            # a valid test name that nothing on the page supports: what a reader says about a page it cannot read
            t["status"] = "needs_check"
            if _norm_name(key) in second_look:
                # found by looking again at an ENLARGED part of the page image (handwriting the text reader cannot read): a test
                # the lab list knows, shown in the list but never accepted
                t["reason"] = SECOND_LOOK + (f": {t['reason']}" if t.get("reason") else "")
            else:
                t["reason"] = UNCONFIRMED + (f": {t['reason']}" if t.get("reason") else "")
        if _norm_name(key) in text_scan and t.get("status") != "rejected":
            t["status"] = "needs_check"
            t["reason"] = text_scan[_norm_name(key)] + ": please check it" + (f" ({t['reason']})" if t.get("reason") else "")
        if alt and rz is not None and t.get("status") != "rejected":
            t["status"] = "needs_check"
            t["reason"] = f"read as '{alt}' (chosen from the reference list by the model; check the page)" + (
                f": {t['reason']}" if t.get("reason") else "")
        if rz is not None and t.get("status") != "rejected":
            # the one standard test this written name stands for (the mapping table, then the national list / table)
            t["standard_name"] = rz.long_name
            t["standard_source"] = {"mapping": "mapping table", "CLCI": "Indian lab list"}.get(rz.source, "abbreviation table")
        if rz is not None and rz.kind == "test" and rz.loinc and not t.get("code") and t.get("status") != "rejected":
            # the name is a test the Indian list knows: give it its standard code (only for a licensed code system)
            up = licensed_only({"code_system": "http://loinc.org", "code": rz.loinc, "code_display": rz.long_name,
                                "code_status": rz.status})
            t.update(code=up.get("code"), code_system=up.get("code_system"), code_display=up.get("code_display"),
                     code_status=up.get("code_status"))

    earlier: dict[str, list[dict[str, Any]]] = {}
    for c in inp.corrections:                      # the replaced reading stays referenced (OUT-S2 AC4)
        earlier.setdefault(str(c["fact_id"]), []).append(
            {"value": c.get("original_value"), "corrected_by": c.get("reviewer_id")})
    for lst in (*buckets.values(), other):
        for i in lst:
            if i["fact_id"] in earlier:
                i["earlier_readings"] = earlier[i["fact_id"]]

    items = [i for lst in (*buckets.values(), other) for i in lst]
    from ..extract import unread
    unread_flags = unread.flags(inp.blocks)                       # a handwritten line nobody could read, where orders are written
    name_v = _patient_name(doc, checks["patient"]["name"])
    n_check = sum(1 for i in items if i["status"] == "needs_check") + len(checks["review"])
    if name_v["status"] == "needs_check" and checks["patient"]["name"]["status"] != "needs_check":
        n_check += 1                   # a read name counts as one value to check until a person confirms it
    quality = _quality(inp.pages, doc)

    if doc.get("status") == "quality_hold":
        status = "held_for_rescan"
    elif doc.get("status") == "error":
        status = "error"
    elif doc.get("status") not in _FINISHED:
        status = "processing"
    elif n_check or unread_flags or any(i["status"] == "rejected" for i in items) or _capture_flagged(inp.pages):
        status = "needs_check"
    elif not items:
        status = "incomplete"          # finished, but nothing could be read
    else:
        status = "complete"

    p, d = checks["patient"], checks["doctor"]
    fu = checks["follow_up"]
    cls = inp.classification or {}
    refused = status in ("held_for_rescan", "error")
    src = inp.source or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": str(doc["id"]),
        "filename": doc.get("original_filename"),
        "source": {"channel": doc.get("source_channel"), "drive": src.get("connector"),
                   "dropped_file_name": src.get("name"), "original_filename": doc.get("original_filename")},
        "refusal": {"refused": refused,
                    "reason": (doc.get("error_detail") or "; ".join(quality["reasons"]) or None) if refused else None},
        "links": {"original": f"api/documents/{doc['id']}/original"},
        "provenance": _provenance(inp.extraction),
        "doc_type": cls.get("doc_type"),
        "is_handwritten": cls.get("is_handwritten"),
        "page_count": doc.get("page_count"),
        "status": status,
        "needs_check_count": n_check,
        "quality": quality,
        "extraction_incomplete": bool(payload.get("_partial")),
        "flags": [*checks["flags"], *unread_flags],
        "patient": {**{k: _v(p[k]) for k in ("name", "age_text", "dob", "sex", "mrn", "phone", "address", "abha_id")},
                    "name": name_v},
        "doctor": {**{k: _v(d[k]) for k in ("name", "reg_no", "department", "designation", "qualification")},
                   "clinic": {k: _v(d["clinic"][k]) for k in ("name", "address", "phone")},
                   "stamp_present": _v(d["stamp_present"]), "signature_present": _v(d["signature_present"])},
        "organization": {k: _v(d["clinic"][k]) for k in ("name", "address", "phone")},
        "intake": _intake(doc, payload),
        "visits": _visits(payload),
        "lab_tests": buckets.pop("lab_tests"),
        "lab_preparation": checks["preparation"],
        "retracted_preparation": checks["retracted"],
        "advice": buckets.pop("advice"),
        "follow_up": {**_v(fu), **fu["detail"], **_next_date(inp.blocks)},
        **buckets,
        "other": other,
        "not_extracted": _not_extracted(doc),
        "notice": NOTICE,
    }


class ResultInvalid(Exception):
    """The built result does not match ``schemas/result.v1.json``: it is never handed out."""


_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "result.v1.json"


@lru_cache(maxsize=1)
def result_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_result(result: dict[str, Any]) -> None:
    """Raise :class:`ResultInvalid` unless ``result`` matches the strict schema (an unknown field fails)."""
    import jsonschema

    errors = sorted(jsonschema.Draft202012Validator(result_schema()).iter_errors(_as_json(result)),
                    key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        where = ".".join(str(p) for p in e.absolute_path) or "(top)"
        raise ResultInvalid(f"result does not match result.v1 at {where}: {e.message[:200]}")


def _as_json(x: Any) -> Any:
    """What the serialised JSON would contain (dates and UUIDs as text), so validation sees the same thing."""
    return json.loads(json.dumps(x, default=str))


def to_bytes(result: dict[str, Any]) -> bytes:
    """The one canonical serialisation: the screen shows it and the download is these exact bytes."""
    return (json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def gather(sess: Any, document_id: str) -> ResultInputs | None:
    doc = repo.get_document(sess, document_id)
    if not doc:
        return None
    facts = repo.list_clinical_facts(sess, document_id=document_id)
    for f in facts:
        if f["fact_type"] == "medication":
            f["medication"] = repo.get_medication_detail(sess, f["id"])
    payload = sess.execute(
        text("SELECT payload FROM extraction WHERE document_id = :d ORDER BY created_at DESC LIMIT 1"),
        {"d": document_id}).scalar_one_or_none()
    src = sess.execute(
        text("SELECT connector, name FROM listener_file WHERE document_id = :d ORDER BY first_seen_at LIMIT 1"),
        {"d": document_id}).mappings().first()
    ext = sess.execute(
        text("SELECT schema_version, prompt_version, engine_versions FROM extraction WHERE document_id = :d "
             "ORDER BY created_at DESC LIMIT 1"), {"d": document_id}).mappings().first()
    corrections = sess.execute(
        text("SELECT fact_id, original_value, reviewer_id FROM correction WHERE document_id = :d "
             "ORDER BY created_at, id"), {"d": document_id}).mappings().all()
    return ResultInputs(doc, repo.get_doc_classification(sess, document_id),
                        repo.list_document_pages(sess, document_id),
                        payload if isinstance(payload, dict) else {}, facts,
                        repo.list_ocr_blocks(sess, document_id),
                        dict(src) if src else {}, [dict(c) for c in corrections], dict(ext) if ext else {})


class OutputConnector(Protocol):
    name: str

    def render(self, document_id: str) -> dict[str, Any] | None: ...


class JsonPlaceholderConnector:
    name = "json_placeholder"

    def render(self, document_id: str) -> dict[str, Any] | None:
        with session_scope() as sess:
            inp = gather(sess, document_id)
        if not inp:
            return None
        result = build_result(inp)
        validate_result(result)              # a result that breaks the contract is never served
        return result


_CONNECTORS: dict[str, type] = {JsonPlaceholderConnector.name: JsonPlaceholderConnector}


def get_connector(name: str | None = None) -> OutputConnector:
    from ..config import settings

    key = name or settings.output_connector
    if key not in _CONNECTORS:
        raise ValueError(f"unknown output connector {key!r} (known: {', '.join(sorted(_CONNECTORS))})")
    return _CONNECTORS[key]()
