"""One flat table for everything the extraction produces: intake, patient, organisation, doctor, doctor booking, lab test (``prescription_flat``).

GRAIN: one row per (prescription x lab test). A prescription with N lab tests makes N rows (the prescription-level columns repeat); one with no lab
test makes ONE row with every ``lab_test_*`` column empty (``row_kind = 'no_lab_test'``). Count prescriptions with ``COUNT(DISTINCT document_id)``.
``raw_result`` keeps the whole result.v1 JSON, so nothing is lost if a column is not enough. ``*_status`` is the screen's own verdict for that value
(checked | needs_check | absent | not_gated): a handwritten name is never "checked" until a person confirms it.

Rows are written when a document has been read (``save_document``), again when the name is confirmed, and replaced as a whole each time (delete by
document, then insert), so a re-read never leaves a stale row. Searched by mobile number or token number (``search``)."""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import text

from ..db import session_scope
from ..logging import get_logger

log = get_logger(__name__)

# column name -> SQL type (the order here is the order of the table)
_COLUMNS: list[tuple[str, str]] = [
    ("document_id", "uuid NOT NULL"), ("row_kind", "text NOT NULL DEFAULT 'lab_test'"), ("lab_test_seq", "integer"),
    ("result_schema_version", "text"), ("filename", "text"), ("source_channel", "text"), ("source_original_filename", "text"),
    ("intake_token_no", "text"), ("intake_phone", "text"), ("intake_patient_name", "text"), ("intake_name_read", "text"),
    ("intake_name_confirmed", "boolean"), ("intake_name_confirmed_by", "text"), ("intake_name_candidates", "jsonb"),
    ("doc_type", "text"), ("is_handwritten", "boolean"), ("page_count", "integer"), ("result_status", "text"), ("needs_check_count", "integer"),
    ("extraction_incomplete", "boolean"), ("quality_passed", "boolean"), ("quality_reasons", "jsonb"), ("flags", "jsonb"),
    ("vlm_served", "text"), ("vlm_revision", "text"), ("prompt_version", "text"), ("printed_ocr_engine", "text"), ("handwriting_engine", "text"),
    ("patient_name", "text"), ("patient_name_status", "text"), ("patient_name_confidence", "numeric"),
    ("patient_age_text", "text"), ("patient_age_status", "text"), ("patient_dob", "text"), ("patient_dob_status", "text"),
    ("patient_sex", "text"), ("patient_sex_status", "text"), ("patient_phone", "text"), ("patient_phone_status", "text"),
    ("patient_address", "text"), ("patient_address_status", "text"), ("patient_mrn", "text"), ("patient_mrn_status", "text"),
    ("patient_abha_id", "text"), ("patient_abha_status", "text"),
    ("org_name", "text"), ("org_name_status", "text"), ("org_address", "text"), ("org_address_status", "text"),
    ("org_phone", "text"), ("org_phone_status", "text"),
    ("doctor_name", "text"), ("doctor_name_status", "text"), ("doctor_department", "text"), ("doctor_department_status", "text"),
    ("doctor_designation", "text"), ("doctor_designation_status", "text"), ("doctor_qualification", "text"), ("doctor_qualification_status", "text"),
    ("doctor_reg_no", "text"), ("doctor_reg_no_status", "text"), ("doctor_clinic_name", "text"), ("doctor_clinic_address", "text"),
    ("doctor_clinic_phone", "text"), ("doctor_stamp_present", "boolean"), ("doctor_signature_present", "boolean"),
    ("booking_needed", "text"), ("booking_with_doctor", "text"), ("booking_as_written", "text"), ("booking_when_text", "text"),
    ("booking_kind", "text"), ("booking_interval_value", "numeric"), ("booking_interval_value_max", "numeric"), ("booking_interval_unit", "text"),
    ("booking_date", "text"), ("booking_next_date_text", "text"), ("booking_next_date_iso", "text"), ("booking_bring_reports", "boolean"),
    ("booking_status", "text"), ("booking_reason", "text"),
    ("lab_test_fact_id", "text"), ("lab_test_as_written", "text"), ("lab_test_text", "text"), ("lab_test_standard_name", "text"),
    ("lab_test_standard_source", "text"), ("lab_test_code", "text"), ("lab_test_code_system", "text"), ("lab_test_code_display", "text"),
    ("lab_test_code_status", "text"), ("lab_test_gate_recognised", "boolean"), ("lab_test_page_support", "numeric"), ("lab_test_status", "text"),
    ("lab_test_reason", "text"), ("lab_test_confidence", "numeric"), ("lab_test_context", "jsonb"), ("lab_test_preparation", "jsonb"),
    ("lab_test_earlier_readings", "jsonb"),
    ("lab_preparation", "jsonb"), ("visits", "jsonb"), ("latest_visit_date", "text"), ("advice", "jsonb"), ("diagnoses", "jsonb"),
    ("medications", "jsonb"), ("vitals", "jsonb"), ("lab_results", "jsonb"), ("other_findings", "jsonb"), ("not_extracted", "jsonb"),
    ("notice", "text"), ("raw_result", "jsonb NOT NULL"),
]
COLUMN_NAMES = [c for c, _t in _COLUMNS]
_JSONB = {c for c, t in _COLUMNS if t.startswith("jsonb")}

DDL = ("CREATE TABLE IF NOT EXISTS prescription_flat (\n    id bigserial PRIMARY KEY,\n    inserted_at timestamptz NOT NULL DEFAULT now(),\n    "
       + ",\n    ".join(f"{c} {t}" for c, t in _COLUMNS)
       + "\n);\n"
       "CREATE UNIQUE INDEX IF NOT EXISTS prescription_flat_doc_test_uq ON prescription_flat (document_id, COALESCE(lab_test_seq, 0));\n"
       "CREATE INDEX IF NOT EXISTS prescription_flat_token_phone ON prescription_flat (intake_token_no, intake_phone);\n"
       "CREATE INDEX IF NOT EXISTS prescription_flat_patient ON prescription_flat (intake_phone, patient_name);\n"
       "CREATE INDEX IF NOT EXISTS prescription_flat_test ON prescription_flat (lab_test_standard_name);\n"
       "CREATE INDEX IF NOT EXISTS prescription_flat_status ON prescription_flat (result_status, lab_test_status);\n")

_ready = False


def ensure_table() -> None:
    """Create the table when it is not there (idempotent; the same SQL as the migration)."""
    global _ready
    if _ready:
        return
    with session_scope() as s:
        for stmt in [x for x in DDL.split(";\n") if x.strip()]:
            s.execute(text(stmt))
    _ready = True


def _v(x: Any) -> Any:
    return x.get("value") if isinstance(x, dict) and "value" in x else x


def _st(x: Any) -> str | None:
    return x.get("status") if isinstance(x, dict) else None


def _bool(x: Any) -> bool | None:
    x = _v(x)
    return x if isinstance(x, bool) else None


def _num(x: Any) -> float | None:
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


_INTERVAL = re.compile(r"(?i)\b(?:after|in|within|r/?v|rev(?:iew)?)\b[^0-9]{0,12}(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*(day|d|week|wk|w|month|mo|m|year|yr|y)s?\b")


def _booking(fu: Any, doctor_name: Any) -> dict[str, Any]:
    """What the screen derives from the follow-up: needed or not, when, and whether to bring reports."""
    fu = fu if isinstance(fu, dict) else {}
    as_written = fu.get("text") or _v(fu) or ""
    as_written = as_written if isinstance(as_written, str) else ""
    kind, iv, unit, ivmax, date = (fu.get(k) for k in ("kind", "interval_value", "interval_unit", "interval_value_max", "date"))
    if kind is None and iv is None and as_written:
        m = _INTERVAL.search(as_written)               # "Review after 2 months", "after 2-3 weeks", "r/v 10 days": the interval as written
        if m:
            kind, iv, ivmax = "interval", float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
            unit = {"d": "days", "w": "weeks", "m": "months", "y": "years"}[m.group(3)[0].lower()]
    if kind == "as_needed":
        needed, when = "only_if_needed", "as needed"
    elif kind == "interval" and iv is not None:
        span = f"{iv:g}-{ivmax:g}" if ivmax else f"{iv:g}"
        needed, when = "yes", f"after {span} {unit or ''}".strip()
    elif kind == "date" and date:
        needed, when = "yes", f"on {date}"
    elif as_written:
        needed, when = "yes", "time not clear: please read the note"
    else:
        needed, when = "none_written", None
    return {"booking_needed": needed, "booking_with_doctor": _v(doctor_name), "booking_as_written": as_written or None, "booking_when_text": when,
            "booking_kind": kind, "booking_interval_value": _num(iv), "booking_interval_value_max": _num(ivmax), "booking_interval_unit": unit,
            "booking_date": date, "booking_next_date_text": fu.get("next_date"), "booking_next_date_iso": fu.get("next_date_iso"),
            "booking_bring_reports": bool(re.search(r"(?i)\b(?:bring|report|result)s?\b", as_written)) if as_written else None,
            "booking_status": _st(fu), "booking_reason": fu.get("reason")}


def flatten(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows (one per lab test the lists place, or one empty-test row) for one result.v1."""
    p, d, o = result.get("patient") or {}, result.get("doctor") or {}, result.get("organization") or {}
    intake, prov, quality, source = result.get("intake") or {}, result.get("provenance") or {}, result.get("quality") or {}, result.get("source") or {}
    eng = prov.get("engines") or {}
    clinic = d.get("clinic") or {}
    visits = result.get("visits") or []
    name = p.get("name")
    base: dict[str, Any] = {
        "document_id": result.get("document_id"), "result_schema_version": result.get("schema_version"), "filename": result.get("filename"),
        "source_channel": source.get("channel"), "source_original_filename": source.get("original_filename"),
        "intake_token_no": intake.get("token_no"), "intake_phone": intake.get("phone"), "intake_patient_name": intake.get("patient_name"),
        "intake_name_read": intake.get("name_read"), "intake_name_confirmed": intake.get("name_confirmed"),
        "intake_name_confirmed_by": intake.get("name_confirmed_by"), "intake_name_candidates": intake.get("name_candidates"),
        "doc_type": result.get("doc_type"), "is_handwritten": result.get("is_handwritten"), "page_count": result.get("page_count"),
        "result_status": result.get("status"), "needs_check_count": result.get("needs_check_count"),
        "extraction_incomplete": result.get("extraction_incomplete"), "quality_passed": quality.get("passed"), "quality_reasons": quality,
        "flags": result.get("flags"), "vlm_served": eng.get("vlm_served"), "vlm_revision": eng.get("vlm_revision"),
        "prompt_version": prov.get("prompt_version"), "printed_ocr_engine": eng.get("printed_ocr"), "handwriting_engine": eng.get("vlm_served") or eng.get("vlm_configured"),
        "patient_name": _v(name), "patient_name_status": _st(name), "patient_name_confidence": _num(name.get("confidence")) if isinstance(name, dict) else None,
        "patient_age_text": _v(p.get("age_text")), "patient_age_status": _st(p.get("age_text")),
        "patient_dob": _v(p.get("dob")), "patient_dob_status": _st(p.get("dob")),
        "patient_sex": _v(p.get("sex")), "patient_sex_status": _st(p.get("sex")),
        "patient_phone": _v(p.get("phone")), "patient_phone_status": _st(p.get("phone")),
        "patient_address": _v(p.get("address")), "patient_address_status": _st(p.get("address")),
        "patient_mrn": _v(p.get("mrn")), "patient_mrn_status": _st(p.get("mrn")),
        "patient_abha_id": _v(p.get("abha_id")), "patient_abha_status": _st(p.get("abha_id")),
        "org_name": _v(o.get("name")), "org_name_status": _st(o.get("name")), "org_address": _v(o.get("address")), "org_address_status": _st(o.get("address")),
        "org_phone": _v(o.get("phone")), "org_phone_status": _st(o.get("phone")),
        "doctor_name": _v(d.get("name")), "doctor_name_status": _st(d.get("name")),
        "doctor_department": _v(d.get("department")), "doctor_department_status": _st(d.get("department")),
        "doctor_designation": _v(d.get("designation")), "doctor_designation_status": _st(d.get("designation")),
        "doctor_qualification": _v(d.get("qualification")), "doctor_qualification_status": _st(d.get("qualification")),
        "doctor_reg_no": _v(d.get("reg_no")), "doctor_reg_no_status": _st(d.get("reg_no")),
        "doctor_clinic_name": _v(clinic.get("name")), "doctor_clinic_address": _v(clinic.get("address")), "doctor_clinic_phone": _v(clinic.get("phone")),
        "doctor_stamp_present": _bool(d.get("stamp_present")), "doctor_signature_present": _bool(d.get("signature_present")),
        **_booking(result.get("follow_up"), d.get("name")),
        "lab_preparation": result.get("lab_preparation"), "visits": visits, "latest_visit_date": next((v.get("date") for v in visits if v.get("is_latest")), None),
        "advice": result.get("advice"), "diagnoses": result.get("diagnoses"), "medications": result.get("medications"), "vitals": result.get("vitals"),
        "lab_results": result.get("lab_results"), "other_findings": result.get("other"), "not_extracted": result.get("not_extracted"),
        "notice": result.get("notice"), "raw_result": result,
    }
    tests = [t for t in result.get("lab_tests") or []
             if t.get("status") != "rejected" and (t.get("gate_recognised") or t.get("standard_name") or t.get("code"))]
    if not tests:
        return [{**base, "row_kind": "no_lab_test", "lab_test_seq": None}]
    rows = []
    for i, t in enumerate(tests, start=1):
        rows.append({**base, "row_kind": "lab_test", "lab_test_seq": i, "lab_test_fact_id": t.get("fact_id"), "lab_test_as_written": t.get("as_written"),
                     "lab_test_text": t.get("text"), "lab_test_standard_name": t.get("standard_name"), "lab_test_standard_source": t.get("standard_source"),
                     "lab_test_code": t.get("code"), "lab_test_code_system": t.get("code_system"), "lab_test_code_display": t.get("code_display"),
                     "lab_test_code_status": t.get("code_status"), "lab_test_gate_recognised": t.get("gate_recognised"),
                     "lab_test_page_support": _num(t.get("page_support")), "lab_test_status": t.get("status"), "lab_test_reason": t.get("reason"),
                     "lab_test_confidence": _num(t.get("confidence")), "lab_test_context": t.get("context"),
                     "lab_test_preparation": t.get("preparation"), "lab_test_earlier_readings": t.get("earlier_readings")})
    return rows


def _param(col: str, value: Any) -> Any:
    if col in _JSONB:
        return None if value is None else json.dumps(value, ensure_ascii=False, default=str)
    return value


def save_result(result: dict[str, Any]) -> int:
    """Replace this document's rows with those of ``result``. Returns the number of rows written."""
    ensure_table()
    rows = flatten(result)
    doc = str(result.get("document_id"))
    with session_scope() as s:
        s.execute(text("DELETE FROM prescription_flat WHERE document_id = CAST(:d AS uuid)"), {"d": doc})
        for r in rows:
            cols = [c for c in COLUMN_NAMES if c in r and r[c] is not None]
            holders = [f"CAST(:{c} AS jsonb)" if c in _JSONB else f"CAST(:{c} AS uuid)" if c == "document_id" else f":{c}" for c in cols]
            s.execute(text("INSERT INTO prescription_flat (" + ", ".join(cols) + ") VALUES (" + ", ".join(holders) + ")"),
                      {c: _param(c, r[c]) for c in cols})
    return len(rows)


def save_document(document_id: str) -> int:
    """Render the document's result and store its rows. Best effort: a failure here never costs the read (it is logged)."""
    try:
        from .json_connector import get_connector
        return save_result(get_connector().render(document_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("flat_table_not_saved", document_id=document_id, error=str(exc)[:200])
        return 0


# the columns the screen shows (every column is in the CSV)
SCREEN_COLUMNS = ["intake_token_no", "intake_phone", "patient_name", "patient_name_status", "patient_age_text", "patient_sex", "doctor_name",
                  "doctor_department", "doctor_clinic_name", "doc_type", "result_status", "latest_visit_date", "lab_test_seq", "lab_test_as_written",
                  "lab_test_standard_name", "lab_test_status", "lab_test_reason", "booking_needed", "booking_when_text", "booking_as_written", "document_id"]


def search(phone: str | None = None, token: str | None = None, limit: int = 500, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Rows for a mobile number and / or a token number (digits only for the number; the token ignores case). Newest first."""
    ensure_table()
    digits = re.sub(r"\D", "", phone or "")[-10:]
    tok = (token or "").strip()
    if not digits and not tok:
        return []
    cols = [c for c in (columns or COLUMN_NAMES) if c in COLUMN_NAMES and c != "raw_result"]
    where, args = [], {"lim": int(max(1, min(limit, 5000)))}
    if digits:
        where.append("(right(regexp_replace(coalesce(intake_phone,''), '[^0-9]', '', 'g'), 10) LIKE :p "
                     "OR right(regexp_replace(coalesce(patient_phone,''), '[^0-9]', '', 'g'), 10) LIKE :p)")
        args["p"] = f"%{digits}%"
    if tok:
        where.append("upper(coalesce(intake_token_no,'')) LIKE upper(:t)")
        args["t"] = f"%{tok}%"
    sql = (f"SELECT {', '.join(cols)}, inserted_at FROM prescription_flat WHERE {' AND '.join(where)} "
           "ORDER BY inserted_at DESC, document_id, lab_test_seq NULLS FIRST LIMIT :lim")
    with session_scope() as s:
        return [dict(r) for r in s.execute(text(sql), args).mappings().all()]
