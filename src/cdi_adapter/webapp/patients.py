"""Finding patients and their prescriptions on the upload screen (no dropdown of thousands: type a part of the mobile number).

A patient here is a (mobile number, name) pair, as the front desk sees it: the mobile number is typed with the upload, the
name is read from the page. Two people who share a phone are two patients. Everything is plain SQL on ``source_document``
(migration 0010); results are capped, so a search never returns thousands of rows.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from ..db import session_scope

SEARCH_LIMIT = 10
LIST_LIMIT = 100
from ..names import alike, name_key  # noqa: F401  (name_key is re-exported for callers of this module)

SAME_NAME = 0.78            # handwriting is read a little differently each time ("Onkar" / "Oukar"): near names are one patient


def same_patient(a: str | None, b: str | None) -> bool:
    """Two readings of a name on the SAME mobile number are one patient when they are alike (an empty name matches only an empty one)."""
    return alike(a, b, SAME_NAME)


def _cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge rows of one mobile number whose names are alike; the shown name is the one read most often."""
    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: -x["prescriptions"]):
        for g in out:
            if g["phone"] == r["phone"] and same_patient(g["name"], r["name"]):
                g["prescriptions"] += r["prescriptions"]
                g["last_uploaded"] = max(g["last_uploaded"] or "", r["last_uploaded"] or "") or None
                g["names"].append(r["name"])
                break
        else:
            out.append({**r, "names": [r["name"]]})
    out.sort(key=lambda g: g["last_uploaded"] or "", reverse=True)
    return out


def _iso(v: Any) -> str | None:
    return v.isoformat() if v is not None else None


def search(q: str) -> list[dict[str, Any]]:
    """Patients whose mobile number STARTS WITH the digits typed, or (when letters are typed) whose name contains the text.
    One row per (mobile, name): ``{phone, name, prescriptions, last_uploaded}``, newest first."""
    q = (q or "").strip()
    digits = re.sub(r"\D", "", q)
    if digits and not re.search(r"[A-Za-z]", q):
        if len(digits) < 2:
            return []
        where, arg = "phone LIKE :a", {"a": digits[:10] + "%"}
    elif len(q) >= 2:
        where, arg = "patient_name ILIKE :a", {"a": "%" + re.sub(r"[%_\\]", "", q)[:60] + "%"}
    else:
        return []
    with session_scope() as sess:
        rows = sess.execute(text(
            f"SELECT phone, coalesce(patient_name, '') AS name, count(*) AS n, max(ingested_at) AS last "
            f"FROM source_document WHERE phone IS NOT NULL AND {where} "
            f"GROUP BY phone, coalesce(patient_name, '') ORDER BY max(ingested_at) DESC LIMIT {SEARCH_LIMIT}"), arg).mappings().all()
    merged = _cluster([{"phone": r["phone"], "name": r["name"] or None, "prescriptions": int(r["n"]), "last_uploaded": _iso(r["last"])}
                       for r in rows])
    return [{k: g[k] for k in ("phone", "name", "prescriptions", "last_uploaded")} for g in merged][:SEARCH_LIMIT]


def prescriptions(phone: str, name: str | None = None) -> list[dict[str, Any]]:
    """The prescriptions of one mobile number (and one name, when given), newest first."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) != 10:
        return []
    sql = ("SELECT id, token_no, original_filename, ingested_at, status, page_count, patient_name, upload_job_id "
           f"FROM source_document WHERE phone = :p ORDER BY ingested_at DESC LIMIT {LIST_LIMIT}")
    with session_scope() as sess:
        rows = sess.execute(text(sql), {"p": digits}).mappings().all()
    if name is not None:                                  # the patient asked for: the names alike to it on this number
        rows = [r for r in rows if same_patient(r["patient_name"], name or None)]
    return [{"document_id": str(r["id"]), "token_no": r["token_no"], "filename": r["original_filename"],
             "uploaded": _iso(r["ingested_at"]), "status": r["status"], "pages": r["page_count"],
             "patient_name": r["patient_name"], "job_id": r["upload_job_id"]} for r in rows]


def confirm_name(document_id: str, name: str, by: str) -> dict[str, Any] | None:
    """A person confirms (or corrects) the patient's name on a prescription. What was READ stays in ``name_read``; the shown name
    becomes the confirmed one and later re-reads never replace it. Returns the new values, or None for an unknown document."""
    import uuid

    from .. import repo

    try:
        uuid.UUID(document_id)
    except ValueError:
        return None
    with session_scope() as sess:
        row = sess.execute(text(
            "UPDATE source_document SET name_read = coalesce(name_read, patient_name), patient_name = :n, "
            "name_confirmed_by = :b, name_confirmed_at = now() WHERE id = :d "
            "RETURNING phone, token_no, name_read, name_confirmed_at"),
            {"n": name, "b": by[:60], "d": document_id}).mappings().first()
        if not row:
            return None
        repo.write_audit(sess, actor=by[:60], action="update", entity="source_document", entity_id=document_id,
                         detail={"patient_name_confirmed": name, "name_read": row["name_read"]})
    try:
        from ..output import flat_table
        flat_table.save_document(document_id)               # the table shows the confirmed name
    except Exception:  # noqa: BLE001 - an extra: never cost the confirmation
        pass
    return {"document_id": document_id, "patient_name": name, "name_read": row["name_read"], "phone": row["phone"],
            "token_no": row["token_no"], "name_confirmed": True, "name_confirmed_by": by[:60]}


def token_conflict(token: str, phone: str) -> dict[str, Any] | None:
    """A token number is issued once a day. When this token was already used TODAY (the clinic's day) for a DIFFERENT mobile number,
    ``{phone, name, when}`` of that earlier use; None otherwise. The same token with the SAME mobile number is not a conflict: that is
    more pages of one patient. Never called for a token that is empty."""
    from ..config import settings

    if not settings.token_unique_per_day or not token:
        return None
    with session_scope() as sess:
        row = sess.execute(text(
            "SELECT phone, patient_name, ingested_at FROM source_document "
            "WHERE upper(token_no) = upper(:t) AND phone IS DISTINCT FROM :p "
            "AND (ingested_at AT TIME ZONE :tz)::date = (now() AT TIME ZONE :tz)::date "
            "ORDER BY ingested_at DESC LIMIT 1"),
            {"t": token, "p": phone, "tz": settings.clinic_timezone}).mappings().first()
    if not row:
        return None
    return {"phone": row["phone"], "name": row["patient_name"], "when": _iso(row["ingested_at"])}


def token_message(token: str, c: dict[str, Any]) -> str:
    """The refusal, as one plain sentence the front desk can act on."""
    who = f"mobile {c['phone'][:5]} {c['phone'][5:]}" if c.get("phone") and len(c["phone"]) == 10 else "another mobile number"
    name = f" ({c['name']})" if c.get("name") else ""
    return (f"Token {token} was already used today for {who}{name}. A token is for one patient: check the token number, or use the "
            f"same mobile number if these are more pages of that patient.")


def existing(phone: str) -> dict[str, Any]:
    """What is already uploaded for this mobile number, so the screen can say so before another one is added."""
    rows = [r for r in prescriptions(phone) if r["status"] not in ("error", "quality_hold")]     # a failed read is not "uploaded"
    return {"count": len(rows), "prescriptions": rows[:20]}
