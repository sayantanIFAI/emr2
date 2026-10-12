"""Hand a read prescription to the lab billing app.

When a prescription has been read (``CDI_BILLING_URL`` set), its patient, doctor and lab tests are sent to ``POST <url>/api/lab-bills`` (header
``X-Api-Token``). The billing app makes a DRAFT lab bill from them; every test stays "to verify" there until a person confirms it. This only SENDS: nothing
here prices, bills or accepts a test, a rejected test is never sent, and a failed call never costs the prescription (it is logged and skipped)."""
from __future__ import annotations

import threading
from typing import Any

import httpx

from ..config import settings
from ..logging import get_logger

log = get_logger(__name__)


def _val(x: Any) -> str | None:
    """The text of a result value object ({"value": ...}) or a plain value; None when empty."""
    v = x.get("value") if isinstance(x, dict) else x
    s = str(v).strip() if v not in (None, "") else ""
    return s or None


def build_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """The billing app's input from a rendered result.v1, or None when the prescription has no lab test to bill."""
    tests = [{"as_written": t.get("as_written"), "standard_name": t.get("standard_name"), "status": t.get("status"), "reason": t.get("reason")}
             for t in result.get("lab_tests") or [] if isinstance(t, dict) and t.get("status") != "rejected" and (t.get("as_written") or t.get("standard_name"))]
    if not tests:
        return None
    intake = result.get("intake") or {}
    patient, doctor = result.get("patient") or {}, result.get("doctor") or {}
    return {
        "document_id": str(result.get("document_id") or "") or None,
        "token_no": intake.get("token_no"),
        "phone": intake.get("phone") or _val(patient.get("phone")),          # the mobile number typed at upload, never one read from the page
        "patient": {"name": intake.get("patient_name") or _val(patient.get("name")), "age_text": _val(patient.get("age_text")), "sex": _val(patient.get("sex")),
                    "address": _val(patient.get("address"))},
        "doctor": {"name": _val(doctor.get("name")), "department": _val(doctor.get("department"))},
        "tests": tests,
    }


def send(result: dict[str, Any], client: httpx.Client | None = None) -> dict[str, Any] | None:
    """POST the prescription to the billing app. Returns its answer, or None when billing is off, there is nothing to bill, or the call failed."""
    url = (settings.billing_url or "").strip().rstrip("/")
    if not url:
        return None
    payload = build_payload(result)
    if payload is None:
        return None
    try:
        c = client or httpx.Client(timeout=settings.billing_timeout_s)
        r = c.post(f"{url}/api/lab-bills", json=payload, headers={"X-Api-Token": settings.billing_token})
        r.raise_for_status()
        out = r.json()
        log.info("billing_sent", document_id=payload["document_id"], tests=len(payload["tests"]), bill=out.get("bill_id"))
        return out
    except Exception as exc:  # noqa: BLE001 - billing is an extra: it must never cost the prescription
        log.warning("billing_failed", document_id=payload.get("document_id"), error=str(exc)[:160])
        return None


def notify(document_id: str) -> None:
    """Render the finished result of a document and send it (called after the result is complete)."""
    if not (settings.billing_url or "").strip():
        return
    try:
        from .json_connector import get_connector

        send(get_connector().render(document_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("billing_render_failed", document_id=document_id, error=str(exc)[:160])


def notify_async(document_id: str) -> None:
    """The same, on a background thread: the reading of the next prescription never waits for the billing app."""
    if (settings.billing_url or "").strip():
        threading.Thread(target=notify, args=(document_id,), name="billing-notify", daemon=True).start()
