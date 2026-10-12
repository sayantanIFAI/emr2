"""The hand-off of a read prescription to the lab billing app (output/billing_hook.py)."""
from __future__ import annotations

import json

import httpx

from cdi_adapter.config import settings
from cdi_adapter.output import billing_hook as H

RESULT = {
    "document_id": "doc-9",
    "intake": {"token_no": "t12", "phone": "9830012345", "patient_name": "Ananya Sen"},
    "patient": {"name": {"value": "Ananya Sen"}, "age_text": {"value": "29 yrs"}, "sex": {"value": "Female"}, "phone": {"value": "0000"}, "address": {"value": None}},
    "doctor": {"name": {"value": "Dr S Chatterjee"}, "department": {"value": "General Medicine"}},
    "lab_tests": [
        {"as_written": "USG (whole Abdomen)", "standard_name": "USG whole abdomen", "status": "needs_check", "reason": "read again"},
        {"as_written": "Ca2+", "standard_name": "Calcium", "status": "needs_check"},
        {"as_written": "Br Av TSH FT4", "standard_name": None, "status": "rejected", "reason": "not confirmed"},
    ],
}


def test_the_payload_has_the_patient_the_doctor_and_only_the_tests_that_are_not_rejected():
    p = H.build_payload(RESULT)
    assert p["document_id"] == "doc-9" and p["token_no"] == "t12" and p["phone"] == "9830012345"       # the typed number, not the page's
    assert p["patient"] == {"name": "Ananya Sen", "age_text": "29 yrs", "sex": "Female", "address": None}
    assert p["doctor"] == {"name": "Dr S Chatterjee", "department": "General Medicine"}
    assert [t["as_written"] for t in p["tests"]] == ["USG (whole Abdomen)", "Ca2+"]


def test_a_prescription_with_no_lab_test_is_not_sent():
    assert H.build_payload({**RESULT, "lab_tests": [{"as_written": "x", "status": "rejected"}]}) is None
    assert H.build_payload({**RESULT, "lab_tests": []}) is None


def test_it_posts_to_the_billing_app_with_the_token_and_returns_its_answer(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["token"], seen["body"] = str(request.url), request.headers.get("x-api-token"), json.loads(request.content)
        return httpx.Response(201, json={"bill_id": 7, "created": True})

    monkeypatch.setattr(settings, "billing_url", "http://127.0.0.1:8090/")
    monkeypatch.setattr(settings, "billing_token", "tok")
    out = H.send(RESULT, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert out == {"bill_id": 7, "created": True}
    assert seen["url"] == "http://127.0.0.1:8090/api/lab-bills" and seen["token"] == "tok" and len(seen["body"]["tests"]) == 2


def test_billing_off_or_down_never_costs_the_prescription(monkeypatch):
    monkeypatch.setattr(settings, "billing_url", "")
    assert H.send(RESULT) is None                                                 # off: nothing is sent
    monkeypatch.setattr(settings, "billing_url", "http://127.0.0.1:9")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert H.send(RESULT, client=httpx.Client(transport=httpx.MockTransport(boom))) is None      # down: logged, skipped
    monkeypatch.setattr(settings, "billing_url", "http://x")
    assert H.send(RESULT, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))) is None
