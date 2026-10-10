"""The deployment surface: sign-in in front of everything, review / FHIR paths closed by default,
the admin upload page, and the file listener."""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from cdi_adapter.config import settings
from cdi_adapter.webapp import app as webapp
from cdi_adapter.webapp import surface


def _basic(user: str, pw: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "s3cret-pass")
    monkeypatch.setattr(settings, "admin_user", "admin")
    monkeypatch.setattr(settings, "review_ui_enabled", False)
    monkeypatch.setattr(settings, "fhir_enabled", False)
    monkeypatch.setattr(surface.asyncio, "sleep", _no_sleep)      # the guess delay is not under test
    return TestClient(webapp.app)


async def _no_sleep(_s: float) -> None:
    return None


def test_everything_needs_sign_in_except_healthz(client):
    assert client.get("/").status_code == 401
    assert client.get("/api/upload/limits").status_code == 401
    assert client.get("/api/jobs/abc").status_code == 401
    assert client.get("/api/jobs/abc/result.json").status_code == 401
    r = client.get("/")
    assert r.headers["www-authenticate"].startswith("Basic")
    assert client.get("/healthz").status_code in (200, 503)         # open (db may be absent in CI)


def test_wrong_credentials_are_refused_right_ones_pass(client):
    assert client.get("/", headers=_basic("admin", "nope")).status_code == 401
    assert client.get("/", headers=_basic("root", "s3cret-pass")).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic !!!not-base64"}).status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer s3cret-pass"}).status_code == 401
    assert client.get("/", headers=_basic("admin", "s3cret-pass")).status_code == 200


def test_the_admin_page_has_upload_only(client):
    html = client.get("/", headers=_basic("admin", "s3cret-pass")).text
    assert "api/jobs" in html and "Read prescriptions" in html
    for word in ("/reviewer", "/admin", "api/registry", "FHIR", "workbasket", "CFP-"):
        assert word not in html


@pytest.mark.parametrize("path", [
    "/review", "/reviewer", "/admin", "/api/reviewer/workbasket", "/api/admin/overview",
    "/api/review/tasks", "/api/facts/x/review", "/api/corrections", "/api/corrections/export",
    "/api/doctors/x/profile", "/api/registry/lookup", "/api/patients/x/fhir",
    "/api/jobs/j1/facts", "/api/jobs/j1/generate", "/api/jobs/j1/fhir", "/api/jobs/j1/fhir/download",
    "/api/documents/d1/bundle", "/api/documents/d1/evidence", "/api/documents/d1/field-records",
])
def test_review_and_fhir_paths_are_closed_by_default(client, path):
    h = _basic("admin", "s3cret-pass")
    assert client.get(path, headers=h).status_code == 404
    assert client.post(path, headers=h).status_code == 404


def test_the_flags_open_their_paths(monkeypatch):
    monkeypatch.setattr(settings, "review_ui_enabled", False)
    monkeypatch.setattr(settings, "fhir_enabled", False)
    assert surface.is_closed("/reviewer") and surface.is_closed("/api/jobs/j/fhir")
    monkeypatch.setattr(settings, "review_ui_enabled", True)
    assert not surface.is_closed("/reviewer") and surface.is_closed("/api/jobs/j/fhir")
    monkeypatch.setattr(settings, "fhir_enabled", True)
    assert not surface.is_closed("/api/jobs/j/fhir")


def test_what_the_admin_needs_is_not_closed(monkeypatch):
    monkeypatch.setattr(settings, "review_ui_enabled", False)
    monkeypatch.setattr(settings, "fhir_enabled", False)
    for p in ("/", "/healthz", "/api/upload/limits", "/api/jobs", "/api/jobs/j1", "/api/jobs/j1/result.json",
              "/api/documents/d1/result.json", "/api/documents/d1/original", "/api/documents/d1/pages/1"):
        assert not surface.is_closed(p), p


def test_no_password_is_allowed_only_in_dev(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(settings, "env", "dev")
    surface.require_auth_configured()
    monkeypatch.setattr(settings, "env", "runpod")
    with pytest.raises(RuntimeError, match="CDI_ADMIN_PASSWORD"):
        surface.require_auth_configured()
    monkeypatch.setattr(settings, "admin_password", "x")
    surface.require_auth_configured()


def test_no_password_in_dev_means_no_sign_in(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "")
    assert TestClient(webapp.app).get("/").status_code == 200


def test_fhir_outbox_is_not_filled_unless_enabled(monkeypatch):
    from cdi_adapter.persist import normalized

    class Boom:
        def execute(self, *a, **k):
            raise AssertionError("must not touch the outbox")

    monkeypatch.setattr(settings, "fhir_enabled", False)
    normalized.enqueue_fhir(Boom(), patient_id="p", document_id="d", reason="validated")


def test_the_upload_screens_patient_look_ups_are_open_while_the_reviewer_surface_stays_closed():
    from cdi_adapter.webapp import surface
    for p in ("/api/intake/search", "/api/intake/prescriptions", "/api/intake/existing", "/api/mappings/lab"):
        assert not surface.is_closed(p), p
    assert surface.is_closed("/api/patients/123/fhir") and surface.is_closed("/api/patients/search")
