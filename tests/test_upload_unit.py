"""Upload screen rules (UP-S1): what the server accepts, in plain words. No database, no object store."""
from __future__ import annotations

import hashlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from cdi_adapter.config import settings
from cdi_adapter.ingest import pages
from cdi_adapter.webapp import app as webapp
from cdi_adapter.webapp import jobs, upload


@pytest.fixture(autouse=True)
def _abha_checks_on(monkeypatch):
    """The pod runs with ABHA off (CDI_ABHA_ENABLED=false); these tests are about the form that still validates one."""
    monkeypatch.setattr(settings, "abha_enabled", True)


def _png(w: int = 900, h: int = 1200, text: str = "Tab Metformin 500 mg") -> bytes:
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).text((20, 20), text, fill="black")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _jpg(w: int = 900, h: int = 1200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG")
    return buf.getvalue()


PDF = b"%PDF-1.7\n1 0 obj<<>>endobj\n"
DOCX = b"PK\x03\x04" + b"\x00" * 50


@pytest.fixture
def client(monkeypatch):
    """The real app with job creation replaced, so nothing touches a database."""
    calls: list[dict] = []

    def fake_create_job(abha, files, patient_ref=None, *, parts=None, idempotency_key=None, token_no=None, phone=None, department=None):
        calls.append({"abha": abha, "files": files, "patient_ref": patient_ref, "parts": parts,
                      "key": idempotency_key, "token_no": token_no, "phone": phone})
        return "job123"

    monkeypatch.setattr(webapp, "create_job", fake_create_job)
    c = TestClient(webapp.app)
    c.calls = calls
    return c


INTAKE = {"token_no": "T-17", "phone": "9830011234"}


def _post(client, files, **data):
    return client.post("/api/jobs", files=[("files", f) for f in files], data={**INTAKE, **data})


# ------------------------------------------------------------------ limits endpoint


def test_limits_come_from_settings_not_from_the_page(client, monkeypatch):
    monkeypatch.setattr(settings, "upload_max_files", 4)
    monkeypatch.setattr(settings, "upload_max_file_bytes", 8_000_000)
    body = client.get("/api/upload/limits").json()
    assert body["max_files"] == 4 and body["max_file_mb"] == 8.0
    assert body["min_short_side_px"] == settings.quality_min_short_side_px
    assert ".pdf" in body["accept"] and ".docx" not in body["accept"]


# ------------------------------------------------------------------ AC3: wrong type


def test_a_docx_is_refused_in_plain_words(client):
    r = _post(client, [("rx.docx", DOCX, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")])
    assert r.status_code == 422
    assert r.json()["detail"].startswith("Please add a photo, PDF or scan.")
    assert not client.calls


def test_the_type_is_read_from_the_bytes_not_the_name(client):
    assert _post(client, [("scan.png", DOCX, "image/png")]).status_code == 422      # renamed docx
    assert _post(client, [("scan.png", b"MZ\x90\x00 exe", "image/png")]).status_code == 422
    assert _post(client, [("photo.txt", _png(), "text/plain")]).status_code == 202   # a real PNG
    assert _post(client, [("anim.gif", b"GIF89a....", "image/gif")]).status_code == 422


def test_pdf_png_jpeg_tiff_are_accepted(client):
    tif = io.BytesIO()
    Image.new("RGB", (700, 900), "white").save(tif, format="TIFF")
    for name, data in (("a.pdf", PDF), ("b.png", _png()), ("c.jpg", _jpg()), ("d.tif", tif.getvalue())):
        assert _post(client, [(name, data)]).status_code == 202, name


# ------------------------------------------------------------------ caps


def test_too_many_files_is_refused_before_any_bytes_are_read(client, monkeypatch):
    monkeypatch.setattr(settings, "upload_max_files", 2)
    r = _post(client, [("a.png", _png()), ("b.png", _png()), ("c.png", _png())])
    assert r.status_code == 422 and "up to 2 files" in r.json()["detail"]


def test_a_file_over_the_limit_is_refused_with_its_name(client, monkeypatch):
    monkeypatch.setattr(settings, "upload_max_file_bytes", 1000)
    r = _post(client, [("big.png", _png(), "image/png")])
    assert r.status_code == 413 and "'big.png' is larger than" in r.json()["detail"]


def test_the_total_is_capped_too(client, monkeypatch):
    one = len(_png())
    monkeypatch.setattr(settings, "upload_max_total_bytes", one + one // 2)
    r = _post(client, [("a.png", _png()), ("b.png", _png())])
    assert r.status_code == 413 and "Together the files are larger" in r.json()["detail"]


def test_an_empty_file_is_named(client):
    r = _post(client, [("empty.png", b"", "image/png")])
    assert r.status_code == 422 and "'empty.png' is empty" in r.json()["detail"]


def test_the_read_is_bounded_by_the_file_limit(monkeypatch):
    """A huge upload must not be read into memory in full: at most limit + 1 bytes per file."""
    asked: list[int] = []

    class Spy:
        filename = "x.png"

        async def read(self, n=-1):
            asked.append(n)
            return b"\x89PNG\r\n\x1a\n" + b"0" * 10

    import asyncio

    monkeypatch.setattr(webapp, "create_job", lambda *a, **k: "j")
    asyncio.run(webapp.submit_job(abha=None, patient_ref=None, token_no="T1", phone="9830011234", grouping="separate",
                                  files=[Spy()], idempotency_key=None))
    assert asked == [settings.upload_max_file_bytes + 1]


# ------------------------------------------------------------------ AC6 and free-text fields


@pytest.mark.parametrize("typed", ["abcd", "14-1111-2222-33A3", "14-1111-2222", "1234567890123456", "١٤-١١١١-٢٢٢٢-٣٣٣٣"])
def test_a_bad_abha_number_is_refused_with_a_plain_message(client, typed):
    r = _post(client, [("a.png", _png())], abha=typed)
    assert r.status_code == 422
    assert r.json()["detail"] == "The ABHA number must have 14 digits, for example 14-1111-2222-3333."
    assert not client.calls


@pytest.mark.parametrize("typed", ["14-1111-2222-3333", "14 1111 2222 3333", "14111122223333", " 14-1111-2222-3333 "])
def test_a_valid_abha_number_is_normalised(client, typed):
    assert _post(client, [("a.png", _png())], abha=typed).status_code == 202
    assert client.calls[-1]["abha"] == "14-1111-2222-3333"


def test_an_empty_abha_is_fine(client):
    assert _post(client, [("a.png", _png())], abha="").status_code == 202
    assert client.calls[-1]["abha"] is None


@pytest.mark.parametrize("ref", ["x" * 41, "CFP<script>", "a;b", "../../etc/passwd", "name\nnewline", "-leading"])
def test_a_patient_reference_is_short_and_plain(client, ref):
    r = _post(client, [("a.png", _png())], patient_ref=ref)
    assert r.status_code == 422 and "patient reference" in r.json()["detail"]


@pytest.mark.parametrize("ref", ["CFP-2026-000901", "14-1111-2222-3333", "9830011234", "jane.doe@abdm"])
def test_the_three_reference_kinds_pass(client, ref):
    assert _post(client, [("a.png", _png())], patient_ref=ref).status_code == 202
    assert client.calls[-1]["patient_ref"] == ref


def test_file_names_in_messages_and_logs_carry_no_path_or_control_characters():
    assert upload.display_name("C:\\Users\\x\\rx.png") == "rx.png"
    assert upload.display_name("../../etc/passwd") == "passwd"
    assert upload.display_name("a\x00b\nc.png") == "abc.png"
    assert len(upload.display_name("x" * 500 + ".png")) == 80
    assert upload.display_name("") == "document"


# ------------------------------------------------------------------ AC2: several pages = ONE prescription


def test_separate_is_todays_behaviour_one_document_per_file(client):
    r = _post(client, [("a.png", _png()), ("b.png", _png(text="page two"))])
    assert r.status_code == 202 and r.json() == {"job_id": "job123", "documents": 2}
    assert [n for n, _ in client.calls[-1]["files"]] == ["a.png", "b.png"]


def test_one_document_makes_the_pictures_the_pages_of_one_prescription(client):
    p1, p2, p3 = _png(900, 1200, "p1"), _jpg(1000, 1300), _png(800, 1100, "p3")
    r = _post(client, [("one.png", p1), ("two.jpg", p2), ("three.png", p3)], grouping="one_document")
    assert r.status_code == 202 and r.json() == {"job_id": "job123", "documents": 1}   # shape unchanged
    call = client.calls[-1]
    ((name, pdf),) = call["files"]
    assert name == "one (3 pages).pdf"
    # the PDF is one document with three pages, in the order sent, each at its own size
    out = pages.render_pages(pdf, "application/pdf")
    assert [(p.width_px, p.height_px) for p in out] == [(900, 1200), (1000, 1300), (800, 1100)]
    # the originals ride along, untouched, in order
    assert call["parts"] == [[("one.png", p1), ("two.jpg", p2), ("three.png", p3)]]


def test_a_single_picture_in_one_document_mode_is_not_wrapped(client):
    data = _png()
    assert _post(client, [("only.png", data)], grouping="one_document").status_code == 202
    assert client.calls[-1]["files"] == [("only.png", data)] and client.calls[-1]["parts"] == [None]


def test_a_pdf_cannot_be_mixed_into_one_prescription(client):
    r = _post(client, [("a.png", _png()), ("b.pdf", PDF)], grouping="one_document")
    assert r.status_code == 422 and "PDF already holds all of its pages" in r.json()["detail"]


def test_an_unknown_grouping_is_refused(client):
    r = _post(client, [("a.png", _png())], grouping="nonsense")
    assert r.status_code == 422 and "one prescription or separate" in r.json()["detail"]


def test_a_picture_that_cannot_be_decoded_reads_as_a_plain_message(client):
    broken = b"\x89PNG\r\n\x1a\n" + b"garbage"
    r = _post(client, [("a.png", _png()), ("b.png", broken)], grouping="one_document")
    assert r.status_code == 422 and "could not be read" in r.json()["detail"]
    assert "Traceback" not in r.text and "PIL" not in r.text


def test_the_same_pictures_assemble_to_the_same_document(client):
    a, b = _png(text="x"), _png(text="y")
    _post(client, [("a.png", a), ("b.png", b)], grouping="one_document")
    _post(client, [("a.png", a), ("b.png", b)], grouping="one_document")
    first, second = client.calls[-2]["files"][0][1], client.calls[-1]["files"][0][1]
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()   # dedupe keeps working


# ------------------------------------------------------------------ AC4: pressed twice -> one job


def test_the_same_idempotency_key_returns_the_same_job_and_creates_it_once(monkeypatch):
    made: list[str] = []
    monkeypatch.setattr(jobs, "_create_job", lambda jid, *a, **k: made.append(jid) or jid)
    monkeypatch.setattr(jobs, "_idem", {})
    first = jobs.create_job(None, [("a.png", b"x")], idempotency_key="send-1")
    again = jobs.create_job(None, [("a.png", b"x")], idempotency_key="send-1")
    other = jobs.create_job(None, [("a.png", b"x")], idempotency_key="send-2")
    assert first == again and other != first and len(made) == 2


def test_two_presses_at_the_same_instant_still_make_one_job(monkeypatch):
    import threading
    import time

    made: list[str] = []

    def slow_create(jid, *a, **k):
        time.sleep(0.05)               # widen the window in which a second request could slip in
        made.append(jid)
        return jid

    monkeypatch.setattr(jobs, "_create_job", slow_create)
    monkeypatch.setattr(jobs, "_idem", {})
    out: list[str] = []
    threads = [threading.Thread(target=lambda: out.append(
        jobs.create_job(None, [("a.png", b"x")], idempotency_key="same"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == 1 and len(made) == 1


def test_a_failed_start_releases_the_key_so_a_retry_works(monkeypatch):
    attempts: list[str] = []

    def flaky(jid, *a, **k):
        attempts.append(jid)
        if len(attempts) == 1:
            raise RuntimeError("database down")
        return jid

    monkeypatch.setattr(jobs, "_create_job", flaky)
    monkeypatch.setattr(jobs, "_idem", {})
    with pytest.raises(RuntimeError):
        jobs.create_job(None, [("a.png", b"x")], idempotency_key="k")
    assert jobs.create_job(None, [("a.png", b"x")], idempotency_key="k") == attempts[1]


def test_old_keys_expire(monkeypatch):
    monkeypatch.setattr(jobs, "_create_job", lambda jid, *a, **k: jid)
    monkeypatch.setattr(jobs, "_idem", {"old": (0.0, "stale-job")})     # long ago
    assert jobs.create_job(None, [("a.png", b"x")], idempotency_key="old") != "stale-job"


def test_the_key_header_is_passed_through_and_a_malformed_one_is_refused(client):
    assert _post(client, [("a.png", _png())]).status_code == 202
    assert client.calls[-1]["key"] is None
    r = client.post("/api/jobs", files=[("files", ("a.png", _png()))], data=INTAKE, headers={"Idempotency-Key": "send-123_x"})
    assert r.status_code == 202 and client.calls[-1]["key"] == "send-123_x"
    bad = client.post("/api/jobs", files=[("files", ("a.png", _png()))], data=INTAKE, headers={"Idempotency-Key": "x" * 80})
    assert bad.status_code == 422 and "reload the page" in bad.json()["detail"]


# ------------------------------------------------------------------ the original parts are kept


def test_store_parts_keeps_every_original_byte_exact_and_audits_their_hashes(monkeypatch):
    stored: dict[str, tuple[bytes, str]] = {}
    audit: list[dict] = []
    monkeypatch.setattr(upload.storage, "put_bytes",
                        lambda key, data, content_type="": stored.setdefault(key, (data, content_type)) and key)

    class Scope:
        def __enter__(self):
            return "sess"

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(upload, "session_scope", lambda: Scope())
    monkeypatch.setattr(upload.repo, "write_audit", lambda sess, **kw: audit.append(kw))
    p1, p2 = _png(text="a"), _jpg()
    upload.store_parts("doc-1", "ab" + "c" * 62, [("one.png", p1), ("two.jpg", p2)])
    assert stored["documents/ab/" + "ab" + "c" * 62 + "/parts/01.png"] == (p1, "image/png")
    assert stored["documents/ab/" + "ab" + "c" * 62 + "/parts/02.jpg"] == (p2, "image/jpeg")
    (entry,) = audit
    assert entry["entity_id"] == "doc-1" and entry["entity"] == "source_document"
    parts = entry["detail"]["original_parts"]
    assert [p["sha256"] for p in parts] == [hashlib.sha256(p1).hexdigest(), hashlib.sha256(p2).hexdigest()]
    assert [p["bytes"] for p in parts] == [len(p1), len(p2)]


def test_the_public_job_view_never_carries_the_originals():
    job = jobs.Job(id="j", abha=None)
    job.docs = [jobs.DocProg(filename="one (2 pages).pdf", parts=[("a.png", b"secret-bytes")])]
    assert "secret-bytes" not in repr(job.public()) and "parts" not in job.public()["documents"][0]


# ---- token number and mobile number (the front desk enters both before anything is uploaded)
@pytest.mark.parametrize("raw,expected", [("9830011234", "9830011234"), ("+91 98300 11234", "9830011234"),
                                          ("919830011234", "9830011234"), ("09830011234", "9830011234"), ("98300-11234", "9830011234")])
def test_a_mobile_number_is_kept_as_its_ten_digits(raw, expected):
    assert upload.clean_phone(raw) == expected


@pytest.mark.parametrize("raw", ["", "12345", "5830011234", "98300112345", "98300x1234", "+1 9830011234"])
def test_a_bad_mobile_number_is_refused_in_plain_words(raw):
    with pytest.raises(upload.UploadError):
        upload.clean_phone(raw)


@pytest.mark.parametrize("raw", ["", " ", "T 17", "../x", "x" * 21, "ü1"])
def test_a_bad_token_is_refused(raw):
    with pytest.raises(upload.UploadError):
        upload.clean_token(raw)


def test_the_upload_is_refused_without_the_token_or_the_mobile_number_and_both_are_passed_on(client):
    r = client.post("/api/jobs", files=[("files", ("a.png", _png()))], data={"phone": "9830011234"})
    assert r.status_code == 422 and "token" in r.json()["detail"].lower() and not client.calls
    r = client.post("/api/jobs", files=[("files", ("a.png", _png()))], data={"token_no": "T-1"})
    assert r.status_code == 422 and "mobile" in r.json()["detail"].lower() and not client.calls
    ok = _post(client, [("a.png", _png())])
    assert ok.status_code == 202 and client.calls[-1]["token_no"] == "T-17" and client.calls[-1]["phone"] == "9830011234"


@pytest.mark.parametrize("raw,expected", [("  Oukar   Chowdhury ", "Oukar Chowdhury"), ("Mr. R. K. Das", "Mr. R. K. Das"), ("D'Souza", "D'Souza"),
                                          ("Anne-Marie Roy", "Anne-Marie Roy"), ("সুমিত্রা দাস", "সুমিত্রা দাস")])
def test_a_typed_patient_name_is_cleaned(raw, expected):
    assert upload.clean_person_name(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "A", "x" * 81, "Asha123", "<b>Asha</b>", "Asha; DROP TABLE", "12345", "..", "Asha\u202e"])
def test_a_bad_patient_name_is_refused(raw):
    with pytest.raises(upload.UploadError):
        upload.clean_person_name(raw)


def test_the_name_confirmation_endpoint_validates_and_passes_who_confirmed(client, monkeypatch):
    from cdi_adapter.webapp import patients
    seen = {}

    def fake(document_id, name, by):
        seen.update(document_id=document_id, name=name, by=by)
        return {"document_id": document_id, "patient_name": name, "name_confirmed": True} if document_id == "doc-1" else None
    monkeypatch.setattr(patients, "confirm_name", fake)
    ok = client.post("/api/intake/name", json={"document_id": "doc-1", "name": "  Oukar   Chowdhury "})
    assert ok.status_code == 200 and seen["name"] == "Oukar Chowdhury" and seen["by"]
    assert client.post("/api/intake/name", json={"document_id": "doc-1", "name": "A1"}).status_code == 422
    assert client.post("/api/intake/name", json={"document_id": "nope", "name": "Asha Rao"}).status_code == 404


# ---- a token number is issued once a day: the same token for another mobile number is refused
def test_the_same_token_today_for_another_mobile_number_is_refused_in_plain_words_and_nothing_is_created(client, monkeypatch):
    from cdi_adapter.webapp import patients
    monkeypatch.setattr(patients, "token_conflict", lambda t, p: {"phone": "9830011234", "name": "Asha Rao", "when": "2026-10-08T05:00:00"})
    r = _post(client, [("a.png", _png())], token_no="T-17", phone="9831122334")
    assert r.status_code == 409 and not client.calls
    msg = r.json()["detail"]
    assert "Token T-17 was already used today" in msg and "98300 11234" in msg and "Asha Rao" in msg and "same mobile number" in msg


def test_the_same_token_with_the_same_mobile_number_is_more_pages_of_one_patient_and_is_allowed(client, monkeypatch):
    from cdi_adapter.webapp import patients
    seen = []
    monkeypatch.setattr(patients, "token_conflict", lambda t, p: seen.append((t, p)))            # None: no clash
    assert _post(client, [("a.png", _png())]).status_code == 202 and seen == [("T-17", "9830011234")]


def test_the_token_check_asks_the_database_for_the_clinics_day_and_a_different_mobile_only(monkeypatch):
    from contextlib import contextmanager
    from cdi_adapter.webapp import patients
    captured = {}

    class Res:
        def __init__(self, row): self.row = row
        def mappings(self): return self
        def first(self): return self.row

    class Sess:
        def execute(self, stmt, params):
            captured["sql"], captured["params"] = str(stmt), params
            return Res({"phone": "9830011234", "patient_name": None, "ingested_at": None})

    @contextmanager
    def scope():
        yield Sess()
    monkeypatch.setattr(patients, "session_scope", scope)
    from cdi_adapter.config import settings
    monkeypatch.setattr(settings, "token_unique_per_day", True)
    got = patients.token_conflict("t-17", "9831122334")
    assert got == {"phone": "9830011234", "name": None, "when": None}
    sql = captured["sql"]
    assert "upper(token_no) = upper(:t)" in sql and "phone IS DISTINCT FROM :p" in sql and "now() AT TIME ZONE :tz" in sql
    assert captured["params"]["tz"] == "Asia/Kolkata"
    monkeypatch.setattr(settings, "token_unique_per_day", False)
    assert patients.token_conflict("t-17", "9831122334") is None and patients.token_conflict("", "9831122334") is None


def test_the_token_check_endpoint_answers_conflict_or_free(client, monkeypatch):
    from cdi_adapter.webapp import patients
    monkeypatch.setattr(patients, "token_conflict", lambda t, p: {"phone": "9830011234", "name": None, "when": None} if t == "T-9" else None)
    bad = client.get("/api/intake/token-check", params={"token": "T-9", "phone": "9831122334"}).json()
    ok = client.get("/api/intake/token-check", params={"token": "T-10", "phone": "9831122334"}).json()
    assert bad["conflict"] and "Token T-9 was already used today" in bad["message"] and ok == {"conflict": False, "message": None}
    assert client.get("/api/intake/token-check", params={"token": "T 9", "phone": "x"}).status_code == 422
