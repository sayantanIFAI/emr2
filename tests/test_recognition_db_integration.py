"""Recognition v2 against a real Postgres (migration 0005): immutable evidence, the
recognition pipeline with fake engines, listener lifecycle + recovery, FHIR outbox,
dispatch. Needs only CDI_DATABASE_URL (object storage is faked)."""
from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import text

pytestmark = pytest.mark.skipif(not os.environ.get("CDI_DATABASE_URL"),
                                reason="CDI_DATABASE_URL not set")


@pytest.fixture()
def sess_scope():
    from cdi_adapter.db import session_scope

    return session_scope


@pytest.fixture()
def fake_store(monkeypatch):
    from cdi_adapter import storage

    blobs: dict[str, bytes] = {}
    monkeypatch.setattr(storage, "put_bytes",
                        lambda k, d, content_type="": blobs.__setitem__(k, d) or f"s3://t/{k}")
    monkeypatch.setattr(storage, "object_uri", lambda k: f"s3://t/{k}")
    monkeypatch.setattr(storage, "get_bytes", lambda k: blobs[k])
    monkeypatch.setattr(storage, "key_from_uri", lambda u: u.split("s3://t/", 1)[-1])
    return blobs


def _page_png() -> bytes:
    im = Image.new("RGB", (1400, 700), "white")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        f = ImageFont.load_default()
    d.text((80, 80), "Tab Telma 40 1-0-1", fill="black", font=f)
    d.line([(100, 400), (160, 380), (220, 420), (300, 390), (380, 415)], fill="black", width=4)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _doc(sess_scope, fake_store) -> tuple[str, dict]:
    from cdi_adapter import repo

    png = _page_png()
    fake_store["pages/p1.png"] = png
    with sess_scope() as s:
        did = str(repo.insert_source_document(
            s, sha256=hashlib.sha256(uuid.uuid4().bytes).hexdigest(), mime_type="image/png",
            object_uri="s3://t/raw", byte_size=len(png), source_channel="test",
            original_filename="t.png"))
        repo.insert_document_page(s, document_id=did, page_no=1, image_uri="s3://t/pages/p1.png",
                                  width_px=1400, height_px=700, dpi=200, preproc={})
        pg = repo.list_document_pages(s, did)[0]
    return did, pg


# ---------------- immutable evidence ----------------
def test_observation_is_append_only(sess_scope, fake_store):
    from cdi_adapter import repo

    did, pg = _doc(sess_scope, fake_store)
    with sess_scope() as s:
        oid = repo.insert_observations(s, [{
            "document_id": did, "page_id": pg["id"], "line_key": "p1:l0",
            "region_kind": "handwritten", "bbox": [1, 2, 3, 4], "engine": "qwen2.5-vl",
            "engine_version": "t", "raw_text": "Telma 40", "raw_confidence": 0.8}])[0]
    for sql in ("UPDATE ocr_observation SET raw_text='x' WHERE id=:i",
                "DELETE FROM ocr_observation WHERE id=:i"):
        with pytest.raises(Exception, match="append-only"):
            with sess_scope() as s:
                s.execute(text(sql), {"i": oid})
    with sess_scope() as s:           # documented erasure path works
        s.execute(text("SET LOCAL cdi.allow_evidence_erasure = 'on'"))
        s.execute(text("DELETE FROM ocr_observation WHERE id=:i"), {"i": oid})


def test_verified_fact_is_append_only(sess_scope):
    from cdi_adapter import repo

    with sess_scope() as s:
        vid = repo.insert_verified_fact(s, fact_id=uuid.uuid4(), fact_type="medication",
                                        governed_value={"x": 1},
                                        verification_method="auto_accepted")
    with pytest.raises(Exception, match="append-only"):
        with sess_scope() as s:
            s.execute(text("UPDATE verified_fact SET confidence=1 WHERE id=:i"), {"i": vid})


# ---------------- recognition pipeline ----------------
class _FakeHost:
    def rapid(self, png):
        from cdi_adapter.ocr.rapid import OcrLine
        return [OcrLine("Tab Telma 40 1-0-1", [80, 80, 460, 120], 0.97, [])]


def test_recognize_document_states_and_evidence(sess_scope, fake_store, monkeypatch):
    state, qwen_text = "single_engine", "Pregabalin 75"
    from cdi_adapter import repo
    from cdi_adapter.recognition import engines, ocrhost_client, pipeline

    did, _pg = _doc(sess_scope, fake_store)
    ocrhost_client.set_ocr_host(_FakeHost())
    monkeypatch.setattr(engines.QwenLineEngine, "recognize", lambda self, crops: [
        engines.Reading("qwen2.5-vl", "fake-qwen", qwen_text, None) for _ in crops])
    try:
        res = pipeline.recognize_document(did)
    finally:
        ocrhost_client.set_ocr_host(None)
    assert res.states.get("printed") and res.states.get(state)
    with sess_scope() as s:
        blocks = repo.list_ocr_blocks(s, did)
        obs = repo.list_current_observations(s, did)
    hw = [b for b in blocks if (b["recognition"] or {}).get("state") == state]
    assert hw and hw[0]["observation_ids"]
    assert {o["engine"] for o in obs} >= {"rapidocr", "qwen2.5-vl"}
    # a second run supersedes, never deletes
    ocrhost_client.set_ocr_host(_FakeHost())
    try:
        pipeline.recognize_document(did)
    finally:
        ocrhost_client.set_ocr_host(None)
    with sess_scope() as s:
        allo = s.execute(text("SELECT count(*) FROM ocr_observation WHERE document_id=:d"),
                         {"d": did}).scalar_one()
        cur = len(repo.list_current_observations(s, did))
    assert allo > cur


def test_adjudication_orders_but_keeps_disagreement(sess_scope, fake_store, monkeypatch):
    from cdi_adapter import repo
    from cdi_adapter.config import settings
    from cdi_adapter.ml import client as mlc
    from cdi_adapter.recognition import engines, ocrhost_client, pipeline

    class Adj:        # prefers the Qwen reading (#1) in both option orders
        def vlm_generate(self, image, prompt, *, max_tokens=8, json_schema=None):
            return "A" if "A: Pregabalin 150" in prompt else "B"

    monkeypatch.setattr(settings, "qwen_adjudication_enabled", True)
    monkeypatch.setattr(mlc, "get_client", lambda: Adj())
    monkeypatch.setattr(engines.QwenLineEngine, "recognize", lambda self, crops: [
        engines.Reading("qwen2.5-vl", "fake-qwen", "Pregabalin 150", None) for _ in crops])
    did, _ = _doc(sess_scope, fake_store)
    ocrhost_client.set_ocr_host(_FakeHost("Pregabalin 75"))
    try:
        pipeline.recognize_document(did)
    finally:
        ocrhost_client.set_ocr_host(None)
    with sess_scope() as s:
        blocks = repo.list_ocr_blocks(s, did)
        obs = repo.list_current_observations(s, did, engine="qwen2.5-vl-adjudicator")
    b = next(b for b in blocks if (b["recognition"] or {}).get("state") == "disagree")
    assert b["text"] == "Pregabalin 150 ⟂ Pregabalin 75"          # preferred first
    assert b["recognition"]["adjudication"]["verdict"] == "prefers"
    assert b["recognition"]["adjudication"]["advisory"] is True
    assert obs and obs[0]["raw_text"] == "B / A"                  # both orders -> reading #1


# ---------------- listener ----------------
def test_listener_lifecycle_and_three_retries(sess_scope, tmp_path, monkeypatch):
    from cdi_adapter.config import settings
    from cdi_adapter.listener import recovery, service
    from cdi_adapter.listener.connectors import LocalConnector

    monkeypatch.setattr(settings, "listener_stable_polls", 2)
    monkeypatch.setattr(settings, "listener_retry_base_seconds", 0)
    monkeypatch.setattr(settings, "listener_batch_wait_seconds", 0)      # flush partial batches
    conn = LocalConnector(str(tmp_path))
    conn.ensure_folders()
    tag = uuid.uuid4().hex[:8]
    (tmp_path / "inbox" / f"ok-{tag}.png").write_bytes(b"png-bytes-ok-" + tag.encode())
    (tmp_path / "inbox" / f"bad-{tag}.png").write_bytes(b"png-bytes-bad-" + tag.encode())
    (tmp_path / "inbox" / "~$lock.png").write_bytes(b"x")          # office lock file: ignored

    def fake_pipeline(raw, name, source_channel="listener"):
        if name.startswith("bad"):
            raise ConnectionError("model gateway timeout")
        return {"document_id": None, "facts": 3}

    monkeypatch.setattr(service, "run_pipeline", fake_pipeline)
    assert service.poll_once(conn) == []                 # first sighting: not stable yet
    assert sorted(service.poll_once(conn)) == ["completed", "error"]     # one batch of 2 (partial: flushed)
    assert (tmp_path / "success" / f"ok-{tag}.png").exists()
    assert (tmp_path / "error" / f"bad-{tag}.png").exists()
    assert not list((tmp_path / "error").glob("*.txt"))                 # reasons live in log/ only
    notes = list((tmp_path / "log").glob(f"bad-{tag}.png.*.run1.log"))
    assert len(notes) == 1 and "model gateway timeout" in notes[0].read_text()
    assert not list((tmp_path / "log").glob(f"ok-{tag}*"))              # successes write no log

    results = []
    for _ in range(3):
        results += [r["result"] for r in recovery.run_once(conn)]
    assert results == ["error", "error", "quarantine"]
    assert (tmp_path / "quarantine" / f"bad-{tag}.png").exists()
    assert recovery.run_once(conn) == []                 # never a 4th attempt
    with sess_scope() as s:
        att = s.execute(text("SELECT attempts, state FROM listener_file WHERE name=:n"),
                        {"n": f"bad-{tag}.png"}).one()
    assert tuple(att) == (3, "quarantine")


def test_listener_data_error_is_not_retried(sess_scope, tmp_path, monkeypatch):
    from cdi_adapter.config import settings
    from cdi_adapter.listener import recovery, service
    from cdi_adapter.listener.connectors import LocalConnector

    monkeypatch.setattr(settings, "listener_stable_polls", 1)
    monkeypatch.setattr(settings, "listener_batch_wait_seconds", 0)
    conn = LocalConnector(str(tmp_path))
    conn.ensure_folders()
    name = f"blurry-{uuid.uuid4().hex[:8]}.jpg"
    (tmp_path / "inbox" / name).write_bytes(b"jpeg" + name.encode())
    monkeypatch.setattr(service, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(
        service.DataError("rescan: page too blurred")))
    service.poll_once(conn)
    service.poll_once(conn)
    assert (tmp_path / "error" / name).exists()
    note = next((tmp_path / "log").glob(f"{name}.*.log")).read_text()
    assert "rescan" in note and "NO automatic retry" in note
    assert recovery.run_once(conn) == []


# ---------------- FHIR outbox agent ----------------
def test_fhir_agent_builds_versions_and_dead_letters(sess_scope, fake_store, monkeypatch):
    from cdi_adapter.agents import fhir_builder
    from cdi_adapter.fhir import service as fsvc
    from cdi_adapter.config import settings
    from cdi_adapter.persist.normalized import enqueue_fhir

    monkeypatch.setattr(settings, "fhir_enabled", True)          # off by default; this test covers the agent
    did, _ = _doc(sess_scope, fake_store)
    pid = str(uuid.uuid4())
    bundle = {"resourceType": "Bundle", "id": did}
    monkeypatch.setattr(fsvc, "project_document", lambda d: {
        "artifact_type": "PrescriptionRecord", "bundle": bundle, "issues": [],
        "bundle_status": "ready_to_share", "asserted_facts": 2, "held_facts": []})
    with sess_scope() as s:
        s.execute(text("UPDATE fhir_outbox SET status='done' WHERE status='pending'"))
        enqueue_fhir(s, patient_id=pid, document_id=did, reason="validated")
        enqueue_fhir(s, patient_id=pid, document_id=did, reason="validated")   # de-duplicated
    out = fhir_builder.run_once()
    assert [o["result"] for o in out] == ["done"] and out[0]["version"] == 1
    assert any(k.endswith("/PrescriptionRecord/v1.json") for k in fake_store)

    monkeypatch.setattr(fsvc, "project_document", lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
    with sess_scope() as s:
        enqueue_fhir(s, patient_id=pid, document_id=did, reason="review_decision")
    seen = []
    for _ in range(3):
        with sess_scope() as s:
            s.execute(text("UPDATE fhir_outbox SET next_attempt_at=now() WHERE document_id=:d"),
                      {"d": did})
        seen += [o["result"] for o in fhir_builder.run_once()]
    assert seen == ["retry", "retry", "dead"]


# ---------------- dispatch ----------------
def test_dispatch_mapping_and_approval_gate(sess_scope):
    from cdi_adapter.agents.dispatch import apply_mapping

    rec = {"patient": {"name": "Ramesh"}, "prescription": {"practitioner_name": "Dr. A. Sen"},
           "diagnoses": [{"text": "LBP"}, {"text": "HTN"}],
           "medications": [{"drug_text": "Telma 40", "strength": 40.0, "frequency_code": "1-0-1"}]}
    mp = {"fields": {"patientName": "patient.name", "doctor": "prescription.practitioner_name",
                     "dx": "diagnoses[*].text|join:, "},
          "lists": {"medicines": {"from": "medications",
                                  "fields": {"name": "drug_text", "freq": "frequency_code"}}}}
    out = apply_mapping(mp, rec)
    assert out == {"patientName": "Ramesh", "doctor": "Dr. A. Sen", "dx": "LBP, HTN",
                   "medicines": [{"name": "Telma 40", "freq": "1-0-1"}]}
    tid = f"t{uuid.uuid4().hex[:6]}"
    with pytest.raises(Exception, match="check"):
        with sess_scope() as s:            # cannot enable without an approver
            s.execute(text("INSERT INTO dispatch_target (id, name, channel, enabled) "
                           "VALUES (:i, 'x', 'rest', true)"), {"i": tid})


def test_practitioner_link_writes_document(sess_scope, fake_store):
    from cdi_adapter import repo
    from cdi_adapter.recognition.practitioner import link_document

    did, _ = _doc(sess_scope, fake_store)
    with sess_scope() as s:
        m = link_document(s, did, name="Dr. A. Sen", reg_no=None,
                          header_texts=["Dr. A. Sen (Reg. No. 12345)"])
        d = repo.get_document(s, did)
    assert m.linked and d["practitioner_link_method"] == "registration_no"
    assert d["practitioner_evidence"]["db_name"] == "Dr. A. Sen"


def test_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "infra/runpod/start_ocrhost.sh").exists()


# ---------------- whole chain ----------------
def test_full_chain_prescription_governance(sess_scope, fake_store, monkeypatch):
    """ingest -> rapid -> classify -> recognition v2 -> extract -> bind -> validate ->
    rx_* + outbox, with deterministic fake engines. Printed Telma on a back-pain Rx is
    flagged by context; the handwritten Pregabalin line the engines disagree on is held."""
    import json as _json

    from cdi_adapter.ml import client as mlc
    from cdi_adapter.listener.service import run_pipeline
    from cdi_adapter.recognition import ocrhost_client

    from cdi_adapter.config import settings

    monkeypatch.setattr(settings, "fhir_enabled", True)          # off by default: this test covers the outbox
    monkeypatch.setattr("cdi_adapter.ingest.service._enqueue_next", lambda d: None)
    monkeypatch.setattr("cdi_adapter.ocr.service._enqueue_extract", lambda d: None)
    monkeypatch.setattr("cdi_adapter.classify.service._enqueue_ocr", lambda d: None)

    class FakeML(mlc.StubMLClient):
        def vlm_generate(self, image, prompt, *, max_tokens=512, json_schema=None):
            if json_schema and json_schema.get("$id") == "cdi:classification.v1":
                return _json.dumps({**mlc._stub_classify(prompt), "doc_type": "prescription",
                                    "is_handwritten": True, "confidence": 0.95})
            if json_schema:   # extraction
                return _json.dumps({
                    "prescriber": {"name": "Dr. A. Sen", "reg_no": "12345"},
                    "diagnoses": [{"text": "Low back pain", "evidence": ["b1"]}],
                    "medications": [
                        {"drug_text": "Telma", "strength": {"value": 40, "unit_text": "mg"},
                         "frequency_text": "1-0-1", "evidence": ["b1"]},
                        {"drug_text": "Pregabalin", "strength": {"value": 75, "unit_text": "mg"},
                         "frequency_text": "0-0-1", "evidence": ["b2"]}],
                    "investigations": [{"text": "CBC", "evidence": ["b1"]}],
                    "extracted_at_confidence": 0.9})
            return "Pregabalin 150 0-0-1"          # qwen per-line crop

    monkeypatch.setattr(mlc, "get_client", lambda: FakeML())
    for mod in ("cdi_adapter.classify.service", "cdi_adapter.extract.service"):
        try:
            monkeypatch.setattr(f"{mod}.get_client", lambda: FakeML())
        except AttributeError:
            pass
    ocrhost_client.set_ocr_host(_FakeHost("Pregabalin 75 0-0-1"))
    monkeypatch.setattr("cdi_adapter.ocr.rapid.run_rapidocr", _FakeHost("").rapid)
    raw = _page_png() + uuid.uuid4().bytes          # unique sha per run (PNG ignores trailing)
    try:
        res = run_pipeline(raw, "rx.png", source_channel="test")
    finally:
        ocrhost_client.set_ocr_host(None)
    did = res["document_id"]
    assert res["facts"] >= 4 and res["in_review"] >= 2, res
    with sess_scope() as s:
        facts = {r["local_text"]: dict(r) for r in s.execute(text(
            "SELECT * FROM clinical_fact WHERE :d = ANY(source_doc_ids)"), {"d": did}).mappings()}
        preg = next(v for k, v in facts.items() if "regabalin" in k)
        telma = next(v for k, v in facts.items() if "elma" in k)
        doc = s.execute(text("SELECT * FROM source_document WHERE id=:d"), {"d": did}).mappings().one()
        rx = s.execute(text("SELECT count(*) FROM rx_medication_order m JOIN rx_prescription p "
                            "ON p.id=m.prescription_id WHERE p.document_id=:d"), {"d": did}).scalar_one()
        inv = s.execute(text("SELECT count(*) FROM rx_investigation_order i JOIN rx_prescription p "
                             "ON p.id=i.prescription_id WHERE p.document_id=:d"), {"d": did}).scalar_one()
        outbox = s.execute(text("SELECT count(*) FROM fhir_outbox WHERE document_id=:d"),
                           {"d": did}).scalar_one()
        enc = s.execute(text("SELECT practitioner_ref FROM encounter WHERE :d = ANY(derived_from)"),
                        {"d": did}).scalar_one()
    assert preg["evidence_state"] == "disagree" and preg["review_state"] == "in_review"
    assert "engines disagree" in (preg["review_note"] or "")
    assert telma["review_state"] == "in_review" and "indicated for" in (telma["review_note"] or "")
    assert preg["decision_trace"] and telma["field_policy"]
    assert doc["practitioner_link_method"] == "registration_no" and enc == "Dr. A. Sen"
    assert rx == 2 and inv == 1 and outbox == 1
    _assert_out_s1(sess_scope, did)


def _assert_out_s1(sess_scope, did: str) -> None:
    """OUT-S1: doctor / patient / context rows with status, provenance beside the extraction, and a
    second sync gives the same rows (no duplicates)."""
    from cdi_adapter.persist.normalized import sync_document

    def snapshot(s):
        rx = s.execute(text("SELECT * FROM rx_prescription WHERE document_id=:d"), {"d": did}).mappings().one()
        ctx = s.execute(text("SELECT test_text, context_text, relation FROM rx_investigation_context "
                             "WHERE prescription_id=:r ORDER BY 1,2"), {"r": rx["id"]}).all()
        prep = s.execute(text("SELECT count(*) FROM rx_investigation_preparation WHERE prescription_id=:r"),
                         {"r": rx["id"]}).scalar_one()
        return dict(rx), [tuple(c) for c in ctx], prep

    with sess_scope() as s:
        rx, ctx, prep = snapshot(s)
        ext = s.execute(text("SELECT prompt_version, engine_versions, raw_answer, schema_version FROM extraction "
                             "WHERE document_id=:d ORDER BY created_at DESC LIMIT 1"), {"d": did}).mappings().one()
    assert rx["doctor_name"] == "Dr. A. Sen" and rx["doctor_reg_no"] == "12345"
    assert rx["field_status"]["doctor.name"]["status"] in ("checked", "needs_check")
    assert rx["provenance"]["prompt_version"] == ext["prompt_version"] and rx["provenance"]["schema_version"] == "v3"
    assert ext["prompt_version"].startswith("p-") and ext["engine_versions"]["pdf_renderer"].startswith("pypdfium2")
    assert ext["raw_answer"] and "Pregabalin" in ext["raw_answer"]          # the model's own text is kept
    assert any(c[0] == "CBC" and c[1] == "Low back pain" for c in ctx)       # test <-> diagnosis, with the relation
    with sess_scope() as s:
        sync_document(s, did)
        rx2, ctx2, prep2 = snapshot(s)
    assert (ctx2, prep2) == (ctx, prep)                                       # same rows twice: no duplicates
    assert {k: v for k, v in rx2.items() if k != "synced_at"} == {k: v for k, v in rx.items() if k != "synced_at"}
