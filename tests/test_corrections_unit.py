"""The correction loop (human correction -> database -> doctor profile -> alias cascade).

The real SQL runs against in-memory SQLite (the tables are SQLAlchemy Core objects shared with the
PostgreSQL migration, and a test keeps the two in step). What SQLite cannot do (the PostgreSQL-only
loader, the append-only trigger) is covered by reading the migration, or by fakes, and is NOT proven
against a live PostgreSQL here.
"""
from __future__ import annotations

import contextlib
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from cdi_adapter.config import settings
from cdi_adapter.corrections import cache as pcache
from cdi_adapter.corrections import models as M
from cdi_adapter.corrections import service as svc
from cdi_adapter.recognition.alias import KB, candidates, norm_alias
from cdi_adapter.webapp import app as webapp
from cdi_adapter.webapp import corrections_api as api

DOC, FACT, DR_A, DR_B = (str(uuid.uuid4()) for _ in range(4))
NOW = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)


@pytest.fixture
def sess():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    M.metadata.create_all(engine)
    with Session(engine) as s:
        s.execute(M.kb_concept.insert(), [
            {"id": "lab:creatinine", "domain": "lab_order", "canonical_name": "Serum Creatinine", "active": True},
            {"id": "lab:kft", "domain": "lab_order", "canonical_name": "KFT", "active": True},
            {"id": "lab:lft", "domain": "lab_order", "canonical_name": "LFT", "active": True},
            {"id": "dx:t2dm", "domain": "diagnosis", "canonical_name": "Type 2 diabetes mellitus", "active": True}])
        s.execute(M.kb_alias.insert(), [
            {"concept_id": "lab:creatinine", "alias": "S. Creatinine", "alias_norm": norm_alias("S. Creatinine"),
             "alias_class": "A", "practitioner_id": None, "source": "seed", "created_at": NOW},
            # two concepts share one global alias: ambiguous for promotion
            {"concept_id": "lab:kft", "alias": "Kidney Liver", "alias_norm": norm_alias("Kidney Liver"),
             "alias_class": "B", "practitioner_id": None, "source": "seed", "created_at": NOW},
            {"concept_id": "lab:lft", "alias": "Kidney Liver", "alias_norm": norm_alias("Kidney Liver"),
             "alias_class": "B", "practitioner_id": None, "source": "seed", "created_at": NOW}])
        s.commit()
        yield s


def ctx(original="S Cr", doctor=DR_A, ftype="investigation_order", **kw):
    return svc.Context(document_id=DOC, fact_id=FACT, field_type=ftype, original_value=original, doctor_id=doctor,
                       qwen_value=kw.pop("qwen", "S Cr"), confidence=0.62,
                       prediction_status="needs_review", crop_hash="a" * 64,
                       crop_ref={"observation_ids": ["o1", "o2"], "page_id": "p1", "bboxes": [[1, 2, 30, 40]]},
                       model_stack={"extractor": "Qwen/Qwen2.5-VL-7B-Instruct"}, **kw)


def correct(s, original="S Cr", to="Serum Creatinine", doctor=DR_A, when=NOW, ftype="investigation_order"):
    return svc.record_correction(s, ctx(original, doctor, ftype), to, "dr.rao", when)


# ------------------------------------------------------------------ what is stored


def test_a_correction_stores_the_whole_picture_append_only(sess):
    out = correct(sess)
    row = sess.execute(select(M.correction)).mappings().one()
    assert out["correction_id"] == row["id"] and out["corrected_value"] == "Serum Creatinine"
    assert (row["original_value"], row["qwen_value"], row["corrected_value"]) == (
        "S Cr", "S Cr", "Serum Creatinine")
    assert (row["doctor_id"], row["field_type"], row["reviewer_id"], row["prediction_status"]) == (
        DR_A, "investigation_order", "dr.rao", "needs_review")
    assert row["crop_hash"] == "a" * 64 and row["crop_ref"]["bboxes"] == [[1, 2, 30, 40]]
    assert row["model_stack"] == {"extractor": "Qwen/Qwen2.5-VL-7B-Instruct"} and float(row["confidence"]) == 0.62
    assert row["created_at"] is not None


def test_the_migration_makes_the_correction_table_append_only_in_the_database():
    sql = Path("db/alembic/versions/0006_corrections.py").read_text(encoding="utf-8")
    assert "CREATE TRIGGER trg_correction_immutable BEFORE UPDATE OR DELETE ON correction" in sql
    assert "cdi_evidence_is_immutable" in sql and "REFERENCES cn_practitioner(id)" in sql


def test_the_core_tables_and_the_migration_have_the_same_columns():
    sql = Path("db/alembic/versions/0006_corrections.py").read_text(encoding="utf-8")
    for table in (M.correction, M.doctor_lexicon):
        block = re.search(rf"CREATE TABLE IF NOT EXISTS {table.name} \((.*?)\n\);", sql, re.DOTALL).group(1)
        cols = {m.group(1) for m in re.finditer(r"^\s{2}([a-z_]+)\s+[a-z]", block, re.MULTILINE)} - {"UNIQUE"}
        assert cols == {c.name for c in table.columns}, table.name


# ------------------------------------------------------------------ what is refused


@pytest.mark.parametrize("value,why", [("", "empty"), ("   ", "empty"), ("x" * 501, "longer than"),
                                       ("S  Cr", "same as the value"), ("s cr", "same as the value"),
                                       (None, "must be text"), (5, "must be text")])
def test_bad_corrected_values_are_refused_in_plain_words(value, why):
    with pytest.raises(svc.CorrectionError) as e:
        svc.clean_value(value, "S Cr")
    assert why in str(e.value) and e.value.status == 422


def test_a_corrected_value_is_stored_as_one_clean_line():
    assert svc.clean_value("  Serum\n\tCreatinine\x00 ", "S Cr") == "Serum Creatinine"


@pytest.mark.parametrize("bad", ["", " ", "x" * 65, "dr;drop", "<script>", None, 7])
def test_the_reviewer_id_is_plain_and_required(bad):
    with pytest.raises(svc.CorrectionError):
        svc.clean_reviewer(bad)
    assert svc.clean_reviewer(" dr.rao@clinic ") == "dr.rao@clinic"


def test_a_refused_correction_stores_nothing(sess):
    with pytest.raises(svc.CorrectionError):
        svc.record_correction(sess, ctx(), "S Cr", "dr.rao", NOW)
    assert sess.execute(select(M.correction)).first() is None and sess.execute(select(M.doctor_lexicon)).first() is None


# ------------------------------------------------------------------ the doctor's lexicon


def test_the_doctor_lexicon_counts_what_the_doctor_writes_and_what_it_means(sess):
    for i in range(2):
        out = correct(sess, when=NOW + timedelta(minutes=i))
    assert out["learned"]["count"] == 2 and out["learned"]["verified_count"] == 2
    row = sess.execute(select(M.doctor_lexicon)).mappings().one()
    assert (row["practitioner_id"], row["raw_example"], row["canonical"], row["count"]) == (DR_A, "S Cr", "Serum Creatinine", 2)
    assert row["first_seen"] < row["last_seen"] and row["concept_id"] is None       # not enough yet to promote


def test_spelling_variants_of_how_the_doctor_writes_it_count_together(sess):
    for raw in ("S Cr", "s  cr", "S CR"):
        svc.record_correction(sess, ctx(raw), "Serum Creatinine", "dr.rao", NOW)
    assert sess.execute(select(M.doctor_lexicon)).mappings().one()["count"] == 3


def test_different_meanings_of_the_same_writing_stay_separate(sess):
    correct(sess, to="Serum Creatinine")
    correct(sess, to="Serum Cortisol")
    assert {r["canonical"] for r in sess.execute(select(M.doctor_lexicon)).mappings()} == {"Serum Creatinine", "Serum Cortisol"}


def test_each_doctor_has_their_own_lexicon(sess):
    correct(sess, doctor=DR_A)
    correct(sess, doctor=DR_B)
    assert {r["practitioner_id"] for r in sess.execute(select(M.doctor_lexicon)).mappings()} == {DR_A, DR_B}


def test_an_unknown_doctor_gets_the_correction_stored_but_no_profile(sess):
    out = correct(sess, doctor=None)
    assert out["learned"] is None and out["doctor_id"] is None
    assert sess.execute(select(M.correction)).first() is not None and sess.execute(select(M.doctor_lexicon)).first() is None


# ------------------------------------------------------------------ promotion to a doctor alias, and its effect


def _kb(s) -> KB:
    concepts = {r["id"]: {**r, "code_system": None, "code": None, "code_display": None, "attrs": {}}
                for r in s.execute(select(M.kb_concept)).mappings()}
    aliases = [dict(r) for r in s.execute(select(M.kb_alias)).mappings()]
    return KB(concepts, aliases)


def test_after_enough_confirmations_it_becomes_this_doctors_alias_and_the_cascade_uses_it_at_once(sess):
    kb0 = _kb(sess)
    assert candidates(kb0, "S Cr", "lab_order", practitioner_id=DR_A) == [] or \
        candidates(kb0, "S Cr", "lab_order", practitioner_id=DR_A)[0]["alias_class"] != "C"
    for i in range(settings.doctor_alias_min_verified):
        out = correct(sess, when=NOW + timedelta(minutes=i))
    assert out["learned"]["promoted_to"] == "lab:creatinine"
    row = sess.execute(select(M.kb_alias).where(M.kb_alias.c.alias_class == "C")).mappings().one()
    assert (row["practitioner_id"], row["concept_id"], row["source"]) == (DR_A, "lab:creatinine", "correction")
    assert sess.execute(select(M.doctor_lexicon)).mappings().one()["concept_id"] == "lab:creatinine"
    kb = _kb(sess)                                                  # the alias engine reloads the knowledge base per document
    mine = candidates(kb, "S Cr", "lab_order", practitioner_id=DR_A)
    assert mine[0]["concept_id"] == "lab:creatinine" and mine[0]["alias_class"] == "C" and mine[0]["resolved_by_level"] == 1
    other = candidates(kb, "S Cr", "lab_order", practitioner_id=DR_B)
    assert all(c["alias_class"] != "C" for c in other)             # another doctor's habit never applies to them


def test_promotion_waits_for_the_threshold_and_is_idempotent(sess, monkeypatch):
    monkeypatch.setattr(settings, "doctor_alias_min_verified", 2)
    assert correct(sess)["learned"]["promoted_to"] is None
    assert correct(sess)["learned"]["promoted_to"] == "lab:creatinine"
    correct(sess)                                                   # a third time: still one alias row
    assert len(sess.execute(select(M.kb_alias).where(M.kb_alias.c.alias_class == "C")).all()) == 1


def test_a_meaning_that_names_two_concepts_or_none_is_never_promoted(sess, monkeypatch):
    monkeypatch.setattr(settings, "doctor_alias_min_verified", 1)
    assert correct(sess, "KL", "Kidney Liver")["learned"]["promoted_to"] is None        # two concepts share it
    assert correct(sess, "XYZ", "Totally Unknown Test")["learned"]["promoted_to"] is None
    assert sess.execute(select(M.kb_alias).where(M.kb_alias.c.alias_class == "C")).first() is None


def test_a_single_letter_and_an_unmapped_field_type_are_never_promoted(sess, monkeypatch):
    monkeypatch.setattr(settings, "doctor_alias_min_verified", 1)
    assert correct(sess, "S", "Serum Creatinine")["learned"]["promoted_to"] is None
    assert correct(sess, "adv", "Serum Creatinine", ftype="advice")["learned"]["promoted_to"] is None


def test_a_diagnosis_correction_promotes_into_the_diagnosis_domain(sess, monkeypatch):
    monkeypatch.setattr(settings, "doctor_alias_min_verified", 1)
    out = correct(sess, "T2DM?", "Type 2 diabetes mellitus", ftype="condition")
    assert out["learned"]["promoted_to"] == "dx:t2dm"


# ------------------------------------------------------------------ reading the profile, with an optional cache


class FakeRedis:
    def __init__(self):
        self.d: dict[str, str] = {}
        self.fail = False

    def get(self, k):
        if self.fail:
            raise ConnectionError("down")
        return self.d.get(k)

    def setex(self, k, ttl, v):
        if self.fail:
            raise ConnectionError("down")
        self.d[k] = v

    def scan_iter(self, match):
        if self.fail:
            raise ConnectionError("down")
        pre = match.rstrip("*")
        return [k for k in self.d if k.startswith(pre)]

    def delete(self, k):
        self.d.pop(k, None)


def test_the_profile_lists_what_the_doctor_writes_most_often_first(sess):
    for _ in range(3):
        correct(sess)
    correct(sess, "HbA", "HbA1c")
    prof = svc.get_profile(sess, DR_A)
    assert [(e["raw"], e["canonical"], e["verified_count"]) for e in prof["entries"]] == [
        ("S Cr", "Serum Creatinine", 3), ("HbA", "HbA1c", 1)]
    assert prof["promoted"] == 1 and prof["cached"] is False
    assert [e["raw"] for e in svc.get_profile(sess, DR_A, "investigation_order")["entries"]] == ["S Cr", "HbA"]
    assert svc.get_profile(sess, DR_A, "advice")["entries"] == [] and svc.get_profile(sess, DR_B)["entries"] == []


def test_the_cache_serves_repeat_reads_and_never_replaces_the_database(sess):
    r = FakeRedis()
    cache = pcache.ProfileCache(r, ttl_s=60)
    correct(sess)
    first = svc.get_profile(sess, DR_A, cache=cache)
    assert first["cached"] is False and any(k.startswith(pcache.PREFIX + DR_A) for k in r.d)
    assert svc.get_profile(sess, DR_A, cache=cache)["cached"] is True
    correct(sess)                                                  # a new correction...
    cache.invalidate(DR_A)                                         # ...drops the cached views of that doctor
    again = svc.get_profile(sess, DR_A, cache=cache)
    assert again["cached"] is False and again["entries"][0]["count"] == 2


def test_a_redis_outage_falls_through_to_the_database(sess):
    r = FakeRedis()
    r.fail = True
    cache = pcache.ProfileCache(r)
    correct(sess)
    assert svc.get_profile(sess, DR_A, cache=cache)["entries"][0]["canonical"] == "Serum Creatinine"
    cache.invalidate(DR_A)                                         # also must not raise


def test_there_is_no_cache_without_redis(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1/0")
    assert pcache.get_cache() is None
    monkeypatch.setattr(settings, "profile_cache_ttl_s", 0)
    assert pcache.get_cache() is None


# ------------------------------------------------------------------ the training export


def test_the_training_export_is_oldest_first_and_can_be_sliced(sess):
    for i, to in enumerate(("Serum Creatinine", "Serum Urea", "Serum Sodium")):
        correct(sess, "S X", to, when=NOW + timedelta(days=i))
    rows = svc.export_training(sess)
    assert [r["corrected_value"] for r in rows] == ["Serum Creatinine", "Serum Urea", "Serum Sodium"]
    assert [r["corrected_value"] for r in svc.export_training(sess, since=NOW + timedelta(days=1))] == ["Serum Urea", "Serum Sodium"]
    assert len(svc.export_training(sess, limit=2)) == 2
    r = rows[0]
    assert {"correction_id", "prescription_id", "field_id", "doctor_id", "field_type", "original_value", "qwen_value",
            "corrected_value", "confidence", "prediction_status", "crop_hash", "crop_ref",
            "reviewer_id", "model_stack", "created_at"} == set(r)
    json.dumps(r)


# ------------------------------------------------------------------ the prediction record the OCR service emits


def test_the_field_record_has_what_the_correction_tool_needs():
    fact = {"id": "F1", "fact_type": "investigation_order", "local_text": "S Cr", "confidence_overall": 0.6234,
            "review_state": "in_review"}
    blocks = [{"observation_ids": ["o1"], "bbox": [1, 2, 3, 4],
               "recognition": {"engines": {"qwen2.5-vl": "S Cr"}}}]
    rec = svc.build_field_record(fact, blocks, DR_A, DOC, {"o1": "c" * 64})
    assert rec == {"prescription_id": DOC, "doctor_id": DR_A, "field_id": "F1", "field_type": "investigation_order",
                   "raw_crop_reference": {"observation_ids": ["o1"], "crop_hashes": ["c" * 64], "bboxes": [[1, 2, 3, 4]]},
                   "qwen_value": "S Cr", "final_value": "S Cr", "confidence": 0.623,
                   "status": "needs_review"}


@pytest.mark.parametrize("state,status", [("auto_accepted", "accepted"), ("clinician_confirmed", "accepted"),
                                          ("corrected", "corrected"), ("in_review", "needs_review"), ("pending", "needs_review"),
                                          ("rejected", "needs_review"), (None, "needs_review")])
def test_prediction_status(state, status):
    assert svc.prediction_status(state) == status


def test_engine_value_joins_the_lines_a_field_came_from_and_skip_what_was_not_read():
    blocks = [{"recognition": {"engines": {"qwen2.5-vl": "Serum"}}},
              {"recognition": {"engines": {"qwen2.5-vl": "Cr"}}}, {"recognition": None}, {}]
    assert svc.engine_value(blocks) == "Serum Cr"
    assert svc.engine_value([]) is None


# ------------------------------------------------------------------ the HTTP API


@pytest.fixture
def client(sess, monkeypatch):
    @contextlib.contextmanager
    def scope():
        yield sess
        sess.commit()

    applied: list[dict] = []
    monkeypatch.setattr(api, "session_scope", scope)
    monkeypatch.setattr(svc, "load_context", lambda s, d, f: ctx())
    monkeypatch.setattr(api.review, "submit_decision", lambda fact, action, **kw: applied.append({"fact": fact, "action": action, **kw}))
    monkeypatch.setattr(api.profile_cache, "get_cache", lambda: None)
    monkeypatch.setattr(settings, "review_ui_enabled", True)     # these paths are closed by default
    c = TestClient(webapp.app)
    c.applied, c.sess = applied, sess
    return c


def body(**kw):
    return {"prescription_id": DOC, "field_id": FACT, "corrected_value": "Serum Creatinine", "reviewer_id": "dr.rao", **kw}


def test_post_corrections_applies_records_and_learns(client):
    r = client.post("/api/corrections", json=body())
    assert r.status_code == 201
    j = r.json()
    assert j["ok"] and j["applied"] and j["recorded"] and j["doctor_id"] == DR_A and j["learned"]["count"] == 1
    assert client.applied == [{"fact": FACT, "action": "correct", "reviewer": "dr.rao",
                               "corrections": {"local_text": "Serum Creatinine"},
                               "note": "correction via the corrections API"}]
    assert client.sess.execute(select(M.correction)).mappings().one()["corrected_value"] == "Serum Creatinine"


def test_the_doctor_profile_grows_with_every_correction(client):
    for _ in range(2):
        client.post("/api/corrections", json=body())
    prof = client.get(f"/api/doctors/{DR_A}/profile").json()
    assert prof["entries"][0]["count"] == 2 and prof["entries"][0]["canonical"] == "Serum Creatinine"


def test_refusals_are_plain_and_change_nothing(client):
    for bad, status, why in ((body(corrected_value="S Cr"), 422, "nothing to correct"),
                             (body(corrected_value=" "), 422, "empty"),
                             (body(reviewer_id=""), 422, "reviewer"),
                             (body(prescription_id="nope"), 404, "prescription was not found"),
                             (body(field_id="nope"), 404, "field was not found")):
        r = client.post("/api/corrections", json=bad)
        assert r.status_code == status and why in r.json()["detail"], bad
    assert client.applied == [] and client.sess.execute(select(M.correction)).first() is None
    assert client.post("/api/corrections", json={"prescription_id": DOC}).status_code == 422


def test_a_field_that_is_not_in_this_prescription_is_a_404(client, monkeypatch):
    def refuse(s, d, f):
        raise svc.CorrectionError("That field does not belong to this prescription.", 404)

    monkeypatch.setattr(svc, "load_context", refuse)
    r = client.post("/api/corrections", json=body())
    assert r.status_code == 404 and "does not belong" in r.json()["detail"] and client.applied == []


def test_if_recording_fails_the_answer_says_so_plainly(client, monkeypatch):
    monkeypatch.setattr(svc, "record_correction", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    r = client.post("/api/corrections", json=body())
    assert r.status_code == 201 and r.json()["applied"] is True and r.json()["recorded"] is False
    assert "could not be recorded" in r.json()["note"]


def test_a_correction_for_an_unknown_doctor_still_succeeds(client, monkeypatch):
    monkeypatch.setattr(svc, "load_context", lambda s, d, f: ctx(doctor=None))
    r = client.post("/api/corrections", json=body())
    assert r.status_code == 201 and r.json()["doctor_id"] is None and r.json()["learned"] is None


def test_the_cache_is_dropped_when_a_doctors_lexicon_changes(client, monkeypatch):
    dropped: list[str] = []

    class C:
        def invalidate(self, pid):
            dropped.append(pid)

    monkeypatch.setattr(api.profile_cache, "get_cache", lambda: C())
    client.post("/api/corrections", json=body())
    assert dropped == [DR_A]


def test_the_export_is_newline_delimited_json(client):
    client.post("/api/corrections", json=body())
    r = client.get("/api/corrections/export")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = r.text.strip().split("\n")
    assert len(lines) == 1 and json.loads(lines[0])["corrected_value"] == "Serum Creatinine"
    assert client.get("/api/corrections/export?since=garbage").status_code == 422
    assert client.get("/api/corrections/export?limit=0").status_code == 422
    assert client.get("/api/corrections/export?since=2099-01-01").text == ""


def test_field_records_come_from_the_service(client, monkeypatch):
    monkeypatch.setattr(svc, "field_records", lambda s, d: [{"field_id": "F1", "prescription_id": d}])
    assert client.get(f"/api/documents/{DOC}/field-records").json() == [{"field_id": "F1", "prescription_id": DOC}]
    assert client.get("/api/documents/nope/field-records").status_code == 404
    assert client.get("/api/doctors/nope/profile").status_code == 404
