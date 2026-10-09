from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .. import repo
from ..classify.service import classify_document
from ..config import settings
from ..db import session_scope
from ..extract.service import extract_document
from ..fhir.service import project_patient
from ..ingest.service import ingest_bytes
from ..logging import get_logger
from sqlalchemy import text

from ..mpi.service import (IdentityCandidate, dob_match, merge_identity_evidence,
                           names_match, parse_age, parse_dob)
from ..ocr.service import ocr_document
from ..terminology.service import bind_document
from ..validate.service import validate_document

log = get_logger(__name__)

STAGES = ["ingest", "classify", "ocr", "extract", "terminology", "validate"]

# CPU-bound stages (ingest / heuristic classify / OCR) run for every document at
# once. The GPU extract stage fans out too, bounded by _extract_sem: with the
# vllm backend those calls batch on one GPU; with `hf` set extract_concurrency=1
# to keep the old serial behaviour.
_pool = ThreadPoolExecutor(max_workers=max(2, settings.job_max_workers),
                           thread_name_prefix="cdi-job")
# Each upload (a "job") gets a thread of its own that waits on its files' work in _pool. They must not
# share _pool: with as many jobs in flight as _pool has threads, every thread would be a job waiting on
# children that can never start (a hang). Jobs wait in this queue; more can be sent while others run.
_job_pool = ThreadPoolExecutor(max_workers=max(1, settings.job_max_concurrent),
                               thread_name_prefix="cdi-run")
# separate pool for the phase-2 fan-out so a _run_job thread waiting on its
# children can never starve _pool (which also hosts _run_job itself)
_stage2_pool = ThreadPoolExecutor(max_workers=max(2, settings.job_max_workers),
                                  thread_name_prefix="cdi-s2")
_extract_sem = threading.Semaphore(max(1, settings.extract_concurrency))
_cand_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()
# Idempotency-Key -> (time, job id): the same Send pressed twice (a double tap, a retry after a
# dropped connection) answers with the SAME job instead of starting a second one. Per process.
_IDEM_TTL_S = 3600.0
_idem: dict[str, tuple[float, str]] = {}


@dataclass
class DocProg:
    filename: str
    document_id: str | None = None
    doc_type: str | None = None
    status: str = "queued"          # queued|running|done|error
    facts: int = 0
    accepted: int = 0
    in_review: int = 0
    error: str | None = None
    stages: dict[str, str] = field(
        default_factory=lambda: {s: "pending" for s in STAGES})
    t0: float = field(default_factory=time.time)
    seconds: float | None = None
    # the untouched originals this document was assembled from (UP-S1 "one prescription"); never
    # part of the public job view
    parts: list[tuple[str, bytes]] | None = None
    job_id: str | None = None       # the Send this document belongs to (saved with the document)
    token_no: str | None = None     # the token and mobile number typed with the upload (saved with the document)
    phone: str | None = None
    patient_name: str | None = None  # the name read from the page, once it is read

    def stage(self, name: str, state: str) -> None:
        self.stages[name] = state


@dataclass
class Job:
    id: str
    abha: str | None
    patient_id: str | None = None
    patient: dict[str, Any] | None = None
    existing: bool = False         # matched an existing registry patient
    state: str = "queued"          # queued|running|review|generating|done|error|mismatch
    created: float = field(default_factory=time.time)
    docs: list[DocProg] = field(default_factory=list)
    error: str | None = None
    mismatch: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    token_no: str | None = None
    phone: str | None = None

    def public(self) -> dict[str, Any]:
        pat = self.patient or (self.result or {}).get("patient")
        if pat is not None:
            pat = {**pat, "is_new": not self.existing}
        # hide the patient identity block until documents have actually been read;
        # on a mismatch nothing is attached, so no id/identity is surfaced either
        show_patient = self.state in ("review", "done")
        return {
            "job_id": self.id,
            "state": self.state,
            "token_no": self.token_no,
            "phone": self.phone,
            "patient_name": next((d.patient_name for d in self.docs if d.patient_name), None),
            "patient_id": self.patient_id if show_patient else None,
            "existing_patient": self.existing,
            "patient": pat if show_patient else None,
            "mismatch": self.mismatch,
            "error": self.error,
            "stages": STAGES,
            "documents": [
                {"filename": d.filename, "document_id": d.document_id, "doc_type": d.doc_type,
                 "status": d.status, "stages": d.stages, "facts": d.facts,
                 "accepted": d.accepted, "in_review": d.in_review,
                 "seconds": round(d.seconds, 1) if d.seconds else None,
                 "error": d.error}
                for d in self.docs
            ],
            "artifact_count": (self.result or {}).get("artifact_count"),
            "ready_to_share": (self.result or {}).get("ready_to_share"),
            "needs_review": (self.result or {}).get("needs_review"),
            "review_open": (self.result or {}).get("review_open"),
            "has_result": self.result is not None,
            "queue_position": queue_position(self),               # prescriptions that start before this one; None once it is being read
        }


class QueueFull(RuntimeError):
    """More prescriptions are in flight (being read or waiting) than ``settings.job_queue_max``."""


def in_flight() -> int:
    """Prescriptions being read or waiting to be read."""
    with _lock:
        return sum(1 for j in _jobs.values() if j.state in ("queued", "running"))


def queue_position(job: "Job") -> int | None:
    """How many prescriptions will start before this one (0 = next); ``None`` once it is being read or finished."""
    if job.state != "queued":
        return None
    with _lock:
        return sum(1 for j in _jobs.values() if j.state == "queued" and j.created < job.created)


def create_job(abha: str | None, files: list[tuple[str, bytes]],
               patient_ref: str | None = None, *,
               parts: list[list[tuple[str, bytes]] | None] | None = None,
               idempotency_key: str | None = None,
               token_no: str | None = None, phone: str | None = None) -> str:
    jid = uuid.uuid4().hex[:12]
    if in_flight() >= settings.job_queue_max and not (idempotency_key and idempotency_key in _idem):
        raise QueueFull(f"{settings.job_queue_max} prescriptions are already being read or waiting. "
                        f"Please wait a minute until one finishes, then send this one.")
    if idempotency_key:
        now = time.time()
        with _lock:
            for k in [k for k, (t, _j) in _idem.items() if now - t > _IDEM_TTL_S]:
                del _idem[k]
            if idempotency_key in _idem:
                return _idem[idempotency_key][1]
            _idem[idempotency_key] = (now, jid)        # reserve before any slow work
        bound = _durable_key(idempotency_key, jid)
        if bound != jid:                               # the same Send was already accepted (even before a restart)
            with _lock:
                _idem[idempotency_key] = (now, bound)
            return bound
    try:
        return _create_job(jid, abha, files, patient_ref, parts, token_no, phone)
    except Exception:
        if idempotency_key:
            with _lock:
                _idem.pop(idempotency_key, None)       # a failed start may be retried
            try:
                with session_scope() as sess:
                    sess.execute(text("DELETE FROM upload_idempotency WHERE idem_key = :k"),
                                 {"k": idempotency_key})
            except Exception:  # noqa: BLE001
                pass
        raise


def _create_job(jid: str, abha: str | None, files: list[tuple[str, bytes]],
                patient_ref: str | None,
                parts: list[list[tuple[str, bytes]] | None] | None,
                token_no: str | None = None, phone: str | None = None) -> str:
    job = Job(id=jid, abha=(abha or "").strip() or None, token_no=token_no, phone=phone)
    job.docs = [DocProg(filename=fn, parts=(parts[i] if parts else None), job_id=jid, token_no=token_no, phone=phone)
                for i, (fn, _) in enumerate(files)]

    ref = (patient_ref or "").strip()
    if ref:
        from ..mpi import registry
        with session_scope() as sess:
            reg = registry.lookup(sess, ref)
            if reg:
                pid, mpi = registry.ensure_identity(sess, reg)
                job.patient_id = pid
                job.existing = True
                job.patient = {
                    "mpi_id": mpi, "name": reg.get("name"),
                    "sex": (reg.get("gender") or "")[:1].upper() or None,
                    "birth_date": reg.get("dob"), "mobile": reg.get("mobile"),
                    "address": reg.get("address"), "abha_number": reg.get("abha_id"),
                    "identity_confidence": 1.0, "provisional": False}
                if not job.abha and reg.get("abha_id"):
                    job.abha = reg["abha_id"]

    with _lock:
        _jobs[jid] = job
    _job_pool.submit(_run_job, jid, files)
    return jid


def get_job(jid: str) -> Job | None:
    return _jobs.get(jid) or _job_from_db(jid)        # a job survives a restart: rebuilt from what was saved


_DONE = ("validated", "projected", "normalized")
_FAILED = ("error", "quality_hold")


def _job_from_db(jid: str) -> Job | None:
    """Rebuild a job's view from its documents (after a restart the in-memory job is gone). Only
    documents that were saved can be listed: a file that never got that far left nothing behind."""
    try:
        with session_scope() as sess:
            rows = sess.execute(text(
                "SELECT id, original_filename, status, error_detail, token_no, phone, patient_name FROM source_document "
                "WHERE upload_job_id = :j ORDER BY ingested_at, id"), {"j": jid}).mappings().all()
    except Exception as exc:  # noqa: BLE001 - e.g. a database without migration 0008
        log.warning("job_lookup_failed", job=jid, error=str(exc)[:150])
        return None
    if not rows:
        return None
    job = Job(id=jid, abha=None, token_no=rows[0]["token_no"], phone=rows[0]["phone"])
    for r in rows:
        done, failed = r["status"] in _DONE, r["status"] in _FAILED
        d = DocProg(filename=r["original_filename"] or "document", document_id=str(r["id"]),
                    status="done" if done else "error" if failed else "running",
                    error=(r["error_detail"] if failed else None), job_id=jid,
                    token_no=r["token_no"], phone=r["phone"], patient_name=r["patient_name"])
        for s in STAGES:
            d.stage(s, "done" if done else "pending")
        job.docs.append(d)
    if any(d.status == "running" for d in job.docs):
        job.state = "running"
    elif any(d.status == "done" for d in job.docs):
        job.state = "review" if settings.review_ui_enabled else "done"
    else:
        job.state = "error"
        job.error = job.error or why_stopped(job)
    _refresh_result(job)
    return job


def why_stopped(job: "Job") -> str | None:
    """The real reason a job ended in error, one line per failed file, from what each document recorded
    ("rescan: " is an internal prefix; the sentence after it is already written for the person). The
    screen shows this instead of a generic "Processing stopped"."""
    out = []
    for d in job.docs:
        if d.error:
            msg = d.error.removeprefix("rescan: ")
            out.append(f"{d.filename}: {msg}")
    return " ".join(out)[:600] or None


def _durable_key(key: str, jid: str) -> str:
    """The job id this Idempotency-Key is bound to, saved in the database so it holds across restarts
    (``jid`` if the key is new). A database without the table falls back to the in-memory rule."""
    try:
        with session_scope() as sess:
            sess.execute(text("DELETE FROM upload_idempotency WHERE created_at < now() - interval '1 day'"))
            got = sess.execute(text(
                "INSERT INTO upload_idempotency (idem_key, job_id) VALUES (:k, :j) "
                "ON CONFLICT (idem_key) DO NOTHING RETURNING job_id"), {"k": key, "j": jid}).first()
            if got:
                return jid
            return sess.execute(text("SELECT job_id FROM upload_idempotency WHERE idem_key = :k"),
                                {"k": key}).scalar_one()
    except Exception as exc:  # noqa: BLE001
        log.warning("idempotency_store_unavailable", error=str(exc)[:150])
        return jid


def _tag_document(document_id: str, job_id: str | None, token_no: str | None = None, phone: str | None = None) -> None:
    if not job_id:
        return
    try:
        with session_scope() as sess:
            sess.execute(text("UPDATE source_document SET upload_job_id = :j, token_no = coalesce(:t, token_no), "
                              "phone = coalesce(:p, phone) WHERE id = :d"),
                         {"j": job_id, "d": document_id, "t": token_no, "p": phone})
    except Exception as exc:  # noqa: BLE001
        log.warning("job_tag_failed", document_id=document_id, error=str(exc)[:150])


# --------------------------------------------------------------------------- #
def _stage1(prog: DocProg, fn: str, raw: bytes, abha: str | None) -> None:
    """ingest -> classify (fast) -> OCR.  CPU-bound; runs in parallel per document."""
    try:
        prog.status = "running"
        prog.stage("ingest", "running")
        scope = f"{prog.phone}|{prog.token_no}" if prog.phone and prog.token_no else None      # the same photo for another patient is another document
        res = ingest_bytes(raw, filename=fn, source_channel="webapp", legacy_patient_ref=abha, dedupe_scope=scope)
        prog.document_id = res.document_id
        _tag_document(res.document_id, prog.job_id, prog.token_no, prog.phone)
        if prog.parts and not res.deduplicated:
            # one prescription built from several pictures: keep each original untouched
            from .upload import store_parts
            store_parts(res.document_id, res.sha256, prog.parts)
        if res.status == "quality_hold":
            # E2-S12: an unreadable capture is held for rescan - never guessed at
            with session_scope() as sess:
                d = repo.get_document(sess, res.document_id) or {}
            prog.stage("ingest", "error")
            raise RuntimeError(d.get("error_detail") or "rescan: image quality below threshold")
        prog.stage("ingest", "done")

        # fast text first: a rapidocr pass the classifier reads directly, so the
        # same page is not OCR'd twice (classifier hint + real stage)
        prog.stage("ocr", "running")
        ocr_document(res.document_id, force_engine="rapidocr")
        prog.stage("ocr", "done")

        prog.stage("classify", "running")
        c = classify_document(res.document_id)
        prog.doc_type = c.doc_type
        prog.stage("classify", "done")

        if settings.prefetch_main_call:
            try:
                from ..extract.service import prefetch_main_call
                prefetch_main_call(res.document_id)       # the main page call runs while the handwritten lines are read below
            except Exception as exc:  # noqa: BLE001 - an early start is an extra: the normal call is made later
                log.warning("prefetch_not_started", error=str(exc)[:200])

        # v2: the region pass always runs - printed lines keep their RapidOCR text and
        # only handwritten/mixed/uncertain line crops go to TrOCR + Qwen (ARCHITECTURE §15).
        # legacy: only a page the classifier calls handwritten gets page-level VLM OCR
        if settings.recognition_v2 or (c.is_handwritten and settings.handwritten_uses_vlm):
            prog.stage("ocr", "running")
            ocr_document(res.document_id, force_engine="vlm")
            prog.stage("ocr", "done")
    except Exception as exc:  # noqa: BLE001
        prog.status = "error"
        prog.error = str(exc)[:400]
        for s in STAGES:
            if prog.stages[s] == "running":
                prog.stage(s, "error")
        raise


def _stage2(job: "Job", prog: DocProg,
            candidates: list[IdentityCandidate]) -> None:
    """extract (GPU) -> terminology -> validate.  May run concurrently across
    documents; the extract call itself is bounded by _extract_sem."""
    if prog.status == "error" or not prog.document_id or job.state == "mismatch":
        return
    try:
        prog.stage("extract", "running")
        with _extract_sem:
            ex = extract_document(prog.document_id, patient_id=job.patient_id,
                                  abha_hint=job.abha)
        prog.facts = ex.n_facts
        if ex.patient_id and not job.patient_id:
            job.patient_id = ex.patient_id

        # existing patient selected -> the document must belong to that person
        if job.existing and ex.identity and (ex.identity.get("name") or "").strip():
            reg = job.patient or {}
            doc_name = ex.identity["name"]
            doc_dob = parse_dob(ex.identity.get("birth_date"))
            reg_dob = parse_dob(reg.get("birth_date"))
            if not names_match(doc_name, reg.get("name")) or not dob_match(doc_dob, reg_dob):
                job.mismatch = {
                    "selected_name": reg.get("name"), "selected_id": reg.get("mpi_id"),
                    "selected_dob": reg.get("birth_date"),
                    "document": prog.filename,
                    "document_name": doc_name,
                    "document_dob": ex.identity.get("birth_date"),
                }
                job.state = "mismatch"
                prog.status = "error"
                prog.error = f"document is for {doc_name}, not {reg.get('name')}"
                prog.stage("extract", "error")
                # undo what this document just wrote against the wrong patient
                try:
                    with session_scope() as s:
                        repo.purge_document_facts(s, prog.document_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("mismatch_purge_failed", error=str(exc)[:150])
                log.warning("patient_mismatch", job=job.id, selected=reg.get("name"),
                            found=doc_name)
                return

        if ex.identity:
            i = ex.identity
            _age_y, bd_age = parse_age(i.get("age_years"))
            with _cand_lock:
                candidates.append(IdentityCandidate(
                    name_full=i.get("name"), sex=i.get("sex"), age_years=i.get("age_years"),
                    birth_date=parse_dob(i.get("birth_date")) or bd_age,
                    abha=job.abha, source_doc_id=prog.document_id))
            job.patient = {"mpi_id": ex.mpi_id, "name": i.get("name"), "sex": i.get("sex"),
                           "birth_date": i.get("birth_date"), "age_years": i.get("age_years"),
                           "abha_number": job.abha, "provisional": True}
        prog.stage("extract", "done")
        if ex.identity and (ex.identity.get("name") or "").strip():
            prog.patient_name = str(ex.identity["name"]).strip()[:120]      # the screen groups by name + mobile number
            try:
                with session_scope() as s:
                    # what was READ is always kept; the shown name changes only while nobody has confirmed one
                    s.execute(text("UPDATE source_document SET name_read = :n, patient_name = CASE WHEN name_confirmed_at IS NULL "
                                   "THEN :n ELSE patient_name END WHERE id = :d"),
                              {"n": prog.patient_name, "d": prog.document_id})
            except Exception as exc:  # noqa: BLE001
                log.warning("patient_name_not_saved", document_id=prog.document_id, error=str(exc)[:150])

        prog.stage("terminology", "running")
        bind_document(prog.document_id)
        prog.stage("terminology", "done")

        prog.stage("validate", "running")
        v = validate_document(prog.document_id)
        prog.accepted, prog.in_review = v.auto_accepted, v.in_review
        prog.stage("validate", "done")

        try:
            from ..output import flat_table
            flat_table.save_document(prog.document_id)        # the flat table row(s) of this prescription (searched by mobile or token number)
        except Exception as exc:  # noqa: BLE001 - the table is an extra: never cost the read
            log.warning("flat_table_failed", document_id=prog.document_id, error=str(exc)[:150])

        prog.status = "done"
        prog.seconds = time.time() - prog.t0
    except Exception as exc:  # noqa: BLE001
        prog.status = "error"
        prog.error = str(exc)[:400]
        for s in STAGES:
            if prog.stages[s] == "running":
                prog.stage(s, "error")
        log.error("job_doc_failed", job=job.id, file=prog.filename,
                  error=str(exc)[:300], tb=traceback.format_exc()[-700:])


def _run_job(jid: str, files: list[tuple[str, bytes]]) -> None:
    job = _jobs[jid]
    job.state = "running"
    candidates: list[IdentityCandidate] = []
    try:
        # --- phase 1: ingest + classify + OCR for every document, concurrently ---
        futs = [_pool.submit(_stage1, prog, fn, raw, job.abha)
                for prog, (fn, raw) in zip(job.docs, files)]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001  (already recorded on the DocProg)
                pass

        # --- phase 2: extract + terminology + validate ---
        # existing-patient jobs stay serial (per-doc mismatch guard must stop the
        # run before later docs write). a new patient with several docs fans out:
        # doc 1 first (it pins the new identity so 2..N don't race to create it),
        # then the rest concurrently - vllm batches the extract calls.
        ready = [p for p in job.docs if p.status != "error" and p.document_id]
        if job.existing or len(ready) <= 1:
            for prog in ready:
                _stage2(job, prog, candidates)
                if job.state == "mismatch":
                    break
        else:
            _stage2(job, ready[0], candidates)
            futs2 = [_stage2_pool.submit(_stage2, job, prog, candidates) for prog in ready[1:]]
            for f in as_completed(futs2):
                try:
                    f.result()
                except Exception:  # noqa: BLE001  (recorded on the DocProg)
                    pass

        if job.state == "mismatch":
            job.error = (f"Uploaded document is for {job.mismatch['document_name']}"
                         f" but you selected {job.mismatch['selected_name']}"
                         f" ({job.mismatch['selected_id']}). Processing stopped -"
                         f" nothing was written for the wrong patient.")
            log.warning("job_stopped_mismatch", job=jid)
            return

        # --- phase 3: finalise identity (skip for an existing registry patient) ---
        if job.patient_id and candidates and not job.existing:
            with session_scope() as sess:
                job.patient = merge_identity_evidence(sess, job.patient_id, candidates)

        ok = any(d.status == "done" for d in job.docs)
        # with the review screens off the job simply ends: "done" (the result JSON is the output)
        job.state = ("review" if settings.review_ui_enabled else "done") if ok else "error"
        if job.state == "error" and not job.error:
            job.error = why_stopped(job)
        _refresh_result(job)
        if job.state == "review":
            try:
                from .reviewer import seed_from_job
                seed_from_job(job)
            except Exception as exc:  # noqa: BLE001
                log.warning("seed_from_job_failed", job=jid, error=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001
        job.state = "error"
        job.error = str(exc)[:500]
        log.error("job_failed", job=jid, error=str(exc)[:400], tb=traceback.format_exc()[-1000:])


def _refresh_result(job: "Job") -> None:
    if not settings.fhir_enabled:
        # no FHIR is built until the owner asks for it: the result is the per-document JSON
        job.result = {"patient": job.patient, "bundles": [], "artifact_count": 0,
                      "ready_to_share": 0, "needs_review": 0}
        return
    if not job.patient_id:
        job.result = {"patient": None, "bundles": [], "artifact_count": 0,
                      "ready_to_share": 0, "needs_review": 0}
        return
    job.result = project_patient(job.patient_id)
    with session_scope() as sess:
        job.result["review_open"] = len(repo.list_review_tasks(sess, patient_id=job.patient_id))


# --------------------------------------------------------------------------- #
def _num(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _med(md: dict[str, Any]) -> dict[str, Any]:
    return {"dose": _num(md.get("dose_num") or md.get("strength_num")),
            "unit": md.get("dose_unit_ucum") or md.get("strength_unit"),
            "freq": md.get("frequency_code"),
            "freq_per_day": _num(md.get("frequency_per_day")),
            "route": md.get("route")}


def job_facts(jid: str) -> dict[str, Any]:
    """All extracted facts for a job, grouped by document, for the inline editor."""
    job = _jobs.get(jid)
    if not job or not job.patient_id:
        raise KeyError(jid)
    out_docs = []
    with session_scope() as sess:
        prow = sess.execute(
            __import__("sqlalchemy").text("SELECT * FROM patient_identity WHERE id=:i"),
            {"i": job.patient_id}).mappings().first()
        for prog in job.docs:
            if not prog.document_id:
                continue
            facts = repo.list_clinical_facts(sess, document_id=prog.document_id)
            pages = repo.list_document_pages(sess, prog.document_id)
            img = (f"api/documents/{prog.document_id}/pages/{pages[0]['page_no']}"
                   if pages else None)
            rows = []
            for f in facts:
                if f["review_state"] in ("rejected",) or not f["is_current"]:
                    continue
                prov = repo.get_fact_provenance(sess, f["id"])
                bbox = next((list(p["bbox_union"]) for p in prov if p.get("bbox_union")), None)
                md = (repo.get_medication_detail(sess, f["id"])
                      if f["fact_type"] == "medication" else None)
                mv = _med(md) if md else None
                row = {
                    "fact_id": str(f["id"]), "fact_type": f["fact_type"],
                    "text": f["local_text"],
                    "value_num": float(f["value_num"]) if f["value_num"] is not None else None,
                    "value_unit_ucum": f["value_unit_ucum"], "value_text": f["value_text"],
                    "freq_text": None,
                    "code_system": f["code_system"], "code": f["code"],
                    "code_display": f["code_display"], "code_status": f["code_status"],
                    "abnormal_flag": f["abnormal_flag"],
                    "confidence": float(f["confidence_overall"] or 0),
                    "review_state": f["review_state"],
                    "medication": mv,
                    "bbox": bbox, "page_width": pages[0]["width_px"] if pages else None,
                    "page_height": pages[0]["height_px"] if pages else None,
                }
                if mv:  # surface dose / frequency into the editable columns
                    row["value_num"] = mv.get("dose")
                    row["value_unit_ucum"] = mv.get("unit")
                    row["freq_text"] = mv.get("freq") or (
                        f"{mv['freq_per_day']:g}/day" if mv.get("freq_per_day") else None)
                rows.append(row)
            out_docs.append({"document_id": prog.document_id, "filename": prog.filename,
                             "doc_type": prog.doc_type, "page_image_url": img,
                             "seconds": round(prog.seconds, 1) if prog.seconds else None,
                             "facts": rows})
    patient = {
        "mpi_id": prow["mpi_id"] if prow else None,
        "name": (prow.get("name_full") if prow else None),
        "sex": prow.get("gender") if prow else None,
        "birth_date": str(prow["birth_date"])[:10] if prow and prow.get("birth_date") else None,
        "age_years": prow.get("age_years") if prow else None,
        "abha_number": prow.get("abha_number") if prow else None,
        "identity_confidence": (float(prow["identity_confidence"])
                                if prow and prow.get("identity_confidence") is not None else None),
        "is_new": not job.existing,
    }
    if prow and prow.get("mpi_id"):
        with session_scope() as sess2:
            reg = sess2.execute(
                __import__("sqlalchemy").text(
                    "SELECT mobile, address FROM patient_registry WHERE patient_id=:m LIMIT 1"),
                {"m": prow["mpi_id"]}).mappings().first()
        if reg:
            patient["mobile"] = reg["mobile"]
            patient["address"] = reg["address"]
    return {"job_id": jid, "state": job.state, "patient": patient, "documents": out_docs}


def apply_edits_and_generate(jid: str, edits: list[dict[str, Any]],
                             reviewer: str = "reviewer") -> dict[str, Any]:
    """Bulk-apply the editor's decisions, then re-project the FHIR bundles (ms)."""
    from . import review as review_svc

    job = _jobs.get(jid)
    if not job or not job.patient_id:
        raise KeyError(jid)
    job.state = "generating"
    t0 = time.time()
    applied = {"keep": 0, "correct": 0, "drop": 0}
    for e in edits:
        fid = e.get("fact_id")
        if not fid:
            continue
        action = (e.get("action") or "keep").lower()
        try:
            if action == "drop":
                review_svc.submit_decision(fid, "reject", reviewer=reviewer)
                applied["drop"] += 1
            elif e.get("corrections"):
                review_svc.submit_decision(fid, "correct", reviewer=reviewer,
                                           corrections=e["corrections"])
                applied["correct"] += 1
            else:
                review_svc.submit_decision(fid, "accept", reviewer=reviewer)
                applied["keep"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("edit_apply_failed", fact=fid, error=str(exc)[:150])
    _refresh_result(job)
    job.state = "done"
    ms = int((time.time() - t0) * 1000)
    log.info("bundles_generated", job=jid, applied=applied, ms=ms)
    return {**job.result, "applied": applied, "generate_ms": ms}
