from __future__ import annotations

import logging

import contextlib
import json
import uuid
from typing import Any

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .. import __version__, repo, storage
from ..config import settings
from ..db import ping as db_ping
from ..db import session_scope
from ..fhir.service import project_document, project_patient
from ..logging import get_logger
from ..ml.client import get_client
from ..storage import ping as s3_ping
from . import review as review_svc
from . import upload
from ..output.json_connector import get_connector, to_bytes
from .jobs import apply_edits_and_generate, create_job, get_job, job_facts
from .page import PAGE
from .review_page import REVIEW_PAGE
from .reviewer import router as reviewer_router
from .reviewer_page import REVIEWER_PAGE
from .admin import router as admin_router
from .corrections_api import router as corrections_router
from .admin_page import ADMIN_PAGE
from .surface import SurfaceMiddleware, reviewer_name
from .upload_page import ADMIN_PAGE as UPLOAD_ONLY_PAGE

log = get_logger(__name__)
# the interactive API docs list every route: only a dev environment shows them
_DEV = settings.env.strip().lower() == "dev"
@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    from .resume import start_in_background
    from .surface import startup_checks

    startup_checks()               # the same checks however the app is started (python -m ... or uvicorn ...:app)
    start_in_background()          # pick up uploads a restart interrupted (OUT-S3)
    try:
        from ..extract import lab_mapping

        lab_mapping.ensure_seed()  # the mapping table's built-in rows (only the ones not there yet)
    except Exception as exc:  # noqa: BLE001 - a database without migration 0010 still serves (the seed answers)
        logging.getLogger(__name__).warning("lab_mapping_seed_failed: %s", str(exc)[:150])
    yield


app = FastAPI(title="CDI-Adapter - scanned docs -> ABDM FHIR", version=__version__, lifespan=_lifespan,
              docs_url="/docs" if _DEV else None, redoc_url=None,
              openapi_url="/openapi.json" if _DEV else None)
app.add_middleware(SurfaceMiddleware)         # sign-in + closed paths (webapp/surface.py)
app.include_router(reviewer_router)
app.include_router(admin_router)
app.include_router(corrections_router)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # the full screen (patient look-up, reviewer links) only when the review screens are on. Never cached: a changed screen is
    # what the next person sees (a cached copy once showed an old layout after an update).
    return HTMLResponse(PAGE if settings.review_ui_enabled else UPLOAD_ONLY_PAGE, headers={"Cache-Control": "no-store"})


@app.get("/healthz")
def healthz() -> Any:
    checks = {"db": db_ping(), "s3": s3_ping()}
    try:
        checks["mlserve"] = get_client().healthz().get("status") == "ok"
    except Exception:  # noqa: BLE001
        checks["mlserve"] = False
    ok = checks["db"] and checks["s3"]
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ok" if ok else "degraded",
                                 "checks": checks, "version": __version__,
                                 "ig_package": settings.ig_package})


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus text (behind the same sign-in as everything else; Prometheus can use basic auth)."""
    if settings.metrics_sink.strip().lower() == "none":
        raise HTTPException(404, "metrics are switched off")
    from ..ops.metrics import render

    with session_scope() as sess:
        body = render(sess)
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/status", response_class=HTMLResponse)
def status_page() -> str:
    from ..ops.status import collect, page

    with session_scope() as sess:
        return page(collect(sess))


@app.get("/api/swap-points")
def swap_points() -> dict[str, Any]:
    """Every part that can be changed by one setting: the setting, what is in use, what else is available."""
    from .. import swap

    return {"swap_points": swap.describe()}


@app.get("/api/listener/health")
def listener_health() -> dict[str, Any]:
    """Last good poll, files waiting / in error / in quarantine, oldest waiting age, stalled or stopped."""
    from ..listener.health import health

    with session_scope() as sess:
        return health(sess)


@app.get("/api/documents")
def find_documents_by_name(filename: str) -> dict[str, Any]:
    """Find the result of an uploaded or dropped file by (part of) its file name."""
    from ..listener.health import find_documents

    name = filename.strip()
    if len(name) < 2:
        raise HTTPException(422, "Type at least two characters of the file name.")
    with session_scope() as sess:
        return {"documents": find_documents(sess, name[:200])}


@app.get("/api/registry/lookup")
def registry_lookup(q: str) -> dict[str, Any]:
    """Resolve an existing clinic patient by CareFlow id / ABHA id / mobile."""
    from ..mpi import registry
    with session_scope() as sess:
        row = registry.lookup(sess, q)
    return {"found": row is not None, "patient": row}


@app.post("/api/registry")
def registry_save(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Persist a new patient's full details for future look-ups."""
    from ..mpi import registry
    req = {k: (body or {}).get(k) for k in
           ("patient_id", "name", "mobile", "dob", "gender", "address", "abha_id")}
    if not req["name"] or not req["mobile"]:
        raise HTTPException(422, "name and mobile are required")
    with session_scope() as sess:
        row = registry.save(sess, **req)
    return {"ok": True, "patient": row}


@app.get("/api/intake/search")
def patients_search(q: str = "") -> dict[str, Any]:
    """Autocomplete: patients (mobile + name) whose mobile number starts with what was typed, or whose name contains it."""
    from . import patients
    return {"patients": patients.search(q)}


@app.get("/api/intake/prescriptions")
def patients_prescriptions(phone: str, name: str | None = None) -> dict[str, Any]:
    from . import patients
    return {"prescriptions": patients.prescriptions(phone, name)}


@app.get("/api/intake/token-check")
def intake_token_check(token: str, phone: str) -> dict[str, Any]:
    """Before anything is uploaded: is this token already used today for another mobile number?"""
    from . import patients
    try:
        t = upload.clean_token(token)
        p = upload.clean_phone(phone)
    except upload.UploadError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    clash = patients.token_conflict(t or "", p or "")
    return {"conflict": bool(clash), "message": patients.token_message(t or "", clash) if clash else None}


@app.get("/api/intake/existing")
def patients_existing(phone: str) -> dict[str, Any]:
    """Prescriptions already uploaded for this mobile number (the upload screen warns before another is added)."""
    from . import patients
    try:
        digits = upload.clean_phone(phone)
    except upload.UploadError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return patients.existing(digits or "")


@app.get("/api/intake/page-image")
def intake_page_image(document_id: str, page: int = 1, view: str = "page", w: int | None = None) -> Response:
    """One page of an uploaded prescription as a picture, for the screen's viewer (``view``: page | original; ``w``: thumbnail width)."""
    from . import images
    got = images.page_image(document_id, page, view, w)
    if got is None:
        raise HTTPException(404, "page not found")
    return Response(content=got[0], media_type=got[1], headers={"Cache-Control": "private, max-age=300", "X-Page-Count": str(got[2])})


@app.get("/api/intake/name-crop")
def intake_name_crop(document_id: str) -> Response:
    """The patient's name line cut out of the page and enlarged, to compare with the typed name."""
    from . import images
    png = images.name_crop(document_id)
    if png is None:
        raise HTTPException(404, "the name line could not be found")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


@app.post("/api/intake/name")
def intake_confirm_name(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """The front desk confirms or corrects the patient's name read from a prescription. A handwritten name is never final
    until a person has done this; the name as read is kept beside it."""
    from . import patients
    from .surface import reviewer_name
    try:
        name = upload.clean_person_name(str(body.get("name") or ""))
    except upload.UploadError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    got = patients.confirm_name(str(body.get("document_id") or ""), name, reviewer_name(request, "admin"))
    if got is None:
        raise HTTPException(404, "unknown prescription")
    return got


@app.get("/api/mappings/lab")
def lab_mappings() -> dict[str, Any]:
    """The whole lab-name mapping table (many written names -> one standard test)."""
    from ..extract import lab_mapping
    with session_scope() as sess:
        return {"rows": lab_mapping.table(sess)}


@app.post("/api/mappings/lab")
def lab_mapping_save(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from ..extract import lab_mapping
    try:
        with session_scope() as sess:
            row = lab_mapping.upsert(sess, str(body.get("alias") or ""), str(body.get("canonical") or ""),
                                     body.get("loinc"), body.get("note"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, "row": row}


@app.post("/api/mappings/lab/{alias_key}/enabled")
def lab_mapping_enabled(alias_key: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from ..extract import lab_mapping
    with session_scope() as sess:
        ok = lab_mapping.set_enabled(sess, alias_key[:80], bool(body.get("enabled")))
    if not ok:
        raise HTTPException(404, "unknown mapping row")
    return {"ok": True}


@app.get("/api/upload/limits")
def upload_limits() -> dict[str, Any]:
    """What the upload screen may tell the person before Send (all from settings)."""
    return upload.limits()


@app.post("/api/jobs", status_code=202)
async def submit_job(
    abha: str | None = Form(default=None),
    patient_ref: str | None = Form(default=None),
    token_no: str | None = Form(default=None),
    phone: str | None = Form(default=None),
    department: str | None = Form(default=None),
    grouping: str = Form(default="separate"),
    files: list[UploadFile] = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    """If ``patient_ref`` (CareFlow id / ABHA / mobile) matches the clinic
    registry, all documents attach to that patient. Otherwise the patient's name,
    sex and DOB are read from the documents and a new CareFlow id is generated.

    ``grouping``: ``separate`` (default) = every file is its own document; ``one_document`` = the
    pictures are the pages of ONE prescription (one document, one result). Every refusal is one
    plain sentence (``detail``); the same ``Idempotency-Key`` returns the same job."""
    try:
        upload.LIMITER.check()                 # before reading any bytes
        abha_n = upload.normalize_abha(abha) if settings.abha_enabled else None   # not asked for: ignored, not validated
        ref = upload.clean_patient_ref(patient_ref)
        need = settings.upload_require_intake              # the front desk enters the token and the mobile number first
        token = upload.clean_token(token_no, required=need)
        mobile = upload.clean_phone(phone, required=need)
        if token and mobile:
            from . import patients
            clash = patients.token_conflict(token, mobile)
            if clash:                                       # the same token for another patient today: stop, say why
                raise upload.UploadError(patients.token_message(token, clash), 409)
        if len(files) > settings.upload_max_files:        # before reading any bytes
            raise upload.UploadError(f"You can send up to {settings.upload_max_files} files at once.")
        cap = settings.upload_max_file_bytes
        raw: list[tuple[str, bytes]] = []
        for f in files:
            raw.append((f.filename or "document", await f.read(cap + 1)))   # bounded memory
        items = upload.prepare(raw, grouping)
    except upload.UploadError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    key = (idempotency_key or "").strip()

    if key and not (len(key) <= 64 and key.replace("-", "").replace("_", "").isalnum()):
        raise HTTPException(422, "Invalid request, please reload the page and try again.")
    from .jobs import QueueFull
    try:
        jid = create_job(abha_n, [(i.name, i.data) for i in items], patient_ref=ref,
                         parts=[i.parts for i in items], idempotency_key=key or None, token_no=token, phone=mobile,
                         department=_dept_hint(department))
    except QueueFull as exc:                           # 10 prescriptions are in flight: a plain "wait a minute", not an error page
        raise HTTPException(429, str(exc)) from exc
    return {"job_id": jid, "documents": len(items)}


@app.get("/api/flat/search")
def flat_search(phone: str = "", token: str = "", fmt: str = "json", limit: int = 500) -> Any:
    """The flat table (``prescription_flat``): one row per prescription x lab test, found by mobile number and / or token number.
    ``fmt=csv`` downloads every column; the default is the columns the Extracted tab shows."""
    from ..output import flat_table
    if not (phone.strip() or token.strip()):
        return {"columns": flat_table.SCREEN_COLUMNS, "rows": []}
    if fmt == "csv":
        import csv
        import io
        rows = flat_table.search(phone, token, limit)
        buf = io.StringIO()
        cols = [c for c in flat_table.COLUMN_NAMES if c != "raw_result"]
        w = csv.writer(buf)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else (r[c] if not isinstance(r[c], (dict, list)) else json.dumps(r[c], ensure_ascii=False)) for c in cols])
        return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="prescriptions.csv"'})
    return {"columns": flat_table.SCREEN_COLUMNS, "rows": flat_table.search(phone, token, limit, columns=flat_table.SCREEN_COLUMNS)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job.public()


def _doc_result(document_id: str) -> dict[str, Any] | None:
    try:
        uuid.UUID(document_id)
    except ValueError:
        return None
    return get_connector().render(document_id)


@app.get("/api/documents/{document_id}/result.json")
def document_result(document_id: str, download: bool = False) -> Response:
    """The connector's JSON for one document (UP-S3 placeholder). ``download=true`` = same bytes
    as a file."""
    result = _doc_result(document_id)
    if result is None:
        raise HTTPException(404, "document not found")
    headers = {"Content-Disposition": f'attachment; filename="result_{document_id}.json"'} if download else {}
    return Response(content=to_bytes(result), media_type="application/json", headers=headers)


@app.get("/api/jobs/{job_id}/result.json")
def job_result(job_id: str) -> Any:
    """Every document of a job, for the screen. A document that failed before it had an id is listed
    with its status so nothing disappears silently."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    results: list[dict[str, Any]] = []
    for d in job.docs:
        res = _doc_result(d.document_id) if d.document_id else None
        results.append(res or {"filename": d.filename, "document_id": d.document_id,
                               "status": "error" if d.status == "error" else d.status,
                               "reason": d.error})
    return {"job_id": job_id, "results": results}


@app.get("/api/jobs/{job_id}/facts")
def job_facts_endpoint(job_id: str) -> Any:
    """Extracted facts per document for the inline human editor (image left, table right)."""
    try:
        return JSONResponse(job_facts(job_id))
    except KeyError:
        raise HTTPException(404, "unknown job or no patient resolved yet") from None


@app.post("/api/jobs/{job_id}/generate")
def job_generate(request: Request, job_id: str, body: dict[str, Any] = Body(default={})) -> Any:
    """Apply the editor's keep/edit/drop decisions, then build the FHIR bundles (ms)."""
    edits = (body or {}).get("edits") or []
    reviewer = reviewer_name(request, (body or {}).get("reviewer"))
    try:
        return JSONResponse(apply_edits_and_generate(job_id, edits, reviewer))
    except KeyError:
        raise HTTPException(404, "unknown job") from None


@app.get("/api/jobs/{job_id}/fhir")
def job_fhir(job_id: str) -> Any:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if not job.patient_id:
        raise HTTPException(409, f"job not started (state={job.state})")
    # re-project live so it reflects any review decisions since the job finished
    return JSONResponse(project_patient(job.patient_id))


@app.get("/api/jobs/{job_id}/fhir/download")
def job_fhir_download(job_id: str) -> Response:
    job = get_job(job_id)
    if not job or not job.patient_id:
        raise HTTPException(404, "no result")
    body = json.dumps(project_patient(job.patient_id), indent=2, default=str)
    return Response(
        content=body, media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="fhir_{job_id}.json"'},
    )


# --------------------------------------------------------------------------- #
# S8 human review
# --------------------------------------------------------------------------- #
@app.get("/review", response_class=HTMLResponse)
def review_index() -> str:
    return REVIEW_PAGE


@app.get("/reviewer", response_class=HTMLResponse)
def reviewer_index() -> str:
    return REVIEWER_PAGE


@app.get("/admin", response_class=HTMLResponse)
def admin_index() -> str:
    return ADMIN_PAGE


@app.get("/api/review/tasks")
def review_tasks(patient_id: str | None = None) -> dict[str, Any]:
    return {"tasks": review_svc.list_open_tasks(patient_id)}


@app.get("/api/review/facts/{fact_id}")
def review_fact(fact_id: str) -> dict[str, Any]:
    try:
        return review_svc.fact_detail(fact_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/facts/{fact_id}/review")
def review_decision(request: Request, fact_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    action = (body or {}).get("action", "")
    try:
        return review_svc.submit_decision(
            fact_id, action, reviewer=reviewer_name(request, body.get("reviewer")),
            corrections=body.get("corrections"), note=body.get("note"),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/documents/{document_id}/original")
def document_original(document_id: str) -> Response:
    with session_scope() as sess:
        doc = repo.get_document(sess, document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    key = storage.key_from_uri(doc["object_uri"])
    return Response(content=storage.get_bytes(key),
                    media_type=doc.get("mime_type") or "application/pdf")


@app.get("/api/documents/{document_id}/pages/{page_no}")
def document_page_image(document_id: str, page_no: int) -> Response:
    with session_scope() as sess:
        pages = {p["page_no"]: p for p in repo.list_document_pages(sess, document_id)}
    if page_no not in pages:
        raise HTTPException(404, "page not found")
    key = storage.key_from_uri(pages[page_no]["image_uri"])
    return Response(content=storage.get_bytes(key), media_type="image/png")


@app.get("/api/patients/{patient_id}/fhir")
def patient_fhir(patient_id: str) -> Any:
    try:
        return JSONResponse(project_patient(patient_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/documents/{document_id}/bundle")
def document_bundle(document_id: str) -> Any:
    try:
        return JSONResponse(project_document(document_id, persist=False)["bundle"])
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/documents/{document_id}/evidence")
def document_evidence(document_id: str) -> dict[str, Any]:
    with session_scope() as sess:
        doc = repo.get_document(sess, document_id)
        if not doc:
            raise HTTPException(404, "document not found")
        cls = repo.get_doc_classification(sess, document_id)
        blocks = repo.list_ocr_blocks(sess, document_id)
        facts = repo.list_clinical_facts(sess, document_id=document_id)
    return {
        "document_id": document_id,
        "doc_type": (cls or {}).get("doc_type"),
        "ocr_block_count": len(blocks),
        "facts": [
            {"type": f["fact_type"], "text": f["local_text"],
             "code_system": f["code_system"], "code": f["code"],
             "code_display": f["code_display"], "code_status": f["code_status"],
             "value_num": float(f["value_num"]) if f["value_num"] is not None else None,
             "unit": f["value_unit_ucum"], "flag": f["abnormal_flag"],
             "confidence": float(f["confidence_overall"])}
            for f in facts
        ],
    }


def main() -> None:
    import uvicorn

    from .surface import startup_checks

    startup_checks()
    uvicorn.run(app, host="0.0.0.0", port=settings.webapp_port,
                log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()


def _dept_hint(value: str | None) -> str | None:
    from ..extract import department as _department

    return _department.normalise_hint(value)
