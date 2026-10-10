# CDI-Adapter — Clinical Document Intelligence → EMR/EHR Adapter

An **AI side-car** that sits on top of a legacy Hospital Management System (HMS) and its
database. The hospital's only medical record today is **scanned images / photocopies of
paper** (prescriptions, lab reports, operative notes, discharge summaries, advice slips,
vitals sheets). This adapter turns that pile into a **structured, longitudinal EMR/EHR** —
**rows in tables** and **validated FHIR R4 JSON**, ABDM/ABHA-ready — **without touching the
legacy database**.

- Full design, architecture, 9-stage flow, DB schema, FHIR/ABDM mapping, 10-country
  research, phased plan and DSLM fine-tuning plan: **[docs/DESIGN.md](docs/DESIGN.md)**.
- Governing principle: **the LLM/VLM is never the source of truth.** Every clinical fact is
  bound to a pixel region + OCR span + model version + confidence, validated by
  deterministic rules and terminology, and low-confidence / conflicting facts go to a
  clinician before entering the record.

## Status

| Phase | Scope | State |
|---|---|---|
| **0 — Foundations** | repo, schema (DDL), storage, config, CI, compose, RunPod bootstrap | **done** |
| **1 — Ingest + Classify + OCR** | S1 folder-watch ingest + page normalization ✅ · S2 VLM classification ✅ · S3 OCR (RapidOCR printed + VLM handwriting) → `ocr_block` w/ bboxes ✅ | **done** (model gateway: `cdi_adapter.mlserve`) |
| 2 — Evidence + Extraction | schema-locked VLM/DSLM extraction + evidence map | next |
| 3 — Terminology + Validation | SNOMED/LOINC/ICD binding + rules + conflict engine | planned |
| 4 — FHIR projection | facts → FHIR + ABDM Composition bundles + rows | planned |
| 5 — Reconciliation + Review UI | LangGraph agent + HITL console | planned |
| 6 — Legacy write-back + ABDM gateway | CDC connector, FHIR façade, Fidelius edge | planned |
| 7 — Eval + hardening + pilot | eval harness, monitoring, shadow-mode pilot | planned |

## What's in this scaffold (Phase 0 + Phase 1 ingest slice)

```
docs/DESIGN.md            the full design document
db/schema.sql             authoritative DDL (13 table groups + read-model views)
db/alembic/               migrations (0001 runs schema.sql + views.sql)
src/cdi_adapter/
  config.py               env-driven settings (CDI_* prefix)
  db.py  storage.py  repo.py   Postgres + S3 object store access
  ingest/                S1  pages.py (deskew/denoise/CLAHE) · service.py (sha256 dedupe,
                             immutable original) · watcher.py (folder-watch + sidecar)
  classify/              S2  prompt.py + service.py -> doc_classification (VLM, schema-locked)
  ocr/                   S3  rapid.py (RapidOCR printed) · vlm_ocr.py (handwriting) ->
                             ocr_block rows (text + bbox + conf + reading order)
  ml/client.py               HTTP client to the model gateway (+ StubMLClient for tests)
  mlserve/                    the gateway: FastAPI, backends = stub | hf (transformers Qwen2.5-VL)
  api.py                     FastAPI: /ingest /healthz /documents/{id}[/classification|/ocr|/evidence]
  worker.py                  Celery chain: ingest -> classify -> ocr -> extract(stub)
schemas/                  JSON Schemas for extraction (prescription/lab_report/vitals v3)
scripts/make_sample_docs.py   synthetic scanned-style test documents
infra/compose/            docker-compose stack + Dockerfile
infra/runpod/             RunPod bootstrap + smoke-test guide
tests/                    unit (no infra) + integration (auto-skip without infra)
```

## Quick start (local)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                          # unit tests, no infra needed

# full stack (Docker):
make up                                            # postgres + redis + object store + api + worker + watcher
python scripts/make_sample_docs.py --out ./data/inbox --count 3
curl -s localhost:8080/healthz | jq
```

## Quick start (RunPod)

See **[infra/runpod/README.md](infra/runpod/README.md)**. The current pilot pod is an
**RTX PRO 4000 Blackwell 24 GB** with **no Docker**; the stack runs as native processes.
`/workspace` (MooseFS) holds the repo, venv, sample data, models and the **object
store** (SeaweedFS) (the real document bytes) plus a periodic `pg_dump`; the live Postgres cluster runs
on the ephemeral overlay and is rebuilt/restored by `infra/runpod/bootstrap_pod.sh` on each
restart. The 24 GB GPU means later model phases use a 4-bit (AWQ) or 8-bit 7B stack,
not full-precision co-resident 7B.

## Recognition v2 (handwriting, evidence, agents)

Design: `docs/ARCHITECTURE.md` §15–§16; what is built: **§17**. In short:
OpenCV quality gate → RapidOCR (printed) → OpenCV line regions → **Qwen2.5-VL per line crop**
(the one handwriting reader) → single-reader verdict → append-only
`ocr_observation` → extraction → alias cascade + clinical-context plausibility → pixel
grounding + per-field policy → `verified_fact` ledger → normalised `rx_*` tables → FHIR
builder agent (blob store). Plus a file listener (OneDrive / SharePoint / Google Drive / local - chosen by config only, batches of 3, 3 retries, `success`/`error`/`log` folders; see `docs/LISTENER.md`),
a doctor master matcher and a dispatch agent for downstream screens (disabled until approved).

```bash
python -m cdi_adapter.ocrhost                       # CPU OCR host on :8079
python -m cdi_adapter.listener.service [--check|--login]   # file listener (CDI_LISTENER_CONNECTOR)
python -m cdi_adapter.listener.recovery             # retries failed files (max 3)
python -m cdi_adapter.agents.fhir_builder           # fhir_outbox -> bundle -> blob
python -m cdi_adapter.agents.dispatch run           # governed data -> approved screens
```

## Configuration

All settings are env vars with the `CDI_` prefix (see `.env.example`). Two jurisdiction
knobs make the pipeline portable: `CDI_IG_PACKAGE` (FHIR IG, default
`nrces.fhir.r4.ndhm#6.5.0`) and `CDI_TERMINOLOGY_PACKAGE`.

## Non-goals / guarantees

- The legacy HMS database is **read-only** to this system. It is never altered.
- No document ever leaves the premises. The ABDM edge (Phase 6) is the only outbound
  component and only sends **consented, Fidelius-encrypted** bundles; encryption keys are
  never persisted.
- No per-document API inference cost — all models are open-source and run on-prem.
