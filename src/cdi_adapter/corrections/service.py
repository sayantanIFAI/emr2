"""The correction loop: human correction -> database -> doctor profile -> alias cascade.

    human correction UI -> POST /api/corrections -> PostgreSQL (source of truth)
        -> profile updater (doctor_lexicon + class-C kb_alias) -> [optional Redis cache] -> resolver

A correction does two things at once, with no model retraining:

1. **runtime knowledge**: "this doctor writes `S Cr` and a reviewer says it means `Serum Creatinine`"
   is counted in the doctor's lexicon; once confirmed often enough and resolving to exactly ONE
   concept it becomes a class-C alias for that doctor, which the alias cascade ranks highest for that
   doctor's documents (never for another doctor's);
2. **training data**: the append-only ``correction`` row (original prediction, both engines' readings,
   the correction, the doctor, the crop reference, the reviewer, the model versions) is the dataset
   for the next offline fine-tuning cycle (``export_training``).

Nothing here changes a reading or creates a concept. An unknown doctor gets the correction stored
(global training data) but no profile entry; a correction that matches no unique concept stays in the
lexicon un-promoted.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects import postgresql, sqlite

from ..config import settings
from ..logging import get_logger
from ..recognition.alias import norm_alias
from .models import correction, doctor_lexicon, kb_alias, kb_concept

log = get_logger(__name__)

# fact type -> knowledge-base domain a promoted alias lives in (anything else is learned but not promoted)
FIELD_DOMAIN = {"investigation_order": "lab_order", "condition": "diagnosis", "medication": "drug"}
_REVIEWER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@\- ]{0,63}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CorrectionError(ValueError):
    """A refusal with a plain message and the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Context:
    """What was predicted for one field and where it came from."""

    document_id: str
    fact_id: str
    field_type: str
    original_value: str
    doctor_id: str | None = None
    qwen_value: str | None = None
    confidence: float | None = None
    prediction_status: str = "needs_review"
    crop_hash: str | None = None
    crop_ref: dict[str, Any] = field(default_factory=dict)
    model_stack: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ the prediction record (what the OCR service emits)


def prediction_status(review_state: str | None) -> str:
    return {"auto_accepted": "accepted", "clinician_confirmed": "accepted",
            "corrected": "corrected"}.get(review_state or "", "needs_review")


def engine_value(blocks: list[dict[str, Any]]) -> str | None:
    """What Qwen read for the lines a fact came from (several lines are joined in order)."""
    qwen = []
    for b in blocks:
        eng = ((b.get("recognition") or {}).get("engines")) or {}
        if eng.get("qwen2.5-vl"):
            qwen.append(str(eng["qwen2.5-vl"]))
    return " ".join(qwen) or None


def build_field_record(fact: dict[str, Any], blocks: list[dict[str, Any]], doctor_id: str | None,
                       document_id: str, crop_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """The per-field record for downstream tools (the correction UI): one dict per extracted value."""
    qwen = engine_value(blocks)
    obs = [str(o) for b in blocks for o in (b.get("observation_ids") or [])]
    hashes = sorted({(crop_hashes or {})[o] for o in obs if o in (crop_hashes or {})})
    return {
        "prescription_id": str(document_id), "doctor_id": doctor_id, "field_id": str(fact["id"]),
        "field_type": fact["fact_type"],
        "raw_crop_reference": {"observation_ids": obs, "crop_hashes": hashes,
                               "bboxes": [list(b["bbox"]) for b in blocks if b.get("bbox")]},
        "qwen_value": qwen,
        "final_value": fact.get("local_text"),
        "confidence": None if fact.get("confidence_overall") is None else round(float(fact["confidence_overall"]), 3),
        "status": prediction_status(fact.get("review_state")),
    }


# ------------------------------------------------------------------ loading (PostgreSQL)


def load_context(sess: Any, document_id: str, fact_id: str) -> Context:
    """Everything known about one field, read from the live tables. Raises ``CorrectionError`` (404)
    when the field does not exist or does not belong to the document."""
    from sqlalchemy import text

    from .. import repo

    row = sess.execute(text("SELECT * FROM clinical_fact WHERE id = :i"), {"i": fact_id}).mappings().first()
    if row is None:
        raise CorrectionError("That field was not found.", 404)
    fact = dict(row)
    if str(document_id) not in [str(d) for d in (fact.get("source_doc_ids") or [])]:
        raise CorrectionError("That field does not belong to this prescription.", 404)
    prov = repo.get_fact_provenance(sess, fact_id)
    block_ids = [b for p in prov for b in (p.get("ocr_block_ids") or [])]
    blocks = repo.get_blocks_by_ids(sess, block_ids)
    obs_ids = [str(o) for b in blocks for o in (b.get("observation_ids") or [])]
    hashes: dict[str, str] = {}
    if obs_ids:
        for r in sess.execute(text("SELECT id, crop_hash FROM ocr_observation WHERE id = ANY(CAST(:ids AS uuid[]))"),
                              {"ids": obs_ids}).mappings():
            if r["crop_hash"]:
                hashes[str(r["id"])] = r["crop_hash"]
    doc = repo.get_document(sess, document_id) or {}
    rec = build_field_record(fact, blocks, str(doc["practitioner_id"]) if doc.get("practitioner_id") else None,
                             str(document_id), hashes)
    first_page = next((p.get("page_id") for p in prov if p.get("page_id")), None)
    return Context(
        document_id=str(document_id), fact_id=str(fact_id), field_type=fact["fact_type"],
        original_value=fact.get("local_text") or "", doctor_id=rec["doctor_id"],
        qwen_value=rec["qwen_value"], confidence=rec["confidence"],
        prediction_status=rec["status"],
        crop_hash=(rec["raw_crop_reference"]["crop_hashes"] or [None])[0],
        crop_ref={**rec["raw_crop_reference"], "page_id": str(first_page) if first_page else None},
        model_stack=(prov[0].get("model_stack") if prov else None) or {})


def field_records(sess: Any, document_id: str) -> list[dict[str, Any]]:
    """The records for every current field of a document (what the OCR service hands to the correction tool)."""
    from sqlalchemy import text

    from .. import repo

    doc = repo.get_document(sess, document_id) or {}
    doctor = str(doc["practitioner_id"]) if doc.get("practitioner_id") else None
    out = []
    for fact in repo.list_clinical_facts(sess, document_id=document_id):
        prov = repo.get_fact_provenance(sess, str(fact["id"]))
        blocks = repo.get_blocks_by_ids(sess, [b for p in prov for b in (p.get("ocr_block_ids") or [])])
        obs = [str(o) for b in blocks for o in (b.get("observation_ids") or [])]
        hashes = {}
        if obs:
            hashes = {str(r["id"]): r["crop_hash"] for r in sess.execute(
                text("SELECT id, crop_hash FROM ocr_observation WHERE id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": obs}).mappings() if r["crop_hash"]}
        out.append(build_field_record(fact, blocks, doctor, str(document_id), hashes))
    return out


# ------------------------------------------------------------------ validation


def clean_value(value: Any, original: str) -> str:
    """The corrected value as it will be stored: one line of plain text, and actually different."""
    if not isinstance(value, str):
        raise CorrectionError("The corrected value must be text.")
    v = re.sub(r"\s+", " ", _CONTROL.sub(" ", value)).strip()
    if not v:
        raise CorrectionError("The corrected value is empty.")
    if len(v) > settings.correction_max_len:
        raise CorrectionError(f"The corrected value is longer than {settings.correction_max_len} characters.")
    if re.sub(r"\s+", " ", original).strip().casefold() == v.casefold():
        raise CorrectionError("The corrected value is the same as the value that was read: nothing to correct.")
    return v


def clean_reviewer(reviewer_id: Any) -> str:
    if not isinstance(reviewer_id, str) or not _REVIEWER.match(reviewer_id.strip()):
        raise CorrectionError("Please give the reviewer's id (letters, numbers, . _ @ - and spaces, up to 64).")
    return reviewer_id.strip()


# ------------------------------------------------------------------ writing


def _insert(sess: Any, table: Any):
    """``INSERT`` with ``ON CONFLICT`` for the database in use (PostgreSQL in production, SQLite in tests)."""
    return (postgresql if sess.get_bind().dialect.name == "postgresql" else sqlite).insert(table)


def record_correction(sess: Any, ctx: Context, corrected_value: str, reviewer_id: str,
                      now: datetime | None = None) -> dict[str, Any]:
    """Store the correction (append-only) and update the doctor's profile. Returns what was done."""
    now = now or datetime.now(UTC)
    value = clean_value(corrected_value, ctx.original_value)
    reviewer = clean_reviewer(reviewer_id)
    cid = str(uuid.uuid4())
    sess.execute(correction.insert().values(
        id=cid, document_id=ctx.document_id, fact_id=ctx.fact_id, doctor_id=ctx.doctor_id,
        field_type=ctx.field_type, original_value=ctx.original_value, qwen_value=ctx.qwen_value,
        corrected_value=value, confidence=ctx.confidence,
        prediction_status=ctx.prediction_status, crop_hash=ctx.crop_hash, crop_ref=ctx.crop_ref,
        reviewer_id=reviewer, model_stack=ctx.model_stack, created_at=now))
    learned = update_lexicon(sess, ctx, value, now) if ctx.doctor_id else None
    return {"correction_id": cid, "doctor_id": ctx.doctor_id, "field_type": ctx.field_type,
            "corrected_value": value, "learned": learned}


def update_lexicon(sess: Any, ctx: Context, canonical: str, now: datetime) -> dict[str, Any] | None:
    """Count "this doctor writes X and it means Y"; promote it to a doctor alias when it has earned it."""
    raw = ctx.original_value.strip()
    raw_norm = norm_alias(raw)
    if not raw_norm or ctx.doctor_id is None:
        return None
    ins = _insert(sess, doctor_lexicon).values(
        practitioner_id=ctx.doctor_id, field_type=ctx.field_type, raw_norm=raw_norm, raw_example=raw,
        canonical=canonical, count=1, verified_count=1, first_seen=now, last_seen=now)
    sess.execute(ins.on_conflict_do_update(
        index_elements=["practitioner_id", "field_type", "raw_norm", "canonical"],
        set_={"count": doctor_lexicon.c.count + 1, "verified_count": doctor_lexicon.c.verified_count + 1,
              "last_seen": now, "raw_example": raw}))
    row = sess.execute(select(doctor_lexicon).where(
        doctor_lexicon.c.practitioner_id == ctx.doctor_id, doctor_lexicon.c.field_type == ctx.field_type,
        doctor_lexicon.c.raw_norm == raw_norm, doctor_lexicon.c.canonical == canonical)).mappings().one()
    concept = row["concept_id"]
    if concept is None and row["verified_count"] >= settings.doctor_alias_min_verified:
        concept = promote(sess, ctx.doctor_id, ctx.field_type, raw, raw_norm, canonical, now)
        if concept:
            sess.execute(update(doctor_lexicon).where(doctor_lexicon.c.id == row["id"]).values(concept_id=concept))
    return {"raw": raw, "canonical": canonical, "count": row["count"], "verified_count": row["verified_count"],
            "promoted_to": concept}


def resolve_concept(sess: Any, field_type: str, canonical: str) -> str | None:
    """The ONE concept the canonical text names, or ``None`` (unknown or ambiguous: never guessed)."""
    domain = FIELD_DOMAIN.get(field_type)
    if domain is None:
        return None
    key = norm_alias(canonical)
    by_name = select(kb_concept.c.id).where(
        kb_concept.c.domain == domain, kb_concept.c.active.is_(True),
        func.lower(kb_concept.c.canonical_name) == canonical.casefold())
    by_alias = (select(kb_alias.c.concept_id).join(kb_concept, kb_concept.c.id == kb_alias.c.concept_id)
                .where(kb_concept.c.domain == domain, kb_concept.c.active.is_(True),
                       kb_alias.c.alias_norm == key, kb_alias.c.alias_class != "C"))   # global aliases only
    ids = {r[0] for r in sess.execute(by_name)} | {r[0] for r in sess.execute(by_alias)}
    return next(iter(ids)) if len(ids) == 1 else None


def promote(sess: Any, doctor_id: str, field_type: str, raw: str, raw_norm: str, canonical: str,
            now: datetime) -> str | None:
    """Make ``raw`` a class-C alias of the concept ``canonical`` names, for this doctor only."""
    if len(raw_norm) < 2:
        return None                                   # a single letter is never a safe alias
    concept = resolve_concept(sess, field_type, canonical)
    if concept is None:
        return None
    ins = _insert(sess, kb_alias).values(concept_id=concept, alias=raw, alias_norm=raw_norm, alias_class="C",
                                         practitioner_id=doctor_id, source="correction", created_at=now)
    sess.execute(ins.on_conflict_do_nothing(
        index_elements=["concept_id", "alias_norm", "alias_class", "practitioner_id"]))
    log.info("doctor_alias_promoted", doctor_id=doctor_id, concept=concept, alias_len=len(raw))
    return concept


# ------------------------------------------------------------------ reading


def get_profile(sess: Any, practitioner_id: str, field_type: str | None = None, cache: Any = None) -> dict[str, Any]:
    """The doctor's lexicon. The database is the source of truth; ``cache`` (optional) only speeds reads."""
    key = f"{practitioner_id}:{field_type or '*'}"
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
    q = select(doctor_lexicon).where(doctor_lexicon.c.practitioner_id == practitioner_id)
    if field_type:
        q = q.where(doctor_lexicon.c.field_type == field_type)
    rows = sess.execute(q.order_by(doctor_lexicon.c.verified_count.desc(), doctor_lexicon.c.raw_norm)).mappings().all()
    entries = [{"field_type": r["field_type"], "raw": r["raw_example"], "canonical": r["canonical"],
                "count": r["count"], "verified_count": r["verified_count"], "promoted_to": r["concept_id"],
                "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None} for r in rows]
    prof = {"practitioner_id": str(practitioner_id), "entries": entries,
            "promoted": sum(1 for e in entries if e["promoted_to"]), "cached": False}
    if cache is not None:
        cache.set(key, {k: v for k, v in prof.items() if k != "cached"})
    return prof


def export_training(sess: Any, since: datetime | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """The training rows (one per correction), oldest first."""
    q = select(correction).order_by(correction.c.created_at, correction.c.id)
    if since is not None:
        q = q.where(correction.c.created_at >= since)
    if limit:
        q = q.limit(limit)
    return [{"correction_id": r["id"], "prescription_id": r["document_id"], "field_id": r["fact_id"],
             "doctor_id": r["doctor_id"], "field_type": r["field_type"], "original_value": r["original_value"],
             "qwen_value": r["qwen_value"], "corrected_value": r["corrected_value"],
             "confidence": None if r["confidence"] is None else float(r["confidence"]),
             "prediction_status": r["prediction_status"], "crop_hash": r["crop_hash"], "crop_ref": r["crop_ref"],
             "reviewer_id": r["reviewer_id"], "model_stack": r["model_stack"],
             "created_at": r["created_at"].isoformat()} for r in sess.execute(q).mappings()]
