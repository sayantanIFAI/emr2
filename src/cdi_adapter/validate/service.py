from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text

from .. import repo
from ..config import settings
from ..db import session_scope
from ..logging import get_logger
from ..recognition.engines import is_fallback_model
from ..recognition.hierarchy import assess_fact
from ..recognition.interpret import fact_candidates
from . import rules as R
from .policy import rule_for

log = get_logger(__name__)


@dataclass
class ValidateResult:
    document_id: str
    auto_accepted: int
    in_review: int
    conflicts: int
    blockers: int


def _calibrate(conf: float, *, partial: bool, n_warn: int, n_block: int) -> float:
    """Placeholder calibration: honest downgrades until a fitted model exists.
    (roadmap: isotonic regression per doc_type x fact_type on adjudicated data)."""
    c = float(conf or 0.0)
    if partial:
        c -= settings.gate_partial_penalty
    if n_block:
        c = min(c, 0.5)
    elif n_warn:
        c = min(c, 0.9)
    return round(max(0.0, min(1.0, c)), 3)


_HANDWRITING_CODES = {"engine-disagreement", "no-reading", "single-engine", "grounding-failed",
                      "page-level-evidence", "candidate-collision"}


def _dominant_kind(findings_by_fact: dict[str, list[R.Finding]]) -> str:
    codes = [c for fs in findings_by_fact.values() for (_s, c, _m) in fs]
    if any(c in _HANDWRITING_CODES for c in codes):
        return "handwriting"
    if any(c.startswith("med-") for c in codes):
        return "dose_check"
    if any(c in ("value-out-of-range", "date-in-future", "date-implausible") for c in codes):
        return "low_confidence"
    if any(c == "unmapped-concept" for c in codes):
        return "unmapped_terminology"
    if any(c in ("no-provenance", "weak-evidence") for c in codes):
        return "low_confidence"
    return "low_confidence"


def fallback_findings(extraction_model: str | None, blocks: list[dict[str, Any]]) -> list[R.Finding]:
    """A blocker for anything the OOM fallback model read (``gate_fallback_review``).

    The fallback is another model generation, loaded in 8-bit, and not yet benchmarked on
    handwritten prescriptions, so a person looks at everything it read: the S4 extraction (the
    model recorded on its pipeline run) and any line crop it transcribed (``recognition.
    fallback_model``). The model is decided from recorded provenance, never guessed."""
    if not settings.gate_fallback_review:
        return []
    models = [extraction_model] if is_fallback_model(extraction_model) else []
    models += [m for b in blocks
               if is_fallback_model(m := (b.get("recognition") or {}).get("fallback_model"))]
    if not models:
        return []
    return [("blocker", "fallback-model", f"read by the fallback model {models[0]}")]


def field_review_items(payload: dict[str, Any] | None, blocks: list[dict[str, Any]], doc: dict[str, Any],
                       extraction_model: str | None) -> list[dict[str, str]]:
    """What a person must look at among the values that are not facts: patient and doctor details,
    lab preparation, the follow-up, text aimed at the system, a cut-off or partial answer, and
    everything the OOM fallback model read. ``[]`` when all is well."""
    from datetime import UTC, date, datetime

    from ..extract import fields

    if not isinstance(payload, dict):
        return []
    d = doc.get("captured_at") or doc.get("ingested_at")
    today = d.date() if isinstance(d, datetime) else d if isinstance(d, date) else datetime.now(UTC).date()
    items = list(fields.build_checks(payload, blocks, today)["review"])
    if payload.get("_truncated"):
        items.append({"field": "document", "reason": "the answer was cut off by the length limit: it is incomplete"})
    elif payload.get("_partial"):
        items.append({"field": "document", "reason": "the answer did not fully match the form"})
    if settings.gate_fallback_review and is_fallback_model(extraction_model):
        items.append({"field": "document", "reason": f"read by the fallback model {extraction_model}"})
    return items


def validate_document(document_id: str | UUID) -> ValidateResult:
    document_id = str(document_id)
    auto = review = conflicts = blockers_total = 0

    with session_scope() as sess:
        doc = repo.get_document(sess, document_id)
        if not doc:
            raise ValueError(f"document {document_id} not found")
        cls = repo.get_doc_classification(sess, document_id)
        facts = repo.list_clinical_facts(sess, document_id=document_id)
        run_id = repo.start_pipeline_run(sess, document_id=document_id, stage="validate",
                                         model_name="rules-v1")

        ext = sess.execute(
            text("SELECT payload FROM extraction WHERE document_id = :d "
                 "ORDER BY created_at DESC LIMIT 1"),
            {"d": document_id},
        ).scalar_one_or_none()
        partial = bool(isinstance(ext, dict) and ext.get("_partial"))
        extraction_model = repo.get_extraction_model(sess, document_id)

        enc_cache: dict[str, dict[str, Any]] = {}

        def enc_of(fid_row: dict[str, Any]) -> dict[str, Any] | None:
            eid = fid_row.get("encounter_id")
            if not eid:
                return None
            eid = str(eid)
            if eid not in enc_cache:
                r = sess.execute(text("SELECT * FROM encounter WHERE id = :i"),
                                 {"i": eid}).mappings().first()
                enc_cache[eid] = dict(r) if r else {}
            return enc_cache[eid]

        held: dict[str, list[R.Finding]] = {}
        held_fact_payload: list[dict[str, Any]] = []

        for f in facts:
            fid = str(f["id"])
            md = repo.get_medication_detail(sess, fid) if f["fact_type"] == "medication" else None
            prov = repo.get_fact_provenance(sess, fid)
            enc = enc_of(f)
            fact_blocks = repo.get_blocks_by_ids(
                sess, [b for p in prov for b in (p.get("ocr_block_ids") or [])])

            findings: list[R.Finding] = []
            findings += R.check_value_range(f)
            findings += R.check_unit_present(f)
            findings += fallback_findings(extraction_model, fact_blocks)
            findings += R.check_dates(f, enc)
            findings += R.check_evidence(f, prov)
            findings += R.check_terminology(f, settings.gate_local_only_review_types)
            if f["fact_type"] == "medication":
                findings += R.check_medication(f, md)

            # recognition v2: evidence hierarchy (pixels, engines, candidates, priors)
            assessment = None
            rule = None
            if settings.recognition_v2:
                assessment = assess_fact(f, md, fact_blocks, fact_candidates(sess, fid))
                findings += assessment.findings
                if settings.gate_policy_enabled:
                    rule = rule_for(f["fact_type"], assessment.evidence_state)

            n_block = sum(1 for s, _c, _m in findings if s == "blocker")
            n_warn = sum(1 for s, _c, _m in findings if s == "warn")
            blockers_total += n_block

            conf = _calibrate(f.get("confidence_overall"), partial=partial,
                              n_warn=n_warn, n_block=n_block)

            note = "; ".join(f"[{s}] {m}" for s, _c, m in findings) or None
            policy_review = bool(rule and (rule.action == "review" or (
                rule.min_conf is not None and conf < rule.min_conf)))
            must_review = (
                n_block > 0
                or partial
                or policy_review
                or conf < settings.gate_review_floor
                or (settings.gate_medication_always_review and f["fact_type"] == "medication"
                    and any(c.startswith("med-") for _s, c, _m in findings))
            )

            if must_review:
                state = "in_review"
                review += 1
                held[fid] = findings
                held_fact_payload.append({
                    "fact_id": fid, "fact_type": f["fact_type"],
                    "text": f["local_text"], "confidence": conf,
                    "findings": [{"severity": s, "code": c, "message": m} for s, c, m in findings],
                })
            elif conf >= settings.gate_auto_accept_conf and not findings:
                state = "auto_accepted"
                auto += 1
            elif conf >= settings.gate_audit_conf and n_block == 0:
                state = "auto_accepted"
                auto += 1
                if random.random() < settings.audit_sample_rate:
                    repo.create_review_task(
                        sess, kind="low_confidence", patient_id=str(f["patient_id"]),
                        document_id=document_id, ref_fact_ids=[fid], priority=5,
                        payload={"reason": "audit sample", "confidence": conf},
                    )
            else:
                state = "in_review"
                review += 1
                held[fid] = findings
                held_fact_payload.append({
                    "fact_id": fid, "fact_type": f["fact_type"],
                    "text": f["local_text"], "confidence": conf, "findings": [],
                })

            repo.set_fact_review(sess, fid, review_state=state,
                                 confidence_overall=conf, review_note=note)
            if assessment is not None:
                trace = assessment.trace + [{
                    "tier": "gate", "state": state, "confidence": conf,
                    "policy": rule.key if rule else None,
                    "policy_action": rule.action if rule else None,
                    "findings": [c for _s, c, _m in findings]}]
                repo.set_fact_decision(sess, fid, evidence_state=assessment.evidence_state,
                                       field_policy=rule.key if rule else None,
                                       decision_trace=trace)

            # cross-document duplicates / contradictions
            if f["fact_type"] in ("condition", "lab_result", "procedure", "medication") and f.get("code"):
                for other in repo.find_similar_current_facts(
                    sess, patient_id=str(f["patient_id"]), fact_type=f["fact_type"],
                    code=str(f["code"]), exclude_id=fid,
                ):
                    same_day = _same_day(f.get("effective_time"), other.get("effective_time"))
                    va, vb = f.get("value_num"), other.get("value_num")
                    if va is not None and vb is not None and same_day and abs(float(va) - float(vb)) > 1e-6:
                        repo.insert_fact_conflict(
                            sess, patient_id=str(f["patient_id"]), fact_a=fid,
                            fact_b=str(other["id"]), conflict_type="value_mismatch",
                            evidence_state="CONTRADICTED", severity="blocker")
                        conflicts += 1
                        if state != "in_review":
                            repo.set_fact_review(sess, fid, review_state="in_review",
                                                 review_note="cross-document value conflict")
                            review += 1
                            auto = max(0, auto - 1)
                            state = "in_review"
                    elif same_day and (va == vb or (not va and not vb)):
                        repo.insert_fact_conflict(
                            sess, patient_id=str(f["patient_id"]), fact_a=fid,
                            fact_b=str(other["id"]), conflict_type="duplicate",
                            evidence_state="SUPPORTED", severity="info",
                            auto_resolution="keep_a")
                        conflicts += 1

            # only a value that is still auto-accepted after every check enters the ledger
            if assessment is not None and state == "auto_accepted":
                record_verified(sess, f, md, assessment, method="auto_accepted",
                                confidence=conf, policy_id=rule.key if rule else None,
                                trace=trace, document_id=document_id)

        if (cls or {}).get("doc_type") == "prescription":
            flagged = field_review_items(ext if isinstance(ext, dict) else None,
                                         repo.list_ocr_blocks(sess, document_id), doc, extraction_model)
            if flagged:
                repo.create_review_task(
                    sess, kind="low_confidence", patient_id=str(facts[0]["patient_id"]) if facts else None,
                    document_id=document_id, ref_fact_ids=[], priority=3,
                    payload={"source": "field-checks", "fields": flagged})
        if held:
            repo.create_review_task(
                sess, kind=_dominant_kind(held),
                patient_id=str(facts[0]["patient_id"]) if facts else None,
                document_id=document_id,
                ref_fact_ids=list(held.keys()), priority=2,
                payload={"doc_type": (cls or {}).get("doc_type"),
                         "held": held_fact_payload, "partial_extraction": partial},
            )

        repo.finish_pipeline_run(sess, run_id, status="ok",
                                 metrics={"auto_accepted": auto, "in_review": review,
                                          "conflicts": conflicts, "blockers": blockers_total,
                                          "partial_extraction": partial})
        repo.set_document_status(sess, document_id, "validated")
        if settings.recognition_v2:
            # E17: normalised rx_* rows + queue the FHIR builder agent (savepoint: the
            # governance decisions above must not be lost to a sync error)
            try:
                from ..persist.normalized import enqueue_fhir, sync_document

                with sess.begin_nested():
                    sync_document(sess, document_id)
                    enqueue_fhir(sess, patient_id=str(facts[0]["patient_id"]) if facts else None,
                                 document_id=document_id, reason="validated")
            except Exception as exc:  # noqa: BLE001
                log.warning("rx_sync_skipped", document_id=document_id, error=str(exc)[:200])
        repo.write_audit(sess, actor="validate-svc", action="update", entity="clinical_fact",
                         entity_id=document_id,
                         patient_id=str(facts[0]["patient_id"]) if facts else None,
                         detail={"auto_accepted": auto, "in_review": review,
                                 "conflicts": conflicts})

    log.info("validated", document_id=document_id, auto_accepted=auto, in_review=review,
             conflicts=conflicts, blockers=blockers_total, partial=partial)
    from ..output.store import write_result_quietly      # after the commit: the file is built from saved rows

    write_result_quietly(document_id)
    return ValidateResult(document_id, auto, review, conflicts, blockers_total)


def _same_day(a: Any, b: Any) -> bool:
    if not isinstance(a, datetime) or not isinstance(b, datetime):
        return False
    return a.date() == b.date()


def governed_value(f: dict[str, Any], md: dict[str, Any] | None) -> dict[str, Any]:
    keys = ("local_text", "value_num", "value_unit_ucum", "value_text", "value_code_display",
            "effective_time", "clinical_status")
    v = {k: f.get(k) for k in keys if f.get(k) is not None}
    if md:
        v["medication"] = {k: md.get(k) for k in (
            "drug_text", "strength_num", "strength_unit", "dose_num", "dose_unit_ucum",
            "frequency_code", "frequency_per_day", "duration_days", "route", "instructions")
            if md.get(k) is not None}
    return v


def record_verified(sess: Any, f: dict[str, Any], md: dict[str, Any] | None, assessment: Any,
                    *, method: str, confidence: float | None, policy_id: str | None,
                    trace: list[dict[str, Any]], document_id: str | None,
                    reviewer_id: str | None = None) -> str:
    """Append the governed value to the immutable verified-fact ledger (E4-S7)."""
    return repo.insert_verified_fact(
        sess, fact_id=f["id"], document_id=document_id, patient_id=f.get("patient_id"),
        fact_type=f["fact_type"],
        observation_ids=getattr(assessment, "observation_ids", []) or [],
        winning_candidate_ids=getattr(assessment, "winning_candidate_ids", []) or [],
        code_system=f.get("code_system"), code=f.get("code"), code_display=f.get("code_display"),
        governed_value=governed_value(f, md), verification_method=method,
        confidence=confidence, evidence_state=getattr(assessment, "evidence_state", None),
        policy_id=policy_id, reviewer_id=reviewer_id,
        model_stack={"recognition": "v2", "vlm": settings.vlm_model_id}, decision_trace=trace)
