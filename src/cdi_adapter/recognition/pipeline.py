"""Recognition v2 (ARCHITECTURE §15): regions -> independent engines -> immutable evidence
-> ocr_block projection.

    page (src render) ──► RapidOCR lines (printed evidence, CPU host)
                     └──► OpenCV line detection + printed/handwritten/mixed classifier
    handwritten | mixed | uncertain line crop ──► Qwen2.5-VL per crop (the one handwriting reader;
                                                  a second read of the same line with other padding
                                                  is the stability check)
    every reading ──► ocr_observation (append-only)
    per line verdict (printed | agree | disagree | single_engine | no_reading)
                 ──► ocr_block.recognition  (S4 reads the text, S6 gates on the state)

A page on which line detection finds nothing (or that errors) falls back to the legacy
page-level VLM transcription, recorded with evidence state ``page_level``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .. import repo, storage
from ..config import settings
from ..db import session_scope
from ..logging import get_logger
from ..ocr.rapid import OcrLine
from .disagreement import AGREE, DISAGREE, NONE, SINGLE, compare_engines, self_consistency
from .drift import disagreement_rate, drift_alarm, recent_rates
from .engines import QwenLineEngine, Reading, is_fallback_model
from . import nontext
from .ocrhost_client import get_ocr_host
from .regions import PRINTED, Region, detect_regions, prepare_crop

log = get_logger(__name__)

PRINTED_STATE, PAGE_LEVEL = "printed", "page_level"
# confidence handed to S4/S6 per evidence state. Disagreement and missing readings are
# forced to review by policy regardless of this number; it only keeps the arithmetic honest.
_STATE_CONF_CAP = {PRINTED_STATE: 1.0, AGREE: 0.98, SINGLE: 0.6, DISAGREE: 0.3,
                   NONE: 0.0, PAGE_LEVEL: 0.55}


@dataclass
class RecognizeResult:
    document_id: str
    n_blocks: int
    n_pages: int
    states: dict[str, int]
    engines: dict[str, Any]


def _decode(png: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("page image could not be decoded")
    return img


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lines_from_observations(obs: list[dict[str, Any]]) -> list[OcrLine]:
    return [OcrLine(text=o["raw_text"], bbox=list(o["bbox"]),
                    conf=float(o["raw_confidence"] or 0.0), polygon=o.get("polygon") or [])
            for o in obs if not o.get("error")]


def record_rapid_observations(sess: Any, *, document_id: str, page: dict[str, Any],
                              lines: list[OcrLine], run_id: Any) -> list[str]:
    """Append the printed-text pass as evidence; supersedes the page's previous pass."""
    prev = [o for o in repo.list_current_observations(sess, document_id, engine="rapidocr")
            if str(o["page_id"]) == str(page["id"])]
    prev_ids = [str(o["id"]) for o in prev]
    rows = []
    for i, ln in enumerate(lines):
        rows.append({
            "document_id": document_id, "page_id": page["id"],
            "line_key": f"p{page['page_no']}:r{i}", "region_kind": PRINTED,
            "bbox": ln.bbox, "polygon": ln.polygon or None, "engine": "rapidocr",
            "engine_version": "rapidocr-onnxruntime", "raw_text": ln.text,
            "raw_confidence": round(float(ln.conf), 4), "run_id": run_id,
            "supersedes": prev_ids if i == 0 else [],
        })
    return repo.insert_observations(sess, rows)


def _region_text_printed(r: Region) -> tuple[str, float]:
    lines = sorted(r.ocr_lines, key=lambda ln: ln.bbox[0])
    txt = " ".join(ln.text for ln in lines).strip()
    conf = min((float(ln.conf) for ln in lines), default=0.0)
    return txt, conf


def _engine_conf(state: str, readings: list[Reading], printed_conf: float | None) -> float:
    cap = _STATE_CONF_CAP[state]
    if state == PRINTED_STATE:
        return round(min(cap, printed_conf or 0.0), 4)
    confs = [r.conf for r in readings if r.ok and r.conf is not None]
    base = min(confs) if confs else 0.5
    return round(min(cap, base), 4)


def recognize_page(*, document_id: str, page: dict[str, Any], rapid_lines: list[OcrLine],
                   run_id: Any, lang: str | None) -> tuple[list[dict[str, Any]], list[dict]]:
    """Returns (ocr_block dicts, observation rows) for one page. No DB access."""
    pre = page.get("preproc") or {}
    src_uri = pre.get("src_uri") or page["image_uri"]
    src = _decode(storage.get_bytes(storage.key_from_uri(src_uri)))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    regions = detect_regions(gray, rapid_lines)
    skipped = nontext.classify_regions(regions, src)         # fabric, table, hand, stray marks: never sent to a reader
    if skipped["non_text"]:
        log.info("non_text_regions_set_aside", document_id=document_id, page=page["page_no"], **skipped)

    hw = [r for r in regions if r.needs_handwriting_engines]
    prepared = [prepare_crop(src, r.bbox) for r in hw]             # the crop standard (IM-S3)
    crops = [p[0] for p in prepared]
    crop_info = [p[1] for p in prepared]
    qwe = QwenLineEngine().recognize(crops) if crops else []
    qwe_b: list[Reading] = []
    again: list[bytes] = []
    if crops and settings.qwen_self_consistency:
        # the same line, cut with different padding: an unstable reading is another disagreement signal
        again = [prepare_crop(src, r.bbox, settings.self_consistency_pad_frac)[0] for r in hw]
        qwe_b = QwenLineEngine().recognize(again)

    blocks: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    hw_idx = {id(r): i for i, r in enumerate(hw)}
    for order, r in enumerate(regions):
        key = f"p{page['page_no']}:l{order}"
        if r.kind == nontext.NON_TEXT:
            continue
        if not r.needs_handwriting_engines:
            txt, conf = _region_text_printed(r)
            if not txt:
                continue
            blocks.append({"page_id": page["id"], "block_type": "line", "text": txt,
                           "bbox": r.bbox, "ocr_conf": _engine_conf(PRINTED_STATE, [], conf),
                           "lang": lang, "model_run_id": run_id, "_line_key": key,
                           "recognition": {"state": PRINTED_STATE, "region_kind": r.kind,
                                           "engines": {"rapidocr": txt},
                                           "features": r.features}})
            continue
        i = hw_idx[id(r)]
        readings = [x for x in (qwe[i] if i < len(qwe) else None,) if x is not None]
        verdict = compare_engines(readings)
        second = qwe_b[i] if i < len(qwe_b) else None
        if second is not None and i < len(qwe) and qwe[i] is not None:
            verdict = self_consistency(verdict, qwe[i], second)
            second = Reading("qwen2.5-vl-b", second.engine_version, second.text, second.conf,
                             second.token_confidences, second.prompt_hash, second.error)
        crop_hash = _sha(crops[i])
        for rd in (readings + ([second] if second is not None else [])):
            obs_rows.append({
                "document_id": document_id, "page_id": page["id"], "line_key": key,
                "region_kind": r.kind, "bbox": r.bbox,
                # the second Qwen read saw a differently padded crop: its own hash
                "crop_hash": _sha(again[i]) if rd is second else crop_hash,
                "engine": rd.engine, "engine_version": rd.engine_version,
                "prompt_hash": rd.prompt_hash, "raw_text": rd.text,
                "raw_confidence": rd.conf, "token_confidences": rd.token_confidences or None,
                "error": rd.error, "run_id": run_id,
            })
        adj = None
        if verdict.state == DISAGREE and settings.qwen_adjudication_enabled:
            # L8, advisory: orders the two readings, never resolves the disagreement
            from .adjudicate import ENGINE, adjudicate, prompt_hash

            texts = list(verdict.detail.get("texts") or [])
            adj = adjudicate(crops[i], texts)
            obs_rows.append({
                "document_id": document_id, "page_id": page["id"], "line_key": key,
                "region_kind": r.kind, "bbox": r.bbox, "crop_hash": crop_hash,
                "engine": ENGINE, "engine_version": settings.vlm_model_id,
                "prompt_hash": prompt_hash(), "raw_text": " / ".join(adj.answers),
                "raw_confidence": None, "error": adj.error, "run_id": run_id,
            })
            if adj.preferred_index == 1 and len(texts) == 2:
                verdict.display_text = f"{texts[1]} ⟂ {texts[0]}"
        printed_txt = " ".join(ln.text for ln in r.ocr_lines).strip()
        rec = {"state": verdict.state, "region_kind": r.kind,
               "adjudication": adj.as_dict() if adj else None,
               "engines": {rd.engine: rd.text for rd in readings if rd.ok},
               "errors": {rd.engine: rd.error for rd in readings if not rd.ok} or None,
               "engine_conf": {rd.engine: rd.conf for rd in readings},
               "rapidocr": printed_txt or None, "detail": verdict.detail,
               "features": r.features, "crop": crop_info[i]}
        served_fb = next((rd.engine_version for rd in readings if is_fallback_model(rd.engine_version)),
                         None)
        if served_fb:
            # the gateway answered this crop with the OOM fallback model: S6 never auto-accepts it
            rec["fallback_model"] = served_fb
        text_out = verdict.display_text or printed_txt
        if not text_out:
            continue
        blocks.append({"page_id": page["id"], "block_type": "line", "text": text_out,
                       "bbox": r.bbox, "ocr_conf": _engine_conf(verdict.state, readings, None),
                       "lang": lang, "model_run_id": run_id, "_line_key": key,
                       "recognition": rec})
    return blocks, obs_rows


def _page_level_fallback(document_id: str, page: dict[str, Any], run_id: Any,
                         lang: str | None, reason: str) -> tuple[list[dict], list[dict]]:
    from ..ocr.vlm_ocr import run_vlm_transcription

    png = storage.get_bytes(storage.key_from_uri(page["image_uri"]))
    lines = run_vlm_transcription(png, page["width_px"], page["height_px"])
    blocks, obs = [], []
    for i, ln in enumerate(lines):
        key = f"p{page['page_no']}:v{i}"
        obs.append({"document_id": document_id, "page_id": page["id"], "line_key": key,
                    "region_kind": "page", "bbox": ln.bbox, "engine": "qwen2.5-vl-page",
                    "engine_version": settings.vlm_model_id, "raw_text": ln.text,
                    "raw_confidence": None, "run_id": run_id})
        blocks.append({"page_id": page["id"], "block_type": "line", "text": ln.text,
                       "bbox": ln.bbox, "ocr_conf": _STATE_CONF_CAP[PAGE_LEVEL],
                       "lang": lang, "model_run_id": run_id, "_line_key": key,
                       "recognition": {"state": PAGE_LEVEL, "region_kind": "page",
                                       "fallback_reason": reason}})
    return blocks, obs


def recognize_document(document_id: str) -> RecognizeResult:
    """Run recognition v2 over every page and rebuild the ocr_block projection."""
    with session_scope() as sess:
        if not repo.get_document(sess, document_id):
            raise ValueError(f"document {document_id} not found")
        cls = repo.get_doc_classification(sess, document_id)
        pages = repo.list_document_pages(sess, document_id)
        if not pages:
            raise ValueError(f"document {document_id} has no rendered pages")
        run_id = repo.start_pipeline_run(
            sess, document_id=document_id, stage="ocr", model_name="recognition-v2",
            params={"qwen_line_mode": settings.qwen_line_mode,
                    "ocrhost": settings.ocrhost_url or "in-process"})
        rapid_obs = repo.list_current_observations(sess, document_id, engine="rapidocr")
    lang = cls["language"][0] if cls and cls.get("language") else None

    all_blocks: list[dict[str, Any]] = []
    all_obs: list[dict[str, Any]] = []
    fresh_rapid: list[tuple[dict[str, Any], list[OcrLine]]] = []
    try:
        for pg in pages:
            pobs = [o for o in rapid_obs if str(o["page_id"]) == str(pg["id"])]
            if pobs:
                lines = _lines_from_observations(pobs)
            else:
                lines = get_ocr_host().rapid(storage.get_bytes(storage.key_from_uri(pg["image_uri"])))
                fresh_rapid.append((pg, lines))
            try:
                blocks, obs = recognize_page(document_id=document_id, page=pg,
                                             rapid_lines=lines, run_id=run_id, lang=lang)
                if not blocks:
                    blocks, obs = _page_level_fallback(document_id, pg, run_id, lang,
                                                       "no lines detected")
            except Exception as exc:  # noqa: BLE001 - one bad page must not sink the doc
                log.error("recognize_page_failed", document_id=document_id,
                          page=pg["page_no"], error=str(exc)[:200])
                blocks, obs = _page_level_fallback(document_id, pg, run_id, lang,
                                                   f"recognition error: {exc}"[:200])
            all_blocks += blocks
            all_obs += obs

        with session_scope() as sess:
            for pg, lines in fresh_rapid:
                record_rapid_observations(sess, document_id=document_id, page=pg,
                                          lines=lines, run_id=run_id)
            # supersede this document's previous handwriting readings (append-only)
            prev = [str(o["id"]) for o in repo.list_current_observations(sess, document_id)
                    if o["engine"] != "rapidocr"]
            if all_obs:
                all_obs[0]["supersedes"] = prev
            obs_ids = repo.insert_observations(sess, all_obs)
            by_key: dict[str, list[str]] = {}
            for o, oid in zip(all_obs, obs_ids):
                by_key.setdefault(o["line_key"], []).append(oid)
            rapid_now = repo.list_current_observations(sess, document_id, engine="rapidocr")
            for order, b in enumerate(all_blocks, start=1):
                b["reading_order"] = order
                ids = list(by_key.get(b.pop("_line_key"), []))
                ids += [str(o["id"]) for o in rapid_now
                        if str(o["page_id"]) == str(b["page_id"])
                        and _inside(list(o["bbox"]), b["bbox"])]
                b["observation_ids"] = ids
            repo.delete_ocr_blocks_for_document(sess, document_id)
            n = repo.insert_ocr_blocks(sess, all_blocks)
            states: dict[str, int] = {}
            for b in all_blocks:
                s = (b.get("recognition") or {}).get("state", "?")
                states[s] = states.get(s, 0) + 1
            engines = {"qwen_line_mode": settings.qwen_line_mode, "ocrhost": settings.ocrhost_url or "in-process"}
            rate, drift = disagreement_rate(states), None
            if rate is not None:
                try:                       # a failed history query must never lose the document
                    with sess.begin_nested():
                        drift = drift_alarm(recent_rates(sess, run_id), rate)
                    if drift["alarm"]:
                        log.warning("disagreement_drift_alarm", document_id=document_id, **drift)
                except Exception as exc:  # noqa: BLE001
                    log.warning("drift_check_skipped", error=str(exc)[:160])
            repo.finish_pipeline_run(sess, run_id, status="ok",
                                     metrics={"blocks": n, "pages": len(pages),
                                              "observations": len(obs_ids), "states": states,
                                              "disagreement_rate": rate, "drift": drift})
            repo.set_document_status(sess, document_id, "ocr_done")
            repo.write_audit(sess, actor="recognition-v2", action="create", entity="ocr_block",
                             entity_id=document_id,
                             detail={"blocks": n, "states": states,
                                     "observations": len(obs_ids)})
        log.info("recognized", document_id=document_id, blocks=n, states=states)
        return RecognizeResult(document_id, n, len(pages), states, engines)
    except Exception as exc:
        log.error("recognition_failed", document_id=document_id, error=str(exc)[:300])
        with session_scope() as sess:
            repo.finish_pipeline_run(sess, run_id, status="failed", error_detail=str(exc)[:400])
            repo.set_document_status(sess, document_id, "error",
                                     error_detail=f"recognition: {exc}"[:400])
        raise


def _inside(inner: list[int], outer: list[int]) -> bool:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ix * iy / area >= 0.5
