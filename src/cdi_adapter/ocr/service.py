from __future__ import annotations

from dataclasses import dataclass

from .. import repo, storage
from ..config import settings
from ..db import session_scope
from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class OcrResult:
    document_id: str
    engine: str
    n_blocks: int
    n_pages: int


def ocr_document(document_id: str, *, force_engine: str | None = None) -> OcrResult:
    """OCR every page and persist the blocks (replacing any previous ones).

    With ``recognition_v2`` (default) any pass other than an explicit ``rapidocr`` one is
    delegated to :func:`cdi_adapter.recognition.pipeline.recognize_document`.

    ``force_engine`` pins the engine ("rapidocr" | "vlm"); without it the engine
    is chosen from the classification (VLM for handwriting). The pipeline runs a
    fast ``rapidocr`` pass before classification and, only for a page the
    classifier calls handwritten, a second ``vlm`` pass that supersedes it.
    """
    with session_scope() as sess:
        doc = repo.get_document(sess, document_id)
        if not doc:
            raise ValueError(f"document {document_id} not found")
        cls = repo.get_doc_classification(sess, document_id)
        pages = repo.list_document_pages(sess, document_id)
        if not pages:
            raise ValueError(f"document {document_id} has no rendered pages")
        run_id = repo.start_pipeline_run(
            sess, document_id=document_id, stage="ocr",
            model_name="rapidocr+vlm",
            params={"handwritten": bool(cls and cls["is_handwritten"]),
                    "forced": force_engine},
        )

    if force_engine == "vlm":
        use_vlm = True
    elif force_engine == "rapidocr":
        use_vlm = False
    else:
        use_vlm = bool(cls and cls["is_handwritten"]) and settings.handwritten_uses_vlm
    engine = "vlm" if use_vlm else settings.ocr_engine

    # v2 always runs the region pass (a printed form can carry handwritten lines); only an
    # explicit rapidocr pass - the fast text the classifier reads - stays printed-only
    if settings.recognition_v2 and force_engine != "rapidocr":
        # recognition v2: line regions -> Qwen per crop -> immutable evidence
        from ..recognition.pipeline import recognize_document

        with session_scope() as sess:
            repo.finish_pipeline_run(sess, run_id, status="ok",
                                     metrics={"delegated": "recognition-v2"})
        rr = recognize_document(document_id)
        _enqueue_extract(document_id)
        return OcrResult(document_id, "recognition-v2", rr.n_blocks, rr.n_pages)

    all_blocks: list[dict] = []
    page_lines: list[tuple[dict, list]] = []
    order = 0
    try:
        for pg in pages:
            png = storage.get_bytes(storage.key_from_uri(pg["image_uri"]))
            if use_vlm:
                from .vlm_ocr import run_vlm_transcription

                lines = run_vlm_transcription(png, pg["width_px"], pg["height_px"])
                block_type = "line"
            elif settings.ocr_engine == "rapidocr":
                from .rapid import run_rapidocr

                lines = run_rapidocr(png)
                block_type = "line"
                page_lines.append((pg, lines))
            else:
                lines = []
                block_type = "line"

            for ln in lines:
                order += 1
                all_blocks.append(
                    {
                        "page_id": pg["id"],
                        "block_type": block_type,
                        "reading_order": order,
                        "text": ln.text,
                        "bbox": ln.bbox,
                        "polygon": ln.polygon or None,
                        "ocr_conf": ln.conf,
                        "lang": (cls["language"][0] if cls and cls.get("language") else None),
                        "model_run_id": run_id,
                    }
                )

        with session_scope() as sess:
            if settings.recognition_v2 and page_lines:
                # the printed pass is evidence too (append-only ocr_observation)
                from ..recognition.pipeline import record_rapid_observations

                for pg, lines in page_lines:
                    record_rapid_observations(sess, document_id=document_id, page=pg,
                                              lines=lines, run_id=run_id)
            repo.delete_ocr_blocks_for_document(sess, document_id)
            n = repo.insert_ocr_blocks(sess, all_blocks)
            repo.finish_pipeline_run(
                sess, run_id, status="ok",
                metrics={"engine": engine, "blocks": n, "pages": len(pages)},
            )
            repo.set_document_status(sess, document_id, "ocr_done")
            repo.write_audit(
                sess, actor="ocr-svc", action="create", entity="ocr_block",
                entity_id=document_id, detail={"engine": engine, "blocks": n},
            )
        log.info("ocr_done", document_id=document_id, engine=engine, blocks=len(all_blocks))
        _enqueue_extract(document_id)
        return OcrResult(document_id, engine, len(all_blocks), len(pages))

    except Exception as exc:  # noqa: BLE001
        log.error("ocr_failed", document_id=document_id, error=str(exc)[:300])
        with session_scope() as sess:
            repo.finish_pipeline_run(sess, run_id, status="failed", error_detail=str(exc)[:400])
            repo.set_document_status(sess, document_id, "error", error_detail=f"ocr: {exc}")
        raise


def _enqueue_extract(document_id: str) -> None:
    try:
        from ..worker import extract_document

        extract_document.delay(document_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("enqueue_extract_skipped", document_id=document_id, error=str(exc))
