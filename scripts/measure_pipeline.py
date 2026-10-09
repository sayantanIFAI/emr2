"""Time the whole read chain of documents already in the database and list the lab tests each one gave, so a speed change can be judged on BOTH
time and recall. Mode ``prefetch`` starts the main page call right after classification (parallel with the line reading); mode ``normal`` does not.

    python scripts/measure_pipeline.py normal|prefetch OUT.json rx01.jpg rx02.jpg ...
The compact answer and the second-look skip follow the settings (CDI_EXTRACT_COMPACT_ANSWER, CDI_SKIP_SECOND_LOOK_WHEN_AGREE)."""
from __future__ import annotations

import json
import sys
import time

from sqlalchemy import text

from cdi_adapter.classify.service import classify_document
from cdi_adapter.config import settings
from cdi_adapter.db import session_scope
from cdi_adapter.extract.service import extract_document, prefetch_main_call
from cdi_adapter.ocr.service import ocr_document
from cdi_adapter.output.json_connector import get_connector
from cdi_adapter.terminology.service import bind_document
from cdi_adapter.validate.service import validate_document


def main(mode: str, out: str, names: list[str]) -> int:
    settings.prefetch_main_call = mode == "prefetch"
    res: dict[str, list] = {}
    times: dict[str, float] = {}
    for fn in names:
        with session_scope() as s:
            did = str(s.execute(text("SELECT id FROM source_document WHERE original_filename=:n ORDER BY ingested_at DESC LIMIT 1"), {"n": fn}).scalar())
        t0 = time.perf_counter()
        ocr_document(did, force_engine="rapidocr")
        c = classify_document(did)
        if settings.prefetch_main_call:
            prefetch_main_call(did)
        if settings.recognition_v2 or c.is_handwritten:
            ocr_document(did, force_engine="vlm")
        extract_document(did)
        bind_document(did)
        validate_document(did)
        r = get_connector().render(did)
        times[fn] = round(time.perf_counter() - t0, 2)
        tests = [{"as_written": t.get("as_written"), "standard_name": t.get("standard_name"), "status": t.get("status"),
                  "gate_recognised": t.get("gate_recognised"), "reason": t.get("reason"), "confidence": t.get("confidence")} for t in r.get("lab_tests") or []]
        res.setdefault(fn, []).append({"document_id": did, "doc_type": r.get("doc_type"), "lab_tests": tests, "seconds": times[fn]})
        print(fn, times[fn], "s", [t["standard_name"] or t["as_written"] for t in tests if t["status"] != "rejected"], flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"mode={mode}: {len(times)} documents, mean {sum(times.values()) / max(1, len(times)):.1f} s, total {sum(times.values()):.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
