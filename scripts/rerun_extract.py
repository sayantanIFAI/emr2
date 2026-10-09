"""Run ONLY the extraction, binding and validation stages again on documents already read (their OCR / handwriting readings are kept), then write
each document's lab tests to a JSON file. For measuring a change to the extraction rules without re-reading the page.

    python scripts/rerun_extract.py OUT.json rx01.jpg rx02.jpg 29.jpg ..."""
from __future__ import annotations

import json
import sys

from sqlalchemy import text

from cdi_adapter.db import session_scope
from cdi_adapter.extract.service import extract_document
from cdi_adapter.output.json_connector import get_connector
from cdi_adapter.terminology.service import bind_document
from cdi_adapter.validate.service import validate_document


def main(out: str, names: list[str]) -> int:
    with session_scope() as s:
        ids = [(str(r[0]), r[1]) for n in names for r in s.execute(text(
            "SELECT id, original_filename FROM source_document WHERE original_filename = :n ORDER BY ingested_at DESC LIMIT 1"),
            {"n": n}).all()]
    res: dict[str, list] = {}
    for did, fn in ids:
        try:
            extract_document(did)
            bind_document(did)
            validate_document(did)
            r = get_connector().render(did)
        except Exception as exc:  # noqa: BLE001
            print("FAILED", fn, str(exc)[:200])
            continue
        tests = [{"as_written": t.get("as_written"), "standard_name": t.get("standard_name"), "status": t.get("status"),
                  "gate_recognised": t.get("gate_recognised"), "reason": t.get("reason"), "confidence": t.get("confidence")}
                 for t in (r.get("lab_tests") or [])]
        res.setdefault(fn, []).append({"document_id": did, "doc_type": r.get("doc_type"), "lab_tests": tests, "follow_up": r.get("follow_up")})
        print(fn, r.get("doc_type"), [t["standard_name"] or t["as_written"] for t in tests if t["status"] != "rejected"])
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
