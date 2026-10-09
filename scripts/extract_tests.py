"""List the lab tests the pipeline extracted for each photo already processed on this host (read from the database, nothing is re-run).

    python scripts/extract_tests.py OUT.json        (run on the pod, main venv, env loaded)"""
from __future__ import annotations

import json
import sys

import sqlalchemy as sa

from cdi_adapter.db import session_scope
from cdi_adapter.output.json_connector import get_connector


def main(out: str) -> int:
    with session_scope() as s:
        cols = [r[0] for r in s.execute(sa.text("select column_name from information_schema.columns where table_name='source_document'"))]
        name_col = next((c for c in ("filename", "original_filename", "file_name", "name") if c in cols), None)
        rows = s.execute(sa.text(f"select id, {name_col} as fn from source_document order by 1")).all() if name_col else []
    res: dict[str, list] = {}
    for did, fn in rows:
        try:
            r = get_connector().render(str(did))
        except Exception as exc:  # noqa: BLE001
            print("render failed", fn, str(exc)[:80])
            continue
        tests = [{"as_written": t.get("as_written") or t.get("text"), "standard_name": t.get("standard_name"),
                  "status": t.get("status"), "gate_recognised": t.get("gate_recognised"), "reason": t.get("reason")}
                 for t in (r.get("lab_tests") or [])]
        res.setdefault(str(fn), []).append({"document_id": str(did), "doc_type": r.get("doc_type"), "lab_tests": tests,
                                            "follow_up": r.get("follow_up")})
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for fn, docs in sorted(res.items()):
        for d in docs:
            print(fn, d["doc_type"], [t["standard_name"] or t["as_written"] for t in d["lab_tests"]])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
