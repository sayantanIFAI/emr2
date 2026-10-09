"""Fill the flat table (prescription_flat) from every document already read. Safe to repeat: each document's rows are replaced as a whole.

    python scripts/backfill_flat.py"""
from __future__ import annotations

import sys

from sqlalchemy import text

from cdi_adapter.db import session_scope
from cdi_adapter.output import flat_table


def main() -> int:
    flat_table.ensure_table()
    with session_scope() as s:
        ids = [str(r[0]) for r in s.execute(text("SELECT id FROM source_document ORDER BY ingested_at")).all()]
    done = rows = 0
    for did in ids:
        n = flat_table.save_document(did)
        done += 1 if n else 0
        rows += n
    print(f"{done} of {len(ids)} documents saved, {rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
