"""Score one or more runs of ``scripts/parallel_pages.py`` against the page-level truth in ``tests/golden/batch13_gold.json``: for each page the tests FOUND,
MISSED and EXTRA (listed but not written). Both sides are put through the same gate and gazetteer (``lab_resolve``) so "Creat" and "Creatinine" are one test.
A test the system marked ``rejected`` is not a listed test.

    python scripts/score_batch.py batch13_after.json batch13_after2.json

The truth is one reader's reading of enlarged crops (see the file): treat a miss or an extra as a question to look at, not as a verdict."""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdi_adapter.extract import lab_resolve  # noqa: E402

GOLD = Path(__file__).resolve().parents[1] / "tests" / "golden" / "batch13_gold.json"
ALIASES = {"xray": "xray", "x-ray": "xray"}


def key(name: str | None) -> str:
    """One key per test: the standard name the lists give it, else its letters."""
    text = str(name or "").strip()
    r = None
    try:
        r = lab_resolve.resolve(text)
    except Exception:  # noqa: BLE001
        r = None
    base = (r.long_name if r and r.long_name else text)
    return re.sub(r"[^a-z0-9]", "", base.casefold())


def score(run: dict, gold: dict) -> tuple[list[dict], dict]:
    rows, tot = [], {"expected": 0, "found": 0, "extra": 0}
    for page in run["pages"]:
        pid = page["image"].split(".")[0]
        if pid not in gold:
            continue
        want = {key(t): t for t in gold[pid]["tests"]}
        got = {key(t.get("as_written") or t.get("standard")): t.get("as_written") for t in page.get("tests", []) if t.get("status") != "rejected"}
        def same(a: str, b: str) -> bool:
            """the same test: one key equals the other, or contains it (a longer reading of the same line), or they are 85% alike"""
            return a == b or (len(a) >= 4 and len(b) >= 4 and (a in b or b in a)) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

        found = [want[k] for k in want if any(same(k, g) for g in got)]
        missed = [want[k] for k in want if not any(same(k, g) for g in got)]
        extra = [got[g] for g in got if not any(same(k, g) for k in want)]
        rows.append({"page": pid, "found": found, "missed": missed, "extra": extra})
        tot["expected"] += len(want)
        tot["found"] += len(found)
        tot["extra"] += len(extra)
    return rows, tot


def main(paths: list[str]) -> int:
    gold = {k: v for k, v in json.loads(GOLD.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    for p in paths:
        run = json.loads(Path(p).read_text(encoding="utf-8"))
        rows, tot = score(run, gold)
        print(f"\n=== {p}: found {tot['found']} of {tot['expected']} written tests ({100 * tot['found'] / max(1, tot['expected']):.0f}%), {tot['extra']} extra (listed, not written)")
        for r in rows:
            print(f"  {r['page']}: found {len(r['found'])}/{len(r['found']) + len(r['missed'])}"
                  f"{'  MISSED ' + str(r['missed']) if r['missed'] else ''}{'  EXTRA ' + str(r['extra']) if r['extra'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
