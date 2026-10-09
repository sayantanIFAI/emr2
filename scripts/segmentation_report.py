"""How often does the line cutter cut a line wrongly? From the owner's labelling progress file ("Save progress" on the labelling page).

    python scripts/segmentation_report.py SET/lines.jsonl labelling-progress.json

Prints, per photo and overall: crops looked at, the share ticked "bad crop" (cut wrongly or printed), "can't read", "has a name", and the share labelled,
with the median crop height and aspect of the bad ones against the good ones, so the cutter can be tuned where it fails."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main(lines_p: str, progress_p: str) -> int:
    base = Path(lines_p).parent
    items = [json.loads(x) for x in Path(lines_p).read_text(encoding="utf-8").splitlines() if x.strip()]
    prog = json.loads(Path(progress_p).read_text(encoding="utf-8"))
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    geo: dict[str, list[tuple[int, float]]] = {"bad": [], "good": []}
    for it in items:
        s = prog.get(it["id"]) or {}
        looked = bool((s.get("t") or "").strip() or s.get("u") or s.get("n") or s.get("b"))
        if not looked:
            continue
        pg = it.get("page", "?")
        per[pg]["looked"] += 1
        for k, name in (("b", "bad_crop"), ("u", "cant_read"), ("n", "has_name")):
            if s.get(k):
                per[pg][name] += 1
        if (s.get("t") or "").strip() and not (s.get("b") or s.get("u") or s.get("n")):
            per[pg]["labelled"] += 1
        try:
            w, h = Image.open(base / it["crop"]).size
            geo["bad" if s.get("b") else "good"].append((h, w / max(1, h)))
        except OSError:
            pass
    tot: dict[str, int] = defaultdict(int)
    for pg, d in sorted(per.items()):
        print(f"{pg:12} looked={d['looked']:3} bad_crop={d['bad_crop']:3} ({d['bad_crop'] / d['looked']:.0%}) cant_read={d['cant_read']:3} has_name={d['has_name']:3} labelled={d['labelled']:3}")
        for k, v in d.items():
            tot[k] += v
    if tot["looked"]:
        print(f"ALL          looked={tot['looked']} bad_crop={tot['bad_crop']} ({tot['bad_crop'] / tot['looked']:.0%}) cant_read={tot['cant_read']} labelled={tot['labelled']}")
    for k, v in geo.items():
        if v:
            print(f"{k}: n={len(v)} median height {statistics.median(a for a, _ in v):.0f}px, median width/height {statistics.median(b for _, b in v):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
