"""Cut handwriting line crops out of real prescription photos and build the labelling page (recognition/labelset.py).

    pages     full photos, read through the pipeline on a pod (its own line detection and printed / handwritten labels):
                  python scripts/make_label_set.py pages  OUTDIR photo1.jpg photo2.jpg ...
    closeups  small close-up crops of handwriting, lines found with OpenCV only (runs anywhere, no model):
                  python scripts/make_label_set.py closeups OUTDIR crop1.png crop2.png ...

Both modes can write into the same OUTDIR (run ``pages`` first, then ``closeups``): the second run adds to the first."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cdi_adapter.recognition import labelset as L


def _stem(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in Path(name).stem)[:40]


def _load_existing(out: Path) -> tuple[list[L.LineItem], dict[str, bytes], dict[str, bytes], list[dict]]:
    items, crops, pages, excluded = [], {}, {}, []
    f = out / "lines.jsonl"
    if f.is_file():
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                items.append(L.LineItem(**json.loads(ln)))
        for it in items:
            crops[it.crop] = (out / it.crop).read_bytes()
        for p in (out / "pages").glob("*"):
            pages[p.name] = p.read_bytes()
    g = out / "excluded_phi.jsonl"
    if g.is_file():
        excluded = [json.loads(x) for x in g.read_text(encoding="utf-8").splitlines() if x.strip()]
    return items, crops, pages, excluded


def run_pages(out: Path, photos: list[str]) -> dict:
    from cdi_adapter import repo, storage
    from cdi_adapter.db import session_scope
    from cdi_adapter.webapp import jobs

    items, crops, pages, excluded = _load_existing(out)
    for path in photos:
        raw = Path(path).read_bytes()
        name = Path(path).name
        t0 = time.time()
        jid = jobs.create_job(None, [(name, raw + b"\x00" + str(time.time()).encode() + os.urandom(4))], token_no="LBL" + str(int(t0) % 100000),
                              phone="9000000007")
        while time.time() - t0 < 600:
            j = jobs.get_job(jid)
            if j and j.state in ("done", "review", "error", "mismatch"):
                break
            time.sleep(1)
        doc = j.docs[0].document_id
        with session_scope() as s:
            pgs = repo.list_document_pages(s, doc)
            blocks = repo.list_ocr_blocks(s, doc)
        pg = pgs[0]
        pre = pg.get("preproc") if isinstance(pg.get("preproc"), dict) else {}
        src = cv2.imdecode(np.frombuffer(storage.get_bytes(storage.key_from_uri(pre.get("src_uri") or pg["image_uri"])), np.uint8), cv2.IMREAD_COLOR)
        ref = cv2.imdecode(np.frombuffer(storage.get_bytes(storage.key_from_uri(pg["image_uri"])), np.uint8), cv2.IMREAD_COLOR)
        sx, sy = src.shape[1] / ref.shape[1], src.shape[0] / ref.shape[0]       # the boxes are in the normalised image's pixels
        pages[name] = raw
        for k, b in enumerate(blocks):
            try:
                bb = [int(float(v)) for v in b["bbox"][:4]]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if bb[2] - bb[0] < 12 or bb[3] - bb[1] < 6:
                continue
            reason = L.exclusion_reason(k, blocks)
            rec = b.get("recognition") or {}
            if reason:
                excluded.append({"page": name, "bbox": bb, "reason": reason, "machine": str(b.get("text") or "")})
                continue
            box = [int(bb[0] * sx), int(bb[1] * sy), int(bb[2] * sx), int(bb[3] * sy)]
            rel = f"crops/{_stem(name)}_{len(items):03d}.png"
            crops[rel] = L.crop_bytes(src, box)
            kind = str(rec.get("kind") or ("printed" if rec.get("state") == "printed" else "handwritten"))
            items.append(L.LineItem(id=f"{_stem(name)}_{len(items):03d}", page=name, kind=kind, bbox=box, crop=rel,
                                    machine=str(b.get("text") or ""), state=str(rec.get("state") or "")))
        print(f"{name}: {len(blocks)} lines read, set now {len(items)} lines, {len(excluded)} left out", flush=True)
    return L.write_set(out, items, crops, pages, excluded)


def run_closeups(out: Path, images: list[str]) -> dict:
    items, crops, pages, excluded = _load_existing(out)
    for path in images:
        raw = Path(path).read_bytes()
        name = Path(path).name
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{name}: not an image, skipped")
            continue
        pages[name] = raw
        boxes = L.closeup_boxes(img)
        for bb in boxes:
            rel = f"crops/{_stem(name)}_{len(items):03d}.png"
            crops[rel] = L.crop_bytes(img, bb)
            items.append(L.LineItem(id=f"{_stem(name)}_{len(items):03d}", page=name, kind="handwritten", bbox=bb, crop=rel))
        print(f"{name}: {len(boxes)} lines, set now {len(items)} lines", flush=True)
    return L.write_set(out, items, crops, pages, excluded)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[1] not in ("pages", "closeups"):
        print(__doc__)
        return 2
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary = (run_pages if argv[1] == "pages" else run_closeups)(out, argv[3:])
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
