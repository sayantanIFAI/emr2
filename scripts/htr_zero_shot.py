"""Zero-shot baseline of off-the-shelf recognizers on the 106 owner-labelled handwriting crops (D:\\Claude\\bake1), scored with the same helper as the
Qwen bake-off (cdi_adapter.recognition.scoring). No training, no tuning: one pass per engine.

    python eval_htr.py qwen                  # rescore the saved Qwen2.5-VL-7B predictions (bake1/out/qwen.jsonl)
    python eval_htr.py doctr crnn_vgg16_bn   # docTR pretrained recognizer
    python eval_htr.py paddle PP-OCRv5_server_rec
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdi_adapter.recognition import scoring as S  # noqa: E402

# The labelled crops are real patients' handwriting: they stay OUTSIDE the repository. EMR2_BAKE_DIR holds lines.jsonl (id, crop, truth), crops/ and out/qwen.jsonl.
BAKE = Path(os.environ.get("EMR2_BAKE_DIR", r"D:\Claude\bake1"))
OUT = Path(os.environ.get("EMR2_HTR_OUT", str(Path(__file__).parent / "out")))       # predictions are written here: not committed
OUT.mkdir(exist_ok=True)


def load():
    rows = [json.loads(x) for x in (BAKE / "lines.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def run_qwen(rows):
    got = {}
    for x in (BAKE / "out" / "qwen.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            d = json.loads(x)
            got[d["id"]] = d.get("text") or ""
    return [got.get(r["id"], "") for r in rows], None


def images(rows):
    import numpy as np
    from PIL import Image

    return [np.array(Image.open(BAKE / r["crop"]).convert("RGB")) for r in rows]


def run_doctr(rows, arch):
    from doctr.models import recognition_predictor

    pred = recognition_predictor(arch, pretrained=True)
    imgs = images(rows)
    t0 = time.time()
    out = []
    for im in imgs:                                   # one crop at a time: the per-line time is what the pipeline would pay
        res = pred([im])
        out.append(res[0][0] if res else "")
    return out, (time.time() - t0) / len(imgs)


def run_paddle(rows, name):
    from paddleocr import TextRecognition

    model = TextRecognition(model_name=name)
    t0 = time.time()
    out = []
    for r in rows:
        res = model.predict(input=str(BAKE / r["crop"]), batch_size=1)
        txt = ""
        for item in res:
            d = item.json if hasattr(item, "json") else item
            d = d.get("res", d) if isinstance(d, dict) else {}
            txt = str(d.get("rec_text") or "")
        out.append(txt)
    return out, (time.time() - t0) / len(rows)


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def report(label, truths, preds, sec):
    s = S.score(truths, preds)
    n = len(truths)
    ex_np = sum(alnum(t) == alnum(p) for t, p in zip(truths, preds))
    ch_edits = sum(S.levenshtein(alnum(t), alnum(p)) for t, p in zip(truths, preds))
    ch_len = sum(len(alnum(t)) for t in truths)
    nonempty = sum(1 for p in preds if p.strip())
    print()
    print(f"== {label}  ({n} lines, {nonempty} non-empty readings, {('%.2f s/line' % sec) if sec else 'time n/a'})")
    print(f"   exact (case and spaces folded, punctuation counts): {round(100 * s['exact_match'], 1)}%")
    print(f"   exact (letters and digits only):                    {round(100 * ex_np / n, 1)}%")
    print(f"   CER (scoring.score, as in the Qwen bake-off):        {s['cer']}")
    print(f"   CER (letters and digits only):                       {round(ch_edits / ch_len, 4)}")
    print(f"   WER: {s['wer']}   critical values (all numbers on a line right): {s['critical_value_accuracy']} over {s['lines_with_numbers']} lines")
    return s


def main():
    which = sys.argv[1]
    rows = load()
    truths = [r["truth"] for r in rows]
    if which == "qwen":
        preds, sec = run_qwen(rows)
        label = "Qwen2.5-VL-7B (saved predictions)"
    elif which == "doctr":
        preds, sec = run_doctr(rows, sys.argv[2])
        label = f"docTR {sys.argv[2]} pretrained, zero-shot"
    else:
        preds, sec = run_paddle(rows, sys.argv[2])
        label = f"PaddleOCR {sys.argv[2]}, zero-shot"
    tag = f"{which}_{sys.argv[2] if len(sys.argv) > 2 else ''}".strip("_")
    (OUT / f"{tag}.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "truth": r["truth"], "pred": p}, ensure_ascii=False) for r, p in zip(rows, preds)), encoding="utf-8")
    report(label, truths, preds, sec)


if __name__ == "__main__":
    main()
