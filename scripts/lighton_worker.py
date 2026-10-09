"""Read line crops with LightOnOCR-2-1B (EVALUATION ONLY; the model is a registered candidate, never a champion).

Runs in its own venv (transformers >= 5, see infra/runpod/install_lighton_eval.sh), apart from the Qwen stack:

    /workspace/lighton-venv/bin/python scripts/lighton_worker.py lines.jsonl out.jsonl

``lines.jsonl`` is the labelled set (``{"id", "crop", ...}``, crops relative to its folder). One output row per line:
``{"id", "text", "seconds", "mean_logprob", "min_logprob", "error"}``. The log-probabilities are the model's own
token probabilities of what it wrote (a trigger signal to be measured, not a calibrated confidence).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL = "lightonai/LightOnOCR-2-1B"
REVISION = "be7c41313c30e940fe5571de59e5fabc779e00c8"      # the exact revision in compliance/models.json


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    import torch
    from PIL import Image
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    src, dst = Path(argv[1]), Path(argv[2])
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = LightOnOcrForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(dev).eval()
    proc = LightOnOcrProcessor.from_pretrained(MODEL, revision=REVISION)
    max_new = int(os.environ.get("LIGHTON_MAX_NEW", "96"))
    with dst.open("w", encoding="utf-8") as out:
        for r in rows:
            rec = {"id": r["id"], "text": "", "seconds": None, "mean_logprob": None, "min_logprob": None, "error": None}
            try:
                img = Image.open(src.parent / r["crop"]).convert("RGB")
                conv = [{"role": "user", "content": [{"type": "image", "image": img}]}]
                inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                inp = {k: v.to(device=dev, dtype=dtype) if v.is_floating_point() else v.to(dev) for k, v in inp.items()}
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False, output_scores=True, return_dict_in_generate=True)
                if dev == "cuda":
                    torch.cuda.synchronize()
                rec["seconds"] = round(time.perf_counter() - t0, 3)
                ids = gen.sequences[0, inp["input_ids"].shape[1]:]
                rec["text"] = proc.decode(ids, skip_special_tokens=True).strip()
                lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(gen.scores, ids)]
                if lps:
                    rec["mean_logprob"] = round(sum(lps) / len(lps), 4)
                    rec["min_logprob"] = round(min(lps), 4)
            except Exception as exc:  # noqa: BLE001  (one bad crop must not stop the run; the error is written)
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    if dev == "cuda":
        print("peak GPU memory GB (MEASURED):", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
