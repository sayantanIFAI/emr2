"""Four controlled Qwen tests on the owner's labelled crops. Everything goes to vLLM directly (temperature 0, same base prompt), so a
difference between two variants is the variant and nothing else.

  1  raw crop            vs  enlarged crop            vs  enlarged + contrast-normalised crop
  2  crop only           vs  crop + surrounding context (the page around the line, the line boxed in red)
  3  crop only           vs  crop + verified exemplars of the SAME doctor (leave-one-out)
  4  crop only           vs  doctor's words + closed candidate list for the field type (drug / lab test, found from the blind reading)

Run on the pod (main venv, vLLM on :8078):
    python scripts/bake_off_variants.py SET/lines.jsonl /workspace/labelset/lines.jsonl /workspace/labelset/pages SET/variants.json

``SET/lines.jsonl`` = the owner's labelled file; ``/workspace/labelset/lines.jsonl`` = the crop list with page and box (for test 2)."""
from __future__ import annotations

import base64
import difflib
import io
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_off import _alnum, _group, _rows, _setup, doctor_vocab, vocab_for  # noqa: E402
from cdi_adapter.recognition.bench_htr import _norm  # noqa: E402
from cdi_adapter.recognition.engines import QWEN_LINE_PROMPT, clean_line  # noqa: E402

URL = "http://127.0.0.1:8078/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SESSION = requests.Session()
SESSION.trust_env = False


def _b64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def ask(images: list[bytes], prompt: str) -> str:
    content = [{"type": "image_url", "image_url": {"url": _b64(i)}} for i in images] + [{"type": "text", "text": prompt}]
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 64, "temperature": 0}
    r = SESSION.post(URL, json=body, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"] or ""
    return clean_line(next((x.strip() for x in txt.splitlines() if x.strip()), ""))


def run_all(jobs: list[tuple[list[bytes], str]], workers: int = 8) -> list[str]:
    def one(j: tuple[list[bytes], str]) -> str:
        try:
            return ask(*j)
        except Exception as exc:  # noqa: BLE001
            return "[error] " + str(exc)[:80]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, jobs))


def png_of(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ------------------------------------------------------------------ test 1: enlarge / normalise
def enlarge(png: bytes, target_h: int = 160, max_w: int = 1800) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(max(1.0, target_h / h), max_w / w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4) if s > 1.01 else img


def normalise(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lo, hi = np.percentile(l, (1, 99))
    l = np.clip((l.astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(out, 1.4, cv2.GaussianBlur(out, (0, 0), 1.2), -0.4, 0)


# ------------------------------------------------------------------ test 2: context
CONTEXT_PROMPT = ("This is a piece of a medical prescription with ONE line marked by a red rectangle. Transcribe exactly the text written "
                  "INSIDE the red rectangle only (the other text is context). Copy letters, numbers, units and dosing notation exactly as they "
                  "appear, do not correct spelling or expand abbreviations. Write an unreadable character as ?. Output only the transcription "
                  "on one line.")


def context_image(page: np.ndarray, bbox: list[int]) -> bytes:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    h = max(1, y1 - y0)
    cx0, cy0 = max(0, x0 - int(1.2 * h)), max(0, y0 - int(2.5 * h))
    cx1, cy1 = min(page.shape[1], x1 + int(1.2 * h)), min(page.shape[0], y1 + int(2.5 * h))
    win = np.ascontiguousarray(page[cy0:cy1, cx0:cx1], dtype=np.uint8)
    cv2.rectangle(win, (x0 - cx0 - 2, y0 - cy0 - 2), (x1 - cx0 + 2, y1 - cy0 + 2), (0, 0, 255), 2)
    return png_of(enlarge(png_of(win), 220, 2000))


# ------------------------------------------------------------------ test 3: exemplars
EXEMPLAR_PROMPT = ("Image 1 is one line from a medical prescription (maybe handwritten): transcribe it exactly, keeping the doctor's own "
                   "spelling. Image 2 shows other lines by the SAME doctor, each with its verified transcription written under it: use them to learn "
                   "this doctor's letter shapes and habitual words, but write only what Image 1 shows. Write an unreadable character as ?. "
                   "Output only the transcription of Image 1 on one line.")


def montage(examples: list[tuple[bytes, str]]) -> bytes:
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    tiles = []
    for png, truth in examples:
        im = Image.open(io.BytesIO(png)).convert("RGB")
        s = min(1.0, 520 / im.width, 90 / im.height) if im.width > 520 or im.height > 90 else 1.0
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        tile = Image.new("RGB", (540, im.height + 30), "white")
        tile.paste(im, (6, 2))
        ImageDraw.Draw(tile).text((8, im.height + 6), "= " + truth[:60], fill=(160, 0, 0), font=font)
        tiles.append(tile)
    out = Image.new("RGB", (540, sum(t.height + 6 for t in tiles)), (235, 235, 235))
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height + 6
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ test 4: doctor words + closed candidate list
DRUG_FORM = re.compile(r"^\W*(?:tab|cap|inj|syp|syr|susp|oint|cream|gel|drops?|neb|inh|t|c)\b[\s.:\-]*", re.I)
CAND_PROMPT = ("This is one line from a medical prescription and may be handwritten. {mem}{cands}Use a word from these lists ONLY if the image "
               "really shows it; otherwise write exactly what you see, keeping the doctor's own spelling. Write an unreadable character as ?. "
               "Output only the transcription on one line.")


def drug_candidates(word: str, k: int = 8) -> list[str]:
    from cdi_adapter.extract import indian_codes as ic

    idx = ic.drugs("/workspace/data")
    if idx is None or len(word) < 3:
        return []
    pool = getattr(drug_candidates, "_pool", None)
    if pool is None:
        pool = sorted({w for ws in idx.buckets.values() for w in ws} | set(idx.brands))
        drug_candidates._pool = pool  # type: ignore[attr-defined]
    w = word.casefold()
    sm = difflib.SequenceMatcher(None, "", w)
    scored = []
    for cand in pool:
        if abs(len(cand) - len(w)) > 2:
            continue
        sm.set_seq1(cand)
        if sm.real_quick_ratio() < 0.6 or sm.quick_ratio() < 0.6:
            continue
        r = sm.ratio()
        if r >= 0.6:
            scored.append((r, cand))
    scored.sort(reverse=True)
    return [c for _r, c in scored[:k]]


def field_and_candidates(blind: str) -> tuple[str, list[str]]:
    from cdi_adapter.extract import indian_codes as ic

    m = DRUG_FORM.match(blind or "")
    if m:
        words = re.findall(r"[A-Za-z]{3,}", blind[m.end():])
        return "drug", drug_candidates(words[0]) if words else []
    labs = ic.suggest_labs(blind, k=8, floor=0.6, path="/workspace/data")
    return ("lab test", labs) if labs else ("other", [])


# ------------------------------------------------------------------ run
def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    lines_p, crops_p, pages_dir, out_p = (Path(a) for a in argv[1:5])
    rows = _rows(lines_p)
    meta = {r["id"]: r for r in _rows(crops_p)}
    crops = {r["id"]: (lines_p.parent / r["crop"]).read_bytes() for r in rows}
    ids = [r["id"] for r in rows]
    T = [r["truth"] for r in rows]
    pages: dict[str, np.ndarray] = {}
    vocab = doctor_vocab(rows)
    rng = random.Random(7)
    t0 = time.perf_counter()

    v0 = run_all([([crops[i]], QWEN_LINE_PROMPT) for i in ids])
    v1a = run_all([([png_of(enlarge(crops[i]))], QWEN_LINE_PROMPT) for i in ids])
    v1b = run_all([([png_of(normalise(enlarge(crops[i])))], QWEN_LINE_PROMPT) for i in ids])

    ctx_jobs, ctx_ok = [], []
    for i in ids:
        m = meta.get(i)
        pf = pages_dir / m["page"] if m else None
        if m and pf and pf.is_file():
            pg = pages.setdefault(m["page"], cv2.imdecode(np.frombuffer(pf.read_bytes(), np.uint8), cv2.IMREAD_COLOR))
            try:
                ctx_jobs.append(([context_image(pg, m["bbox"])], CONTEXT_PROMPT))
                ctx_ok.append(True)
            except cv2.error:
                ctx_jobs.append(([crops[i]], QWEN_LINE_PROMPT))
                ctx_ok.append(False)
        else:
            ctx_jobs.append(([crops[i]], QWEN_LINE_PROMPT))
            ctx_ok.append(False)
    v2 = run_all(ctx_jobs)

    ex_jobs, n_ex = [], []
    by_doc: dict[str, list[str]] = {}
    for i in ids:
        by_doc.setdefault(_group(i), []).append(i)
    truth_of = dict(zip(ids, T))
    for i in ids:
        others = [j for j in by_doc[_group(i)] if j != i and len(_alnum(truth_of[j])) >= 3]
        rng.shuffle(others)
        pick = sorted(others[:4])
        n_ex.append(len(pick))
        if pick:
            ex_jobs.append(([crops[i], montage([(crops[j], truth_of[j]) for j in pick])], EXEMPLAR_PROMPT))
        else:
            ex_jobs.append(([crops[i]], QWEN_LINE_PROMPT))
    v3 = run_all(ex_jobs)

    kinds, cands, v4_jobs = [], [], []
    for i, blind in zip(ids, v0):
        kind, c = field_and_candidates(blind)
        words = vocab_for(i, vocab)
        kinds.append(kind)
        cands.append(c)
        mem = f"The same doctor has written these words before: {', '.join(words)[:700]}. " if words else ""
        cl = f"Possible {kind} names from the reference list: {', '.join(c)}. " if c else ""
        v4_jobs.append(([crops[i]], CAND_PROMPT.format(mem=mem, cands=cl)))
    v4 = run_all(v4_jobs)

    base = {"0_raw": v0, "1a_enlarged": v1a, "1b_enlarged_contrast": v1b, "2_with_context": v2, "3_same_doctor_exemplars": v3,
            "4_doctor_words_plus_closed_list": v4}
    report = {"lines": len(ids), "status": "MEASURED on the owner's labelled crops, Qwen2.5-VL-7B temperature 0, same base prompt; owner sets margins",
              "seconds_total": round(time.perf_counter() - t0, 1), "variants": {}}
    for name, preds in base.items():
        s = _setup(name, T, preds)
        s["fixed_vs_raw"] = sum(_norm(t) != _norm(a) and _norm(t) == _norm(p) for t, a, p in zip(T, v0, preds))
        s["broke_vs_raw"] = sum(_norm(t) == _norm(a) and _norm(t) != _norm(p) for t, a, p in zip(T, v0, preds))
        s["fixed_vs_raw_ignoring_punctuation"] = sum(_alnum(t) != _alnum(a) and _alnum(t) == _alnum(p) for t, a, p in zip(T, v0, preds))
        s["broke_vs_raw_ignoring_punctuation"] = sum(_alnum(t) == _alnum(a) and _alnum(t) != _alnum(p) for t, a, p in zip(T, v0, preds))
        report["variants"][name] = s
    report["notes"] = {
        "context_lines_with_page_window": sum(ctx_ok),
        "exemplar_lines_with_examples": sum(1 for n in n_ex if n), "mean_examples": round(sum(n_ex) / max(1, len(n_ex)), 2),
        "field_types_found_from_blind_reading": {k: kinds.count(k) for k in sorted(set(kinds))},
        "candidate_list_contained_a_true_word": sum(1 for t, c in zip(T, cands) if c and any(w.casefold() in _norm(t) for w in c)),
        "lines_given_a_candidate_list": sum(1 for c in cands if c),
    }
    report["per_line"] = [{"id": i, "truth": t, **{n: p[k] for n, p in base.items()}} for k, (i, t) in enumerate(zip(ids, T))]
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ("cer", "wer", "exact_match", "exact_ignoring_punctuation_and_spacing", "critical_value_accuracy", "fixed_vs_raw", "broke_vs_raw")
    for n, s in report["variants"].items():
        print(n, {k: s.get(k) for k in keys})
    print(json.dumps(report["notes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
