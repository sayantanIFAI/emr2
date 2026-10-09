"""A labelled set of handwriting line crops: cut the lines out of real prescription photos, leave out the patient-name rows, and build a
labelling page where a person types the true text. The labelled file is the input of ``recognition/bench_htr.py``
(``{"id", "crop", "truth"}`` per line).

Nothing here reads handwriting. A line's machine reading (the pipeline's own) is kept in ``lines.jsonl`` for reference only and is hidden on
the page by default, so it does not steer the person typing the truth.

Names: the pipeline's reading of a name row is unreliable, so a line is left out when it is on the same row as a name label ("Patient",
"Name", "Mr", "Mrs" ...), when it holds an age/sex or a phone number, or when it is a name title. Those lines go to ``excluded_phi.jsonl``
with the reason; nothing is blanked silently."""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import regions

_NAME_LABEL = re.compile(r"(?i)\bpatient\b|\bname\b|\bnam e\b|\bmr\b\.?|\bmrs\b\.?|\bms\b\.?|\bsmt\b|\bshri\b|\bmaster\b|\bpt\.? name\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)|\bm\s*[:.]\s*\d{8,}|\bph(?:one)?\b\s*[:.]?\s*\d{6,}|\bmob(?:ile)?\b\s*[:.]?\s*\d{6,}", re.I)
_AGE_SEX = re.compile(r"(?i)\(\s*\d{1,3}\s*(?:y|yr|yrs|years?)?\s*/\s*(?:male|female|m|f)\s*\)|\bage\b\s*[:.\-]?\s*\d|\bsex\b\s*[:.\-]?\s*(?:m|f|male|female)\b")


@dataclass
class LineItem:
    id: str
    page: str                      # the photo's file name
    kind: str                      # printed | handwritten | mixed | uncertain (the pipeline's own label; closeups: handwritten)
    bbox: list[int]
    crop: str                      # path of the crop, relative to the set's folder
    machine: str = ""              # the pipeline's reading, for reference only
    state: str = ""                # the pipeline's evidence state (single_engine | no_reading | ...)
    truth: str = ""


def exclusion_reason(index: int, blocks: list[dict[str, Any]]) -> str | None:
    """Why block ``index`` must not go into the set (it is, or may be, the patient's name / phone / age), else ``None``."""
    b = blocks[index]
    text = str(b.get("text") or "")
    if _PHONE.search(text):
        return "phone number"
    if _AGE_SEX.search(text):
        return "age / sex"
    try:
        y0, y1 = float(b["bbox"][1]), float(b["bbox"][3])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    for j, other in enumerate(blocks):
        if j == index or not _NAME_LABEL.search(str(other.get("text") or "")):
            continue
        try:
            oy0, oy1 = float(other["bbox"][1]), float(other["bbox"][3])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        overlap = min(y1, oy1) - max(y0, oy0)
        if overlap >= 0.5 * min(y1 - y0, oy1 - oy0):
            return "on the same row as a name label"
    if _NAME_LABEL.search(text):
        return "name label / title"
    return None


def closeup_boxes(img_bgr: np.ndarray) -> list[list[int]]:
    """The text rows of a small close-up crop (OpenCV, CPU), top to bottom. The page line detector is made for whole pages and cuts a
    close-up into fragments, so rows are found from the ink itself: the ink is counted row by row, long rules and the page's edge lines are
    taken out first, the rows with ink make a run, runs a few pixels apart are one row, and each row is cut to where its ink is."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    block = max(15, (min(h, w) // 8) | 1)
    ink = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12) > 0
    ink = ink.astype(np.uint8)
    ink[:, ink.mean(axis=0) > 0.5] = 0                             # a vertical rule (the margin line of a form)
    ink[ink.mean(axis=1) > 0.6, :] = 0                             # a horizontal rule
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    profile = np.convolve(ink.sum(axis=1).astype(float), np.ones(max(3, h // 60)) / max(3, h // 60), mode="same")
    on = profile > max(2.0, 0.06 * profile.max())
    runs: list[list[int]] = []
    y = 0
    while y < h:
        if on[y]:
            y0 = y
            while y < h and on[y]:
                y += 1
            if runs and y0 - runs[-1][1] <= max(3, h // 40):
                runs[-1][1] = y
            else:
                runs.append([y0, y])
        else:
            y += 1
    out: list[list[int]] = []
    for y0, y1 in runs:
        if y1 - y0 < max(8, int(0.07 * h)):
            continue
        cols = np.where(ink[y0:y1].sum(axis=0) > 0)[0]
        if len(cols) < 0.05 * w or ink[y0:y1].sum() < 0.004 * h * w:
            continue
        out.append([int(cols[0]), y0, int(cols[-1]) + 1, y1])
    return out


def crop_bytes(src: np.ndarray, bbox: list[int]) -> bytes:
    return regions.crop_png(src, bbox, margin_frac=0.12, min_margin_px=4)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label the handwriting</title>
<style>
:root{--bg:#f6f7f9;--fg:#1b1f24;--muted:#5d6672;--card:#fff;--line:#d7dbe0;--ok:#1a7f4b;--warn:#a15c00;--acc:#1f5fd6}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e8eaed;--muted:#9aa3ad;--card:#1d2127;--line:#343a42;--ok:#4cc38a;--warn:#e0a24a;--acc:#6ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header b{font-size:17px} .bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok);width:0}
button{font:inherit;padding:7px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}button.p{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:980px;margin:0 auto;padding:12px 16px 80px}
h2{margin:26px 0 6px;font-size:18px}.pg{max-width:100%;max-height:340px;border:1px solid var(--line);border-radius:8px}
.hint{color:var(--muted);font-size:14px}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0;display:grid;grid-template-columns:minmax(0,1fr);gap:8px}
.row.done{border-color:var(--ok)} .row img{max-width:100%;max-height:160px;object-fit:contain;background:#fff;border-radius:6px;border:1px solid var(--line)}
.row input[type=text]{font:inherit;width:100%;box-sizing:border-box;padding:9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.opts{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;color:var(--muted)} .m{font-size:13px;color:var(--muted)}
details>summary{cursor:pointer;color:var(--muted)}
</style></head><body>
<header><b>Label the handwriting</b><span id="cnt" class="hint"></span><div class="bar"><i id="bar"></i></div>
<button class="p" id="dl">Download lines.jsonl</button><button id="dlp">Save progress</button><label class="hint">Load progress <input type="file" id="ld" accept=".json"></label></header>
<main>
<p class="hint">Type exactly what is written, letter by letter, as you read it. Do not correct spelling. Tick <b>can't read</b> if you cannot tell, <b>has a name</b> if the crop shows a person's name (it is left out), and <b>bad crop</b> if the line is cut wrongly (part of it missing, or two lines in one). Your work is saved in this browser as you type. When done, press <b>Download lines.jsonl</b> and give me that file.</p>
__BODY__
</main>
<script>
const ITEMS=__ITEMS__;
const KEY="labelset:"+__KEY__;
let S={}; try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){} upd()}
function upd(){const n=ITEMS.length;let d=0;for(const it of ITEMS){const s=S[it.id]||{};if((s.t&&s.t.trim())||s.u||s.n||s.b)d++}
 document.getElementById("cnt").textContent=d+" of "+n+" done";document.getElementById("bar").style.width=(100*d/Math.max(n,1))+"%";
 for(const it of ITEMS){const s=S[it.id]||{};const el=document.getElementById("r-"+it.id);if(el)el.classList.toggle("done",!!((s.t&&s.t.trim())||s.u||s.n||s.b))}}
for(const it of ITEMS){const s=S[it.id]||{};
 const t=document.getElementById("t-"+it.id),u=document.getElementById("u-"+it.id),n=document.getElementById("n-"+it.id),bd=document.getElementById("b-"+it.id);
 t.value=s.t||"";u.checked=!!s.u;n.checked=!!s.n;bd.checked=!!s.b;
 const set=()=>{S[it.id]={t:t.value,u:u.checked,n:n.checked,b:bd.checked};save()};
 t.addEventListener("input",set);u.addEventListener("change",set);n.addEventListener("change",set);bd.addEventListener("change",set)}
function dl(name,text){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type:"application/json"}));a.download=name;a.click()}
document.getElementById("dl").onclick=()=>{const out=[];for(const it of ITEMS){const s=S[it.id]||{};if(s.n||s.u||s.b||!(s.t&&s.t.trim()))continue;out.push(JSON.stringify({id:it.id,crop:it.crop,truth:s.t.trim()}))}
 dl("lines.jsonl",out.join("\\n")+"\\n");alert(out.length+" labelled lines downloaded.")};
document.getElementById("dlp").onclick=()=>dl("labelling-progress.json",JSON.stringify(S));
document.getElementById("ld").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(x=>{try{S=JSON.parse(x);save();location.reload()}catch(err){alert("not a progress file")}})};
upd();
</script></body></html>
"""


def build_html(items: list[LineItem], pages: list[str], key: str) -> str:
    """The self-contained labelling page (crops are read from ``crops/`` next to it). The pipeline's reading is behind a closed <details>."""
    body: list[str] = []
    by_page: dict[str, list[LineItem]] = {}
    for it in items:
        by_page.setdefault(it.page, []).append(it)
    order = {"handwritten": 0, "mixed": 1, "uncertain": 2, "printed": 3}
    for page in pages:
        group = sorted(by_page.get(page, []), key=lambda i: (order.get(i.kind, 4), i.bbox[1], i.bbox[0]))
        if not group:
            continue
        body.append(f'<h2>{html.escape(page)}</h2><p class="hint">{len(group)} lines. The whole photo, for context:</p>'
                    f'<a href="pages/{html.escape(page)}" target="_blank"><img class="pg" src="pages/{html.escape(page)}" alt="the whole photo"></a>')
        for it in group:
            kind = "printed (the reader's label; type it only if it is handwritten)" if it.kind == "printed" else it.kind
            body.append(
                f'<div class="row" id="r-{html.escape(it.id)}"><img src="{html.escape(it.crop)}" alt="line crop" loading="lazy">'
                f'<input type="text" id="t-{html.escape(it.id)}" placeholder="what is written here" autocomplete="off" spellcheck="false">'
                f'<div class="opts"><label><input type="checkbox" id="u-{html.escape(it.id)}"> can\'t read</label>'
                f'<label><input type="checkbox" id="n-{html.escape(it.id)}"> has a name (leave out)</label>'
                f'<label><input type="checkbox" id="b-{html.escape(it.id)}"> bad crop (cut wrongly)</label><span class="m">{html.escape(kind)}</span></div>'
                f'<details><summary>machine reading (optional, hidden so it does not steer you)</summary><span class="m">{html.escape(it.machine or "(none)")}</span></details></div>')
    items_js = json.dumps([{"id": i.id, "crop": i.crop} for i in items], ensure_ascii=False)
    return _PAGE.replace("__BODY__", "\n".join(body)).replace("__ITEMS__", items_js).replace("__KEY__", json.dumps(key))


def write_set(out_dir: Path, items: list[LineItem], crops: dict[str, bytes], page_images: dict[str, bytes],
              excluded: list[dict[str, Any]]) -> dict[str, int]:
    """Write the folder: ``crops/``, ``pages/``, ``lines.jsonl``, ``excluded_phi.jsonl``, ``labelling.html``, ``README.txt``."""
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    for path, data in crops.items():
        (out_dir / path).write_bytes(data)
    for name, data in page_images.items():
        (out_dir / "pages" / name).write_bytes(data)
    (out_dir / "lines.jsonl").write_text("".join(json.dumps(asdict(i), ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    (out_dir / "excluded_phi.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in excluded), encoding="utf-8")
    (out_dir / "labelling.html").write_text(build_html(items, list(page_images), key=str(len(items)) + ":" + ",".join(sorted(page_images))),
                                            encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "Open labelling.html in a browser (double-click it). Type the true text for each crop. Press 'Download lines.jsonl' and give that file back.\n"
        "lines.jsonl here is the list of crops (truth empty). excluded_phi.jsonl lists the lines left out (name rows, phone, age/sex) and why.\n"
        "The photos in pages/ are real patients' prescriptions: keep this folder private.\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for i in items:
        kinds[i.kind] = kinds.get(i.kind, 0) + 1
    return {"lines": len(items), "excluded": len(excluded), **{f"kind_{k}": v for k, v in kinds.items()}}
