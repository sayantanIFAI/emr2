"""Terminology-assisted candidate ranking on the 106 labelled lines: words not in the lexicon are snapped to the closest lexicon word.

The lexicon is the national lists only (drug brand words, lab-test words and aliases): a dictionary is a candidate generator, not ground truth.
For every strictness level it counts the lines fixed, the lines made worse, and the words that were already right and were changed.
    python lexicon_snap.py <codes dir>
"""
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdi_adapter.recognition import scoring as S  # noqa: E402

CODES = Path(sys.argv[1])
import os

BAKE = Path(os.environ.get("EMR2_BAKE_DIR", r"D:\Claude\bake1"))               # real patients' crops: outside the repository
OUT = Path(os.environ.get("EMR2_HTR_OUT", str(Path(__file__).parent / "out")))   # holds paddle_PP-OCRv5_server_rec.jsonl from htr_zero_shot.py

L = json.loads((CODES / "clci_labs.json").read_text(encoding="utf-8"))["tests"]
D = json.loads((CODES / "cdci_drugs.json").read_text(encoding="utf-8"))
words: set[str] = set()
for name in D["brands"]:
    words.update(w for w in re.findall(r"[a-z]+", name) if len(w) >= 4)
for g in D["generic_names"] if not isinstance(D["generic_names"], dict) else D["generic_names"].values():
    words.update(w for w in re.findall(r"[a-z]+", str(g).lower()) if len(w) >= 4)
for t in L:
    for n in [t["name"], *t.get("aliases", [])]:
        words.update(w for w in re.findall(r"[a-z]+", n.lower()) if len(w) >= 4)
vocab = sorted(words)
print("lexicon words:", len(vocab))

al = lambda s: re.sub(r"[^a-z0-9]", "", s.casefold())  # noqa: E731
rows = [json.loads(x) for x in (BAKE / "lines.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
truths = [r["truth"] for r in rows]


def snap(text: str, cutoff: float) -> str:
    def one(m):
        w = m.group(0)
        if len(w) < 4 or w.lower() in words:
            return w
        hit = process.extractOne(w.lower(), vocab, scorer=fuzz.ratio, score_cutoff=cutoff)
        if not hit:
            return w
        c = hit[0]
        return c.upper() if w.isupper() else c.capitalize() if w[0].isupper() else c
    return re.sub(r"[A-Za-z]+", one, text)


def evaluate(label: str, preds: list[str]) -> None:
    base_ex = sum(al(t) == al(p) for t, p in zip(truths, preds))
    base_cer = sum(S.levenshtein(al(t), al(p)) for t, p in zip(truths, preds)) / sum(len(al(t)) for t in truths)
    print(f"\n== {label}: no snapping -> exact {base_ex}/106, CER(letters+digits) {base_cer:.3f}")
    print("   cutoff | changed lines | fixed (wrong->right) | harmed (right->wrong) | right words changed | exact | CER")
    for cutoff in (92, 88, 84, 80, 75):
        new = [snap(p, cutoff) for p in preds]
        changed = [i for i, (a, b) in enumerate(zip(preds, new)) if a != b]
        fixed = sum(1 for i in changed if al(truths[i]) != al(preds[i]) and al(truths[i]) == al(new[i]))
        harmed = sum(1 for i in changed if al(truths[i]) == al(preds[i]) and al(truths[i]) != al(new[i]))
        bad_words = 0
        for i in changed:
            tw = set(re.findall(r"[a-z]+", truths[i].lower()))
            for a, b in zip(re.findall(r"[A-Za-z]+", preds[i]), re.findall(r"[A-Za-z]+", new[i])):
                if a != b and a.lower() in tw:
                    bad_words += 1
        ex = sum(al(t) == al(p) for t, p in zip(truths, new))
        cer = sum(S.levenshtein(al(t), al(p)) for t, p in zip(truths, new)) / sum(len(al(t)) for t in truths)
        print(f"   {cutoff:5d}  | {len(changed):12d}  | {fixed:19d}  | {harmed:20d}  | {bad_words:18d}  | {ex:3d}   | {cer:.3f}")
        if cutoff == 84:
            for i in changed[:10]:
                print(f"      e.g. truth {truths[i]!r:34s} read {preds[i]!r:30s} -> {new[i]!r}")


q = {}
for x in (BAKE / "out" / "qwen.jsonl").read_text(encoding="utf-8").splitlines():
    if x.strip():
        d = json.loads(x)
        q[d["id"]] = d.get("text") or ""
evaluate("Qwen2.5-VL-7B", [q[r["id"]] for r in rows])
pad = {json.loads(x)["id"]: json.loads(x)["pred"] for x in (OUT / "paddle_PP-OCRv5_server_rec.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
evaluate("PaddleOCR PP-OCRv5 server", [pad[r["id"]] for r in rows])
