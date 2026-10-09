"""How well does the yes / no probability (extract/verify.py) separate real ordered tests from wrong candidates? MEASURED on the labelled pages.

    python scripts/measure_verify.py TESTS.json TRUTH.json PAGES_DIR OUT.json
TESTS.json = output of rerun_extract.py; TRUTH.json = {photo: [test keys]}; PAGES_DIR holds the photos."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from cdi_adapter.extract.resolve_llm import page_views
from cdi_adapter.extract.verify import verify_tests

RULES = [("complete blood count|^cbc|cbs", "cbc"), ("c-reactive", "crp"), ("liver function", "lft"), ("kidney|renal function|kft", "kft"),
         ("creatinine", "creatinine"), ("fasting blood|fasting glu|^fbs|fpg|blood glucose", "fbs"), ("post-prandial|ppbs|postprandial", "ppbs"),
         ("vitamin d", "vitamin d"), ("hba1c|glycated", "hba1c"), ("^tsh|thyroid stim", "tsh"), ("free t4", "ft4"), ("prothrombin|^pt$", "pt"),
         ("urea", "urea"), ("sodium and potassium|electrolyt", "sodium+potassium"), ("lipid", "lipid profile"), ("fever profile", "fever profile"),
         ("d-dimer", "ddimer"), ("il6|il-6|interleukin", "il6"), ("urine routine", "urine re"), ("lipase", "lipase"), ("sgpt|alt .sgpt", "sgpt"),
         ("hiv", "hiv"), ("hepatitis c", "hcv"), ("covid|rt-pcr|sars", "covid pcr"), ("hepatitis b|hbsag", "hbsag"), ("bleeding time", "bt"),
         ("clotting time", "ct"), ("aptt|npt", "aptt"), ("platelet", "platelet")]


def key(n: str | None) -> str:
    s = (n or "").lower()
    for p, k in RULES:
        if re.search(p, s):
            return k
    return s


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def main(tests_p: str, truth_p: str, pages: str, out_p: str) -> int:
    ext = json.load(open(tests_p, encoding="utf-8"))
    tru = json.load(open(truth_p, encoding="utf-8"))
    rows = []
    for fn, docs in ext.items():
        if fn not in tru:
            continue
        img = Path(pages, fn).read_bytes()
        views = [img, *page_views(img)[:1]]
        want = set(tru[fn])
        names = []
        for d in docs:
            for t in d["lab_tests"]:
                if t.get("status") != "rejected" and (t.get("standard_name") or t.get("as_written")):
                    names.append(t.get("standard_name") or t["as_written"])
        names = sorted(set(names))
        for view_name, vs in (("full_page", views[:1]), ("full_page_and_enlarged", views)):
            got = verify_tests(vs, names)
            for n in names:
                k = key(n)
                good = k in want or (k == "sodium+potassium" and want & {"sodium", "potassium"}) or (k == "tsh" and "tft" in want)
                rows.append({"photo": fn, "view": view_name, "name": n, "true_test": bool(good), "p_yes": got.get(n)})
    rep = {}
    for view in ("full_page", "full_page_and_enlarged"):
        sub = [r for r in rows if r["view"] == view and r["p_yes"] is not None]
        pos = [r["p_yes"] for r in sub if r["true_test"]]
        neg = [r["p_yes"] for r in sub if not r["true_test"]]
        rep[view] = {"true": len(pos), "wrong": len(neg), "auc": auc(pos, neg),
                     "mean_p_true": round(sum(pos) / len(pos), 3) if pos else None, "mean_p_wrong": round(sum(neg) / len(neg), 3) if neg else None,
                     "wrong_below_0.5": sum(p < 0.5 for p in neg), "true_below_0.5": sum(p < 0.5 for p in pos)}
    json.dump({"summary": rep, "rows": rows}, open(out_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
