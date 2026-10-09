"""Qwen alone vs LightOnOCR-2-1B alone vs both together, on the owner's labelled handwriting crops.

    1. Qwen readings (pod, main venv):   python scripts/bake_off.py qwen    SET/lines.jsonl SET/qwen.jsonl
    2. LightOn readings (its own venv):  /workspace/lighton-venv/bin/python scripts/lighton_worker.py SET/lines.jsonl SET/lighton.jsonl
    3. The comparison:                   python scripts/bake_off.py compare SET/lines.jsonl SET/qwen.jsonl SET/lighton.jsonl SET/report.json

``SET/lines.jsonl`` is the file downloaded from the labelling page (``{"id", "crop", "truth"}``).
Every number printed is MEASURED on those lines; pass margins are NOT decided here - the report gives the data
and the owner sets the margins."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from cdi_adapter.recognition.bench_htr import _norm, score, wilson_lower
from cdi_adapter.recognition.disagreement import AGREE, compare_engines
from cdi_adapter.recognition.engines import Reading


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_qwen(lines: Path, out: Path) -> None:
    from cdi_adapter.recognition.engines import QwenLineEngine

    rows = _rows(lines)
    crops = [(lines.parent / r["crop"]).read_bytes() for r in rows]
    t0 = time.perf_counter()
    got = QwenLineEngine().recognize(crops)
    secs = (time.perf_counter() - t0) / max(1, len(rows))
    with out.open("w", encoding="utf-8") as fh:
        for r, g in zip(rows, got):
            fh.write(json.dumps({"id": r["id"], "text": g.text if g.ok else "", "seconds": round(secs, 3),
                                 "error": g.error, "engine_version": g.engine_version}, ensure_ascii=False) + "\n")
    print(f"qwen: {len(rows)} lines, {secs:.3f} s per line (batched, MEASURED)")


def _setup(name: str, truths: list[str], preds: list[str]) -> dict:
    ok = sum(_norm(t) == _norm(p) for t, p in zip(truths, preds))
    n = len(truths)
    return {"name": name, **score(truths, preds), "accepted": n, "coverage": 1.0, "exact": ok, "silent_errors": n - ok,
            "silent_error_rate": round((n - ok) / n, 4) if n else None,
            "precision_lower_bound": round(wilson_lower(ok, n), 4)}


def compare(lines: Path, qwen: Path, lighton: Path, out: Path) -> dict:
    truth = {r["id"]: r["truth"] for r in _rows(lines)}
    q = {r["id"]: r for r in _rows(qwen)}
    lo = {r["id"]: r for r in _rows(lighton)}
    ids = [i for i in truth if i in q and i in lo]
    T = [truth[i] for i in ids]
    qt = [q[i]["text"] for i in ids]
    lt = [lo[i]["text"] for i in ids]
    qr = [Reading("qwen", "q", q[i]["text"], None, error=q[i].get("error")) for i in ids]
    lr = [Reading("lighton", "l", lo[i]["text"], None, error=lo[i].get("error")) for i in ids]

    acc_t, acc_p, review, caught = [], [], 0, 0
    for t, a, b in zip(T, qr, lr):
        v = compare_engines([a, b])
        if v.state == AGREE:
            acc_t.append(t)
            acc_p.append(v.display_text)
        else:
            review += 1
            if _norm(t) != _norm(a.text) or _norm(t) != _norm(b.text):
                caught += 1
    wrong = sum(_norm(t) != _norm(p) for t, p in zip(acc_t, acc_p))
    both = {"name": "both together (accept only when they agree)", **score(acc_t, acc_p), "accepted": len(acc_p),
            "coverage": round(len(acc_p) / len(ids), 4) if ids else None, "silent_errors": wrong,
            "silent_error_rate_of_accepted": round(wrong / len(acc_p), 4) if acc_p else None,
            "precision_lower_bound": round(wilson_lower(len(acc_p) - wrong, len(acc_p)), 4),
            "sent_to_a_person": review, "wrong_lines_caught_by_disagreement": caught}
    both_wrong = sum(_norm(t) != _norm(a) and _norm(t) != _norm(b) for t, a, b in zip(T, qt, lt))

    # does the model's own token probability warn about its mistakes? (a signal to measure, not a confidence)
    lp = [(lo[i].get("mean_logprob"), _norm(truth[i]) == _norm(lo[i]["text"])) for i in ids if lo[i].get("mean_logprob") is not None]
    lp.sort()
    bands = []
    for k in range(4):
        part = lp[k * len(lp) // 4:(k + 1) * len(lp) // 4]
        if part:
            bands.append({"quartile_of_mean_logprob (1 = least sure)": k + 1, "lines": len(part),
                          "wrong": sum(not ok for _, ok in part), "mean_logprob_range": [part[0][0], part[-1][0]]})
    report = {
        "lines_compared": len(ids), "status": "MEASURED on the owner's labelled crops; margins NOT applied (owner decides)",
        "seconds_per_line": {"qwen_batched": round(sum(q[i]["seconds"] or 0 for i in ids) / max(1, len(ids)), 3),
                             "lighton_one_at_a_time": round(sum(lo[i]["seconds"] or 0 for i in ids) / max(1, len(ids)), 3)},
        "qwen_versions": sorted({q[i].get("engine_version", "") for i in ids}),
        "A_qwen_alone": _setup("Qwen alone", T, qt),
        "B_lighton_alone": _setup("LightOnOCR-2-1B alone", T, lt),
        "C_both_together": both,
        "lines_wrong_in_both_readers": both_wrong,
        "lighton_logprob_vs_errors": bands,
        "per_line": [{"id": i, "truth": truth[i], "qwen": q[i]["text"], "lighton": lo[i]["text"],
                      "lighton_mean_logprob": lo[i].get("mean_logprob")} for i in ids],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    short = {k: v for k, v in report.items() if k != "per_line"}
    print(json.dumps(short, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "qwen":
        run_qwen(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) == 6 and argv[1] == "compare":
        compare(*(Path(a) for a in argv[2:6]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
