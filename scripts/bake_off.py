"""Qwen alone vs LightOnOCR-2-1B alone vs both together, on the owner's labelled handwriting crops.

    1. Qwen readings (pod, main venv):   python scripts/bake_off.py qwen    SET/lines.jsonl SET/qwen.jsonl
    2. LightOn readings (its own venv):  /workspace/lighton-venv/bin/python scripts/lighton_worker.py SET/lines.jsonl SET/lighton.jsonl
    2b. Qwen shown LightOn's reading:    python scripts/bake_off.py qwenfed SET/lines.jsonl SET/lighton.jsonl SET/qwenfed.jsonl
    2c. Qwen told the doctor's words:    python scripts/bake_off.py memory  SET/lines.jsonl SET/qwenmem.jsonl
    3. The comparison:                   python scripts/bake_off.py compare SET/lines.jsonl SET/qwen.jsonl SET/lighton.jsonl SET/report.json [SET/qwenfed.jsonl [SET/qwenmem.jsonl]]

``SET/lines.jsonl`` is the file downloaded from the labelling page (``{"id", "crop", "truth"}``).
Every number printed is MEASURED on those lines; pass margins are NOT decided here - the report gives the data
and the owner sets the margins."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from cdi_adapter.recognition.bench_htr import _norm, score, wilson_lower
from cdi_adapter.recognition.disagreement import AGREE, compare_engines
from cdi_adapter.recognition.engines import Reading


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").casefold())


def lighton_line(raw: str, unwrap_latex: bool = False) -> str:
    """LightOnOCR writes a page's worth of markdown and often repeats itself on a one-line crop. It gets the same treatment Qwen's
    engine gets: the first non-empty line. ``unwrap_latex`` also takes the words out of dollar-text and mathcal wrappers
    (the most favourable reading of what it wrote)."""
    line = next((x.strip() for x in (raw or "").splitlines() if x.strip()), "")
    if unwrap_latex:
        line = re.sub(r"\\(?:text|mathcal|mathrm|mathbf|textbf)\{([^{}]*)\}", r"\1", line)
        line = re.sub(r"\*\*|\$|\\[a-zA-Z]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
    return line


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


FED_PROMPT = (
    "This is one line from a medical prescription and may be handwritten. A first reader proposed: \"{cand}\". "
    "The proposal may be wrong. Look at the image and transcribe exactly what is written, keeping the doctor's own spelling. "
    "Write an unreadable character as ?. Output only the transcription on one line."
)


def run_qwen_fed(lines: Path, lighton: Path, out: Path) -> None:
    """The controlled test: the same Qwen, but shown the other reader's candidate (anchoring risk is what this measures)."""
    from cdi_adapter.ml.client import get_client
    from cdi_adapter.recognition.engines import clean_line

    rows = _rows(lines)
    cand = {r["id"]: r["text"] for r in _rows(lighton)}
    client = get_client()
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            err = None
            try:
                txt, _ = client.vlm_generate_ex((lines.parent / r["crop"]).read_bytes(),
                                                FED_PROMPT.format(cand=lighton_line(cand.get(r["id"], ""), True).replace('"', "'")[:120]), max_tokens=64)
                text = clean_line(next((x.strip() for x in (txt or "").splitlines() if x.strip()), ""))
            except Exception as exc:  # noqa: BLE001
                text, err = "", str(exc)[:200]
            fh.write(json.dumps({"id": r["id"], "text": text, "error": err}, ensure_ascii=False) + "\n")
    print(f"qwen fed: {len(rows)} lines, {(time.perf_counter() - t0) / max(1, len(rows)):.3f} s per line (one at a time, MEASURED)")


def _group(line_id: str) -> str:
    """The doctor = the prescription photo the line came from (ids are <photo>_<n>)."""
    return line_id.rsplit("_", 1)[0]


def doctor_vocab(rows: list[dict]) -> dict[str, list[tuple[str, set[str]]]]:
    """Per doctor: the words in that doctor's OTHER confirmed lines. Used leave-one-out, so a line never sees its own truth."""
    out: dict[str, list[tuple[str, set[str]]]] = {}
    for r in rows:
        toks = {t for t in re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", r["truth"]) if len(t) >= 2}
        out.setdefault(_group(r["id"]), []).append((r["id"], {t.casefold() for t in toks}))
    return out


def vocab_for(line_id: str, vocab: dict[str, list[tuple[str, set[str]]]]) -> list[str]:
    words: set[str] = set()
    for other, toks in vocab.get(_group(line_id), []):
        if other != line_id:
            words |= toks
    return sorted(words)


MEMORY_PROMPT = (
    "This is one line from a medical prescription and may be handwritten. The same doctor has written these words before: {words}. "
    "They are only hints: use one only if the image really shows it, otherwise write exactly what you see, keeping the doctor's own "
    "spelling. Write an unreadable character as ?. Output only the transcription on one line."
)


def run_qwen_memory(lines: Path, out: Path) -> None:
    """The controlled test: the same Qwen, told the words this doctor has confirmed elsewhere (leave-one-out)."""
    from cdi_adapter.ml.client import get_client
    from cdi_adapter.recognition.engines import clean_line

    rows = _rows(lines)
    vocab = doctor_vocab(rows)
    client = get_client()
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            words = vocab_for(r["id"], vocab)
            err = None
            try:
                txt, _ = client.vlm_generate_ex((lines.parent / r["crop"]).read_bytes(),
                                                MEMORY_PROMPT.format(words=", ".join(words)[:900] or "(none yet)"), max_tokens=64)
                text = clean_line(next((x.strip() for x in (txt or "").splitlines() if x.strip()), ""))
            except Exception as exc:  # noqa: BLE001
                text, err = "", str(exc)[:200]
            fh.write(json.dumps({"id": r["id"], "text": text, "error": err, "vocab_size": len(words)}, ensure_ascii=False) + chr(10))
    print(f"qwen with doctor memory: {len(rows)} lines, {(time.perf_counter() - t0) / max(1, len(rows)):.3f} s per line (one at a time, MEASURED)")


def snap_to_vocab(text: str, words: list[str]) -> str:
    """Deterministic memory: a word one or two letters away from exactly ONE of the doctor's confirmed words becomes that word."""
    from cdi_adapter.recognition.bench_htr import levenshtein

    def fix(tok: str) -> str:
        core = tok.casefold()
        if len(core) < 3 or core in words:
            return tok
        lim = 1 if len(core) < 6 else 2
        near = [w for w in words if abs(len(w) - len(core)) <= lim and levenshtein(core, w) <= lim]
        return near[0] if len(near) == 1 else tok
    return re.sub(r"[A-Za-z][A-Za-z0-9]*", lambda m: fix(m.group(0)), text)


def _setup(name: str, truths: list[str], preds: list[str]) -> dict:
    ok = sum(_norm(t) == _norm(p) for t, p in zip(truths, preds))
    n = len(truths)
    ok_alnum = sum(_alnum(t) == _alnum(p) for t, p in zip(truths, preds))
    return {"name": name, **score(truths, preds), "accepted": n, "coverage": 1.0, "exact": ok, "silent_errors": n - ok,
            "exact_ignoring_punctuation_and_spacing": round(ok_alnum / n, 4) if n else None,
            "silent_error_rate_ignoring_punctuation_and_spacing": round((n - ok_alnum) / n, 4) if n else None,
            "silent_error_rate": round((n - ok) / n, 4) if n else None,
            "precision_lower_bound": round(wilson_lower(ok, n), 4)}


def compare(lines: Path, qwen: Path, lighton: Path, out: Path, fed: Path | None = None, memory: Path | None = None) -> dict:
    truth = {r["id"]: r["truth"] for r in _rows(lines)}
    q = {r["id"]: r for r in _rows(qwen)}
    lo = {r["id"]: r for r in _rows(lighton)}
    ids = [i for i in truth if i in q and i in lo]
    T = [truth[i] for i in ids]
    qt = [q[i]["text"] for i in ids]
    lt = [lighton_line(lo[i]["text"]) for i in ids]
    lt_u = [lighton_line(lo[i]["text"], True) for i in ids]
    qr = [Reading("qwen", "q", q[i]["text"], None, error=q[i].get("error")) for i in ids]

    def together(second: list[str], label: str) -> dict:
        lr = [Reading("lighton", "l", x, None, error=lo[i].get("error")) for i, x in zip(ids, second)]
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
        return {"name": label, **score(acc_t, acc_p), "accepted": len(acc_p),
                "coverage": round(len(acc_p) / len(ids), 4) if ids else None, "silent_errors": wrong,
                "silent_error_rate_of_accepted": round(wrong / len(acc_p), 4) if acc_p else None,
                "precision_lower_bound": round(wilson_lower(len(acc_p) - wrong, len(acc_p)), 4),
                "sent_to_a_person": review, "wrong_lines_caught_by_disagreement": caught}

    both = together(lt, "both together (accept only when they agree; LightOn first line)")
    both_u = together(lt_u, "both together (accept only when they agree; LightOn LaTeX removed)")
    both_wrong = sum(_norm(t) != _norm(a) and _norm(t) != _norm(b) for t, a, b in zip(T, qt, lt_u))

    # does the model's own token probability warn about its mistakes? (a signal to measure, not a confidence)
    lp = [(lo[i].get("mean_logprob"), _alnum(truth[i]) == _alnum(lighton_line(lo[i]["text"], True))) for i in ids if lo[i].get("mean_logprob") is not None]
    lp.sort()
    bands = []
    for k in range(4):
        part = lp[k * len(lp) // 4:(k + 1) * len(lp) // 4]
        if part:
            bands.append({"quartile_of_mean_logprob (1 = least sure)": k + 1, "lines": len(part),
                          "wrong": sum(not ok for _, ok in part), "mean_logprob_range": [part[0][0], part[-1][0]]})
    extra = {}
    if fed is not None and fed.is_file():
        f = {r["id"]: r for r in _rows(fed)}
        D = [f[i]["text"] if i in f else "" for i in ids]
        extra["D_qwen_shown_lighton_candidate"] = _setup("Qwen shown LightOn's reading", T, D)
        extra["D_changed_a_correct_qwen_reading"] = sum(_norm(t) == _norm(a) and _norm(t) != _norm(d) for t, a, d in zip(T, qt, D))
        extra["D_fixed_a_wrong_qwen_reading"] = sum(_norm(t) != _norm(a) and _norm(t) == _norm(d) for t, a, d in zip(T, qt, D))
        extra["D_copied_lighton_wrongly"] = sum(_norm(t) != _norm(l) and _norm(d) == _norm(l) for t, l, d in zip(T, lt, D))
        extra["_fed_texts"] = D
    if memory is not None and memory.is_file():
        m = {r["id"]: r for r in _rows(memory)}
        E = [m[i]["text"] if i in m else "" for i in ids]
        vocab = doctor_vocab(_rows(lines))
        S = [snap_to_vocab(a, vocab_for(i, vocab)) for i, a in zip(ids, qt)]
        extra["E1_qwen_told_the_doctors_words"] = _setup("Qwen told the doctor's other confirmed words", T, E)
        extra["E1_changed_a_correct_blind_reading"] = sum(_norm(t) == _norm(a) and _norm(t) != _norm(e) for t, a, e in zip(T, qt, E))
        extra["E1_fixed_a_wrong_blind_reading"] = sum(_norm(t) != _norm(a) and _norm(t) == _norm(e) for t, a, e in zip(T, qt, E))
        extra["E2_qwen_blind_then_snapped_to_doctor_words"] = _setup("Qwen blind, then words snapped to the doctor's confirmed words", T, S)
        extra["E2_changed_a_correct_blind_reading"] = sum(_norm(t) == _norm(a) and _norm(t) != _norm(x) for t, a, x in zip(T, qt, S))
        extra["E2_fixed_a_wrong_blind_reading"] = sum(_norm(t) != _norm(a) and _norm(t) == _norm(x) for t, a, x in zip(T, qt, S))
        extra["E_note"] = "leave-one-out: each line sees only the OTHER confirmed lines of the same photo (doctor); small sets overstate how much a doctor repeats"
        extra["_mem_texts"] = E
        extra["_snap_texts"] = S
    report = {
        **{k: v for k, v in extra.items() if not k.startswith("_")},
        "lines_compared": len(ids), "status": "MEASURED on the owner's labelled crops; margins NOT applied (owner decides)",
        "seconds_per_line": {"qwen_batched": round(sum(q[i]["seconds"] or 0 for i in ids) / max(1, len(ids)), 3),
                             "lighton_one_at_a_time": round(sum(lo[i]["seconds"] or 0 for i in ids) / max(1, len(ids)), 3)},
        "qwen_versions": sorted({q[i].get("engine_version", "") for i in ids}),
        "A_qwen_alone": _setup("Qwen alone", T, qt),
        "B_lighton_alone_first_line": _setup("LightOnOCR-2-1B alone (first line of its answer)", T, lt),
        "B2_lighton_alone_latex_unwrapped": _setup("LightOnOCR-2-1B alone (first line, LaTeX wrappers removed)", T, lt_u),
        "C_both_together": both,
        "C2_both_together_latex_removed": both_u,
        "lines_wrong_in_both_readers": both_wrong,
        "lighton_logprob_vs_errors": bands,
        "per_line": [{"id": i, "truth": truth[i], "qwen": q[i]["text"], "lighton": lo[i]["text"],
                      "lighton_mean_logprob": lo[i].get("mean_logprob"),
                      **({"qwen_fed": extra["_fed_texts"][k]} if "_fed_texts" in extra else {}),
                      **({"qwen_doctor_words": extra["_mem_texts"][k], "qwen_snapped": extra["_snap_texts"][k]} if "_mem_texts" in extra else {})}
                     for k, i in enumerate(ids)],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    short = {k: v for k, v in report.items() if k != "per_line"}
    print(json.dumps(short, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "qwen":
        run_qwen(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) == 5 and argv[1] == "qwenfed":
        run_qwen_fed(Path(argv[2]), Path(argv[3]), Path(argv[4]))
        return 0
    if len(argv) == 4 and argv[1] == "memory":
        run_qwen_memory(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) in (6, 7, 8) and argv[1] == "compare":
        compare(*(Path(a) for a in argv[2:6]), fed=Path(argv[6]) if len(argv) >= 7 else None,
                memory=Path(argv[7]) if len(argv) == 8 else None)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
