import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bake_off  # noqa: E402


def _w(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_compare_three_setups(tmp_path):
    _w(tmp_path / "lines.jsonl", [{"id": str(i), "crop": "x.png", "truth": t} for i, t in enumerate(["CBC", "LFT", "FBS", "TSH"])])
    _w(tmp_path / "q.jsonl", [{"id": str(i), "text": t, "seconds": 0.1} for i, t in enumerate(["CBC", "LFT", "FBS", "TSH"][:3] + ["TSX"])])
    _w(tmp_path / "l.jsonl", [{"id": str(i), "text": t, "seconds": 1.0, "mean_logprob": lp}
                              for i, (t, lp) in enumerate([("CBC", -0.1), ("LFT", -0.2), ("FPS", -2.0), ("TSX", -0.3)])])
    r = bake_off.compare(tmp_path / "lines.jsonl", tmp_path / "q.jsonl", tmp_path / "l.jsonl", tmp_path / "r.json")
    assert r["lines_compared"] == 4
    assert r["A_qwen_alone"]["silent_errors"] == 1 and r["B_lighton_alone_first_line"]["silent_errors"] == 2
    c = r["C_both_together"]
    assert c["accepted"] == 3 and c["sent_to_a_person"] == 1
    assert c["silent_errors"] == 1          # both said TSX: agreement does not make it right
    assert r["lines_wrong_in_both_readers"] == 1
