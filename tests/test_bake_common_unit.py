import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bake_common as b  # noqa: E402


def test_vocabulary_is_leave_one_out_and_per_doctor():
    rows = [{"id": "p_1", "truth": "Tab Montina Fx"}, {"id": "p_2", "truth": "Tab Sompraz D"}, {"id": "q_1", "truth": "Zincovit"}]
    v = b.doctor_vocab(rows)
    assert b.vocab_for("p_1", v) == ["sompraz", "tab"]          # never its own words, never another doctor's
    assert b.vocab_for("q_1", v) == []


def test_snap_only_when_exactly_one_word_is_near():
    assert b.snap_to_vocab("Tab. Somprax 5", ["tab", "sompraz"]) == "Tab. sompraz 5"
    assert b.snap_to_vocab("Montana fx", ["montina fx", "montana"]) == "Montana fx"     # already a known word


def test_setup_counts_silent_errors():
    r = b._setup("x", ["CBC", "LFT"], ["CBC", "LPT"])
    assert r["silent_errors"] == 1 and r["exact"] == 1
