import math

from cdi_adapter.extract import verify


def test_p_yes_is_yes_over_yes_plus_no():
    top = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": "yes", "logprob": math.log(0.2)}, {"token": "No", "logprob": math.log(0.2)},
           {"token": "maybe", "logprob": math.log(0.5)}]
    assert abs(verify._p_yes(top) - 0.8) < 1e-9
    assert verify._p_yes([{"token": "perhaps", "logprob": -0.1}]) is None


def test_off_without_the_vllm_backend(monkeypatch):
    monkeypatch.setattr(verify.settings, "mlserve_backend", "stub")
    assert verify.verify_tests([b"x"], ["CBC"]) == {}
