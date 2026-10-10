"""ENT-S1 / SW-S2 / SW-S7: licence audit, SBOM, NOTICE, model registry and the gate in front of every swap."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cdi_adapter.compliance import licences as L
from cdi_adapter.compliance import models as M
from cdi_adapter.config import Settings

POLICY = L.load_policy()
AGPL_PDF = "some-agpl-pdf-lib"          # a stand-in for any package under a network-copyleft licence


# ------------------------------------------------------------------ licence classification
@pytest.mark.parametrize("lic,cls,state", [
    ("MIT", [], "allowed"), ("BSD-3-Clause", [], "allowed"), ("Apache-2.0", [], "allowed"),
    ("Apache 2.0", [], "allowed"), ("MIT OR Apache-2.0", [], "allowed"), ("Apache-2.0 OR BSD-2-Clause", [], "allowed"),
    ("BSD-3-Clause, Apache-2.0, dependency licenses", [], "allowed"), ("MPL-2.0", [], "allowed"),
    ("", ["OSI Approved", "MIT License"], "allowed"), ("Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSL-1.0", [], "allowed"),
    ("LGPL-3.0-only", [], "review"), ("", ["GNU Lesser General Public License v3 (LGPLv3)"], "review"),
    ("GPL-3.0-or-later", [], "banned"), ("GNU General Public License v3", [], "banned"), ("AGPL-3.0", [], "banned"),
    ("GNU Affero General Public License v3", [], "banned"), ("SSPL-1.0", [], "banned"), ("BUSL-1.1", [], "banned"),
    ("CC-BY-NC-4.0", [], "banned"), ("Non-Commercial Use Only", [], "banned"), ("qwen-research", [], "banned"),
    ("Research License", [], "banned"), ("", [], "unknown"), ("Some Homemade Licence v9", [], "unknown"),
])
def test_licences_are_sorted_into_the_four_states(lic, cls, state):
    assert L.classify(lic, cls, POLICY)[0] == state


def test_lgpl_is_not_mistaken_for_gpl():
    assert L.classify("LGPL-2.1", [], POLICY)[0] == "review"
    assert L.classify("GNU Lesser General Public License", [], POLICY)[0] == "review"


# ------------------------------------------------------------------ the gate (fake installs)
class _D:
    def __init__(self, name, version, lic="", classifiers=()):
        self.version = version
        self.metadata = type("MD", (), {
            "__getitem__": lambda s, k: name if k == "Name" else None,
            "get": lambda s, k, d=None: {"License": lic}.get(k, d),
            "get_all": lambda s, k, d=None: [f"License :: {c}" for c in classifiers] if k == "Classifier" else d})()


def _report(*dists, policy=None):
    return L.audit(policy or POLICY, list(dists))


def test_a_strong_copyleft_package_fails_the_build_until_a_decision_is_recorded():     # SW-S7 AC2
    rep = _report(_D(AGPL_PDF, "1.24", "GNU AFFERO GENERAL PUBLIC LICENSE v3"))
    assert any(p.startswith("BANNED: " + AGPL_PDF + "") for p in L.problems(rep))
    approved = copy.deepcopy(POLICY)
    approved["decisions"][AGPL_PDF] = {"status": "approved", "reason": "exception signed by counsel", "by": "x", "date": "d"}
    assert L.problems(_report(_D(AGPL_PDF, "1.24", "GNU AFFERO GENERAL PUBLIC LICENSE v3"), policy=approved)) == []


def test_review_licences_need_a_decision_and_strict_wants_it_approved():
    d = _D("somelib", "1.0", "LGPL-3.0-only")
    assert any("NEEDS A DECISION" in p for p in L.problems(_report(d)))
    pol = copy.deepcopy(POLICY)
    pol["decisions"]["somelib"] = {"status": "proposed", "reason": "dynamic link, unmodified"}
    assert L.problems(_report(d, policy=pol)) == []                                    # passes with a warning
    assert any("NOT YET APPROVED" in p for p in L.problems(_report(d, policy=pol), strict=True))
    pol["decisions"]["somelib"]["status"] = "approved"
    assert L.problems(_report(d, policy=pol), strict=True) == []
    pol["decisions"]["somelib"]["status"] = "rejected"
    assert any("REJECTED" in p for p in L.problems(_report(d, policy=pol)))


def test_an_unknown_licence_is_never_waved_through():
    assert any("NEEDS A DECISION" in p for p in L.problems(_report(_D("mystery", "0.1", ""))))


def test_the_projects_own_installed_environment_passes_the_gate():
    rep = L.audit(POLICY)
    assert L.problems(rep) == [], L.problems(rep)
    assert not rep.by_state("banned") and not rep.by_state("unknown")


# ------------------------------------------------------------------ SBOM and NOTICE
def test_the_sbom_is_stable_and_complete():
    rep = _report(_D("b-lib", "2.0", "MIT"), _D("a-lib", "1.0", "Apache-2.0"))
    s = L.sbom(rep)
    assert [c["name"] for c in s["components"]] == ["a-lib", "b-lib"]
    assert s["components"][0]["licenses"][0]["license"]["name"] == "Apache-2.0"
    assert json.dumps(L.sbom(rep)) == json.dumps(s)                                    # same install = same bytes


def test_a_swap_is_visible_as_an_sbom_diff():                                          # SW-S7 AC2
    old = L.sbom(_report(_D("keep", "1.0", "MIT"), _D("gone", "1.0", "MIT"), _D("bump", "1.0", "MIT")))
    new = L.sbom(_report(_D("keep", "1.0", "MIT"), _D("new-dep", "3.1", "GPL-3.0"), _D("bump", "2.0", "BSD-3-Clause")))
    d = L.diff(old, new)
    assert d["added"] == ["new-dep 3.1 (GPL-3.0)"] and d["removed"] == ["gone 1.0"]
    assert d["version_changed"] == ["bump: 1.0 -> 2.0"] and d["licence_changed"] == ["bump: MIT -> BSD-3-Clause"]


def test_the_notice_lists_every_attribution_licence_and_the_models():                  # SW-S7 AC3
    n = L.notice(_report(_D("alpha", "1.2", "MIT"), _D("beta", "3", "Apache-2.0"), _D("gamma", "9", "LGPL-3.0")), POLICY)
    assert "- alpha 1.2: MIT" in n and "- beta 3: Apache-2.0" in n and "- gamma 9: LGPL-3.0" in n
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in n


def test_a_model_licence_notice_line_reaches_the_notice_file(monkeypatch):
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["licence"]["notice_line"] = "Built with Example Model"
    monkeypatch.setattr(M, "load_registry", lambda path=None: reg)
    assert "Built with Example Model" in L.notice(_report(), POLICY)


# ------------------------------------------------------------------ model registry
def test_the_registry_entries_are_complete_and_pinned():
    reg = M.load_registry()
    for m in reg["models"]:
        assert len(m["revision"]) == 40 and m["weights_format"] == "safetensors" and m["trust_remote_code"] is False
        assert any(f.endswith(".safetensors") and meta["sha256"] and len(meta["sha256"]) == 64 for f, meta in m["files"].items())
        assert M.licence_problem(m) is None, (m["model_id"], M.licence_problem(m))
    assert M.champion("vlm")["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_the_default_settings_pass_the_gate():
    assert M.check_settings(Settings(mlserve_backend="hf")) == [
        "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]


def test_a_model_that_is_not_registered_is_refused_naming_the_setting():                # SW-S2 AC1
    with pytest.raises(M.ModelRefused, match=r"not registered \(CDI_VLM_MODEL_ID\)"):
        M.check_settings(Settings(mlserve_backend="hf", vlm_model_id="Someone/Unregistered-VL"))


def test_a_model_with_no_licence_text_cannot_be_champion(tmp_path):                      # SW-S2 AC3
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["licence"]["text_file"] = "licences/does-not-exist.txt"
    with pytest.raises(M.ModelRefused, match="no copy of the licence text"):
        M.check_model(reg["models"][0]["model_id"], "CDI_VLM_MODEL_ID", reg=reg)
    reg["models"][0]["licence"]["text_file"] = "licences/Apache-2.0.txt"
    (tmp_path / "licences").mkdir()
    (tmp_path / "licences" / "Apache-2.0.txt").write_text("   \n")
    with pytest.raises(M.ModelRefused, match="no copy of the licence text"):
        M.check_model(reg["models"][0]["model_id"], "CDI_VLM_MODEL_ID", reg=reg, root=tmp_path)    # an empty file is no copy


@pytest.mark.parametrize("name", ["qwen-research", "CC-BY-NC-4.0", "Research License", "AGPL-3.0"])
def test_a_research_only_or_copyleft_licence_is_refused(name):                           # SW-S7 AC1
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["licence"]["name"] = name
    with pytest.raises(M.ModelRefused, match="licence"):
        M.check_model(reg["models"][0]["model_id"], "CDI_VLM_MODEL_ID", reg=reg)


def test_revision_format_and_remote_code_are_enforced():
    reg = copy.deepcopy(M.load_registry())
    mid = reg["models"][0]["model_id"]
    reg["models"][0]["revision"] = "main"
    with pytest.raises(M.ModelRefused, match="exact revision"):
        M.check_model(mid, "S", reg=reg)
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["trust_remote_code"] = True
    with pytest.raises(M.ModelRefused, match="remote code"):
        M.check_model(mid, "S", reg=reg)
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["weights_format"] = "pickle"
    with pytest.raises(M.ModelRefused, match="safetensors"):
        M.check_model(mid, "S", reg=reg)


def test_a_candidate_is_not_a_champion_until_the_registry_says_so():
    reg = copy.deepcopy(M.load_registry())
    reg["models"][0]["status"] = "candidate"
    with pytest.raises(M.ModelRefused, match="not champion"):
        M.check_model(reg["models"][0]["model_id"], "CDI_VLM_MODEL_ID", reg=reg)


def test_changed_files_are_refused_even_when_the_pinned_revision_is_the_same(tmp_path):   # SW-S2 AC4 / SW-S7 AC4
    good = b"weights-v1"
    reg = {"models": [{"model_id": "a/b", "revision": "0" * 40,
                       "files": {"model.safetensors": {"sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good)},
                                 "config.json": {"sha256": None, "bytes": 2}}}]}
    (tmp_path / "model.safetensors").write_bytes(good)
    (tmp_path / "config.json").write_text("{}")
    assert M.verify_files("a/b", tmp_path, reg=reg) == 1
    (tmp_path / "model.safetensors").write_bytes(b"weights-TAMPERED")
    with pytest.raises(M.ModelIntegrityError, match=r"model.safetensors does not match the registered checksum"):
        M.verify_files("a/b", tmp_path, reg=reg)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(M.ModelIntegrityError, match="is missing"):
        M.verify_files("a/b", tmp_path, reg=reg)


def test_prepare_load_returns_the_pinned_revision_and_enforces_unless_switched_off(monkeypatch):
    rev = M.pinned_revision("Qwen/Qwen2.5-VL-7B-Instruct")
    assert M.prepare_load("Qwen/Qwen2.5-VL-7B-Instruct", setting_name="S", verify=False) == rev
    with pytest.raises(M.ModelRefused):
        M.prepare_load("nobody/nothing", setting_name="CDI_VLM_MODEL_ID", verify=False)
    from cdi_adapter.config import settings
    monkeypatch.setattr(settings, "model_registry_enforce", False)
    assert M.prepare_load("nobody/nothing", setting_name="S") == "main"                  # dev only: not enforced
    assert M.require_registered(Settings(model_registry_enforce=False, vlm_model_id="x/y")) == []


def test_results_name_the_model_revision_and_prompt_version():                            # SW-S2 AC2, SW-S1 AC4
    from cdi_adapter import provenance

    v = provenance.engine_versions("Qwen/Qwen2.5-VL-7B-Instruct")
    assert v["vlm_revision"] == M.pinned_revision("Qwen/Qwen2.5-VL-7B-Instruct") and len(v["vlm_revision"]) == 40
    assert v["pdf_renderer"].startswith("pypdfium2") and provenance.prompt_version().startswith("p-")
    fb = provenance.engine_versions("Qwen/Qwen2-VL-7B-Instruct")                          # the fallback answered
    assert fb["vlm_revision"] != v["vlm_revision"]
