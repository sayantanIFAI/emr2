"""SW-S1 / SW-S6: the swap-point registry and the contract every implementation must pass."""
from __future__ import annotations

import io
import os
import uuid

import pytest
from PIL import Image

from cdi_adapter import swap
from cdi_adapter.config import settings
from cdi_adapter.ingest import pages
from cdi_adapter.ingest.pdfgen import make_text_pdf
from cdi_adapter.storage_fs import FilesystemStore


# ------------------------------------------------------------------ the registry
def test_every_swap_point_has_a_setting_a_current_choice_and_alternatives_listed():
    d = {s["swap_point"]: s for s in swap.describe()}
    assert {"pdf_renderer", "object_store", "job_queue", "printed_ocr", "vlm", "terminology",
            "drive_connector", "signin", "metrics_sink"} <= set(d)
    for s in d.values():
        assert s["setting"].startswith("CDI_") and s["current"] in s["available"] and s["about"]
    assert d["object_store"]["available"] == ["filesystem", "s3"] and d["job_queue"]["available"] == ["redis", "valkey"]
    assert d["pdf_renderer"]["current"] == "pypdfium2" and d["pdf_renderer"]["version"]


def test_the_default_setup_resolves_at_start_up():
    assert "pdf_renderer=pypdfium2" in swap.check_all()


def test_a_wrong_name_stops_the_service_and_names_the_setting(monkeypatch):
    monkeypatch.setattr(settings, "pdf_renderer", "some-other-renderer")      # not an option
    with pytest.raises(swap.SwapConfigError, match=r"CDI_PDF_RENDERER=.*pypdfium2"):
        swap.check_all()
    swap.reset()


def test_a_missing_library_stops_the_service_and_names_the_setting(monkeypatch):
    p = swap.POINTS["pdf_renderer"]
    monkeypatch.setitem(p.choices, "pypdfium2", swap.Choice("pypdfium2", lambda: None, ("no_such_library_xyz",),
                                                              'pip install "pypdfium2"'))
    with pytest.raises(swap.SwapConfigError, match=r"CDI_PDF_RENDERER='pypdfium2' needs the library 'no_such_library_xyz'.*pip install"):
        swap.check_all()
    assert [s for s in swap.describe() if s["swap_point"] == "pdf_renderer"][0]["usable"] is False


def test_a_custom_drive_class_is_accepted_but_a_typo_is_not(monkeypatch):
    monkeypatch.setattr(settings, "listener_connector", "mypackage.drives:BoxConnector")
    swap.check_all()
    monkeypatch.setattr(settings, "listener_connector", "onedrve")
    with pytest.raises(swap.SwapConfigError, match="CDI_LISTENER_CONNECTOR"):
        swap.check_all()


def test_changing_the_setting_changes_the_implementation_without_touching_other_code(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "object_store", "filesystem")
    monkeypatch.setattr(settings, "object_store_dir", str(tmp_path))
    swap.reset("object_store")
    assert swap.resolve("object_store").name == "filesystem"
    monkeypatch.setattr(settings, "object_store", "s3")
    swap.reset("object_store")
    assert swap.resolve("object_store").name == "s3"
    swap.reset("object_store")


# ------------------------------------------------------------------ contract: PdfRenderer
@pytest.fixture(params=sorted(swap.POINTS["pdf_renderer"].choices) if swap.POINTS.get("pdf_renderer") else ["pypdfium2"])
def renderer(request, monkeypatch):
    monkeypatch.setattr(settings, "pdf_renderer", request.param)
    swap.reset("pdf_renderer")
    yield swap.resolve("pdf_renderer")
    swap.reset("pdf_renderer")


def _size(png: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png)) as im:
        return im.size


def test_pdf_renderer_contract_page_count_and_dpi(renderer):
    pdf = make_text_pdf([["page one"], ["page two"], ["page three"]])
    out = renderer.render_pngs(pdf, 150)
    assert len(out) == 3 and all(p.startswith(b"\x89PNG") for p in out)
    w150, h150 = _size(out[0])
    w300, h300 = _size(renderer.render_pngs(pdf, 300)[0])
    assert abs(w300 - w150 * 2) <= 1 and abs(h300 - h150 * 2) <= 1            # the dpi really scales the picture (page size is rounded up)
    assert abs(w150 - 1275) <= 2 or abs(w150 - 1240) <= 2                      # US Letter / A4 at 150 dpi


def test_pdf_renderer_contract_text_is_visible_and_white_page(renderer):
    out = renderer.render_pngs(make_text_pdf([["HbA1c 7.8 %", "Creatinine 1.1 mg/dL"]]), 150)
    with Image.open(io.BytesIO(out[0])) as im:
        assert im.convert("RGB").getpixel((2, 2)) == (255, 255, 255)
        assert sum(1 for px in im.convert("L").getdata() if px < 100) > 500


@pytest.mark.parametrize("blob", [b"", b"not a pdf at all", b"%PDF-1.7\n" + b"junk" * 40])
def test_pdf_renderer_contract_bad_input_is_a_data_error_not_a_crash(renderer, blob):
    with pytest.raises(ValueError):                                           # PdfReadError is a ValueError
        renderer.render_pngs(blob, 150)


def test_pdf_renderer_contract_too_long_is_refused_whole_never_cut(renderer, monkeypatch):
    monkeypatch.setattr(settings, "max_pages", 2)
    with pytest.raises(ValueError, match="3 pages"):
        renderer.render_pngs(make_text_pdf([["a"], ["b"], ["c"]]), 100)


def test_the_renderer_name_and_version_are_saved_with_the_result():           # SW-S1 AC1
    from cdi_adapter import provenance

    assert provenance.engine_versions()["pdf_renderer"].startswith("pypdfium2 ")


# ------------------------------------------------------------------ contract: ObjectStore
def _s3_store():
    from cdi_adapter.storage import S3Store

    st = S3Store()
    if not os.environ.get("CDI_S3_ENDPOINT_URL") or not st.ping():
        pytest.skip("no live S3-API store (set CDI_S3_ENDPOINT_URL)")
    st.ensure()
    return st


@pytest.fixture(params=["filesystem", "s3"])
def store(request, tmp_path):
    return FilesystemStore(tmp_path / "objects") if request.param == "filesystem" else _s3_store()


def _key(name: str = "x") -> str:
    return f"contract/{uuid.uuid4().hex}/{name}"


def test_object_store_contract_round_trip_is_byte_exact(store):
    store.ensure()
    store.ensure()                                                           # safe to call again
    data = bytes(range(256)) * 4001                                          # ~1 MB of every byte value
    k = _key("original.bin")
    uri = store.put(k, data, "application/octet-stream")
    assert store.get(k) == data and store.key_from_uri(uri) == k and store.uri(k) == uri
    store.put(k, b"replaced")
    assert store.get(k) == b"replaced"                                       # the same key overwrites
    store.put(_key("empty"), b"")
    assert store.ping() is True


def test_object_store_contract_a_missing_object_is_a_keyerror(store):
    store.ensure()
    with pytest.raises(KeyError):
        store.get(_key("never-written"))


def test_a_second_object_store_adapter_serves_the_existing_ingest_code(monkeypatch, tmp_path):     # SW-S6 AC1, SW-S1 AC2
    """The module API every caller uses (put_bytes / get_bytes / object_uri / key_from_uri / ping) now runs on a folder."""
    from cdi_adapter import storage

    monkeypatch.setattr(settings, "object_store", "filesystem")
    monkeypatch.setattr(settings, "object_store_dir", str(tmp_path / "objs"))
    storage.ensure_bucket()
    key = f"documents/ab/{uuid.uuid4().hex}/original.pdf"
    uri = storage.put_bytes(key, b"%PDF-scan bytes", "application/pdf")
    assert uri == f"file://{key}" and storage.key_from_uri(uri) == key and storage.get_bytes(key) == b"%PDF-scan bytes"
    assert storage.ping() is True
    with pytest.raises(NotImplementedError):
        storage.presign_get(key)                                              # said plainly, not a silent wrong answer


@pytest.mark.parametrize("bad", ["../escape", "a/../../etc/passwd", "/abs/path", "a//b", "a/./b", "", "a\x00b", "..\\x"])
def test_the_folder_store_never_leaves_its_folder(tmp_path, bad):
    st = FilesystemStore(tmp_path / "objs")
    with pytest.raises(ValueError):
        st.put(bad, b"x")
    with pytest.raises(ValueError):
        st.get(bad)


def test_the_folder_store_refuses_a_symlink_that_points_out(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "objs"
    root.mkdir()
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privilege on this system")
    with pytest.raises(ValueError, match="leaves the store"):
        FilesystemStore(root).get("link/secret.txt")


def test_a_failed_write_leaves_no_half_object_and_no_temp_file(tmp_path, monkeypatch):
    st = FilesystemStore(tmp_path / "objs")
    st.put("a/b.bin", b"old")
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        st.put("a/b.bin", b"new-data-that-must-not-appear")
    monkeypatch.undo()
    assert st.get("a/b.bin") == b"old" and [p.name for p in (tmp_path / "objs" / "a").iterdir()] == ["b.bin"]


# ------------------------------------------------------------------ job queue: Redis or Valkey, same protocol
class _FakeRedis:
    def __init__(self, info): self._info = info
    def ping(self): return True
    def info(self, section): return self._info


@pytest.mark.parametrize("name,info,server", [
    ("redis", {"redis_version": "7.0.15"}, "redis 7.0.15"),
    ("valkey", {"redis_version": "7.2.4", "valkey_version": "8.0.1"}, "valkey 8.0.1")])
def test_queue_contract_redis_and_valkey_answer_the_same_calls(monkeypatch, name, info, server):
    import redis

    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url, **kw: _FakeRedis(info)))
    monkeypatch.setattr(settings, "queue_backend", name)
    swap.reset("job_queue")
    q = swap.resolve("job_queue")
    assert q.name == name and q.ping() is True and q.server() == server
    swap.reset("job_queue")


def test_a_queue_that_is_down_reports_down_not_an_exception(monkeypatch):
    import redis

    def boom(cls, url, **kw): raise ConnectionError("refused")
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(boom))
    swap.reset("job_queue")
    assert swap.resolve("job_queue").ping() is False
    swap.reset("job_queue")


# ------------------------------------------------------------------ terminology: licensed code systems only (SW-S6 AC3)
def test_without_a_snomed_or_icd_licence_a_value_keeps_its_text_and_the_canonical_term_but_no_external_code(monkeypatch):
    from cdi_adapter.terminology import service as T

    monkeypatch.setattr(settings, "licensed_code_systems", ("LOINC", "UCUM"))
    up = T.bind_fact({"fact_type": "condition", "local_text": "type 2 diabetes mellitus", "value_code_display": None})
    assert up["code_system"] is None and up["code"] is None and up["code_status"] == "local_only"
    lab = T.bind_fact({"fact_type": "lab_result", "local_text": "HbA1c", "value_unit_ucum": None})
    assert lab["code_system"] == "http://loinc.org" and lab["code"] == "4548-4"       # LOINC is free: kept


def test_a_customer_with_the_licence_gets_the_code_back(monkeypatch):
    from cdi_adapter.terminology import service as T

    monkeypatch.setattr(settings, "licensed_code_systems", ("LOINC", "UCUM", "SNOMED"))
    up = T.bind_fact({"fact_type": "condition", "local_text": "type 2 diabetes mellitus", "value_code_display": None})
    assert up["code_system"] == "http://snomed.info/sct" and up["code_status"] == "bound"


def test_the_term_that_matched_is_kept_when_the_code_is_dropped(monkeypatch):
    from cdi_adapter.terminology import service as T

    monkeypatch.setattr(settings, "licensed_code_systems", ("LOINC", "UCUM"))
    up = T.bind_fact({"fact_type": "condition", "local_text": "type 2 diabetes mellitus", "value_code_display": None})
    assert up.get("code_display")                                                         # the canonical term stays
    assert T.system_short("http://snomed.info/sct") == "SNOMED" and T.system_short("http://hl7.org/fhir/sid/icd-10") == "ICD10"
    assert {"SNOMED", "ICD10"}.isdisjoint(swap.resolve("terminology").code_systems())
