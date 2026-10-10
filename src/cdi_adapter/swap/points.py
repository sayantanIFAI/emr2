"""Every swap point and its choices (SW-S1, SW-S6). Importing this registers them; nothing else knows the
implementation names. To add an alternative: write the adapter, pass its contract test
(tests/test_contracts_unit.py), then register it here."""
from __future__ import annotations

from typing import Any, Protocol

from . import choice, point


# ----------------------------------------------------------------------------- interfaces
class PdfRenderer(Protocol):
    name: str

    def render_pngs(self, raw: bytes, dpi: int) -> list[bytes]:
        """One PNG per page, ``ceil(points * dpi / 72)`` pixels on each side. Raises ``PdfReadError`` (a
        ``ValueError``) for a corrupt, protected or too-long PDF; never returns a partial document."""


class ObjectStore(Protocol):
    name: str

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store ``data`` under ``key`` (overwrite = the same key). Returns the stored object's URI."""

    def get(self, key: str) -> bytes:
        """The exact bytes stored under ``key``; ``KeyError`` when there is no such object."""

    def uri(self, key: str) -> str: ...

    def key_from_uri(self, uri: str) -> str: ...

    def ping(self) -> bool: ...

    def ensure(self) -> None:
        """Create what is needed to store objects (a bucket / a folder); safe to call again."""


class TerminologyProvider(Protocol):
    name: str

    def code_systems(self) -> frozenset[str]:
        """The external code systems this provider may attach to a result (a licence decision)."""


class JobQueue(Protocol):
    name: str

    def ping(self) -> bool: ...


# ----------------------------------------------------------------------------- the points
point("pdf_renderer", setting="CDI_PDF_RENDERER", attr="pdf_renderer",
      doc="turns a PDF into one picture per page (pypdfium2: Apache-2.0 / BSD-3; no AGPL renderer)")
point("object_store", setting="CDI_OBJECT_STORE", attr="object_store",
      doc="where the original documents and page pictures are kept (S3 API: SeaweedFS, cloud S3 ... | a folder)")
point("job_queue", setting="CDI_QUEUE_BACKEND", attr="queue_backend",
      doc="the queue behind background jobs (Redis 7.2 or older, or Valkey: same protocol)")
point("printed_ocr", setting="CDI_OCR_ENGINE", attr="ocr_engine", doc="reads printed text lines")
point("vlm", setting="CDI_MLSERVE_BACKEND", attr="mlserve_backend",
      doc="the vision-language model server: stub (tests) | hf (transformers) | vllm")
point("terminology", setting="CDI_TERMINOLOGY_PROVIDER", attr="terminology_provider",
      doc="the vocabulary results are coded with: the internal one by default; external code systems only where licensed")
point("drive_connector", setting="CDI_LISTENER_CONNECTOR", attr="listener_connector",
      doc="the drive the file listener watches (OneDrive | SharePoint | Google Drive | local | a custom class)",
      custom_ok=True)
point("signin", setting="CDI_SIGNIN_PROVIDER", attr="signin_provider",
      doc="who may use the web app: one admin password today; an OIDC provider plugs in here")
point("metrics_sink", setting="CDI_METRICS_SINK", attr="metrics_sink",
      doc="where operational numbers go (Prometheus text endpoint; no dashboard is bundled)")


# ----------------------------------------------------------------------------- factories (lazy: no import until used)
def _pdfium() -> PdfRenderer:
    from ..ingest.pages import PdfiumRenderer

    return PdfiumRenderer()


def _s3() -> ObjectStore:
    from ..storage import S3Store

    return S3Store()


def _fs() -> ObjectStore:
    from ..storage_fs import FilesystemStore

    return FilesystemStore()


class _RedisProtocolQueue:
    """Redis and Valkey speak the same protocol: one adapter, two names."""

    def __init__(self, name: str) -> None:
        self.name = name

    def ping(self) -> bool:
        import redis

        from ..config import settings

        try:
            return bool(redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping())
        except Exception:  # noqa: BLE001
            return False

    def server(self) -> str | None:
        """``'redis 7.0.15'`` / ``'valkey 8.0.1'`` as the server reports itself (None when unreachable)."""
        import redis

        from ..config import settings

        try:
            info: dict[str, Any] = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).info("server")
        except Exception:  # noqa: BLE001
            return None
        if info.get("valkey_version"):
            return f"valkey {info['valkey_version']}"
        return f"redis {info.get('redis_version')}"


class _InternalTerminology:
    name = "internal"

    def code_systems(self) -> frozenset[str]:
        from ..config import settings

        return frozenset(s.upper() for s in settings.licensed_code_systems)


class _SignIn:
    def __init__(self, name: str) -> None:
        self.name = name


class _Metrics:
    name = "prometheus"


choice("pdf_renderer", "pypdfium2", _pdfium, requires=("pypdfium2",), install='pip install "pypdfium2>=4.30,<6"', dist="pypdfium2")
choice("object_store", "s3", _s3, requires=("boto3",), install="pip install boto3", dist="boto3")
choice("object_store", "filesystem", _fs, note="a folder on an encrypted volume; no server needed")
choice("job_queue", "redis", lambda: _RedisProtocolQueue("redis"), requires=("redis",), install="pip install redis", dist="redis")
choice("job_queue", "valkey", lambda: _RedisProtocolQueue("valkey"), requires=("redis",), install="pip install redis", dist="redis")
choice("printed_ocr", "rapidocr", lambda: __import__("cdi_adapter.ocr.rapid", fromlist=["x"]),
       requires=("rapidocr_onnxruntime",), install='pip install ".[ocr]"', dist="rapidocr-onnxruntime")
choice("printed_ocr", "none", lambda: None, note="no printed-text reader: every line goes to the vision-language model")
choice("vlm", "stub", lambda: None, note="canned answers for tests")
choice("vlm", "hf", lambda: None, requires=("transformers", "torch"), install='pip install ".[ml]"', dist="transformers")
choice("vlm", "vllm", lambda: None, requires=("httpx",), dist="httpx", note="a separate `vllm serve` process")
choice("terminology", "internal", lambda: _InternalTerminology())
choice("drive_connector", "local", lambda: None)
choice("drive_connector", "onedrive", lambda: None, requires=("msal",), install='pip install ".[listener]"', dist="msal")
choice("drive_connector", "sharepoint", lambda: None, requires=("msal",), install='pip install ".[listener]"', dist="msal")
choice("drive_connector", "gdrive", lambda: None, requires=("google.auth",), install='pip install ".[gdrive]"', dist="google-auth")
choice("signin", "basic", lambda: _SignIn("basic"), note="HTTP Basic, one admin (webapp/surface.py)")
choice("signin", "none", lambda: _SignIn("none"), note="no sign-in: refused outside CDI_ENV=dev")
choice("metrics_sink", "prometheus", lambda: _Metrics())
choice("metrics_sink", "none", lambda: None)
