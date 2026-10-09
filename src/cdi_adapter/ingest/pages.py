"""Page rendering + image normalization.

- PDFs -> one PNG per page at a target DPI (pypdfium2 / PDFium, Apache-2.0 / BSD-3).
- Images -> a single normalized PNG (Pillow; EXIF orientation applied, never through PDFium).
- Normalization: grayscale -> deskew (min-area-rect on ink pixels) -> optional
  denoise -> CLAHE contrast. The ORIGINAL bytes are never altered; these are
  derived render artifacts used by OCR/VLM downstream.
"""
from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageOps

from .._cpu import THREADS_PER_TASK
from ..config import settings

try:
    cv2.setNumThreads(THREADS_PER_TASK)
except Exception:  # noqa: BLE001
    pass
from ..logging import get_logger

log = get_logger(__name__)

# A PDF page that would need more pixels than this at the render DPI is refused instead of
# allocating gigabytes (a 14400 x 14400 pt page at 200 dpi is ~1.6 gigapixels). A0 at 200 dpi is
# ~62 megapixels, so real documents are never affected.
MAX_PDF_PAGE_PIXELS = 100_000_000

# PDFium is not thread-safe (pypdfium2 README): the API, the worker pool and Celery all ingest on
# threads, so every call into the library is serialised by this lock. PNG encoding is the slow
# part and happens outside it.
_PDFIUM_LOCK = threading.Lock()

# Intermediate page PNGs are read once by normalize_with_source() and thrown away, so they are
# written at the fastest zlib level (measured on a local PC: level 6 / 1 = 696 / 240 ms per page).
INTERMEDIATE_PNG_LEVEL = 1


class PdfReadError(ValueError):
    """The PDF is corrupt, password-protected, zero-sized or too large to render.
    A property of the file, so retrying cannot help (the listener classifies it as data)."""


@dataclass
class RenderedPage:
    page_no: int
    png_bytes: bytes             # normalised (gray, denoised, CLAHE) - what RapidOCR/classify read
    width_px: int
    height_px: int
    dpi: int
    preproc: dict[str, Any] = field(default_factory=dict)
    # pixel-faithful colour render in the SAME coordinates as png_bytes (deskew applied, no
    # filtering). Line crops and grounding are cut from this - never from the filtered image.
    src_png: bytes | None = None


def _pil_to_png(img: Image.Image, compress_level: int = 6) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def _np_to_png(arr: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    if not ok:  # pragma: no cover
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def _row_score(ink: np.ndarray, angle: float) -> float:
    h, w = ink.shape
    r = cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h), flags=cv2.INTER_NEAREST)
    return float(np.var(r.sum(axis=1, dtype=np.float64)))


def _estimate_skew_deg(gray: np.ndarray) -> float:
    """Angle (deg) to rotate the page so text lines become horizontal.

    Projection profile: the angle at which the rows of ink are most uneven is the angle at which the
    text lines are level. It looks at the text only (the outer 2 % frame is ignored, so a scanner's
    dark border or a page edge cannot decide it) and does not depend on how a given OpenCV version
    reports ``minAreaRect`` angles. The earlier fit of a rectangle to ALL ink pixels was fooled by
    any such edge: on a page cut out of a photo it turned a level page 4 degrees."""
    h, w = gray.shape[:2]
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ink = cv2.threshold(cv2.bitwise_not(small), 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1].astype(np.uint8)
    b = max(2, int(min(ink.shape) * 0.02))
    ink[:b], ink[-b:], ink[:, :b], ink[:, -b:] = 0, 0, 0, 0
    if int(ink.sum()) < 200:
        return 0.0
    limit = settings.max_deskew_deg + 5.0
    coarse = np.arange(-limit, limit + 0.01, 0.5)
    best = max(coarse, key=lambda a: _row_score(ink, float(a)))
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    best = max(fine, key=lambda a: _row_score(ink, float(a)))
    if _row_score(ink, float(best)) <= 1.02 * _row_score(ink, 0.0):      # no clearer than level: leave it
        return 0.0
    return float(round(best, 2))


def _rotation_matrix(w: int, h: int, angle_deg: float) -> np.ndarray:
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)


def _rotate(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = arr.shape[:2]
    m = _rotation_matrix(w, h, angle_deg)
    return cv2.warpAffine(
        arr, m, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _hold(meta: dict[str, Any], code: str, message: str) -> None:
    """Add a blocking reason (and its code) to the page's quality record: the page is held for a retake."""
    q = meta.get("quality")
    if q is None:                                           # the quality check is switched off: nothing to hold on
        return
    q["passed"] = False
    q.setdefault("reasons", []).append(message)
    q.setdefault("reason_codes", []).append(code)


def _warn(meta: dict[str, Any], code: str, message: str) -> None:
    """Record a non-blocking finding on the page's quality record: the page still goes on."""
    q = meta.get("quality")
    if q is None:
        return
    q.setdefault("warnings", []).append(message)
    q.setdefault("warning_codes", []).append(code)


def _straighten(arr: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """IM-S2: cut the page out of a photo, turn it upright, record every step as one matrix.

    Order matters: perspective first (so the page is a clean rectangle), then upright (a sideways
    page is judged on the page alone, not on a photo's background), then the small deskew. Each step
    changes nothing unless it is sure; what is unsure is recorded or held, never guessed."""
    from ..recognition.quality import ORIENTATION_UNCERTAIN, PAGE_EDGES_NOT_FOUND
    from . import geometry as G

    h0, w0 = arr.shape[:2]
    t = G.new_transform(w0, h0)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    is_photo = G.photo_like(gray)

    if settings.perspective_enabled:
        quad = G.find_page_quad(gray)
        if quad is None and is_photo and not G.page_fills_frame(gray):
            quad = G.find_page_quad_loose(gray)      # a busy / partly bright background (patterned bedding)
        if quad is not None and G.quad_moves_enough(quad, w0, h0):
            arr, m = G.rectify(arr, quad)
            G.add_step(t, {"op": "perspective", "quad": [[round(float(x), 1) for x in p] for p in quad]}, m,
                       (arr.shape[1], arr.shape[0]))
            meta["steps"].append("perspective")
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif quad is None and is_photo and not G.page_fills_frame(gray) and (cut := G.page_cutout(arr)) is not None:
            # no clean corners (the page runs off the frame, a fold, a shadow): cut the page out by its colour, paint
            # the floor / bedspread / hand white and crop to the page, so nothing but the prescription goes on
            arr, (x0, y0, x1, y1), info = cut
            m = G.affine3(np.array([[1, 0, -x0], [0, 1, -y0]], float))
            G.add_step(t, {"op": "crop_to_page", "box": [x0, y0, x1, y1], **info}, m, (x1 - x0, y1 - y0))
            meta["steps"] += ["crop_to_page", "mask_background"]
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif quad is None and is_photo and not G.page_fills_frame(gray):
            # the picture is still readable, so it is never refused: crop to the writing so the background is
            # not read, and flag it (the document is marked "needs check" in the result). A page that fills
            # the frame has nothing to cut away and is not flagged.
            box = G.find_content_box(gray)
            if box is not None:
                x0, y0, x1, y1 = box
                arr = np.ascontiguousarray(arr[y0:y1, x0:x1])
                m = G.affine3(np.array([[1, 0, -x0], [0, 1, -y0]], float))
                G.add_step(t, {"op": "crop_to_content", "box": [x0, y0, x1, y1]}, m, (x1 - x0, y1 - y0))
                meta["steps"].append("crop_to_content")
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            _warn(meta, PAGE_EDGES_NOT_FOUND,
                  "the exact page edges could not be found, so the picture was cropped to the writing - "
                  "check the result, or retake it on a plain background" if box is not None else
                  "the edges of the page could not be found, so the whole picture was read as it is - "
                  "check the result, or retake it on a plain background")

    def turn(arr: np.ndarray, k: int, info: dict[str, Any]) -> np.ndarray:
        """Turn the page by ``k`` quarter turns (numpy.rot90) and record it in the page transform."""
        h, w = arr.shape[:2]
        out = np.ascontiguousarray(np.rot90(arr, k))
        op = "rotate180" if k == 2 else "rotate90"
        G.add_step(t, {"op": op, **({"k": k} if k != 2 else {}), **info}, G.rot90_matrix(w, h, k), (out.shape[1], out.shape[0]))
        meta["steps"].append(op)
        return out

    # Which way up? The PRINTED text is the witness (ingest/orient_ocr.py): the page is read at the four turns and the turn where the
    # print reads best wins. For a phone photo it is asked first (the ink-shape rule is fooled by a photo's background); for a flat
    # scan it checks the ink-shape rule when that rule wants to turn the page.
    from . import orient_ocr

    voted: tuple[int | None, dict[str, Any]] | None = None
    # The ink-shape rule is only trusted when it is decisive (upright pages score below 0.5, sideways ones above ``quality_sideways_ratio``). In
    # between, and for a page not judged a photo, the PRINTED text decides. MEASURED on a real sideways photo that was not judged a photo: the ink
    # rule said 1.2 (undecided), the printed-text vote was never asked, the page was read sideways and every word came out as garbage; asked, the
    # vote said "turn once" (print read 278 against 148 / 23 / 23).
    ratio0 = _orientation_ratio(gray)
    ambiguous = ratio0 is None or 0.5 <= ratio0 <= settings.quality_sideways_ratio
    if settings.orient_enabled and settings.orient_ocr_check and (is_photo or ambiguous):
        voted = orient_ocr.vote(arr)
        meta["orientation_vote"] = voted[1]
    if voted is not None and voted[0] is not None:
        if voted[0]:
            arr = turn(arr, voted[0], {"by": "printed text", **voted[1]})
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # a photo's background fools the axis test: judge orientation on a flat scan or a page cut out of a photo
    elif settings.orient_enabled and (not is_photo or "perspective" in meta["steps"] or "crop_to_page" in meta["steps"]):
        ratio = _orientation_ratio(gray)
        if ratio is not None and ratio > settings.quality_sideways_ratio:          # lines run vertically
            k, info = G.decide_sideways(gray)
            if settings.orient_ocr_check:                                          # the rule knows the page is SIDEWAYS; the print says which way
                voted = orient_ocr.vote(arr, candidates=(1, 3))
                meta["orientation_vote"] = voted[1]
                if voted[0] is not None:
                    k, info = voted[0], {"by": "printed text", **voted[1]}
            if k is None:
                _hold(meta, ORIENTATION_UNCERTAIN, "the page is sideways but which way is up could not be decided "
                      "- please retake it with the page upright")
                t["steps"].append({"op": "sideways_undecided", **info})
            else:
                arr = turn(arr, k, info)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif settings.orient_upside_down:
            score, n = G.upright_score(gray)
            if score is not None and score < settings.upside_down_threshold:
                flip = True
                if settings.orient_ocr_check:                                      # the rule says upside down; the print confirms or overrules
                    voted = orient_ocr.vote(arr, candidates=(0, 2))
                    meta["orientation_vote"] = voted[1]
                    flip = voted[0] != 0                                           # decided "upright": do not turn it
                if flip:
                    arr = turn(arr, 2, {"upright_score": round(score, 4), "lines": n})
                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr, t


def _orientation_ratio(gray: np.ndarray) -> float | None:
    from ..recognition.quality import orientation_ratio

    return orientation_ratio(gray)


def normalize_image(png_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return (normalized_png, preproc_metadata)."""
    norm, _src, meta = normalize_with_source(png_bytes)
    return norm, meta


def normalize_with_source(png_bytes: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    """Return (normalized_png, source_png, meta). ``source_png`` is the colour render with the
    same deskew rotation but no denoise/CLAHE, so its pixels line up with OCR bboxes.
    ``meta['quality']`` is the E2-S12 gate verdict measured on the UNPROCESSED render."""
    from ..recognition.quality import assess

    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode page image")

    meta: dict[str, Any] = {"steps": []}
    if settings.quality_gate_mode != "off":
        meta["quality"] = assess(arr).as_dict()
    arr, transform = _straighten(arr, meta)
    from . import enhance

    f = enhance.upscale_factor(*arr.shape[:2])
    if f > 1.001:                                  # a small photo: enlarge it (bicubic) before it is read; recorded in the transform
        from . import geometry as G

        arr = enhance.upscale(arr, f)
        G.add_step(transform, {"op": "upscale", "factor": round(f, 3)}, G.affine3(np.array([[f, 0, 0], [0, f, 0]], float)),
                   (arr.shape[1], arr.shape[0]))
        meta["steps"].append("upscale")
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    skew = 0.0
    if settings.deskew_enabled:
        skew = _estimate_skew_deg(gray)
        if abs(skew) > 0.15 and abs(skew) <= settings.max_deskew_deg:
            from . import geometry as G

            hh, ww = gray.shape[:2]
            m23 = _rotation_matrix(ww, hh, skew)
            gray = _rotate(gray, skew)
            arr = _rotate(arr, skew)
            G.add_step(transform, {"op": "deskew", "angle_deg": round(skew, 3)}, G.affine3(m23), (ww, hh))
            meta["steps"].append("deskew")
    meta["skew_deg"] = round(skew, 3)
    meta["transform"] = transform                  # original render -> straightened copy (IM-S2)

    if settings.denoise_enabled:
        gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
        meta["steps"].append("denoise")

    gray, arr, sinfo = enhance.sharpen(gray, arr)    # soft pictures only, checked; the verdict is kept in the page meta
    meta["sharpen"] = sinfo
    if sinfo["applied"]:
        meta["steps"].append("sharpen")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    meta["steps"].append("clahe")

    h, w = gray.shape[:2]
    meta["width_px"], meta["height_px"] = int(w), int(h)
    return _np_to_png(gray), _np_to_png(arr), meta


def _upright(img: Image.Image) -> Image.Image:
    """Apply the EXIF Orientation tag. A phone photo of a prescription is often stored sideways
    with only the tag saying so; the OCR models read the raw pixels and never see the tag.
    Best effort: a damaged EXIF block leaves the pixels as they are."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        return img


def _open_pdf(raw: bytes) -> pdfium.PdfDocument:
    """Open ``raw`` (caller holds the PDFium lock); unreadable input becomes ``PdfReadError``."""
    try:
        doc = pdfium.PdfDocument(raw)
    except pdfium.PdfiumError as exc:
        # pypdfium2 4.x sets no usable error code for this case: fall back to the message text
        if (
            getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD
            or "password" in str(exc).lower()
        ):
            raise PdfReadError("PDF is password-protected") from exc
        raise PdfReadError(f"cannot open PDF: {exc}") from exc
    try:
        if doc.get_formtype() != pdfium_c.FORMTYPE_NONE:
            doc.init_forms()  # draw filled AcroForm / XFA field values onto the page image
    except pdfium.PdfiumError:
        pass  # form drawing is best effort: the page content itself still renders
    return doc


def _render_pdf_page(doc: pdfium.PdfDocument, index: int, scale: float) -> Image.Image:
    """Render page ``index`` (0-based) at ``scale`` (caller holds the PDFium lock)."""
    page = doc[index]
    try:
        width, height = page.get_size()  # points, after the page's own /Rotate
        if width <= 0 or height <= 0:
            raise PdfReadError(f"PDF page {index + 1} has no size")
        pixels = math.ceil(width * scale) * math.ceil(height * scale)
        if pixels > MAX_PDF_PAGE_PIXELS:
            raise PdfReadError(
                f"PDF page {index + 1} is too large to render ({pixels:,} pixels at this DPI)"
            )
        bitmap = page.render(scale=scale)
        try:
            return bitmap.to_pil().convert("RGB")  # convert() returns a copy: safe after close()
        finally:
            bitmap.close()
    except pdfium.PdfiumError as exc:
        raise PdfReadError(f"cannot render PDF page {index + 1}: {exc}") from exc
    finally:
        page.close()


def render_pdf_pngs(raw: bytes, dpi: int) -> list[bytes]:
    """One intermediate PNG per page, by the renderer ``CDI_PDF_RENDERER`` names (swap point ``pdf_renderer``)."""
    from .. import swap

    return swap.resolve("pdf_renderer").render_pngs(raw, dpi)


class PdfiumRenderer:
    """The pypdfium2 adapter behind the ``PdfRenderer`` interface (the only renderer: the AGPL one is not an option)."""

    name = "pypdfium2"

    def render_pngs(self, raw: bytes, dpi: int) -> list[bytes]:
        return _pdfium_render_pngs(raw, dpi)


def _pdfium_render_pngs(raw: bytes, dpi: int) -> list[bytes]:
    """One intermediate PNG per page. A PDF with more than ``max_pages`` pages is REFUSED (LS-S3), never
    cut short: a half-read prescription would look complete. A page is ``ceil(points * dpi / 72)``
    pixels on each side."""
    # PDF user space is 72 units per inch. The tiny factor stops float rounding from adding a pixel
    # when points * dpi / 72 is an exact integer (US Letter at 150 dpi is 1650 px, not 1651).
    scale = dpi / 72.0 * (1 - 1e-9)
    with _PDFIUM_LOCK:
        doc = _open_pdf(raw)
        count = len(doc)
        if count > settings.max_pages:
            doc.close()
            raise PdfReadError(f"PDF has {count} pages; the limit is {settings.max_pages}. "
                               "Nothing was read: split the file or send fewer pages")
    pngs: list[bytes] = []
    try:
        for index in range(count):
            with _PDFIUM_LOCK:
                image = _render_pdf_page(doc, index, scale)
            # outside the lock; one page in memory at a time
            pngs.append(_pil_to_png(image, INTERMEDIATE_PNG_LEVEL))
    finally:
        with _PDFIUM_LOCK:
            doc.close()
    return pngs


def render_pages(raw: bytes, mime_type: str, *, dpi: int | None = None) -> list[RenderedPage]:
    dpi = dpi or settings.page_dpi
    pages: list[RenderedPage] = []

    if mime_type == "application/pdf" or raw[:5] == b"%PDF-":
        for i, png in enumerate(render_pdf_pngs(raw, dpi)):
            norm, src, meta = normalize_with_source(png)
            meta["source"] = "pdf"
            pages.append(
                RenderedPage(
                    page_no=i + 1,
                    png_bytes=norm,
                    width_px=meta["width_px"],
                    height_px=meta["height_px"],
                    dpi=dpi,
                    preproc=meta,
                    src_png=src,
                )
            )
        return pages

    if mime_type.startswith("image/") or _looks_like_image(raw):
        img = Image.open(io.BytesIO(raw))
        if getattr(img, "n_frames", 1) > 1:  # multi-page TIFF
            if img.n_frames > settings.max_pages:
                raise PdfReadError(f"image has {img.n_frames} pages; the limit is {settings.max_pages}. "
                                   "Nothing was read: split the file or send fewer pages")
            for i in range(img.n_frames):
                img.seek(i)
                frame = _upright(img).convert("RGB")
                norm, src, meta = normalize_with_source(
                    _pil_to_png(frame, INTERMEDIATE_PNG_LEVEL))
                meta["source"] = "tiff"
                pages.append(
                    RenderedPage(i + 1, norm, meta["width_px"], meta["height_px"], dpi, meta, src)
                )
        else:
            frame = _upright(img).convert("RGB")
            norm, src, meta = normalize_with_source(_pil_to_png(frame, INTERMEDIATE_PNG_LEVEL))
            meta["source"] = "image"
            pages.append(RenderedPage(1, norm, meta["width_px"], meta["height_px"], dpi, meta, src))
        return pages

    raise ValueError(f"unsupported mime_type for rendering: {mime_type!r}")


def _looks_like_image(raw: bytes) -> bool:
    sigs = (b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"GIF8", b"BM")
    return any(raw.startswith(s) for s in sigs)


def make_thumbnail(png_bytes: bytes, max_side: int = 480) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    img.thumbnail((max_side, max_side))
    return _pil_to_png(img)
