from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override via env vars (prefix ``CDI_``) or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CDI_", extra="ignore", case_sensitive=False
    )

    env: str = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- datastore ---
    database_url: str = "postgresql+psycopg://cdi:cdi@localhost:5432/cdi"
    redis_url: str = "redis://localhost:6379/0"

    # --- object storage: any S3 API (SeaweedFS in the stack, docs/object-store.md) ---
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "cdiadmin"
    s3_secret_key: str = "cdiadminsecret"
    s3_bucket: str = "cdi-documents"
    s3_region: str = "us-east-1"
    s3_use_path_style: bool = True

    # --- ingestion ---
    inbox_dir: str = "./data/inbox"
    processed_dir: str = "./data/processed"
    failed_dir: str = "./data/failed"
    watch_settle_seconds: float = 2.0
    page_dpi: int = 200
    # A file with more pages than this is refused, not cut short (LS-S3). 20 is a PLACEHOLDER: the story
    # assumes about 10 and the real longest prescription is not known; the owner confirms the number.
    max_pages: int = 20
    allowed_mime_prefixes: tuple[str, ...] = ("image/", "application/pdf")

    # --- swap points (SW-S1/S6): one setting picks each implementation; swap/points.py lists the choices ---
    pdf_renderer: str = "pypdfium2"
    object_store: str = "s3"                      # s3 (SeaweedFS / any S3 API) | filesystem
    object_store_dir: str = "./data/objects"      # the folder when object_store=filesystem (an encrypted volume)
    server_licence_enforce: bool = True           # the queue server's version is checked at start-up (compliance/servers.py)
    queue_backend: str = "redis"                  # redis (7.2 or older) | valkey
    terminology_provider: str = "internal"
    # External code systems that may be attached to a result. SNOMED CT and ICD are NOT in the default: they are
    # enabled per customer who holds the licence (SW-S6); without them a value is saved as the text as written plus
    # the canonical term, with no external code. LOINC and UCUM are free to use.
    licensed_code_systems: tuple[str, ...] = ("LOINC", "UCUM")
    signin_provider: str = "basic"
    metrics_sink: str = "prometheus"

    # --- output connector (UP-S3): what a finished document is turned into for downstream ---
    output_connector: str = "json_placeholder"      # the real HIS / EMR contract replaces it (result.v2)
    # where the finished result JSON is also written when a document is validated (OUT-S2). Blank / false =
    # not written: the endpoint still builds it from the database on demand.
    output_dir: str = ""
    output_object_store: bool = False

    # --- correction loop: human corrections -> per-doctor lexicon -> class-C alias ---
    correction_max_len: int = 500
    # a doctor's "X means Y" becomes an alias for that doctor after this many confirmations by
    # reviewers (and only if Y names exactly one concept). PLACEHOLDER: tune on real corrections.
    doctor_alias_min_verified: int = 3
    profile_cache_ttl_s: int = 300        # Redis read cache of a doctor's lexicon; the database stays the source of truth

    # --- upload screen (UP-S1) ---
    # What the web upload accepts, decided from the file's bytes (never its name): JPG, PNG, TIFF, PDF.
    upload_mime_types: tuple[str, ...] = ("application/pdf", "image/png", "image/jpeg", "image/tiff")
    # PLACEHOLDER limits (not measured): sized for phone photos (a few MB) and scanned PDFs, and to
    # bound memory per request (the whole upload is read into memory). Tune on pilot data.
    upload_max_files: int = 10
    upload_max_file_bytes: int = 50_000_000
    upload_max_total_bytes: int = 150_000_000

    # --- image preprocessing ---
    # which way up a page is, decided by reading its PRINTED text at the four turns (ingest/orient_ocr.py); the ink-shape rule stays as
    # the fallback when there is no printed text. A real page was turned the wrong way by the ink rule and then read as garbage.
    orient_ocr_check: bool = True
    orient_ocr_min_score: float = 40.0     # the best turn must have read at least this much real text
    orient_ocr_min_ratio: float = 1.3      # ... and at least this many times the second best
    deskew_enabled: bool = True
    denoise_enabled: bool = True
    # make a small / soft photo easier to read before anything reads it (ingest/enhance.py): enlarge it, then sharpen it by
    # how soft it measures. Both are checked and undone when they would make the picture worse. PLACEHOLDER thresholds until
    # measured on the pilot's own photos.
    # OFF: measured on a real photo, enlarging + sharpening made the model read the patient's surname "Broadway" in 6 of 6 runs
    # (and "Chowdhury" with it off). Switch on only after a labelled set shows it helps.
    enhance_enabled: bool = False
    # which picture of a page the vision model is given for the full-page read and the focused looks: "source" = the colour page
    # (cut out and straightened, no contrast / sharpening change) or "normalized" = the grey, contrast-stretched copy
    model_image: str = "normalized"
    enhance_target_long_side: int = 2000   # a picture shorter than this (long side, px) is enlarged towards it
    enhance_max_scale: float = 2.5
    enhance_soft_below: float = 150.0      # Laplacian variance (at 1000 px) at or under which sharpening is at full strength
    enhance_crisp_above: float = 900.0     # ... at or over which none is applied
    enhance_max_amount: float = 1.0        # unsharp-mask amount at full strength
    enhance_sigma: float = 1.6             # unsharp-mask radius (px, after enlarging)
    enhance_max_new_clipping: float = 0.01  # sharpening may not clip more than 1% extra of the page to pure black / white
    enhance_max_gain: float = 12.0         # a sharpness gain above this x is ringing / halos, not detail: refused
    max_deskew_deg: float = 15.0
    # page geometry (IM-S2). All thresholds are PLACEHOLDERS measured on synthetic pages only.
    orient_enabled: bool = True           # turn a sideways page (90 / 270 degrees) upright when the way up is clear
    orient_upright_margin: float = 0.02   # how much clearer one way must be (ink-centroid score); else hold for retake
    # 180 degrees: needs a clear negative score on mixed-case text. OFF: the signal was measured only on
    # synthetic text and real handwriting may bias it, and a wrong 180 turn would ruin an upright page.
    orient_upside_down: bool = False
    upside_down_threshold: float = -0.008
    perspective_enabled: bool = True      # cut the page out of a photo and flatten it when four clear corners are found
    perspective_min_side_px: int = 400    # smaller pictures are not looked at for a page edge
    perspective_min_area_frac: float = 0.30   # the page must cover at least this share of the picture
    perspective_min_move_frac: float = 0.03   # corners at least this far (share of the diagonal) from the picture's corners
    photo_border_contrast: int = 25       # outer frame vs middle (grey levels): bigger = a photo with a background
    photo_border_texture: int = 25        # ... or a frame this busy (grey-level spread) where a scan's margin is flat

    # --- pipeline / jurisdiction packages ---
    ig_package: str = "nrces.fhir.r4.ndhm#6.5.0"
    terminology_package: str = "in-snomed-loinc-icd10"

    # --- web app (upload UI + FHIR API) ---
    webapp_port: int = 8080   # RunPod proxies external 8081 -> localhost:8080
    # Deployment surface. Default = one admin who uploads images and reads the result JSON.
    # The review / reviewer / correction screens and every FHIR path stay shut until the owner
    # switches them on (webapp/surface.py answers 404 for them; the FHIR builder agent is not
    # started and nothing is queued for it).
    review_ui_enabled: bool = False
    fhir_enabled: bool = False
    # HTTP Basic sign-in in front of the whole web app (except /healthz). Blank = no sign-in, which
    # is accepted only while CDI_ENV=dev: any other environment refuses to start without a password.
    admin_user: str = "admin"
    admin_password: str = ""

    # --- model gateway (cdi_adapter.mlserve) ---
    mlserve_url: str = "http://127.0.0.1:8077"
    mlserve_port: int = 8077
    mlserve_backend: str = "stub"          # stub | hf | vllm
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_fallback_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"   # OOM-only; replaced the 3B (licence)
    # The OOM fallback is as large as the primary: in bf16 it would run out of memory in exactly
    # the situation it exists for, so it loads in 8-bit (bitsandbytes LLM.int8, hf backend only;
    # "" = bf16, for a host that can hold two bf16 7B models in turn). Not measured on the target
    # GPU yet: docs/fallback-model.md.
    vlm_fallback_quantize: Literal["", "8bit"] = "8bit"
    # the model registry gate (compliance/models.json): a model that is not registered, licensed and pinned to an
    # exact revision is refused, and downloaded files are checked against the registered sha256 before use
    model_registry_enforce: bool = True
    model_verify_checksums: bool = True
    vlm_max_pixels_classify: int = 1_000_000
    vlm_max_pixels_ocr: int = 2_000_000   # keep activations modest on a 24 GB card
    vlm_dtype: str = "bfloat16"
    # --- vllm backend (separate `vllm serve` process, OpenAI-compatible) ---
    vllm_url: str = "http://127.0.0.1:8078/v1"
    vllm_model: str = ""                    # blank -> use vlm_model_id
    vllm_timeout_s: float = 240.0
    vllm_guided: bool = True                # token-level JSON-schema decoding (xgrammar).
                                            # adds grammar-mask cost per token; turn off to
                                            # rely on client-side repair + retry instead.
    vllm_retry_on_length: bool = True         # an answer that hits the token limit is retried once (a repetition loop)
    vllm_retry_repetition_penalty: float = 1.2   # 1.05 did not break a loop that wrote the same line 42 times (MEASURED on a real page)
    vllm_guided_api: str = "structured_outputs"   # structured_outputs (vLLM >= 0.12) | guided_json (older: newer ones ignore it)
    vllm_guided_backend: str = "xgrammar"

    # --- future: vLLM OpenAI endpoint for the DSLM / guided decoding ---
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "not-needed-local"
    dslm_model: str = "cdi-dslm"

    # --- OCR ---
    ocr_engine: str = "rapidocr"          # rapidocr | none
    ocr_min_conf: float = 0.30
    handwritten_uses_vlm: bool = True

    # --- recognition v2 (ARCHITECTURE §15): regions -> independent engines -> evidence ---
    recognition_v2: bool = True           # False = legacy page-level VLM transcription (§5 S3)
    # CPU OCR host (RapidOCR). Blank = run RapidOCR in-process.
    ocrhost_url: str = ""                 # e.g. http://127.0.0.1:8079
    ocrhost_port: int = 8079
    ocrhost_timeout_s: float = 120.0
    # RapidOCR on the GPU needs onnxruntime-gpu with a CUDA build that supports the card; if the
    # CUDA provider cannot start it falls back to the CPU and logs which one it is using.
    rapidocr_use_cuda: bool = False

    # crop standard (IM-S3): what every handwriting crop looks like when it reaches a reader.
    # PLACEHOLDERS, not tuned: accuracy by crop size is not measured yet (the size of every crop is
    # recorded with the line so it can be). 0 turns the upscaling off.
    # a region with no thin pen/print strokes, or a background that is not the page's paper colour (fabric, table, hand),
    # is not writing and is not sent to a reader (recognition/nontext.py). PLACEHOLDERS from three real photos.
    nontext_blackhat_threshold: int = 25
    nontext_min_thin_ink: float = 0.03
    nontext_min_paper_overlap: float = 0.30      # share of a region that must lie on the page's paper
    nontext_max_paper_distance: float = 20.0
    nontext_min_height_px: int = 9
    nontext_min_width_px: int = 12
    crop_pad_frac: float = 0.04           # padding on each side, as a share of the box, so strokes are not cut
    crop_pad_min_px: int = 4              # ... and never less than this
    crop_min_height_px: int = 32          # a crop shorter than this is upscaled before it is read
    crop_max_upscale: float = 4.0         # ... by at most this factor; still shorter = flagged below_standard
    # self-consistency (RD-S3): read each handwriting crop a second time with different padding; a
    # mismatch between the two Qwen readings is another disagreement signal. OFF by default: it
    # doubles the Qwen calls per line (latency and GPU cost are not measured yet).
    qwen_self_consistency: bool = False
    self_consistency_pad_frac: float = 0.12
    # disagreement-rate drift alarm: needs this many earlier runs, then alarms when the share of
    # disagreeing lines moves by more than max(sigma x the usual spread, the absolute tolerance)
    drift_min_runs: int = 20
    drift_sigma: float = 3.0
    drift_abs_tolerance: float = 0.15
    qwen_line_mode: str = "crop"          # crop = independent read per line crop | off
    qwen_line_max_tokens: int = 48
    qwen_line_concurrency: int = 1        # lines read at once; 1 for the serial hf gateway, ~12 with vLLM
    # an engine reading is "the same" as another when, after numeric-context
    # normalisation, every number matches exactly AND the text similarity is >= this
    engine_agree_similarity: float = 0.85
    # L8 Qwen adjudication of a disagreement - ADVISORY: orders the two readings for the
    # reviewer, never resolves the disagreement (the fact still goes to review)
    qwen_adjudication_enabled: bool = False
    qwen_adjudication_max_tokens: int = 8

    # --- image quality gate (E2-S12) ---
    quality_gate_mode: str = "enforce"    # enforce = hold for rescan | warn = record only | off
    quality_min_blur_var: float = 25.0    # variance of the Laplacian; sharp 200-dpi scans >> 100
    quality_min_short_side_px: int = 600
    # Text too small to read reliably (IM-S1): the p85 height of a glyph in pixels, measured on the page
    # as sent to the readers. PLACEHOLDER, not tuned on real photos or handwriting. Basis: on clean
    # SYNTHETIC printed text RapidOCR read >= 97% of characters down to a 8 px font (glyph p85 ~ 6 px),
    # so 8 only rejects pictures far below that; handwriting very likely needs more. 0 turns the check off.
    quality_min_text_height_px: int = 8
    # a page looks sideways when the vertical-line score exceeds the horizontal-line score by this
    # factor (synthetic pages: upright 0.08-0.31, turned 3.2-12; tune on real pictures)
    quality_sideways_ratio: float = 2.0
    quality_max_glare_frac: float = 0.25  # share of page area in saturated blobs
    quality_max_dark_frac: float = 0.60   # share of page that is near-black (clipped/underexposed)

    # --- pixel grounding (E4-S4) ---
    grounding_enabled: bool = True
    grounding_margin_frac: float = 0.15
    grounding_min_similarity: float = 0.72

    # --- per-field gate policy (E4-S2/S3). Values are ASSUMED until fitted on adjudicated
    # data; engine disagreement, single-engine handwriting and failed grounding always review.
    gate_policy_enabled: bool = True
    gate_policy_path: str = ""            # optional JSON overriding validate/policy.DEFAULT_POLICY

    # --- practitioner link (E6-S11 / E18) ---
    practitioner_min_link_conf: float = 0.90

    # --- file listener (E16) ---
    listener_connector: str = "local"     # local | onedrive | sharepoint | gdrive | pkg.module:Class
    listener_root: str = "./data/listener"            # local: folder; graph/gdrive: folder path in the drive
    listener_inbox: str = "inbox"         # "." = the root folder itself is the inbox
    listener_processing: str = "processing"
    listener_completed: str = "success"   # logical name stays "completed"
    listener_error: str = "error"
    listener_quarantine: str = "quarantine"
    listener_log: str = "log"             # failure-reason notes only
    listener_poll_seconds: float = 10.0
    listener_batch_size: int = 3          # files picked and processed together as one batch
    listener_batch_wait_seconds: float = 60.0   # a partial batch is flushed after this long (0 = never wait)
    listener_stable_polls: int = 2        # size/etag unchanged across this many polls = upload done
    listener_lease_seconds: int = 900
    listener_max_bytes: int = 50_000_000
    listener_patterns: tuple[str, ...] = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    listener_max_attempts: int = 3        # hard cap, also enforced by a DB CHECK
    listener_pipeline: str = "inline"     # inline = run every stage in the listener | celery
    listener_retry_base_seconds: int = 60 # recovery agent back-off: base * 2^(attempt-1)
    # the drive answers "too many requests" (429) / "busy" with a wait time: wait that long (up to this cap) and
    # try again, up to this many tries; a throttle is never counted as a failed file (LS-S1)
    listener_throttle_attempts: int = 6
    listener_throttle_max_wait_seconds: float = 300.0
    # no good poll for this long = stalled (health view and the alert rule; 300 is an ASSUMPTION, the owner sets it)
    listener_stall_seconds: int = 300
    graph_auth: str = "app"               # app (client credentials, admin consent) | device_code (sign in once)
    graph_scopes: tuple[str, ...] = ("Files.ReadWrite",)   # delegated scopes (device_code only)
    graph_token_cache: str = "./data/graph_token_cache.json"   # device_code: refresh-token cache (keep on a persistent disk)
    graph_tenant_id: str = ""
    graph_client_id: str = ""
    graph_client_secret: str = ""         # prefer certificate auth in production
    graph_cert_path: str = ""
    graph_cert_thumbprint: str = ""
    graph_drive_id: str = ""              # OneDrive: the drive id (or leave blank + graph_user_id)
    graph_user_id: str = ""
    graph_site_id: str = ""               # SharePoint: site id; library resolved to its default drive
    # Google Drive connector (CDI_LISTENER_CONNECTOR=gdrive; pip install '.[gdrive]')
    gdrive_auth: str = "service_account"  # service_account | oauth
    gdrive_credentials_file: str = ""     # service-account key (JSON)
    gdrive_impersonate_user: str = ""     # domain-wide delegation: act as this user
    gdrive_client_id: str = ""            # oauth mode
    gdrive_client_secret: str = ""
    gdrive_refresh_token: str = ""
    gdrive_root_folder_id: str = ""       # listener root folder id (else CDI_LISTENER_ROOT as a path)
    gdrive_drive_id: str = ""             # a shared drive id (optional)

    # --- FHIR builder agent + blob store (E17) ---
    fhir_agent_poll_seconds: float = 5.0
    fhir_agent_max_attempts: int = 3

    # --- downstream screen dispatch (E20) - disabled per target until approved ---
    dispatch_poll_seconds: float = 30.0
    dispatch_max_attempts: int = 3

    # --- throughput ---
    # an interrupted web upload (restart, crash) is picked up again on start; after this many pick-ups a
    # document that still does not finish is parked as an error (OUT-S3)
    # sends accepted per minute (sliding window, per process); 0 = no limit. 30 is a PLACEHOLDER, not measured
    upload_rate_per_minute: int = 30
    resume_on_start: bool = True
    resume_max_attempts: int = 3
    resume_batch: int = 50
    job_max_workers: int = 5              # documents ingested/classified/OCR'd concurrently;
                                         # _cpu.py sizes native thread pools to
                                         # (cpu_budget - 1) / this  (keep them in sync)
    abha_enabled: bool = True             # ask the page for an ABHA / ABDM identity, check it, and match patients on it. Off
                                          # (CDI_ABHA_ENABLED=false) the deployment never asks for, reads or validates one
    medicine_lexicon_path: str = ""          # list of medicine names (scripts/build_medicine_lexicon.py); empty = not used
    indian_codes_dir: str = ""              # CLCI + CDCI indexes (scripts/build_indian_codes.py); empty = not used
    # a second, focused look at the page for tests written WITH the follow-up instruction ("review after 2 wks {HbA1c / FBS
    # / TSH}"): the full-page answer often misses them. One short extra call, only when a follow-up is written.
    followup_second_look: bool = True
    lab_tests_only: bool = False         # OWNER'S DECISION (2026-10): MRI, CT, USG, echo, EEG, ECG, X-ray and OPG are tests of this system and are listed. ON would reject imaging / ECG / EEG entries
                                         # the lab lists do not place (extract/not_lab.py); physiotherapy, results ("MRI Brain - N") and a printed list of services stay rejected either way
    checklist_marks_enabled: bool = True # a PRINTED CHECKLIST page (a menu of tests the doctor strikes, ticks or circles) is read for pen marks, only when the page's text shows such a menu (extract/checklist.py)
    upload_assume_prescription: bool = True   # a page uploaded on the admin screen that the classifier calls 'other' is handled as a prescription (the screen takes prescriptions; a cropped page with no letterhead, name or doctor looks like 'other' and used to skip extraction completely: MEASURED on a real page, 0 tests)
    recall_reroute: bool = True          # a page typed other / operative_note / lab_report whose own text holds ordered tests is handled as a prescription (extract/recall.py)
    extract_compact_answer: bool = False  # OFF: MEASURED on 17 pages it saved 0.3 s and recall fell 81% -> 73% in one run (not settled: run-to-run noise is about 3 points). The main page answer leaves out the evidence lists and the empty system / code fields; the program links the OCR blocks afterwards (extract/evidence.py): fewer tokens written
    skip_second_look_when_agree: bool = False   # OFF: MEASURED no time saved (15.3 s with and without): the text and the answer rarely name exactly the same tests. The enlarged second look is skipped when the page text scan and the main answer already name the same tests
    prefetch_main_call: bool = False     # start the main page call right after classification, in parallel with the line reading (extract/prefetch.py); ON only after it is measured not to lower recall
    age_sex_reread: bool = True          # the age / sex written beside the name ("74/F") is read again from its own crop at three sizes; used only when two readings agree (extract/age_sex.py)
    unseen_ok_p: float = 0.85            # ASSUMPTION (owner sets): a test no line reader saw is still listed when the model, asked directly whether it is written on the page, says yes with at least this probability. MEASURED on the 17 pages: of 8 unseen tests the direct question gave 0.94 / 0.97 / 0.88 for three true ones and 0.12 / 0.07 for two wrong or unreadable ones (0.96 for one wrong)
    list_unseen_tests: bool = False      # OFF: a test that no line reader saw and no second look agrees with (only the whole-page answer said it) is not listed as a test; ON lists it as needs-check
    verify_tests_enabled: bool = True    # each test candidate is put to the page as a yes / no question and the model's probability of yes is kept (extract/verify.py)
    verify_low_p: float = 0.30           # ASSUMPTION (owner sets): below this the test is flagged "check this first". MEASURED on 64 placed tests of 17 pages: all 7 wrong ones were below 0.30, so were 12 of 57 right ones; none of the 45 at or above 0.30 was wrong
    marks_enabled: bool = False          # pen marks on a pre-printed list of tests (extract/marks.py). OFF: MEASURED on one real page it credited a whole
                                         # printed line as marked, missed a tick and took a handwritten result value for a mark; turn on only after it is
                                         # measured on labelled pages
    name_surname_votes: bool = True      # the surname is put to the model as a choice among the readings and the closest common surnames
    second_look_min_views: int = 2       # a test the second look adds must be read in this many DIFFERENT views of the page (one view alone can
                                         # make a test up: "PT / APTT" was read as "PT/INR" by the whole-page view alone, 5 of 6 times)
    second_look_repeats: int = 2         # each focused view is asked this many times and the answers are pooled: the model server does not
                                         # answer the same picture the same way twice, so one unlucky answer must not lose a test
    # the patient's name is read again from its own line at three sizes; readings that disagree are flagged (a name is never final
    # until the front desk confirms it)
    name_reread: bool = True
    name_choice_votes: bool = True       # the first name is also put to the model as a choice among spellings (readings + letter confusions); suggestions only
    # medicines are not what this product is judged on (lab tests are): the model's choice among reference medicine names
    # costs a call per page, so it is off by default
    llm_resolve_medicines: bool = False
    token_unique_per_day: bool = True      # a token number already used TODAY for another mobile number is refused (tokens are issued per day)
    clinic_timezone: str = "Asia/Kolkata"  # what "today" means for the token rule
    upload_require_intake: bool = True     # the token and mobile number must be sent with an upload from the screen
    lab_mapping_db: bool = True            # the lab-name mapping table is read from the database (False: the built-in seed only)
    llm_resolve_enabled: bool = True       # the model may CHOOSE among reference names for a misread test (resolve_llm.py)
    job_max_concurrent: int = 5           # prescriptions ("jobs") READ at once; the next ones wait their turn in the order they were sent
    job_queue_max: int = 10               # prescriptions in flight (being read + waiting); one more is refused with a plain message until one finishes
    fast_classify: bool = True           # try the scored heuristic classifier first; it only
                                         # short-circuits the VLM on an unambiguous, cleanly
                                         # OCR'd page - everything else still goes to the VLM
    extract_retries: int = 1             # VLM extraction re-tries on schema failure (schemas
                                         # were relaxed so a first-pass slip is now rare)
    extract_concurrency: int = 4         # concurrent VLM extract calls in flight (vllm backend
                                         # batches them; 1 = the old serial behaviour for `hf`)
    # What Qwen is asked to WRITE for a prescription / consultation note (the time of the extraction is its output length):
    #   mlp1 = patient details, doctor name / department / designation, the lab tests with their preparation, the diagnoses
    #          (for context), advice and follow-up. Medicines, vitals, registration number, qualification, clinic and
    #          stamp / signature are NOT asked for.
    #   full = everything the schema has (the earlier behaviour).
    extract_profile: str = "mlp1"
    extract_max_tokens_mlp1: int = 800     # a normal answer is 300-600 tokens; a repetition loop is cut after ~9 s, not ~17 s
    extract_per_page: bool = True          # a paper of several pages: each page is read on its own, then the latest dated visit is picked
    extract_max_tokens: int = 1400       # base; long doc types get more (see extract/prompt.py)

    # --- S6 governance gate (fact -> auto_accepted | in_review) ---
    gate_auto_accept_conf: float = 0.985  # >= this AND clean -> auto_accepted
    gate_audit_conf: float = 0.95         # >= this AND clean -> auto_accepted + audit sample
    gate_review_floor: float = 0.85       # < gate_audit_conf -> in_review
    # Anything the OOM fallback model read is never auto-accepted: it is an older model
    # generation, loaded in 8-bit, and not benchmarked on handwritten prescriptions. Turn off
    # only after the benchmark in docs/fallback-model.md shows parity with the primary.
    gate_fallback_review: bool = True
    gate_partial_penalty: float = 0.30    # confidence subtracted when extraction was _partial
    gate_medication_always_review: bool = True   # any med line with missing dose/route/freq -> review
    gate_local_only_review_types: tuple[str, ...] = ("condition", "medication", "allergy", "procedure")
    audit_sample_rate: float = 0.10       # fraction of gate_audit tier pulled for QA


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
