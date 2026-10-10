# Reading engines working together (epic RD): status against the code

MEASURED = run here (local PC, synthetic). UNVERIFIED = not run. Nothing below was measured on real
prescription handwriting.

| Story | Status | Notes |
|---|---|---|
| **RD-S1** printed text (RapidOCR) | built before; unchanged | `ocr/rapid.py`, `ocrhost`; line evidence is append-only (`record_rapid_observations`, DB triggers). Clean synthetic printed text read at >= 97 % of characters down to an 8 px font (MEASURED, `recognition` sweep, docs/image-preparation.md). Real printed prescriptions: unmeasured |
| **RD-S2** Qwen reads every handwritten line | built; the second recognizer was **removed** | Qwen2.5-VL is the one handwriting reader (RapidOCR reads print). Every handwritten line is a single reading, so a person checks it; nothing is auto-accepted. `recognition/scoring.py` holds the error-rate helpers (CER, WER, critical-value accuracy, Wilson lower bound) used by the evaluation scorer and the measurement scripts. No second model is downloaded or run |
| **RD-S3** disagreement, grounding, second opinion | built before; two gaps **closed** | compare / grounding / hierarchy / advisory adjudication unchanged. **Self-consistency:** `CDI_QWEN_SELF_CONSISTENCY` (off by default: it doubles the Qwen calls) reads each handwriting crop again with different padding; a mismatch turns an `agree` line into `disagree` and the second reading is kept as evidence (`qwen2.5-vl-b`, its own crop hash). **Drift alarm:** the share of disagreeing lines is stored with every run and compared with earlier runs (`recognition/drift.py`, `CDI_DRIFT_*`); a jump logs `disagreement_drift_alarm`. Thresholds are placeholders, not tuned on real data |
| **RD-S4** Qwen fills the form | built before; gaps **closed** except two | see docs/extraction-fields.md: only-this-page and page-is-data prompt policy, instruction-like page text flagged, `Page N:` labels, cut-off answers marked incomplete, nulls. The out-of-memory fallback is **Qwen2-VL-7B in 8-bit with every fact held for review** (docs/fallback-model.md): the story text still says "queue and alert, 3B removed", which is the older decision. Still open: the maximum-pixels sweep (needs the GPU) and "quote first" retraction of a value whose cited line does not contain it (numbers are already grounded against the page by `grounding.py`) |
| **RD-S5** ten at once in one minute | **not built, not measurable here** | needs a GPU pod: the vLLM backend is dormant, no 10-parallel run exists. Nothing in this repo changes that |

Not built anywhere in the repo: the per-doctor layer (DoctorNode, vocabulary priors, exemplar memory,
novelty detector, correction-fed profiles). See the architecture doc §15.12-15.14 (HW-Phases C-E).
