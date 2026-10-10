# The OOM fallback model: Qwen2-VL-7B-Instruct

**The main OCR/vision model is Qwen2.5-VL-7B-Instruct** (`CDI_VLM_MODEL_ID`). It reads every
handwritten line crop and does S4 extraction. The fallback
(`CDI_VLM_FALLBACK_MODEL_ID`, default **Qwen2-VL-7B-Instruct**) is loaded by the gateway
(`mlserve`, `hf` backend) only when the primary fails to load (out of memory), so work is never
dropped. It replaced Qwen2.5-VL-3B-Instruct (non-commercial licence).

Pipeline for handwriting (unchanged): pdfium (PDF) / Pillow (images, EXIF-upright) -> OpenCV
normalise -> printed lines: RapidOCR; handwritten / mixed lines: Qwen per crop ->
evidence -> S4 extraction -> S6 gate.

## What the change costs, and what the code does about it

1. **Memory.** The fallback is as large as the primary, so in bf16 it would run out of memory in
   exactly the situation it exists for. It loads in **8-bit** (`CDI_VLM_FALLBACK_QUANTIZE=8bit`,
   bitsandbytes LLM.int8, about 9 to 11 GB). `start_mlserve.sh` installs bitsandbytes best effort.
   The fallback loads only **after** the failed primary load's exception has ended and its memory
   has been released (`HFQwenVLBackend._ensure/_release`): inside the `except` block the traceback
   still holds the half-built primary and the fallback would run out of memory too.
2. **Accuracy and hallucination.** An older model generation, quantised, not benchmarked on
   handwritten prescriptions. The gateway names the model in every response; the client records it
   (`vlm_generate_ex`, `vlm_json_ex`), so provenance is exact:
   - S4: `pipeline_run.model_version` of the extract run (`repo.set_run_model_version`);
   - each Qwen line reading: `Reading.engine_version` = the model that answered, and the block's
     `recognition.fallback_model`.
   S6 (`validate/service.py: fallback_findings`) adds a **blocker `fallback-model`** to every fact
   whose extraction or any source line the fallback read (`CDI_GATE_FALLBACK_REVIEW=true`), so it
   is never auto-accepted. It is a floor under the existing gate.
3. **Security.** Every request carries a fixed system message (`mlserve/backends.py:
   SYSTEM_NOTICE`): the page image and the OCR text are untrusted data, never instructions, and an
   unreadable value is left out or null. Defence in depth: the model has no tools and no database
   access; its output is schema-checked, grounded against the pixels and gated.
4. **No patient value in a log or an error.** `vlm_json_retry` / `vlm_json_partial` log where
   validation failed (`error_at`), not the quoted value; `MLError` for a non-JSON answer says how
   many characters, not what they were; the final `vlm_json failed` error names the location only.

## NOT verified yet (needs a GPU pod)

- bitsandbytes LLM.int8 on the pod's Blackwell GPU, and the 8-bit fallback's latency.
- Qwen2-VL-7B accuracy on real handwritten prescriptions vs the primary.
- Whether the system message changes the primary's accuracy (expected: no, unmeasured).
- Whether the bbox/grounding conventions of Qwen2-VL differ from Qwen2.5-VL (not used by the
  current line-crop engine, which sends already-cropped lines).

## Promoting the fallback

Do not set `CDI_GATE_FALLBACK_REVIEW=false` until a benchmark on de-identified real prescriptions
(with a clinician's answer key) shows field-level parity with the primary: drug, strength, dose,
frequency, duration, the hallucination rate, and p50/p95 latency.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CDI_VLM_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | main model |
| `CDI_VLM_FALLBACK_MODEL_ID` | `Qwen/Qwen2-VL-7B-Instruct` | loaded only on a failed (OOM) load |
| `CDI_VLM_FALLBACK_QUANTIZE` | `8bit` | `8bit` or empty (bf16); `hf` backend only |
| `CDI_GATE_FALLBACK_REVIEW` | `true` | every fact the fallback read goes to review |
