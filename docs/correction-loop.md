# The correction loop: human corrections update the doctor's knowledge instantly, models change only offline

```
human correction tool -> POST /api/corrections -> PostgreSQL (source of truth)
     -> profile updater (doctor_lexicon, class-C kb_alias) -> [optional Redis read cache] -> alias cascade
```

Immediate adaptation = the doctor's lexicon. Neural adaptation = an occasional offline fine-tune from the
exported corrections. A correction never retrains anything and never rewrites a reading.

## What the OCR service emits (`GET /api/documents/{id}/field-records`)

One record per extracted field: `prescription_id` (the document id), `doctor_id`, `field_id` (the fact id),
`field_type`, `raw_crop_reference` (observation ids, crop hashes, boxes), `qwen_value`,
`final_value`, `confidence`, `status` (`accepted` / `needs_review` / `corrected`).

## What the correction tool calls (`POST /api/corrections`)

```json
{"prescription_id": "<document id>", "field_id": "<field id>", "corrected_value": "Serum Creatinine", "reviewer_id": "dr.rao"}
```

1. refused in one plain sentence if the field is not in that prescription, the value is empty / too long /
   identical to what was read, or the reviewer id is not plain text; nothing changes on a refusal;
2. the correction is applied to the field through the existing review path (`webapp/review.submit_decision`),
   so the immutable ledger and the audit trail see it (the as-written text stays in the immutable evidence and
   in the correction row);
3. the `correction` row is stored, append-only (database trigger): original prediction, both engines' readings,
   the correction, doctor, crop hash and reference, reviewer, model versions, time;
4. the doctor's `doctor_lexicon` row `(doctor, field type, how it is written, what it means)` is counted
   (`count`, `verified_count`);
5. when a mapping has `verified_count >= CDI_DOCTOR_ALIAS_MIN_VERIFIED` (3, **PLACEHOLDER**) **and** the meaning
   names exactly ONE concept in the right domain (lab order / diagnosis / drug) it becomes a **class-C alias for
   that doctor** in `kb_alias`. The alias engine reloads the knowledge base per document and ranks class C
   highest, so the next document of that doctor resolves it at level 1; no other doctor is affected. Meanings
   that name no concept or several, single-letter writings, and field types without a domain are learned but
   never promoted. An unknown doctor gets the correction stored (global training data) and no profile.

If step 3 fails after step 2 the answer says so plainly (`applied: true, recorded: false`).

## Reads

- `GET /api/doctors/{id}/profile[?field_type=]`: the lexicon, most-confirmed first. Optional Redis read cache
  (`CDI_PROFILE_CACHE_TTL_S`, 0 = off); a miss or an outage falls through to the database; the cache is dropped
  when that doctor's lexicon changes.
- `GET /api/corrections/export[?since=&limit=]`: NDJSON, oldest first: the dataset for the next offline
  fine-tuning cycle.

## What is and is not proven

- The SQL runs against in-memory SQLite in `tests/test_corrections_unit.py` (52 tests, including the effect on
  the real alias cascade: promoted for the doctor, not for another). **Not run on PostgreSQL**: migration
  `0006_corrections.py` (and its trigger) and the PostgreSQL-only loader (`load_context`, `field_records`) are
  checked by reading and by a column-parity test only.
- No authentication on these endpoints (neither does the rest of this web app): `reviewer_id` is whatever the
  caller sends. Put them behind the same access control as the reviewer console before exposing them.
- Not built from the wider design (HW-Phases C-E): handwriting prototypes / embeddings per doctor, confusion
  statistics, layout priors, the novelty detector, and any per-doctor model adapter. The lexicon is the first
  slice of the DoctorNode, and it only helps where a doctor repeats the same writing for the same thing.
- The 3-confirmation threshold and the cold-start numbers in the architecture doc (about 20 / 50 / 100
  reviewed prescriptions) are assumptions, not measured.
