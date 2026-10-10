# Clinical Document Intelligence → EMR/EHR Adapter
### Production design, architecture, implementation plan & flow

**Codename:** `CDI-Adapter` (Clinical Document Intelligence Adapter)
**Nature:** An AI **adapter / side-car** that sits on top of a legacy Hospital Management System (HMS) and its database. It does **not** replace or modify the legacy core. It ingests the hospital's *only* medical record today — **scanned images / photocopies / PDFs of paper documents** — and turns them into a **structured, longitudinal EMR/EHR**: rows in relational tables **and** validated **FHIR R4 JSON**, ABDM/ABHA-ready.
**Deployment target:** Fully **offline / on-premise**, open-source models, **no per-document API cost** (no Gemini / OpenAI / Anthropic inference in the hot path). RunPod is used only for development, fine-tuning and load-testing.
**Date:** 2026-09-07 · **Status:** Design baseline for build.

---

## 0. Executive summary

The hospital runs a legacy HMS that stores **no discrete clinical data** — no coded diagnoses, no lab result rows, no medication list. For every patient it keeps a pile of **scanned documents**: prescriptions (mostly handwritten), lab reports, radiology reports, operative notes, discharge summaries, advice slips, vitals/symptom intake sheets.

The adapter converts this into a real clinical record through a **9-stage pipeline**:

```
pixels → evidence → structured extraction → terminology normalization →
clinical validation → conflict reconciliation → human adjudication →
provenance-bound clinical facts → FHIR resources → EMR rows + JSON bundle
```

**The governing principle:** the LLM/VLM is **never the source of truth**. The source of truth is
*original scan (immutable) + deterministic structured clinical facts + bound terminology codes + provenance + FHIR-validated resources.*
AI is the **extractor, normalizer and reasoning assistant** between the pixels and those controlled structures. Every clinical fact is traceable to a pixel region, an OCR span, a model version and a confidence score, and every low-confidence or conflicting fact is routed to a clinician before it becomes part of the record.

**Two synchronized outputs per care context** (as requested):

1. **Relational rows** — `clinical_fact`, `fhir_resource`, `encounter`, … in the adapter's own PostgreSQL (legacy DB untouched).
2. **FHIR R4 JSON** — an ABDM Composition "record artifact" `Bundle` (e.g. `PrescriptionRecord`, `DiagnosticReportRecord`, `DischargeSummaryRecord`) ready to hand to the ABDM HIP/HRP gateway under patient consent.

---

# PART A — Research: how the top HMS/EMR solutions structure EMR/EHR data

## A.1 The one pattern everybody converges on

Regardless of vendor, a mature clinical record has the same logical spine:

| Layer | What it holds | Examples of the "unit" |
|---|---|---|
| **Demographic / identity** | one durable identity per person, independent of any single visit | MPI/EMPI record, ABHA number, MRN |
| **Encounter / episode** | a contact with the health system; everything clinical hangs off it | outpatient visit, admission, lab order episode |
| **Clinical statements** | atomic, coded, time-stamped assertions | a diagnosis, one lab result, one drug order, one procedure, one vital sign |
| **Documents** | the human-readable narrative + the source artifact | discharge summary, scanned prescription |
| **Provenance / audit** | who/what/when asserted each statement, from what source | signer, device, source document, author |
| **Terminology** | the controlled vocabularies the codes come from | SNOMED CT, LOINC, ICD, a drug dictionary, UCUM units |

Every system below is a different physical realization of that same spine. The scanned-only hospital has **only the "Documents" layer** and needs the adapter to synthesize the other four.

## A.2 Data-model archetypes of the leading systems

### Epic — *Chronicles + Clarity/Caboodle*
- **Operational store:** **Chronicles**, a hierarchical **MUMPS** database (now on InterSystems IRIS), behaving like a NoSQL document store to avoid relational-join latency at the bedside. Structure: `Chronicles → Master File → Record → Contact → Item/Value`. A **Master File** holds all data about one *type* of entity (patient `EPT`, encounter, provider, order, diagnosis), each record with a unique ID; a **Contact** is one time-stamped update to that record. ([mindbowser][epic1], [docsity][epic2])
- **Analytics store:** **Clarity** (relational, SQL Server/Oracle) and **Caboodle** (dimensional warehouse) — Chronicles data is ETL'd out because it is not natively SQL-queryable. ([sdrfoundation][epic3])
- **Exchange:** HL7 v2, C-CDA, FHIR R4 (USCDI via US Core), "Care Everywhere" for cross-org.
- **Lesson for us:** separate a fast **operational** representation from a **query/analytics/reporting** representation; keep a per-fact "contact" history.

### Oracle Health (Cerner) — *Millennium*
- **Person-centric** model: `PERSON → PATIENT (facility-specific) → ENCOUNTER → VISIT`. Core tables `PERSON`, `ENCOUNTER`, `ENCNTR_ALIAS`, `ORDER`, `ORDER_DETAIL`, and the central **`CLINICAL_EVENT`** table — nearly every discrete clinical observation (labs, vitals, documents, notes) is a row in `CLINICAL_EVENT` keyed by an event code, with results in `CE_*` child tables. Queried via **CCL** (Cerner Command Language, an Oracle-SQL dialect). FHIR R4 resources surface it externally. ([boristyukin][cerner1], [tactionsoft][cerner2])
- **Lesson for us:** a single **generic "clinical event / observation" table** (EAV-ish) with a strong code on every row scales to arbitrary document content without schema churn — this is exactly what a document-derived record needs.

### InterSystems IRIS for Health / TrakCare
- **FHIR-native**; internal canonical model is **SDA** (Summary Document Architecture), auto-transformed to/from FHIR, HL7 v2, C-CDA. Ships a built-in FHIR repository + terminology.
- **Lesson for us:** treat **FHIR as the canonical interchange model**, keep an internal normalized model, and generate FHIR by deterministic transform — not by asking a model to emit FHIR directly.

### OpenMRS + Bahmni (dominant open-source stack in India, Africa)
- Heart of the model is a **concept dictionary**: every clinical "question" and "answer" is a `concept`. Observations live in a single **`obs`** table using an **Entity-Attribute-Value (EAV)** design — `obs(person_id, encounter_id, concept_id, obs_datetime, value_numeric|value_coded|value_text|value_datetime, obs_group_id)` — so new kinds of data need **no schema change**. Hierarchy: `Patient → Visit → Encounter → Obs (grouped)`. Diagnoses, drug orders, lab results, chief complaints are all concepts. Analytics via **Bahmni Mart** (flattened) and terminology via **OCL / Dictionary Manager**. ([openmrs wiki][omrs1], [bahmni wiki][omrs2])
- **Lesson for us:** the **EAV `obs` model is the right internal shape** for "we don't know in advance what the document will contain." Our `clinical_fact` table is essentially `obs` + provenance + confidence + review-state.

### OpenEHR (used in national programs: parts of UK, Norway, Brazil, Catalonia)
- Two-level modelling: a small stable **reference model** + clinically-authored **archetypes/templates**; data stored as versioned **compositions** with full audit. Very strong provenance and change history.
- **Lesson for us:** every clinical fact must be **versioned with an author and a reason for change** — matters for medico-legal defensibility.

### OpenEMR (small-clinic open source)
- Straightforward **relational** schema: `patient_data`, `form_encounter`, `lists` (problems/allergies/medications as typed list rows), `procedure_result`, `prescriptions`. Simple, readable, but rigid.
- **Lesson for us:** offer a **flattened "list" projection** (problem list, medication list, allergy list, results table) on top of the EAV store — that is what clinicians actually read.

## A.3 The standards layer everyone exchanges on

| Concern | Standard(s) |
|---|---|
| Messaging (legacy) | HL7 v2.x (ADT, ORU, ORM) |
| Documents | HL7 CDA / C-CDA (US), IHE XDS for document repositories |
| Modern API | **HL7 FHIR R4 / R4B** — resources: `Patient`, `Encounter`, `Condition`, `Observation`, `Procedure`, `MedicationRequest/Statement`, `AllergyIntolerance`, `DiagnosticReport`, `DocumentReference`, `Composition`, `Provenance`, `Bundle` |
| Problems / findings / procedures | **SNOMED CT** (national editions) |
| Lab & clinical observations | **LOINC** |
| Billing / statistics / mortality | **ICD-10**, migrating to **ICD-11** |
| Drugs | RxNorm (US), dm+d (UK), ATC, national drug DBs; India: no single mandated dictionary → use SNOMED CT drug hierarchy + a local formulary map |
| Units | **UCUM** |

`DiagnosticReport` references its individual `Observation` results; `MedicationRequest` carries medication + subject + encounter + requester + dosage — this resource-oriented decomposition is exactly what the extractor must produce. ([hapifhir][fhir1])

## A.4 India — ABDM / ABHA in depth (primary target)

**Architecture (federated, consent-first):**

- **ABHA** — 14-digit health account number = the person identity the longitudinal record is keyed to.
- **HFR / HPR** — Health Facility Registry / Healthcare Professional Registry (the `Organization` / `Practitioner` identities).
- **HIP (Health Information Provider)** — role our hospital plays: creates health info and shares it digitally under consent.
- **HRP (Health Repository Provider)** — the bridge service between the HIP/HMS and the ABDM gateway; the hospital gets an **HFR ID** and links **care contexts**.
- **HIU (Health Information User)** — a consumer (another hospital, the patient's app) that *pulls* records.
- **Consent Manager (HIE-CM / ABHA app)** — patient grants a **consent artifact**: a signed JSON stating which care contexts, which HI types, which date range, retention, and a **one-time encryption keypair**. Data is exchanged as **FHIR bundles encrypted with Fidelius (ECDH)**; keys are never stored. ([coronasafe/abdm docs][abdm1], [nirmitee HIP guide][abdm2], [bahmni HIP][abdm3])

**Content standard:** **FHIR R4** per the **NRCeS "FHIR Implementation Guide for ABDM"** (current line v6.5.x → v7.x). Clinical artifacts are **profiles on the `Composition` resource**, one per document type:

| ABDM record artifact (Composition profile) | Our source document |
|---|---|
| `OPConsultRecord` | OPD advice / consultation / symptoms & vitals sheet |
| `PrescriptionRecord` | prescription (handwritten or printed) |
| `DiagnosticReportRecord` | lab report, radiology/imaging report, pathology report |
| `DischargeSummaryRecord` | discharge summary (+ embedded operative note) |
| `ImmunizationRecord` | vaccination card |
| `WellnessRecord` | vitals / general checkup |
| `HealthDocumentRecord` | anything unstructured — always available as a fallback carrying the scan as `DocumentReference` |
| `InvoiceRecord` | billing (out of clinical scope, noted) |

Terminologies for India: **SNOMED CT (India edition, free national licence via NRCeS)**, **LOINC**, **ICD-10** (→ ICD-11), **UCUM**. ([nrces ABDM IG][abdm4], [nrces DiagnosticReportRecord][abdm5])

**Compliance for the data pipeline:** DPDP Act 2023 (India) for personal data; de-identification of any training corpus; audit trail; RBAC; encryption at rest & in transit; consent enforced **before** any external disclosure. Bahmni already ships an HIP reference implementation — useful as an integration blueprint. ([bahmni HIP][abdm3])

## A.5 Country deep-dives — "what a scanned-only hospital must comply with"

For each country: national exchange architecture, content standard, terminology, and the **adapter delta** — what the same scanned-only hospital would need to produce there. This confirms the adapter design is portable: swap the **terminology package** and the **profile/IG package**, keep the pipeline.

| Country | National record / exchange | Content standard the hospital must emit | Terminology | Adapter delta (vs. India build) |
|---|---|---|---|---|
| **India** | ABDM: ABHA + HIP/HRP + consent manager, federated (no central store) | **FHIR R4**, NRCeS ABDM IG — `Composition`-based record artifacts, encrypted bundles (Fidelius) | SNOMED CT-IN, LOINC, ICD-10→11, UCUM | baseline |
| **USA** | ONC-certified EHRs + TEFCA/QHINs; USCDI is the mandated data floor | **FHIR R4 US Core** (+ C-CDA for documents); USCDI v3/v4 data classes; "USCDI+" for domains | SNOMED CT-US, LOINC, ICD-10-CM, **RxNorm**, CVX | swap IG→US Core, drug map→RxNorm, add C-CDA generator for TEFCA doc exchange ([healthit USCDI][us1], [hl7 US Core][us2]) |
| **UK (NHS England)** | GP Connect + Transfer of Care + NHS App; moving CareConnect→**UK Core** | **FHIR (STU3→R4)**, GP Connect *Access Record: Structured* & *Access Document*; sections keyed by fixed SNOMED codes | **SNOMED CT-UK**, **dm+d** (drugs), UK LOINC subset | swap IG→UK Core, drug map→dm+d, section codes→GP Connect SNOMED set ([nhs gp connect structured][uk1], [nhs interoperability][uk2]) |
| **Japan** | **SS-MIX2** standardized storage at 1,200+ hospitals; FHIR "JP Core" + IPS emerging | **HL7 v2.5** files in SS-MIX2 folder layout (patient, prescriptions, labs) for the *standardized* store; extension store for the rest | JLAC10/JLAC11 (labs), HOT/YJ (drugs), ICD-10 (MEDIS), SNOMED limited | add an **SS-MIX2 HL7 v2.5 writer** (folder-per-datatype) alongside FHIR; map labs→JLAC ([jami SS-MIX2][jp1], [springer SS-MIX2][jp2]) |
| **South Korea** | 3 MOHW programs: **EMR certification**, HIE project, **MyHealthWay** (national PHR) | **FHIR R4 "KR Core"** aligned to **KR CDI**; CDA still ~21% of tertiary hospitals, FHIR ~10% and rising | KOSTOM, EDI codes, KCD (Korean ICD), SNOMED CT (member), LOINC | swap IG→KR Core, terminology→KCD/KOSTOM; support EMR-certification export format ([e-hir KR standardization][kr1], [e-hir MyHealthWay][kr2]) |
| **China** | National **WS/WS·T** standards: *Basic dataset of EMR* (**WS 445**), *EMR shared-document spec*, *hospital information platform based on EMR spec*; ~150 standards; regional HIE platforms | **CDA-style shared documents** per WS·T spec (XML), provincial platform APIs; FHIR pilots only | ICD-10 (national clinical version), ICD-9-CM-3 procedures, national drug codes, TCM datasets | add a **WS 445 / WS·T CDA-XML generator**; map to national ICD clinical version ([pmc china CDA][cn1], [wiley china EMR][cn2]) |
| **Germany** | **gematik TI** + **ePA "für alle"** (opt-out, live 2025); hospital connectors via **ISiK**; **E-Rezept** | **FHIR R4**: **MIOs** (Medizinische Informationsobjekte) for ePA use cases, **ISiK** profiles for hospital systems, KBV profiles for prescribing | SNOMED CT-DE, LOINC, **ICD-10-GM**, **OPS** (procedures), PZN/ATC (drugs) | swap IG→ISiK+MIO, procedures→OPS, drugs→PZN; integrate via ISiK "Basismodul" ([health-samurai germany][de1], [gematik ePA][de2]) |
| **France** | **Mon espace santé** (contains **DMP**, opt-out, auto-created 2022); **CI-SIS** interop framework by ANS; **INS** national health identity | **CDA + IHE XDS** today, transitioning to **FHIR (FR Core)**; documents fed from Ségur-referenced software | CIM-10 (ICD-10-FR), CCAM (procedures), LOINC, SNOMED (limited), **INS** identity | add **CDA/XDS document feeder** to DMP + **INS identity** resolution; plan FHIR FR Core path ([gnius DMP][fr1], [esante DMP référentiel][fr2]) |
| **Canada** | Province-led; **Canada Health Infoway** **CA Core+** = CACDI as FHIR profiles; **PS-CA** patient summary | **FHIR R4 CA Core+ / PS-CA** | SNOMED CT-CA, **pCLOCD** (Canadian LOINC), CCDD (drugs), ICD-10-CA / CCI | swap IG→CA Core+, labs→pCLOCD, drugs→CCDD; per-province endpoints ([infoway CA Core+][ca1], [infoway pCLOCD][ca2]) |
| **Australia** | **My Health Record** (national, opt-out); moving CDA→FHIR via **AUCDI / AU Core** | **CDA documents** today (Shared Health Summary, Event Summary, Discharge Summary), FHIR **AU Core** for new content | **SNOMED CT-AU** + **AMT** (medicines), LOINC, ICD-10-AM | keep CDA document packages for MHR upload + AU Core FHIR; drugs→AMT; NCTS terminology refresh ≤30 days ([digitalhealth AU interoperability][au1], [healthterminologies AU TIG][au2]) |

**Common denominator across all ten:** SNOMED CT + LOINC + an ICD variant + a national drug dictionary + a document/composition wrapper + a provenance/consent layer. The adapter's design isolates every country-specific piece into two swappable packages: **(1) Terminology Package**, **(2) Profile/IG + Document-format Package**.

---

# PART B — The canonical target data structure (rows **and** JSON)

## B.1 Design stance

- **Internal store = normalized EAV clinical-fact model** (OpenMRS `obs` / Cerner `CLINICAL_EVENT` lineage) + first-class **provenance**, **confidence**, **review-state**, **temporal** attributes.
- **Interchange model = FHIR R4**, generated by a **deterministic mapping engine**, then **validated** against core + national profiles.
- **Read model = flattened lists** (problem list, medication list, allergy list, results grid, encounter timeline) as SQL views / materialized views.
- The legacy DB is **read-only** to us. We never alter its schema.

## B.2 Adapter database — DDL (PostgreSQL 16, `jsonb` throughout)

```sql
-- ========== 1. INGESTION & SOURCE EVIDENCE (immutable) ==========

CREATE TABLE source_document (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  legacy_ref         text,                       -- legacy doc/blob id or file path
  legacy_patient_ref text,                       -- legacy MRN as-found (unverified)
  sha256             char(64) NOT NULL UNIQUE,   -- dedupe identical scans
  mime_type          text NOT NULL,
  page_count         int  NOT NULL,
  object_uri         text NOT NULL,              -- MinIO/S3 path of original bytes
  captured_at        timestamptz,               -- when the hospital scanned it (if known)
  ingested_at        timestamptz NOT NULL DEFAULT now(),
  source_channel     text NOT NULL,             -- 'legacy_cdc' | 'folder_watch' | 'batch' | 'manual'
  status             text NOT NULL DEFAULT 'received'  -- received|classified|extracted|projected|error
);

CREATE TABLE document_page (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id    uuid NOT NULL REFERENCES source_document(id),
  page_no        int  NOT NULL,
  width_px       int, height_px int, dpi int,
  image_uri      text NOT NULL,                 -- normalized/deskewed page render
  preproc        jsonb NOT NULL DEFAULT '{}',   -- rotation, denoise, binarize params
  UNIQUE (document_id, page_no)
);

CREATE TABLE pipeline_run (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   uuid NOT NULL REFERENCES source_document(id),
  stage         text NOT NULL,                  -- classify|ocr|extract|normalize|validate|project|reconcile
  status        text NOT NULL,                  -- ok|retry|failed|skipped
  model_name    text, model_version text, model_sha text,
  params        jsonb NOT NULL DEFAULT '{}',
  started_at    timestamptz NOT NULL DEFAULT now(),
  ended_at      timestamptz,
  metrics       jsonb NOT NULL DEFAULT '{}',    -- latency, tokens, page count, error
  input_sha     text, output_sha text
);

-- ========== 2. CLASSIFICATION ==========

CREATE TABLE doc_classification (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   uuid NOT NULL REFERENCES source_document(id),
  doc_type      text NOT NULL,   -- prescription|lab_report|radiology_report|discharge_summary|
                                 -- operative_note|opd_note|vitals_sheet|referral|consent_form|other
  specialty     text,            -- cardiology|ortho|...
  language      text[],          -- ['en','bn','hi']
  is_handwritten bool NOT NULL DEFAULT false,
  page_spans    jsonb NOT NULL DEFAULT '[]',   -- [{page_no, doc_type}] for multi-doc scans
  confidence    numeric(4,3) NOT NULL,
  model_run_id  uuid REFERENCES pipeline_run(id),
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- ========== 3. OCR / LAYOUT EVIDENCE ==========

CREATE TABLE ocr_block (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  page_id       uuid NOT NULL REFERENCES document_page(id),
  block_type    text NOT NULL,        -- line|word|cell|kv_key|kv_value|table|title|signature|stamp
  reading_order int,
  text          text NOT NULL,
  bbox          int[]  NOT NULL,      -- [x0,y0,x1,y1] in page px
  polygon       jsonb,               -- optional quad for skewed text
  ocr_conf      numeric(4,3) NOT NULL,
  lang          text,
  table_ref     uuid,                -- groups cells belonging to one table
  row_idx int, col_idx int,
  model_run_id  uuid REFERENCES pipeline_run(id)
);
CREATE INDEX ON ocr_block (page_id);
CREATE INDEX ON ocr_block USING gin (to_tsvector('simple', text));

-- ========== 4. RAW MODEL EXTRACTION (pre-normalization, auditable) ==========

CREATE TABLE extraction (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   uuid NOT NULL REFERENCES source_document(id),
  schema_name   text NOT NULL,       -- 'prescription.v3' | 'lab_report.v3' | ...
  schema_version text NOT NULL,
  payload       jsonb NOT NULL,      -- schema-locked JSON straight from the DSLM/VLM
  evidence_map  jsonb NOT NULL,      -- json-path -> [ocr_block_id...] + char spans
  model_run_id  uuid REFERENCES pipeline_run(id),
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- ========== 5. PATIENT IDENTITY (adapter-side MPI) ==========

CREATE TABLE patient_identity (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  legacy_mrn    text,
  abha_number   text, abha_address text,
  name_given    text, name_family text,
  gender        text, birth_date date, birth_date_est bool DEFAULT false,
  phone_hash    text, address_hash text,
  match_status  text NOT NULL DEFAULT 'unlinked',  -- unlinked|auto|clerk_confirmed|abha_verified
  match_score   numeric(4,3),
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE patient_identity_alias (           -- every spelling seen on a document
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES patient_identity(id),
  document_id uuid REFERENCES source_document(id),
  raw_name text, raw_mrn text, raw_dob text, raw_age text, raw_phone text,
  ocr_block_ids uuid[]
);

-- ========== 6. ENCOUNTERS (derived) ==========

CREATE TABLE encounter (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid NOT NULL REFERENCES patient_identity(id),
  class         text NOT NULL,       -- AMB|IMP|EMER|virtual
  period_start  timestamptz, period_end timestamptz,
  period_precision text DEFAULT 'day',   -- year|month|day|minute  (scans are often date-only)
  facility_ref  text, practitioner_ref text, specialty text,
  derived_from  uuid[] NOT NULL,     -- source_document ids that evidenced this encounter
  confidence    numeric(4,3) NOT NULL,
  review_state  text NOT NULL DEFAULT 'pending',
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- ========== 7. CLINICAL FACTS (the EAV core: "rows in a table") ==========

CREATE TABLE clinical_fact (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id      uuid NOT NULL REFERENCES patient_identity(id),
  encounter_id    uuid REFERENCES encounter(id),
  fact_type       text NOT NULL,   -- condition|symptom|finding|lab_result|vital_sign|
                                   -- procedure|medication|allergy|immunization|
                                   -- diagnostic_report|advice|care_plan|referral|document
  -- coded concept
  code_system     text,            -- 'http://snomed.info/sct' | 'http://loinc.org' | 'icd-10' | ...
  code            text,
  code_display    text,
  code_status     text NOT NULL DEFAULT 'unmapped',  -- unmapped|candidate|bound|local_only
  local_text      text NOT NULL,   -- verbatim clinical phrase from the document
  -- value (only the relevant columns are filled, OpenMRS-style)
  value_kind      text,            -- quantity|codeable|string|boolean|datetime|range|ratio
  value_num       numeric,
  value_unit_ucum text,
  value_code_system text, value_code text, value_code_display text,
  value_text      text,
  value_bool      bool,
  value_low_num numeric, value_high_num numeric,
  ref_range_low numeric, ref_range_high numeric, ref_range_text text,
  abnormal_flag   text,            -- H|L|HH|LL|A|N
  -- clinical status / context
  clinical_status text,            -- active|resolved|inactive|completed|stopped|entered-in-error
  verification    text,            -- confirmed|provisional|differential|unconfirmed
  onset           timestamptz, onset_precision text,
  effective_time  timestamptz, effective_precision text,   -- when the observation/act applies
  asserted_time   timestamptz,                              -- when written on the document
  -- extraction quality & lifecycle
  extraction_id   uuid REFERENCES extraction(id),
  source_doc_ids  uuid[] NOT NULL,
  confidence_overall numeric(4,3) NOT NULL,   -- fused OCR × extraction × terminology
  confidence_ocr     numeric(4,3),
  confidence_extract numeric(4,3),
  confidence_terminology numeric(4,3),
  review_state    text NOT NULL DEFAULT 'pending', -- pending|auto_accepted|in_review|
                                                   -- clinician_confirmed|corrected|rejected
  reviewed_by     text, reviewed_at timestamptz, review_note text,
  supersedes      uuid REFERENCES clinical_fact(id),  -- versioning / correction chain
  is_current      bool NOT NULL DEFAULT true,
  dedup_key       text,            -- hash(patient, type, code, value, effective_day) for merge
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON clinical_fact (patient_id, fact_type, is_current);
CREATE INDEX ON clinical_fact (dedup_key);
CREATE INDEX ON clinical_fact (review_state) WHERE review_state = 'pending';

-- medication specifics (kept relational for the medication list / e-prescribing)
CREATE TABLE medication_detail (
  fact_id        uuid PRIMARY KEY REFERENCES clinical_fact(id),
  drug_text      text NOT NULL,
  rxlike_system  text, rxlike_code text,         -- SNOMED CT drug / local formulary
  form           text, strength_num numeric, strength_unit text,
  dose_num       numeric, dose_unit_ucum text,
  route          text, frequency_code text,      -- BID/TID/HS/SOS ...
  frequency_per_day numeric, duration_days int,
  prn            bool, instructions text,
  intent         text NOT NULL DEFAULT 'order'   -- order (MedicationRequest) | record (MedicationStatement)
);

-- ========== 8. PROVENANCE (first-class) ==========

CREATE TABLE fact_provenance (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  fact_id       uuid NOT NULL REFERENCES clinical_fact(id),
  source_doc_id uuid NOT NULL REFERENCES source_document(id),
  page_id       uuid REFERENCES document_page(id),
  ocr_block_ids uuid[] NOT NULL,
  bbox_union    int[],                 -- highlight region for the reviewer UI
  extracted_text text NOT NULL,        -- exact substring the fact came from
  pipeline_run_ids uuid[] NOT NULL,    -- classify+ocr+extract+normalize chain
  model_stack   jsonb NOT NULL,        -- {classifier, ocr, vlm, dslm, linker} names+versions+sha
  prompt_sha    text,
  agent         text NOT NULL DEFAULT 'system',  -- 'system' | clinician user id
  recorded_at   timestamptz NOT NULL DEFAULT now()
);

-- ========== 9. CONFLICTS & RECONCILIATION ==========

CREATE TABLE fact_conflict (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid NOT NULL REFERENCES patient_identity(id),
  fact_a        uuid NOT NULL REFERENCES clinical_fact(id),
  fact_b        uuid REFERENCES clinical_fact(id),
  conflict_type text NOT NULL,   -- value_mismatch|code_mismatch|temporal_impossible|
                                 -- dose_out_of_range|contradiction|duplicate|unit_mismatch
  evidence_state text NOT NULL,  -- SUPPORTED | CONTRADICTED | UNKNOWN_NOT_MENTIONED
  severity      text NOT NULL,   -- info|warn|blocker
  auto_resolution text,          -- keep_a|keep_b|merge|both_provisional|null
  resolved_by   text, resolved_at timestamptz, resolution_note text,
  status        text NOT NULL DEFAULT 'open'   -- open|auto_resolved|clinician_resolved
);

-- ========== 10. FHIR PROJECTION ("and in json") ==========

CREATE TABLE fhir_resource (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid REFERENCES patient_identity(id),
  encounter_id  uuid REFERENCES encounter(id),
  resource_type text NOT NULL,     -- Patient|Encounter|Condition|Observation|Procedure|
                                   -- MedicationRequest|AllergyIntolerance|DiagnosticReport|
                                   -- DocumentReference|Composition|Provenance
  fhir_id       text NOT NULL,
  version_id    int  NOT NULL DEFAULT 1,
  profile       text[],            -- national profile URLs asserted
  resource      jsonb NOT NULL,    -- the resource JSON
  derived_from_facts uuid[] NOT NULL,
  validation_status text NOT NULL DEFAULT 'pending',  -- pending|valid|warning|error
  validation_issues jsonb NOT NULL DEFAULT '[]',
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (resource_type, fhir_id, version_id)
);

CREATE TABLE fhir_bundle (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid NOT NULL REFERENCES patient_identity(id),
  encounter_id  uuid REFERENCES encounter(id),
  care_context  text,              -- ABDM care-context reference
  artifact_type text NOT NULL,     -- OPConsultRecord|PrescriptionRecord|DiagnosticReportRecord|
                                   -- DischargeSummaryRecord|WellnessRecord|ImmunizationRecord|
                                   -- HealthDocumentRecord
  bundle        jsonb NOT NULL,    -- Bundle(type=document): Composition + all referenced resources + Provenance
  bundle_hash   char(64) NOT NULL,
  validation_status text NOT NULL DEFAULT 'pending',
  fhir_version  text NOT NULL DEFAULT '4.0.1',
  ig_package    text NOT NULL,     -- e.g. 'nrces.fhir.r4.ndhm#6.5.0'
  status        text NOT NULL DEFAULT 'draft',  -- draft|validated|ready_to_share|shared|superseded
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- ========== 11. HUMAN-IN-THE-LOOP REVIEW ==========

CREATE TABLE review_task (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid REFERENCES patient_identity(id),
  document_id   uuid REFERENCES source_document(id),
  kind          text NOT NULL,     -- low_confidence|conflict|identity_match|dose_check|
                                   -- unmapped_terminology|schema_reject|handwriting
  ref_fact_ids  uuid[], ref_conflict_id uuid REFERENCES fact_conflict(id),
  priority      int NOT NULL DEFAULT 3,   -- 1 highest
  sla_due       timestamptz,
  assignee      text, status text NOT NULL DEFAULT 'queued', -- queued|in_progress|done|escalated
  payload       jsonb NOT NULL DEFAULT '{}',
  created_at    timestamptz NOT NULL DEFAULT now(),
  closed_at     timestamptz
);

-- ========== 12. ABDM LINKAGE / CONSENT / TRANSFER ==========

CREATE TABLE abdm_care_context (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  encounter_id uuid NOT NULL REFERENCES encounter(id),
  patient_id uuid NOT NULL REFERENCES patient_identity(id),
  care_context_ref text NOT NULL UNIQUE,   -- opaque string registered with ABDM
  hi_types text[] NOT NULL,                -- ['Prescription','DiagnosticReport',...]
  display  text NOT NULL,
  linked_at timestamptz, link_status text NOT NULL DEFAULT 'pending'
);
CREATE TABLE abdm_consent (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  consent_artifact_id text NOT NULL,
  patient_id uuid NOT NULL REFERENCES patient_identity(id),
  hi_types text[] NOT NULL, date_from date, date_to date,
  expiry timestamptz, purpose_code text,
  artifact jsonb NOT NULL,                 -- signed consent JSON (keys NOT persisted)
  received_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE abdm_transfer (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  consent_id uuid REFERENCES abdm_consent(id),
  bundle_id uuid NOT NULL REFERENCES fhir_bundle(id),
  transaction_id text, hiu_id text,
  status text NOT NULL DEFAULT 'prepared', -- prepared|encrypted|pushed|acknowledged|failed
  pushed_at timestamptz, ack_at timestamptz, error text
);

-- ========== 13. AUDIT ==========

CREATE TABLE audit_log (
  id bigserial PRIMARY KEY,
  ts timestamptz NOT NULL DEFAULT now(),
  actor text NOT NULL,           -- service name or user id
  action text NOT NULL,          -- read|create|update|accept|correct|reject|share|login
  entity text NOT NULL, entity_id text,
  patient_id uuid,
  detail jsonb NOT NULL DEFAULT '{}',
  ip inet, request_id text
);
```

### Read-model views (what clinicians see)

```sql
CREATE MATERIALIZED VIEW v_problem_list AS
  SELECT patient_id, code_system, code, code_display, local_text,
         clinical_status, verification, min(onset) onset, max(effective_time) last_seen,
         array_agg(DISTINCT id) fact_ids, max(confidence_overall) confidence
  FROM clinical_fact
  WHERE fact_type='condition' AND is_current AND review_state IN ('auto_accepted','clinician_confirmed','corrected')
  GROUP BY patient_id, code_system, code, code_display, local_text, clinical_status, verification;

-- analogous: v_medication_list, v_allergy_list, v_results_grid (pivot by LOINC × date),
--            v_encounter_timeline, v_immunization_list
```

## B.3 FHIR resource set & ABDM profile mapping

| `fact_type` / entity | FHIR resource | Key bindings | ABDM Composition section it lands in |
|---|---|---|---|
| identity | `Patient` (+ `Identifier` for ABHA) | ABHA system URI | subject of every record |
| facility / clinician | `Organization` (HFR), `Practitioner`/`PractitionerRole` (HPR) | — | author / custodian |
| encounter | `Encounter` | class = AMB/IMP/EMER | context |
| condition, symptom (as problem) | `Condition` | SNOMED CT; ICD-10 as secondary | *Chief complaints* / *Diagnosis* / *Medical history* |
| symptom / finding (as observation) | `Observation` (category=exam or symptom) | SNOMED CT | *Symptoms* |
| vital sign | `Observation` (category=vital-signs) | LOINC + UCUM | *Vital signs* (WellnessRecord / OPConsult) |
| lab result | `Observation` (category=laboratory) + `DiagnosticReport` | LOINC + UCUM, interpretation codes | `DiagnosticReportRecord` |
| radiology / path report | `DiagnosticReport` (+ `Observation`, `Media`/`DocumentReference` for images) | LOINC/SNOMED, RadLex text | `DiagnosticReportRecord` |
| procedure / surgery | `Procedure` | SNOMED CT; ICD-9-CM-3/national as secondary | *Procedures* (DischargeSummary) |
| medication (order) | `MedicationRequest` | SNOMED CT drug / local formulary; UCUM dose; timing | `PrescriptionRecord` |
| medication (reported/taking) | `MedicationStatement` | same | *Medications* (history) |
| allergy | `AllergyIntolerance` | SNOMED CT substance + reaction | *Allergies* |
| immunization | `Immunization` | vaccine code | `ImmunizationRecord` |
| advice / plan | `CarePlan` / `Composition.section(text)` | narrative | *Advisory notes* / *Follow-up* |
| the scan itself | `DocumentReference` + `Binary` | doc-type LOINC, `content.attachment` = original scan | *always attached* (HealthDocumentRecord fallback) |
| every fact's origin | `Provenance` | `target` → resource, `entity` → DocumentReference, `agent` = device + software + (clinician on review) | bundled |
| the record wrapper | `Composition` → `Bundle(type=document)` | one of the 8 ABDM profiles | the deliverable JSON |

**Rule:** the original `DocumentReference`/`Binary` (the scan) is **always** included in the bundle. Even if every structured fact is rejected in review, the hospital still produces a compliant `HealthDocumentRecord` carrying the image — the record is never worse than "the scan, shared digitally."

## B.4 Provenance & evidence model (non-negotiable)

Every `clinical_fact` row has ≥1 `fact_provenance` row binding it to:
`source_document → page → ocr_block(s) → bbox → exact substring → pipeline_run chain → model stack (name+version+sha) → prompt hash → asserting agent → timestamp`.

This yields, for any fact: *"Metformin 500 mg BD"* → highlightable region on page 2 of `IMG_004`, OCR confidence 0.71, extraction confidence 0.88, terminology bind confidence 0.93, produced by `qwen2.5-vl-7b@<sha>` + `cdi-dslm-0.4@<sha>`, reviewed & confirmed by `dr.sen` at `2026-09-07T11:04Z`. Required for clinician review, medico-legal defence, audit, model evaluation, and **reprocessing** when a better model ships.

## B.5 Confidence & review-state model

**Fused confidence** `c = w1·c_ocr + w2·c_extract + w3·c_terminology` (weights tuned per `doc_type` × `fact_type`; also a learned calibrator — isotonic regression — so the number means "P(correct)").

Routing (starting thresholds — **must be re-derived empirically per class**, see Part F eval):

| Condition | Action |
|---|---|
| `c ≥ 0.985` and no rule violation | `auto_accepted`, written to read model, no queue |
| `0.95 ≤ c < 0.985` | `auto_accepted` + **audit sample** (10% pulled for QA) |
| `0.85 ≤ c < 0.95` | `review_task(low_confidence)` |
| `c < 0.85` | **mandatory** clinician validation |
| any `fact_conflict.severity = blocker` | mandatory review, fact stays `provisional`, excluded from read model |
| medication dose/route/frequency ambiguous | mandatory review (safety) |
| terminology `code_status = unmapped` for condition/med/allergy | mandatory review or `local_only` binding |
| handwritten prescription, any medication line | minimum "review sample" even at high confidence until per-prescriber accuracy is proven |

Nothing enters `v_problem_list` / `v_medication_list` / `v_allergy_list` unless `review_state ∈ {auto_accepted, clinician_confirmed, corrected}`.

---

# PART C — Architecture

## C.1 The pipeline (9 stages) + component view

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ LEGACY HMS  (unchanged)   Oracle/SQL Server + document blob store / file share │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  (read-only)  CDC on doc table  |  folder watch  |  nightly batch
                ▼
        ┌───────────────┐   S1  INGEST & NORMALIZE
        │ Ingestion svc │   dedupe(sha256) · split PDF · deskew/denoise/binarize
        │  + MinIO      │   → source_document, document_page   (original bytes immutable)
        └───────┬───────┘
                ▼
        ┌───────────────┐   S2  CLASSIFY
        │ Classifier    │   doc_type · specialty · language · handwritten? · page spans
        │ (VLM-7B / DiT)│   → doc_classification
        └───────┬───────┘
                ▼
        ┌───────────────┐   S3  OCR / LAYOUT / HANDWRITING  (evidence layer)
        │ OCR ensemble  │   PaddleOCR PP-StructureV3 (print+tables) · Surya (layout/order)
        │ + VLM reader  │   · Qwen2.5-VL (handwriting + context) · docTR fallback
        └───────┬───────┘   → ocr_block (text + bbox + conf + table cells)
                ▼
        ┌───────────────┐   S4  STRUCTURED EXTRACTION  (schema-locked)
        │ Clinical DSLM │   per doc_type JSON schema · constrained decoding (XGrammar)
        │ (fine-tuned)  │   VLM for spatial/handwritten fields · DSLM for narrative NER+relations
        └───────┬───────┘   → extraction.payload + evidence_map
                ▼
        ┌───────────────┐   S5  TERMINOLOGY NORMALIZATION
        │ Terminology   │   SapBERT/BioLORD bi-encoder → FAISS over SNOMED-IN/LOINC/ICD
        │ service       │   + rule reranker + UCUM unit parse + drug→formulary map
        └───────┬───────┘   → clinical_fact (code_system/code, code_status)
                ▼
        ┌───────────────┐   S6  CLINICAL VALIDATION + CONFLICT ENGINE  (deterministic)
        │ Rules engine  │   value/unit sanity · dose ranges · temporal possibility ·
        │ (no LLM)      │   SUPPORTED / CONTRADICTED / UNKNOWN_NOT_MENTIONED
        └───────┬───────┘   → fact_conflict, confidence fusion + calibration
                ▼
        ┌───────────────┐   S7  RECONCILIATION AGENT  (LLM only where reasoning needed)
        │ LangGraph     │   cross-document merge · dedup_key · temporal ordering ·
        │ agent + tools │   propose (never auto-commit) resolutions for true conflicts
        └───────┬───────┘   → merged clinical_fact set + review_task
                ▼
        ┌───────────────┐   S8  HUMAN ADJUDICATION
        │ Review console│   highlight bbox ⇄ fact · accept/correct/reject · identity match
        │ (React)       │   corrections captured as training data
        └───────┬───────┘   → review_state transitions, fact_provenance(agent=clinician)
                ▼
        ┌───────────────┐   S9  FHIR PROJECTION + VALIDATION + DELIVERY
        │ Mapping engine│   deterministic facts→FHIR · national profiles · Provenance ·
        │ + HAPI valid. │   Bundle(type=document) per ABDM artifact
        └───────┬───────┘   → fhir_resource, fhir_bundle   (rows + JSON)
                ▼
   ┌────────────┴─────────────┬───────────────────────────┐
   ▼                          ▼                           ▼
FHIR façade (HAPI)      Legacy write-back           ABDM HIP/HRP gateway
read model / views     (optional: summary PDF,      care-context link · consent ·
for hospital UI         discrete codes via API)     Fidelius-encrypted bundle push
```

**Cross-cutting services:** identity/MPI, object store (MinIO), job queue (Redis + Celery / Temporal for durability), model gateway (vLLM), terminology server, secrets/KMS, observability (OpenTelemetry → Prometheus/Grafana/Loki), audit.

## C.2 Deployment topology

| Environment | Purpose | Hardware |
|---|---|---|
| **RunPod (dev/training)** | fine-tune DSLM, build FAISS indices, benchmark, load-test | 1× A100 80GB (or 2× A6000 48GB) pod; **persistent network volume** for models + datasets + Postgres data (container disk is wiped on restart; do **not** put PGDATA on MooseFS-mounted `/workspace` — chown fails; use a real block volume or an external managed PG) |
| **On-prem staging** | integration test against a legacy DB copy | 1 GPU box (RTX 6000 Ada 48GB / L40S) + 1 app node |
| **On-prem production** | live, air-gapped or DMZ-only for the ABDM gateway | **GPU node**: 1–2× L40S / A100 40GB (vLLM: VLM-7B ~18 GB + DSLM-7B ~16 GB + embedders ~4 GB → fits 48 GB with paged-attention; 2 GPUs for throughput/HA). **App node(s)**: 16–32 vCPU, 64–128 GB RAM (ingestion, OCR CPU ops, rules, mapping, API). **Data node**: PostgreSQL 16 (primary + standby), MinIO (3-node), Redis. |
| **ABDM edge** | only component allowed outbound; holds HRP client, does Fidelius crypto | small hardened VM in DMZ; mTLS to gateway; no PHI at rest beyond transient bundle |

Everything containerized (Docker/Podman), orchestrated with Docker Compose for a single-site hospital or K3s for multi-node. **No inference call leaves the premises.** Model weights are baked into images or pulled once from an internal registry.

## C.3 Model stack — all open-source, offline, no per-doc cost

| Stage | Primary (recommended) | License | Fallback / alt | Why |
|---|---|---|---|---|
| **Doc classifier** | Qwen2.5-VL-7B-Instruct (same gateway model; a keyword heuristic runs first), few-shot label prompt → later distilled to a **DiT / ViT** classifier on your labelled scans | Apache-2.0 | LayoutLMv3, or logistic head on Surya layout features | zero-shot to bootstrap, cheap CNN once you have ~2k labelled docs |
| **OCR – printed + tables** | **PaddleOCR PP-StructureV3** (detection + recognition + layout + table structure) | Apache-2.0 | docTR, RapidOCR | best open table extraction; Indic + English models |
| **Layout / reading order** | **Surya** | GPL-3.0 (check for prod; alt below) | PP-Structure layout, or `unstructured` | strong layout + order; if GPL is a blocker use PP-Structure only |
| **Handwriting + context** | **Qwen2.5-VL-7B-Instruct** (32B for hard docs) | Apache-2.0 | InternVL2.5-8B, MiniCPM-V 2.6 | VLM reads messy handwriting *with* layout context |
| **Indic scripts** (bn/hi/ta/te…) | PaddleOCR Indic models + **Bhashini/IndicOCR** where needed | Apache-2.0 / govt | Qwen2.5-VL (multilingual) | Bengali/Hindi prescriptions & stamps |
| **Clinical DSLM** (extraction, NER, relations, normalization prompts) | **Qwen2.5-7B-Instruct** + **LoRA** fine-tune → `cdi-dslm` | Apache-2.0 | Meditron3-8B or Llama-3.1-8B-Instruct (Llama license) or OpenBioLLM-8B (Llama3 license) | Apache-2.0 base = clean commercial on-prem; medical bases evaluated as alternates ([meditron][ml1], [openbiollm][ml2]) |
| **Terminology linker** | **SapBERT** (`cambridgeltl/SapBERT-...`) or **BioLORD-2023** bi-encoder + **FAISS** index over SNOMED-IN / LOINC / ICD-10 | Apache-2.0 / MIT | fuzzy + `pymedtermino`, or a cross-encoder reranker | SOTA concept normalization without task supervision ([sapbert][ml3]) |
| **Structured decoding** | **vLLM + XGrammar** (JSON-schema / grammar constrained) | Apache-2.0 | Outlines, lm-format-enforcer | guarantees parseable, schema-valid JSON from the DSLM/VLM ([vllm struct][ml4]) |
| **FHIR validation** | **HAPI FHIR validator** + `hl7.fhir.r4.core` + `nrces.fhir.r4.ndhm` IG package | Apache-2.0 | `fhir.resources` (pydantic) for shape, HAPI for profile | authoritative profile validation |
| **Terminology server** | **Snowstorm-lite** (SNOMED) + LOINC/ICD in Postgres, exposed via **HAPI FHIR terminology** (`$lookup`, `$validate-code`, `$translate`) | Apache-2.0 | HAPI-only with custom CodeSystems | offline `$validate-code` / `$translate` |
| **Orchestration** | **LangGraph** state machine (deterministic DAG; agent nodes only at S7 & ambiguity triage) | MIT | Prefect / custom asyncio FSM | explicit, inspectable control flow; not "an agent decides everything" |
| **LLM serving** | **vLLM** (OpenAI-compatible) | Apache-2.0 | TGI, SGLang | paged-attention, multi-LoRA, structured outputs |
| **Datastore** | PostgreSQL 16 (+ `pgvector`), MinIO, Redis | OSI | — | JSONB + relational + vectors in one place |

> "100% open source" ≠ "zero cost." Honest framing: **on-premise, open-source AI architecture with no per-document API inference cost.** Real costs remain: GPU hardware, power, storage, backup/DR, engineering, fine-tuning, annotation, terminology-licence administration (SNOMED national licence, LOINC terms of use), monitoring, security, HA.

## C.4 Orchestration — the agent graph

LangGraph graph, one instance per `source_document`, with a **document-set super-graph** per patient for S7:

```
ingest → classify → route_by_type ─┬─ ocr_print ──┐
                                   ├─ ocr_hand ───┤→ merge_evidence → extract(schema) →
                                   └─ ocr_table ──┘        │ (XGrammar-constrained)
                                                           ▼
                              normalize_terminology → validate_rules → fuse_confidence →
                                                           │
                          ┌────────────── conflict? ───────┤
                          ▼ (true conflict / low conf)     ▼ (clean, high conf)
                  reconciliation_agent (tools:              project_fhir
                   term lookup, prior facts,                     │
                   dose DB, calendar math) ── proposes ──> review_task
                          │                                      │
                          ▼                                      ▼
                   human_adjudication ───────────────────> project_fhir → validate_fhir →
                                                            build_bundle → (consent) → deliver
```

**Agent nodes are bounded:** they call **tools** (deterministic term lookup, prior-fact query, drug-dose reference, date arithmetic, unit conversion), must cite `ocr_block` ids for every claim, and can only **propose** — writes to `clinical_fact` from an agent are always `review_state='in_review'`. No agent free-form-merges clinical data.

**Three-valued evidence** everywhere: `SUPPORTED` / `CONTRADICTED` / `UNKNOWN_NOT_MENTIONED`. "Document B doesn't mention the appendectomy" is `UNKNOWN_NOT_MENTIONED`, **not** a contradiction — retain fact A. "Document B says cholecystectomy on the same date" is `CONTRADICTED` → blocker → human.

## C.5 Terminology service (detailed)

Responsibilities kept separate (not "a table of dictionaries"):

1. **Storage** — SNOMED CT India edition (RF2) in Snowstorm-lite; LOINC + ICD-10 (+ ICD-11 MMS) + UCUM + local formulary in Postgres CodeSystems.
2. **Candidate generation** — SapBERT/BioLORD embedding of `local_text` → FAISS top-k per target system.
3. **Reranking** — features: string sim, semantic type match (is this a disorder? a substance? an observable?), section context, prior-use frequency at this hospital.
4. **Validation** — `$validate-code` against the target ValueSet for the FHIR element.
5. **Mapping** — `$translate` via ConceptMaps (SNOMED↔ICD-10, local drug↔SNOMED drug, local lab↔LOINC).
6. **Versioning** — every bind records `code_system` **version**; re-bind job when a new edition loads.
7. **Language / synonyms** — Indic synonym lists; brand→generic drug map.
8. **Local codes** — when nothing binds: `code_status='local_only'`, store `local_text`, mint a local CodeSystem entry, queue a curation task. The record still ships (text + local code) and improves later.

```
FHIR core ValueSets
      │
      ▼
India / ABDM profiles + ValueSets  (nrces.fhir.r4.ndhm)
      │
      ▼
Hospital-specific: local formulary, local lab catalogue, local order set
      │
      ▼
bound Clinical Concept on clinical_fact
```

## C.6 Legacy HMS integration (the "adapter" contract)

**Inbound (read-only, pick per site):**
- **CDC** on the legacy "documents/attachments" table (Debezium / logical replication / trigger table) → event per new scan.
- **Folder watch** on the scanner output share (`inotify`/polling) with a manifest join back to MRN.
- **Nightly batch** export for backfill of historical documents.
- Optional **HL7 v2 ADT feed** (if the legacy emits it) to seed `patient_identity` and `encounter` skeletons — reduces identity guessing.

**Outbound (all optional, site-configurable, never destructive):**
- **FHIR façade** — HAPI FHIR server exposing the adapter's validated resources for the hospital's own apps / a new front-end.
- **Legacy write-back** — (a) a generated **structured summary PDF** filed back as a *new* document against the same MRN; (b) discrete codes (problem list, meds) pushed into legacy **custom fields** *only if* the legacy exposes a safe write API; (c) a flat CSV/HL7 ORU for labs if the legacy can import.
- **ABDM gateway** — the DMZ edge service: register HFR/HPR, link care contexts, receive consent artifacts, build + **Fidelius-encrypt** bundles, push to HIU, post status. Keys never persisted (`abdm_consent.artifact` stores metadata, not the keypair).

**Identity resolution (MPI):** blocking on (name-phonetic, DOB/age, gender, phone-hash) → score → `auto` (≥0.95) / `review_task(identity_match)` (0.80–0.95) / new patient (<0.80). ABHA verification, when available, promotes to `abha_verified` and becomes the durable key.

## C.7 Security, privacy, compliance

- **Data residency:** all PHI + inference on-prem; only the DMZ edge talks outbound, only encrypted consented bundles.
- **DPDP Act 2023 (India):** lawful basis = care + consent for sharing; purpose limitation; retention policy on `source_document`; data-principal rights workflow (export/erase) implemented against the adapter store.
- **De-identification** for the training corpus: Presidio + clinical regex + a NER de-id model; date-shifting per patient; manual QA on a sample; training data lives only on the RunPod/on-prem training volume, never leaves.
- **Crypto:** TLS/mTLS in transit; LUKS + Postgres TDE/`pgcrypto` for sensitive columns at rest; MinIO SSE; secrets in Vault/KMS.
- **RBAC:** roles = ingest-svc, reviewer (clinician), curator (terminology), admin, auditor; every state transition in `audit_log`; reviewer actions signed.
- **Model governance:** model registry with version + SHA + eval report; canary + rollback; drift monitors on confidence distributions and human-override rate.
- **Safety:** deterministic rules gate anything reaching the read model; medication changes always human-confirmed until per-prescriber accuracy proven; "entered-in-error" path everywhere (`clinical_status='entered-in-error'`, FHIR `Provenance` of the retraction).

---

# PART D — Worked example: one patient, seven documents

**Patient:** "Anjali Das", DOB approx 1979, MRN `LG-88213`. Seven scans arrive from folder watch.

| # | Scan | S2 class | S3 OCR path | S4 schema | Facts extracted (examples) |
|---|---|---|---|---|---|
| 1 | Handwritten OPD prescription, 12-Mar-2024, Bengali + English | `prescription`, cardiology, handwritten, `[bn,en]` | Qwen2.5-VL-7B (hand) + PaddleOCR (letterhead/stamp) | `prescription.v3` | Dx "HTN" ; Rx *Amlodipine 5 mg OD*, *Metformin 500 mg BD*, *Atorvastatin 10 mg HS* ; advice "low salt diet, review 2 wks" |
| 2 | Printed lab report, 14-Mar-2024 | `lab_report`, path, printed, `[en]` | PP-StructureV3 (table) | `lab_report.v3` | HbA1c 7.8 % ; FBS 142 mg/dL ; Creatinine 0.9 mg/dL ; LDL 132 mg/dL — each with ref range + flag |
| 3 | Radiology report (USG abdomen), 14-Mar-2024 | `radiology_report`, printed | PP-Structure + VLM narrative | `radiology.v3` | Impression "Grade I fatty liver" ; no focal lesion |
| 4 | Discharge summary, admit 02-Jun-2024 / disch 05-Jun-2024 | `discharge_summary`, surgery, printed | PP-Structure + DSLM NER | `discharge_summary.v3` | Dx "Acute appendicitis" ; **Procedure: Laparoscopic appendectomy, 03-Jun-2024** ; meds on discharge ; follow-up |
| 5 | Operative note (same admission) | `operative_note` | VLM + DSLM | `operative_note.v3` | Procedure "Laparoscopic appendectomy" ; surgeon ; findings "inflamed appendix" ; no complications |
| 6 | OPD follow-up advice slip, 20-Jun-2024, handwritten | `opd_note`, handwritten | Qwen2.5-VL-7B | `opd_note.v3` | "wound healthy, sutures out" ; continue Metformin ; **Atorvastatin stopped** |
| 7 | Vitals/symptom intake sheet, 20-Jun-2024 | `vitals_sheet`, printed form | PP-Structure (KV + checkboxes) | `vitals.v3` | BP 138/86 ; HR 78 ; Wt 71 kg ; Temp 98.4 °F ; symptom "mild incision-site pain" |

**S5 normalization (samples):**
- "HTN" → SNOMED `38341003 |Hypertensive disorder|` (conf 0.97); ICD-10 `I10` via `$translate`.
- "Metformin 500 mg" → SNOMED drug `109081006` + local formulary `FORM-METF-500`; dose 500 mg (UCUM `mg`), freq `BID` → 2/day.
- "HbA1c" → LOINC `4548-4`, unit `%` (UCUM), ref 4–5.6, flag `H`.
- "Laparoscopic appendectomy" → SNOMED `174041007`; ICD-9-CM-3 `47.01` secondary.
- "Grade I fatty liver" → SNOMED `442685003 |Steatosis of liver|` + qualifier; kept as `DiagnosticReport` conclusion + `Observation`.

**S6 validation & conflicts:**
- Doc 4 says appendectomy; Doc 1/2/3 don't mention it → `evidence_state = UNKNOWN_NOT_MENTIONED` → **no conflict**, retain the procedure.
- Doc 5 procedure == Doc 4 procedure, same date window → `dedup_key` match → **merge** into one `Procedure` fact, provenance from both docs.
- Doc 6 "Atorvastatin stopped" vs Doc 1 "Atorvastatin 10 mg HS" → **not a contradiction**: temporal — `MedicationRequest` (12-Mar) then `clinical_status='stopped'` effective 20-Jun. Medication list shows Atorvastatin = stopped.
- HbA1c 7.8 % passes range sanity (0–20); FBS 142 passes; Creatinine 0.9 passes. No blocker.
- Metformin dose 500 mg BID within adult range → OK, but **handwritten prescription med line** → forced "review sample" → `review_task(handwriting)` low-priority.

**S7 reconciliation:** builds the longitudinal set; orders encounters (2024-03-12 OPD, 2024-06-02→05 IP, 2024-06-20 OPD); links each to a care context.

**S8 review:** clinician opens the 3 queued tasks (identity match for "Anjali Das" vs an existing "Anjali Das" MRN `LG-88213`; the Metformin handwriting sample; one unmapped local lab code "Sr. Bilirubin (T)" → binds LOINC `1975-2`). Accepts all with one correction (OCR read "Amlodipine 5" that a rule flagged as possibly "0.5" — clinician confirms 5 mg).

**S9 output — rows:**
- `clinical_fact`: 1 `condition` (HTN), 1 `condition` (appendicitis, resolved), 1 `procedure`, 3 `medication` (+`medication_detail`), 4 `lab_result` (report 2) + 1 `diagnostic_report` (radiology), 4 `vital_sign`, 1 `symptom`, 2 `advice`. Each with provenance rows + confidence + `review_state`.
- `encounter`: 3 rows.
- `fhir_resource`: `Patient`, 3 `Encounter`, 2 `Condition`, 1 `Procedure`, 3 `MedicationRequest` + 1 `MedicationStatement` (Atorvastatin stopped), 5 `Observation` (labs) + 4 `Observation` (vitals) + 1 `Observation` (symptom), 2 `DiagnosticReport`, 7 `DocumentReference` (+`Binary`), N `Provenance`.

**S9 output — JSON (bundles):**
- `PrescriptionRecord` bundle for the 12-Mar encounter (Composition + MedicationRequests + Condition + DocumentReference of scan #1 + Provenance).
- `DiagnosticReportRecord` for 14-Mar (labs + radiology + scans #2,#3).
- `DischargeSummaryRecord` for the June admission (Condition + Procedure + meds + follow-up + scans #4,#5).
- `OPConsultRecord` + `WellnessRecord` for 20-Jun (advice + vitals + symptom + scans #6,#7).

Each `fhir_bundle` row: `bundle` JSON, `bundle_hash`, `ig_package='nrces.fhir.r4.ndhm#6.5.0'`, `validation_status='valid'`, `status='ready_to_share'`. On patient consent, `abdm_transfer` encrypts and pushes.

---

# PART E — Implementation plan (phased, RunPod-tested)

Each phase: **scope → deliverables → models/data → acceptance gate**. Phases 1–4 give an end-to-end thin slice; 5–8 harden to production.

### Phase 0 — Foundations (1–1.5 wk)
- **Scope:** repo, IaC, Postgres schema (Part B DDL), MinIO, Redis, vLLM up on RunPod, model registry, CI, synthetic data generator, de-id pipeline, eval harness skeleton.
- **Deliverables:** `docker-compose.yml` (app) + RunPod pod spec; Alembic migrations; `make dev` brings up the stack; 50 hand-labelled gold documents (mix of types, languages, handwriting) + 500 synthetic.
- **Gate:** `pytest` green; a scan can be POSTed and lands as `source_document` + pages in MinIO; vLLM serves Qwen2.5-VL-7B and Qwen2.5-7B with an OpenAI call.

### Phase 1 — Ingestion + Classification + OCR (1.5–2 wk)
- **Scope:** S1–S3. Legacy connector (folder-watch first), PDF split, deskew/denoise, dedupe; classifier (VLM few-shot); OCR ensemble (PaddleOCR PP-StructureV3 + Surya + Qwen2.5-VL for handwriting) writing `ocr_block` with bbox.
- **Models:** Qwen2.5-VL-7B (classify + hand), PP-StructureV3, Surya.
- **Gate:** doc-type accuracy ≥ 0.92 on gold; printed-text CER ≤ 0.03; table cell F1 ≥ 0.85; handwriting word accuracy ≥ 0.70 (baseline) with bbox for every block; full evidence viewer (image + overlaid boxes).

### Phase 2 — Evidence store + Structured extraction (2–3 wk)
- **Scope:** S4. Per-`doc_type` JSON schemas (`prescription.v3`, `lab_report.v3`, `radiology.v3`, `discharge_summary.v3`, `operative_note.v3`, `opd_note.v3`, `vitals.v3`); XGrammar-constrained decoding; `evidence_map` (every field → ocr_block ids + char spans); VLM path for spatial/handwritten fields, DSLM path for narrative NER + relations. **First LoRA fine-tune** of `cdi-dslm` (Part F) on synthetic + de-identified gold.
- **Gate:** JSON schema-valid 100%; field-level extraction F1 ≥ 0.85 macro (≥ 0.90 for printed labs/vitals); every non-null field has ≥1 evidence span; hallucination rate (fields with no supporting OCR span) ≤ 1%.

### Phase 3 — Terminology + Clinical validation (2 wk)
- **Scope:** S5–S6. Snowstorm-lite + LOINC/ICD CodeSystems; SapBERT/BioLORD + FAISS candidate gen + reranker; UCUM parsing; drug→formulary map; deterministic rules engine (value ranges, dose ranges by drug/age, unit checks, temporal possibility, duplicate detection); three-valued evidence; confidence fusion + isotonic calibrator.
- **Gate:** terminology top-1 accuracy ≥ 0.85 (conditions), ≥ 0.90 (labs→LOINC), ≥ 0.88 (drugs); `$validate-code` passes for all bound codes; rules catch 100% of a seeded error set (wrong units, impossible dates, 10× dose); calibration ECE ≤ 0.05.

### Phase 4 — FHIR projection + Bundle + rows (2 wk)
- **Scope:** S9. Deterministic `clinical_fact` → FHIR mapping engine; `Provenance` for every resource; ABDM `Composition` builders for the 8 artifacts; HAPI validator with `hl7.fhir.r4.core` + `nrces.fhir.r4.ndhm`; write `fhir_resource`, `fhir_bundle`; read-model materialized views.
- **Gate:** 100% of generated bundles pass HAPI validation against the ABDM IG (zero `error`, warnings triaged); round-trip test (bundle → parse → compare to `clinical_fact`) lossless for coded data; the Part D 7-doc fixture reproduces the expected rows + 4 bundles.

### Phase 5 — Reconciliation agent + Review console (2–3 wk)
- **Scope:** S7–S8. LangGraph patient super-graph; reconciliation agent with bounded tools (term lookup, prior-fact query, dose DB, date math), cite-or-drop; `review_task` queue with priorities + SLA; React review console (image+bbox ⇄ fact, accept/correct/reject, identity match, terminology curation); corrections logged as training data.
- **Gate:** on a 30-patient multi-doc set: conflict precision/recall ≥ 0.90 / ≥ 0.85; zero auto-committed agent writes reach the read model; reviewer throughput ≥ 20 docs/hr; every correction produces a training record.

### Phase 6 — Legacy write-back + ABDM gateway (2–3 wk)
- **Scope:** CDC connector for the real legacy doc table; FHIR façade (HAPI) over validated resources; optional summary-PDF write-back; DMZ edge service: HFR/HPR registration, care-context linking, consent-artifact intake, Fidelius encryption, HIU push, status callback; `abdm_care_context/consent/transfer`.
- **Gate:** ABDM sandbox: successful ABHA link, consent flow, and encrypted bundle delivery for all 4 artifact types; keys never persisted (audited); write-back files a valid new document against the correct MRN without touching existing rows.

### Phase 7 — Eval, monitoring, hardening, pilot (3–4 wk)
- **Scope:** full eval harness (extraction F1, terminology acc, FHIR validity, hallucination, calibration, end-to-end record accuracy vs clinician gold); load test on RunPod (throughput, GPU headroom, queue depth); observability dashboards; drift monitors (confidence dist, override rate, unmapped-code rate); DR runbook; security review; **shadow-mode pilot** on real historical documents with clinician adjudication.
- **Gate:** end-to-end coded-fact accuracy ≥ 0.95 post-review on a blinded 100-patient set; ≤ 15% of facts require human review at target thresholds; p95 per-document latency ≤ 90 s on the prod GPU; sign-off from clinical lead.

**Indicative timeline:** ~16–22 weeks to production pilot with a 3–4 person team (1 ML, 1–2 backend, 1 clinical informatics / part-time clinician reviewers).

---

# PART F — Fine-tuning plan for the DSLM (`cdi-dslm`)

**"DSLM" = Domain-Specific (Small) Language Model** — a 7–8B model tuned for exactly three jobs: (1) schema-locked extraction from OCR text + layout hints, (2) clinical NER + relation/temporal extraction from narrative, (3) generating normalization queries + short rationales for the reconciliation agent. It is **not** a diagnostic model and never asserts clinical facts unsupported by evidence.

| Item | Choice |
|---|---|
| **Base model** | `Qwen2.5-7B-Instruct` (Apache-2.0, clean for on-prem commercial). Evaluate `Meditron3-8B` and `Llama-3.1-8B-Instruct` as alternates; pick on eval, not vibes. |
| **Method** | **QLoRA** (4-bit NF4) → LoRA rank 16–32, α 32, dropout 0.05, target all attention + MLP proj; 1–3 epochs; lr 1e-4 cosine; seq len 8k (labs/discharge can be long); pack sequences. |
| **Serving** | vLLM with the LoRA adapter hot-loaded (multi-LoRA if per-doc-type adapters help); **XGrammar** grammar per `doc_type` schema so output is always valid JSON. |
| **Task format** | instruction + `OCR_BLOCKS` (text with `[b123]` ids and coarse coords) + `TARGET_SCHEMA` → JSON where every value carries `evidence: ["b123", ...]`. Reject-and-retry on schema failure; abstain (`null` + reason) when unsupported. |
| **Training data** | (a) **Synthetic** — template + LLM-generated prescriptions/labs/discharge notes in Indian formats, English + Bengali/Hindi, with programmatic gold JSON (thousands, cheap, covers rare drugs/units). (b) **De-identified real** — gold-labelled by clinical informatics using the review console; date-shifted, PHI-stripped; start ~500, grow with Phase 5 corrections (active learning on low-confidence + overridden cases). (c) **Hard negatives** — ambiguous doses ("500?"), smudged units, conflicting dates — labelled with the correct **abstention**. |
| **Terminology** | Do **not** fine-tune codes into the model. The model emits `local_text`; SapBERT+FAISS binds. (Optionally train a small reranker on hospital-confirmed binds.) |
| **Eval (gate for every checkpoint)** | field-level **extraction F1** per doc_type; **numeric exactness** (dose, result value, unit) — must be ~1.0 for labs/vitals; **evidence grounding** (% fields with a real supporting span) ≥ 99%; **hallucination rate** ≤ 1%; **abstention correctness** on hard negatives ≥ 0.9; downstream **FHIR validity** ≥ 99%; **calibration** ECE ≤ 0.05. Held-out set is clinician-gold, never seen in training. |
| **RunPod recipe** | 1× A100 80GB; `axolotl` or `trl` for QLoRA; dataset + adapters on the persistent network volume; nightly eval job writes a report to the model registry; promote adapter only if it beats current on the gate metrics with no safety regression. |
| **Governance** | every adapter = version + SHA + training-data manifest + eval report; canary at 10% traffic; auto-rollback if human-override rate rises > 20% relative. |

---

# PART G — Tech stack summary & repo layout

**Stack:** Python 3.11 · FastAPI · Celery/Redis (or Temporal) · PostgreSQL 16 + pgvector · MinIO · vLLM · PaddleOCR PP-StructureV3 · Surya · Qwen2.5-VL-7B · `cdi-dslm` (Qwen2.5-7B + LoRA) · SapBERT/BioLORD + FAISS · XGrammar · LangGraph · HAPI FHIR (validator + terminology + façade) · Snowstorm-lite · React (review console) · OpenTelemetry/Prometheus/Grafana/Loki · Docker/Podman + K3s · Vault.

```
clinical-emr-adapter/
├── docs/                     DESIGN.md (this), ADRs, IG notes, country packs
├── infra/
│   ├── compose/              app stack compose files
│   ├── runpod/               pod spec, volume layout, bootstrap
│   └── k3s/                  prod manifests
├── db/
│   ├── migrations/           Alembic (Part B DDL)
│   └── views/                read-model materialized views
├── services/
│   ├── ingest/               S1 legacy connectors, dedupe, page render
│   ├── classify/             S2
│   ├── ocr/                  S3 ensemble + evidence writer
│   ├── extract/              S4 schemas/, xgrammar grammars/, prompts/
│   ├── terminology/          S5 linker, FAISS build, Snowstorm client, ConceptMaps
│   ├── validate/             S6 rules engine, three-valued evidence, confidence+calibration
│   ├── reconcile/            S7 LangGraph graphs, agent tools
│   ├── review/               S8 API + React app
│   ├── fhir/                 S9 mapping engine, Composition builders, HAPI validation client
│   ├── abdm/                 DMZ edge: HRP client, Fidelius, consent, transfer
│   └── common/               db, models registry, auth/RBAC, audit, telemetry
├── ml/
│   ├── data/                 synthetic generators, de-id, gold manifest
│   ├── train/                QLoRA configs (axolotl/trl)
│   └── eval/                 harness + metrics + reports
├── schemas/                  JSON Schemas: prescription.v3.json, lab_report.v3.json, ...
├── fixtures/                 Part D 7-doc golden fixture + expected rows/bundles
└── tests/                    unit + integration + e2e (the fixture)
```

**Key API contracts (internal):**
- `POST /ingest` `{legacy_ref, bytes|uri, legacy_patient_ref, channel}` → `source_document.id`
- `GET /documents/{id}/evidence` → pages + `ocr_block[]` for the viewer
- `GET /patients/{id}/record` → read-model (problem/med/allergy lists, results grid, timeline)
- `GET /patients/{id}/bundles?artifact=PrescriptionRecord` → validated FHIR `Bundle` JSON
- `POST /review/tasks/{id}/resolve` `{action, corrections[]}` → state transition + training record
- `POST /abdm/care-contexts/link`, `POST /abdm/consents`, `POST /abdm/transfers/{id}/push`

---

# PART H — What "production-grade & succeed" actually requires

1. **Evidence + provenance are first-class, not a feature.** Every fact → pixel. Without it there is no clinician trust, no audit, no reprocessing.
2. **The LLM is never the source of truth.** Deterministic rules + terminology + FHIR validation + human review gate the record. Agents propose; they don't commit.
3. **Three-valued reasoning** (`SUPPORTED`/`CONTRADICTED`/`UNKNOWN_NOT_MENTIONED`) — absence of mention is not contradiction.
4. **Per-class empirical thresholds**, calibrated. A printed HbA1c and a handwritten drug dose are not the same risk.
5. **Don't build one giant longitudinal FHIR blob.** Build per-encounter resources + bundles; derive the longitudinal summary as a view.
6. **The scan always ships.** Worst case = a valid `HealthDocumentRecord` with the image. The record is never worse than today.
7. **Terminology is an architecture, not a table** — storage/candidate/rerank/validate/map/version/local-codes, all swappable per country.
8. **Two swappable packages** isolate every jurisdiction difference: Terminology Package + Profile/IG & Document-format Package. Same pipeline for India → US/UK/JP/KR/CN/DE/FR/CA/AU.
9. **Honest cost framing:** on-prem OSS, **no per-document API cost** — but real hardware/engineering/annotation/terminology-licence/ops cost.
10. **Human-in-the-loop is the product**, not a fallback. The review console + its correction stream is what makes accuracy climb over time.
11. **Safety valves:** `entered-in-error` everywhere; medication changes human-confirmed; drift monitors on override rate; canary + rollback on models.
12. **Pilot in shadow mode** on real historical docs with clinician adjudication before anything is trusted or shared.

---

## Sources

- Epic Chronicles / Clarity: [mindbowser][epic1], [docsity — master files][epic2], [sdrfoundation — Clarity][epic3]
- Oracle/Cerner Millennium: [boristyukin — Millennium data model][cerner1], [tactionsoft — Cerner/Oracle integration][cerner2]
- OpenMRS / Bahmni: [OpenMRS Wiki — Data Model][omrs1], [Bahmni Wiki — metadata management][omrs2]
- FHIR resource model: [HAPI FHIR docs][fhir1]
- ABDM / NRCeS: [NRCeS FHIR IG for ABDM][abdm4], [NRCeS DiagnosticReportRecord][abdm5], [ABDM FHR framework components][abdm1], [Nirmitee — HIP M2 guide][abdm2], [Bahmni as HIP][abdm3]
- USA: [HealthIT — USCDI][us1], [HL7 US Core IG][us2]
- UK: [NHS — GP Connect Access Record: Structured][uk1], [NHS England — Interoperability][uk2]
- Japan: [JAMI — SS-MIX2 specs][jp1], [Springer — SS-MIX Structured Standardized Storage][jp2]
- South Korea: [e-HIR — Health data standardization in Korea][kr1], [e-HIR — MyHealthWay status][kr2]
- China: [PMC — Clinical Document Standards in China][cn1], [Wiley — EMR standard compliance (TCM)][cn2]
- Germany: [Health-Samurai — FHIR in Germany (ISiK/MII)][de1], [gematik — ePA für alle][de2]
- France: [G_NIUS — DMP / Mon espace santé][fr1], [ANS — DMP security & interoperability référentiel][fr2]
- Canada: [Canada Health Infoway — CA Core+][ca1], [Infoway — pCLOCD][ca2]
- Australia: [Australian Digital Health Agency — Interoperability][au1], [NCTS — SNOMED CT-AU Technical Implementation Guide][au2]
- Models/tooling: [Meditron / Meta][ml1], [OpenBioLLM-8B][ml2], [SapBERT concept normalization][ml3], [vLLM structured decoding][ml4], [8 open-source OCR models compared][ml5], [Qwen2.5-VL][ml6]

[epic1]: https://www.mindbowser.com/epic-chronicles-the-backbone-of-epic-ehr-data-architecture/
[epic2]: https://www.docsity.com/en/docs/epic-cln251-252-exam-100-correct-answers/11076514/
[epic3]: https://sdrfoundation.org/epic-clarity-data-model-guide
[cerner1]: https://boristyukin.com/healthcare-analytics-with-cerner-part-2-cerner-millennium-data-model/
[cerner2]: https://www.tactionsoft.com/blog/cerner-oracle-health-integration-guide/
[omrs1]: https://wiki.openmrs.org/display/docs/Data%20Model
[omrs2]: https://bahmni.atlassian.net/wiki/spaces/BAH/pages/9633805/OpenMRS+meta+data+management
[fhir1]: https://hapifhir.io/hapi-fhir/docs/
[abdm1]: https://docs.coronasafe.network/abdm-documentation/overview-of-fhr-framework/fhr-framework-components-and-roles
[abdm2]: https://nirmitee.io/blog/building-abdm-hip-from-scratch-m2-flow-reference-architecture/
[abdm3]: https://bahmni.atlassian.net/wiki/spaces/BAH/pages/2901114904/Bahmni+as+Health+Information+Provider+ABDM+NDHM
[abdm4]: https://www.nrces.in/ndhm/fhir/r4/
[abdm5]: https://nrces.in/ndhm/fhir/r4/StructureDefinition-DiagnosticReportRecord.html
[us1]: https://healthit.gov/test-method/united-states-core-data-for-interoperability-uscdi/
[us2]: https://build.fhir.org/ig/HL7/US-Core/uscdi.html
[uk1]: https://digital.nhs.uk/developer/api-catalogue/gp-connect-access-record-structured-fhir
[uk2]: https://www.england.nhs.uk/digitaltechnology/connecteddigitalsystems/interoperability/
[jp1]: https://www.jami.jp/en/jamistd/ssmix2/
[jp2]: https://link.springer.com/chapter/10.1007/978-981-16-6376-5_10
[kr1]: https://e-hir.org/journal/view.php?doi=10.4258%2Fhir.2024.30.2.93
[kr2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11098772/
[cn1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3259555/
[cn2]: https://onlinelibrary.wiley.com/doi/10.1155/2020/8865264
[de1]: https://www.health-samurai.io/articles/fhir-adoption-in-germany
[de2]: https://fachportal.gematik.de/anwendungen/epa-fuer-alle
[fr1]: https://gnius.esante.gouv.fr/en/regulations/regulation-profiles/dossier-medical-partage-dmp
[fr2]: https://esante.gouv.fr/produits-services/referentiel-dmp
[ca1]: https://accelero.infoway-inforoute.ca/en/initiatives/pan-canadian-core-specifications
[ca2]: https://accelero.infoway-inforoute.ca/en/standards/terminology-standards/pclocd
[au1]: https://www.digitalhealth.gov.au/interoperability
[au2]: https://www.healthterminologies.gov.au/library/dh-3243-2020-snomed-ct-au-australian-technical-implementation-guide-v4-0.pdf
[ml1]: https://ai.meta.com/blog/llama-2-3-meditron-yale-medicine-epfl-open-source-llm/
[ml2]: https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B
[ml3]: https://journals.sagepub.com/doi/10.1177/20552076241288681
[ml4]: https://blog.vllm.ai/2025/01/14/struct-decode-intro.html
[ml5]: https://modal.com/blog/8-top-open-source-ocr-models-compared
[ml6]: https://github.com/QwenLM/Qwen2.5-VL
