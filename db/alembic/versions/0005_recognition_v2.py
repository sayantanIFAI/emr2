"""recognition v2 (ARCHITECTURE emr 6 §15-§16): immutable evidence, candidates, verified-fact
ledger, practitioner link + sample doctor master, drug / lab-order knowledge with alias classes,
normalized prescription tables, FHIR outbox + blob index, file-listener lifecycle, dispatch.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-29
"""
from __future__ import annotations

from alembic import op


def _exec(sql: str) -> None:
    # raw driver SQL: the seed JSON/ids contain ':40'-style text that sa.text() would
    # misread as bind parameters
    # straight to the DBAPI cursor (no params => psycopg does not parse % or :name)
    with op.get_bind().connection.dbapi_connection.cursor() as cur:
        cur.execute(sql)

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    _exec(r"""
-- =====================================================================
-- A. IMMUTABLE EVIDENCE  (E4-S7)  -  append-only, enforced by the database
-- =====================================================================
CREATE TABLE IF NOT EXISTS ocr_observation (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id       uuid NOT NULL REFERENCES source_document(id),
  page_id           uuid NOT NULL REFERENCES document_page(id),
  line_key          text NOT NULL,          -- groups the engines that read the same line crop
  region_kind       text NOT NULL CHECK (region_kind IN ('printed','handwritten','mixed','uncertain','page')),
  bbox              int[] NOT NULL CHECK (array_length(bbox, 1) = 4),
  polygon           jsonb,
  crop_hash         char(64),               -- sha256 of the crop cut from the SOURCE render
  field_domain      text,                   -- text|strength|dose|frequency|duration|lab_value|date|null
  engine            text NOT NULL,          -- rapidocr|qwen2.5-vl|...
  engine_version    text NOT NULL,
  prompt_hash       text,
  raw_text          text NOT NULL,
  raw_confidence    numeric(5,4),           -- NULL when the engine gives none (VLM) - never invented
  token_confidences jsonb,
  error             text,                   -- engine failed/unavailable for this crop
  run_id            uuid REFERENCES pipeline_run(id),
  supersedes        uuid[] NOT NULL DEFAULT '{}',   -- supersession is a pointer, never a delete
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_obs_document ON ocr_observation (document_id, page_id);
CREATE INDEX IF NOT EXISTS ix_obs_line ON ocr_observation (line_key);

CREATE OR REPLACE FUNCTION cdi_evidence_is_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'UPDATE' THEN
    RAISE EXCEPTION '% is append-only: UPDATE is not allowed (row %)', TG_TABLE_NAME, OLD.id
      USING ERRCODE = 'restrict_violation';
  END IF;
  -- DELETE only for a documented erasure (DPDP / E11-S7), set per transaction:
  --   SET LOCAL cdi.allow_evidence_erasure = 'on';
  IF coalesce(current_setting('cdi.allow_evidence_erasure', true), 'off') <> 'on' THEN
    RAISE EXCEPTION '% is append-only: DELETE requires cdi.allow_evidence_erasure', TG_TABLE_NAME
      USING ERRCODE = 'restrict_violation';
  END IF;
  RETURN OLD;
END $$;

DROP TRIGGER IF EXISTS trg_obs_immutable ON ocr_observation;
CREATE TRIGGER trg_obs_immutable BEFORE UPDATE OR DELETE ON ocr_observation
  FOR EACH ROW EXECUTE FUNCTION cdi_evidence_is_immutable();

CREATE OR REPLACE VIEW v_ocr_observation_current AS
  SELECT o.* FROM ocr_observation o
   WHERE NOT EXISTS (SELECT 1 FROM ocr_observation n WHERE o.id = ANY (n.supersedes));

-- ocr_block stays the working projection S4 reads (rebuildable); it now points at evidence
ALTER TABLE ocr_block ADD COLUMN IF NOT EXISTS observation_ids uuid[] NOT NULL DEFAULT '{}';
ALTER TABLE ocr_block ADD COLUMN IF NOT EXISTS recognition jsonb;

CREATE TABLE IF NOT EXISTS interpretation_candidate (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  fact_id           uuid REFERENCES clinical_fact(id) ON DELETE CASCADE,
  observation_ids   uuid[] NOT NULL DEFAULT '{}',
  domain            text NOT NULL,          -- drug|lab_order|lab_result|diagnosis|vital|practitioner
  concept_id        text,
  normalized_text   text NOT NULL,
  code_system       text, code text, code_display text,
  source            text NOT NULL CHECK (source IN ('literal','verified_alias','normalized_alias',
                      'generated_alias','doctor_alias','fuzzy','confusion_map','doctor_exemplar',
                      'global_exemplar','embedding','reranker','qwen_adjudication','reviewer','seed')),
  alias_class       char(1) CHECK (alias_class IN ('A','B','C')),
  resolved_by_level smallint CHECK (resolved_by_level BETWEEN 1 AND 8),
  score             numeric(5,4) NOT NULL DEFAULT 0,
  is_selected       bool NOT NULL DEFAULT false,
  collision         bool NOT NULL DEFAULT false,
  eliminated_by     text,
  evidence          jsonb NOT NULL DEFAULT '{}',
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_cand_fact ON interpretation_candidate (fact_id);

-- governed decisions: an immutable LEDGER (no FK to clinical_fact on purpose - re-extraction
-- may rebuild facts but can never erase what was verified, by whom, on what evidence)
CREATE TABLE IF NOT EXISTS verified_fact (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  fact_id               uuid NOT NULL,
  document_id           uuid,
  patient_id            uuid,
  fact_type             text NOT NULL,
  observation_ids       uuid[] NOT NULL DEFAULT '{}',
  winning_candidate_ids uuid[] NOT NULL DEFAULT '{}',
  code_system           text, code text, code_display text,
  governed_value        jsonb NOT NULL,
  verification_method   text NOT NULL CHECK (verification_method IN
                          ('auto_accepted','clinician_confirmed','corrected','rejected')),
  confidence            numeric(5,4),
  evidence_state        text,
  policy_id             text,
  reviewer_id           text,
  model_stack           jsonb NOT NULL DEFAULT '{}',
  decision_trace        jsonb NOT NULL DEFAULT '[]',
  verified_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_vf_fact ON verified_fact (fact_id, verified_at DESC);
DROP TRIGGER IF EXISTS trg_vf_immutable ON verified_fact;
CREATE TRIGGER trg_vf_immutable BEFORE UPDATE OR DELETE ON verified_fact
  FOR EACH ROW EXECUTE FUNCTION cdi_evidence_is_immutable();

ALTER TABLE clinical_fact ADD COLUMN IF NOT EXISTS evidence_state text;   -- printed|agree|disagree|single_engine|page_level
ALTER TABLE clinical_fact ADD COLUMN IF NOT EXISTS field_policy   text;
ALTER TABLE clinical_fact ADD COLUMN IF NOT EXISTS decision_trace jsonb;

-- =====================================================================
-- B. PRACTITIONER LINK (E6-S11) + SAMPLE DOCTOR MASTER (E18-S1) on cn_practitioner
-- =====================================================================
ALTER TABLE cn_practitioner ADD COLUMN IF NOT EXISTS designation  text;
ALTER TABLE cn_practitioner ADD COLUMN IF NOT EXISTS name_aliases text[] NOT NULL DEFAULT '{}';
ALTER TABLE cn_practitioner ADD COLUMN IF NOT EXISTS council      text;
ALTER TABLE cn_practitioner ADD COLUMN IF NOT EXISTS clinic       text;
ALTER TABLE cn_practitioner ADD COLUMN IF NOT EXISTS is_sample    bool NOT NULL DEFAULT false;
CREATE UNIQUE INDEX IF NOT EXISTS ux_cn_pract_reg
  ON cn_practitioner (council, registration_number) WHERE registration_number IS NOT NULL;

ALTER TABLE source_document ADD COLUMN IF NOT EXISTS practitioner_id uuid REFERENCES cn_practitioner(id);
ALTER TABLE source_document ADD COLUMN IF NOT EXISTS practitioner_link_method text
  CHECK (practitioner_link_method IN ('encounter','registration_no','printed_name','template','manual','unknown'));
ALTER TABLE source_document ADD COLUMN IF NOT EXISTS practitioner_link_confidence numeric(5,4);
ALTER TABLE source_document ADD COLUMN IF NOT EXISTS practitioner_evidence jsonb;

-- SAMPLE data (is_sample = true): fictional doctors for demos/tests only. Production mode refuses
-- to link to them. Confusable pairs (same surname / initials) are deliberate test fixtures.
INSERT INTO cn_practitioner (registration_number, council, name_prefix, name_given, name_family,
  name_full, name_aliases, qualification, specialty, department, designation, clinic, is_sample)
VALUES
 ('12345','WBMC','Dr','A.','Sen','Dr. A. Sen','{"A Sen","Arijit Sen","Dr A Sen"}','{MBBS,MD}','{General Medicine}','General Medicine','Consultant Physician','CareFlow Polyclinic',true),
 ('12346','WBMC','Dr','Arindam','Sen','Dr. Arindam Sen','{"A. Sen (Ortho)","Arindam Sen"}','{MBBS,MS (Ortho)}','{Orthopaedics}','Orthopaedics','Consultant Orthopaedic Surgeon','CareFlow Polyclinic',true),
 ('22001','WBMC','Dr','R.','Iyer','Dr. R. Iyer','{"R Iyer","Ramesh Iyer"}','{MBBS,MD (Biochem)}','{Biochemistry}','Laboratory','Consultant Biochemist','CareFlow Polyclinic',true),
 ('30011','WBMC','Dr','Sudipta','Banerjee','Dr. Sudipta Banerjee','{"S. Banerjee","S Banerjee"}','{MBBS,DM (Cardiology)}','{Cardiology}','Cardiology','Senior Consultant Cardiologist','CareFlow Polyclinic',true),
 ('30012','WBMC','Dr','Sayani','Banerjee','Dr. Sayani Banerjee','{"S. Banerjee (Derm)"}','{MBBS,MD (Derm)}','{Dermatology}','Dermatology','Consultant Dermatologist','CareFlow Polyclinic',true),
 ('30101','WBMC','Dr','Priyanka','Ghosh','Dr. Priyanka Ghosh','{"P. Ghosh"}','{MBBS,DNB (Endocrinology)}','{Endocrinology,Diabetology}','Endocrinology','Consultant Diabetologist','CareFlow Polyclinic',true),
 ('30102','WBMC','Dr','Pradip','Ghosh','Dr. Pradip Ghosh','{"P. Ghosh (ENT)"}','{MBBS,MS (ENT)}','{ENT}','ENT','Consultant ENT Surgeon','CareFlow Polyclinic',true),
 ('30201','WBMC','Dr','Anirban','Mukherjee','Dr. Anirban Mukherjee','{"A. Mukherjee"}','{MBBS,MD (Pulmonology)}','{Pulmonology}','Pulmonology','Consultant Pulmonologist','CareFlow Polyclinic',true),
 ('30202','WBMC','Dr','Tanushree','Chatterjee','Dr. Tanushree Chatterjee','{"T. Chatterjee"}','{MBBS,MS (OBG)}','{Obstetrics and Gynaecology}','OBG','Consultant Gynaecologist','CareFlow Polyclinic',true),
 ('30203','WBMC','Dr','Subhajit','Roy','Dr. Subhajit Roy','{"S. Roy"}','{MBBS,MS (Ortho)}','{Orthopaedics,Spine}','Orthopaedics','Consultant Spine Surgeon','CareFlow Polyclinic',true),
 ('30204','WBMC','Dr','Moumita','Das','Dr. Moumita Das','{"M. Das"}','{MBBS,MD (Paediatrics)}','{Paediatrics}','Paediatrics','Consultant Paediatrician','CareFlow Polyclinic',true),
 ('30205','WBMC','Dr','Kaushik','Bose','Dr. Kaushik Bose','{"K. Bose"}','{MBBS,DM (Neurology)}','{Neurology}','Neurology','Consultant Neurologist','CareFlow Polyclinic',true),
 ('30206','WBMC','Dr','Rituparna','Dutta','Dr. Rituparna Dutta','{"R. Dutta"}','{MBBS,DM (Gastroenterology)}','{Gastroenterology}','Gastroenterology','Consultant Gastroenterologist','CareFlow Polyclinic',true),
 ('30207','WBMC','Dr','Abhishek','Chakraborty','Dr. Abhishek Chakraborty','{"A. Chakraborty"}','{MBBS,MD (Psychiatry)}','{Psychiatry}','Psychiatry','Consultant Psychiatrist','CareFlow Polyclinic',true),
 ('30208','WBMC','Dr','Sanchita','Paul','Dr. Sanchita Paul','{"S. Paul"}','{MBBS,MS (Ophthalmology)}','{Ophthalmology}','Ophthalmology','Consultant Ophthalmologist','CareFlow Polyclinic',true),
 ('30209','WBMC','Dr','Debasish','Mondal','Dr. Debasish Mondal','{"D. Mondal"}','{MBBS,DM (Nephrology)}','{Nephrology}','Nephrology','Consultant Nephrologist','CareFlow Polyclinic',true),
 ('30210','WBMC','Dr','Nandini','Saha','Dr. Nandini Saha','{"N. Saha"}','{MBBS,MD (General Medicine)}','{General Medicine}','General Medicine','Senior Resident','CareFlow Polyclinic',true),
 ('30211','WBMC','Dr','Rahul','Sarkar','Dr. Rahul Sarkar','{"R. Sarkar"}','{MBBS,MS (General Surgery)}','{General Surgery}','Surgery','Consultant Surgeon','CareFlow Polyclinic',true),
 ('30212','WBMC','Dr','Indrani','Bhattacharya','Dr. Indrani Bhattacharya','{"I. Bhattacharya"}','{MBBS,MD (Rheumatology)}','{Rheumatology}','Rheumatology','Consultant Rheumatologist','CareFlow Polyclinic',true),
 ('30213','WBMC','Dr','Soumen','Halder','Dr. Soumen Halder','{"S. Halder"}','{MBBS,MD (Radiology)}','{Radiology}','Radiology','Consultant Radiologist','CareFlow Polyclinic',true),
 ('30214','WBMC','Dr','Arpita','Sen','Dr. Arpita Sen','{"A. Sen (Paed)"}','{MBBS,DCH}','{Paediatrics}','Paediatrics','Junior Consultant','CareFlow Polyclinic',true),
 ('30215','WBMC','Dr','Biswajit','Pal','Dr. Biswajit Pal','{"B. Pal"}','{MBBS,MD (Anaesthesia)}','{Anaesthesiology,Pain Medicine}','Pain Clinic','Consultant, Pain Medicine','CareFlow Polyclinic',true),
 ('30216','WBMC','Dr','Mitali','Kundu','Dr. Mitali Kundu','{"M. Kundu"}','{MBBS,MD (Physiology)}','{General Medicine}','OPD','Medical Officer','CareFlow Polyclinic',true),
 ('30217','WBMC','Dr','Tapas','Majumdar','Dr. Tapas Majumdar','{"T. Majumdar"}','{MBBS,MD (Cardiology)}','{Cardiology}','Cardiology','Associate Consultant','CareFlow Polyclinic',true),
 ('30218','WBMC','Dr','Shreya','Basu','Dr. Shreya Basu','{"S. Basu"}','{MBBS,MD (Pathology)}','{Pathology}','Laboratory','Consultant Pathologist','CareFlow Polyclinic',true)
ON CONFLICT DO NOTHING;

-- =====================================================================
-- C. KNOWLEDGE: drug products + lab/investigation orders + alias classes (E3-S6/S7/S8, E19)
--    SAMPLE seed; LOINC/SNOMED codes are only those already used in the as-built seed or
--    standard LOINC panel codes - verify against licensed releases (E3-S1) before production.
-- =====================================================================
CREATE TABLE IF NOT EXISTS kb_concept (
  id             text PRIMARY KEY,                   -- 'drug:telma-40', 'lab:kft'
  domain         text NOT NULL CHECK (domain IN ('drug','lab_order','diagnosis')),
  canonical_name text NOT NULL,
  code_system    text, code text, code_display text,
  attrs          jsonb NOT NULL DEFAULT '{}',        -- drug: generic,strength,unit,form,class,indications
                                                     -- lab: kind(test|panel),specimen,components
  is_sample      bool NOT NULL DEFAULT true,
  active         bool NOT NULL DEFAULT true
);
CREATE TABLE IF NOT EXISTS kb_alias (
  id              bigserial PRIMARY KEY,
  concept_id      text NOT NULL REFERENCES kb_concept(id) ON DELETE CASCADE,
  alias           text NOT NULL,
  alias_norm      text NOT NULL,
  alias_class     char(1) NOT NULL CHECK (alias_class IN ('A','B','C')),  -- A verified, B generated, C observed-doctor
  practitioner_id uuid REFERENCES cn_practitioner(id),
  source          text NOT NULL DEFAULT 'seed',
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (concept_id, alias_norm, alias_class, practitioner_id)
);
CREATE INDEX IF NOT EXISTS ix_kb_alias_norm ON kb_alias (alias_norm);
CREATE TABLE IF NOT EXISTS kb_indication_group (
  code      text PRIMARY KEY,
  label     text NOT NULL,
  keywords  text[] NOT NULL
);

INSERT INTO kb_indication_group (code, label, keywords) VALUES
 ('cardiac','Cardiac / chest pain / CAD','{chest pain,angina,cad,coronary,ihd,mi,myocardial,acs,heart failure,palpitation,arrhythmia}'),
 ('hypertension','Hypertension','{hypertension,htn,high blood pressure,raised bp}'),
 ('diabetes','Diabetes','{diabetes,t2dm,dm,hyperglycaemia,hyperglycemia,sugar}'),
 ('lipid','Dyslipidaemia','{dyslipidaemia,dyslipidemia,hyperlipidaemia,hyperlipidemia,cholesterol}'),
 ('msk_spine','Spine / back / musculoskeletal pain','{back pain,low back,lbp,lumbar,cervical,spondylosis,spondylitis,sciatica,pivd,disc,spine,radiculopathy,neck pain,myalgia,muscle spasm,sprain}'),
 ('pain_fever','Pain / fever (general)','{pain,fever,pyrexia,headache,body ache}'),
 ('gi_acid','Acid-peptic / GI','{gastritis,gerd,acidity,dyspepsia,ulcer,reflux,heartburn}'),
 ('allergy_resp','Allergy / respiratory','{rhinitis,allergy,allergic,asthma,urticaria,sneezing,cough}'),
 ('infection','Infection','{infection,uti,pharyngitis,tonsillitis,bronchitis,pneumonia,sinusitis,cellulitis}'),
 ('thyroid','Thyroid','{hypothyroidism,thyroid,hashimoto}'),
 ('neuropathic','Neuropathic pain','{neuropathy,neuropathic,neuralgia,radiculopathy,sciatica,tingling,numbness}')
ON CONFLICT DO NOTHING;

INSERT INTO kb_concept (id, domain, canonical_name, code_system, code, code_display, attrs) VALUES
 ('drug:telma-40','drug','Telma 40','http://snomed.info/sct','395892000','Telmisartan','{"generic":"telmisartan","strength":40,"unit":"mg","form":"tablet","class":"ARB","indications":["hypertension","cardiac"],"strengths_available":[20,40,80]}'),
 ('drug:telma-20','drug','Telma 20','http://snomed.info/sct','395892000','Telmisartan','{"generic":"telmisartan","strength":20,"unit":"mg","form":"tablet","class":"ARB","indications":["hypertension","cardiac"],"strengths_available":[20,40,80]}'),
 ('drug:telmikind-40','drug','Telmikind 40','http://snomed.info/sct','395892000','Telmisartan','{"generic":"telmisartan","strength":40,"unit":"mg","form":"tablet","class":"ARB","indications":["hypertension","cardiac"],"strengths_available":[20,40,80]}'),
 ('drug:metformin-500','drug','Metformin 500','http://snomed.info/sct','372567009','Metformin','{"generic":"metformin","strength":500,"unit":"mg","form":"tablet","class":"biguanide","indications":["diabetes"],"strengths_available":[500,850,1000]}'),
 ('drug:glycomet-500','drug','Glycomet 500','http://snomed.info/sct','372567009','Metformin','{"generic":"metformin","strength":500,"unit":"mg","form":"tablet","class":"biguanide","indications":["diabetes"],"strengths_available":[250,500,850,1000]}'),
 ('drug:amlodipine-5','drug','Amlodipine 5','http://snomed.info/sct','386864001','Amlodipine','{"generic":"amlodipine","strength":5,"unit":"mg","form":"tablet","class":"CCB","indications":["hypertension","cardiac"],"strengths_available":[2.5,5,10]}'),
 ('drug:atorvastatin-10','drug','Atorvastatin 10','http://snomed.info/sct','373444002','Atorvastatin','{"generic":"atorvastatin","strength":10,"unit":"mg","form":"tablet","class":"statin","indications":["lipid","cardiac"],"strengths_available":[10,20,40,80]}'),
 ('drug:ecosprin-75','drug','Ecosprin 75','http://snomed.info/sct','387458008','Aspirin','{"generic":"aspirin","strength":75,"unit":"mg","form":"tablet","class":"antiplatelet","indications":["cardiac"],"strengths_available":[75,150]}'),
 ('drug:sorbitrate-5','drug','Sorbitrate 5',NULL,NULL,NULL,'{"generic":"isosorbide dinitrate","strength":5,"unit":"mg","form":"tablet","class":"nitrate","indications":["cardiac"],"strengths_available":[5,10]}'),
 ('drug:clopidogrel-75','drug','Clopidogrel 75',NULL,NULL,NULL,'{"generic":"clopidogrel","strength":75,"unit":"mg","form":"tablet","class":"antiplatelet","indications":["cardiac"],"strengths_available":[75]}'),
 ('drug:pantocid-40','drug','Pantocid 40','http://snomed.info/sct','395728002','Pantoprazole','{"generic":"pantoprazole","strength":40,"unit":"mg","form":"tablet","class":"PPI","indications":["gi_acid"],"strengths_available":[20,40]}'),
 ('drug:pan-40','drug','Pan 40','http://snomed.info/sct','395728002','Pantoprazole','{"generic":"pantoprazole","strength":40,"unit":"mg","form":"tablet","class":"PPI","indications":["gi_acid"],"strengths_available":[20,40]}'),
 ('drug:paracetamol-650','drug','Paracetamol 650','http://snomed.info/sct','387517004','Paracetamol','{"generic":"paracetamol","strength":650,"unit":"mg","form":"tablet","class":"analgesic","indications":["pain_fever","msk_spine"],"strengths_available":[500,650]}'),
 ('drug:aceclofenac-100','drug','Aceclofenac 100',NULL,NULL,NULL,'{"generic":"aceclofenac","strength":100,"unit":"mg","form":"tablet","class":"NSAID","indications":["msk_spine","pain_fever"],"strengths_available":[100]}'),
 ('drug:thiocolchicoside-4','drug','Thiocolchicoside 4',NULL,NULL,NULL,'{"generic":"thiocolchicoside","strength":4,"unit":"mg","form":"tablet","class":"muscle relaxant","indications":["msk_spine"],"strengths_available":[4,8]}'),
 ('drug:pregabalin-75','drug','Pregabalin 75',NULL,NULL,NULL,'{"generic":"pregabalin","strength":75,"unit":"mg","form":"capsule","class":"gabapentinoid","indications":["neuropathic","msk_spine"],"strengths_available":[50,75,150]}'),
 ('drug:etoricoxib-90','drug','Etoricoxib 90',NULL,NULL,NULL,'{"generic":"etoricoxib","strength":90,"unit":"mg","form":"tablet","class":"NSAID","indications":["msk_spine","pain_fever"],"strengths_available":[60,90,120]}'),
 ('drug:montek-lc','drug','Montek LC',NULL,NULL,NULL,'{"generic":"montelukast + levocetirizine","strength":null,"unit":null,"form":"tablet","class":"antiallergic","indications":["allergy_resp"]}'),
 ('drug:azithromycin-500','drug','Azithromycin 500','http://snomed.info/sct','387531004','Azithromycin','{"generic":"azithromycin","strength":500,"unit":"mg","form":"tablet","class":"macrolide","indications":["infection"],"strengths_available":[250,500]}'),
 ('drug:thyronorm-50','drug','Thyronorm 50','http://snomed.info/sct','710809001','Levothyroxine','{"generic":"levothyroxine","strength":50,"unit":"mcg","form":"tablet","class":"thyroid hormone","indications":["thyroid"],"strengths_available":[25,50,75,100]}'),
 ('lab:cbc','lab_order','Complete blood count','http://loinc.org','58410-2','CBC panel - Blood by Automated count','{"kind":"panel","specimen":"blood"}'),
 ('lab:kft','lab_order','Kidney function test (KFT/RFT)','http://loinc.org','24362-6','Renal function 2000 panel - Serum or Plasma','{"kind":"panel","specimen":"serum","components_likely":["urea","creatinine"],"components_optional":["uric acid","sodium","potassium"],"note":"composition varies by lab - never expanded into ordered tests"}'),
 ('lab:lft','lab_order','Liver function test','http://loinc.org','24325-3','Hepatic function 2000 panel - Serum or Plasma','{"kind":"panel","specimen":"serum"}'),
 ('lab:lipid','lab_order','Lipid profile','http://loinc.org','57698-3','Lipid panel with direct LDL - Serum or Plasma','{"kind":"panel","specimen":"serum"}'),
 ('lab:hba1c','lab_order','HbA1c','http://loinc.org','4548-4','Hemoglobin A1c/Hemoglobin.total in Blood','{"kind":"test","specimen":"blood"}'),
 ('lab:fbs','lab_order','Fasting blood sugar','http://loinc.org','1558-6','Fasting glucose [Mass/volume] in Serum or Plasma','{"kind":"test","specimen":"serum"}'),
 ('lab:ppbs','lab_order','Post-prandial blood sugar','http://loinc.org','1521-4','Glucose [Mass/volume] in Serum or Plasma --2 hours post meal','{"kind":"test","specimen":"serum"}'),
 ('lab:tsh','lab_order','TSH','http://loinc.org','3016-3','Thyrotropin [Units/volume] in Serum or Plasma','{"kind":"test","specimen":"serum"}'),
 ('lab:s-creatinine','lab_order','Serum creatinine','http://loinc.org','2160-0','Creatinine [Mass/volume] in Serum or Plasma','{"kind":"test","specimen":"serum"}'),
 ('lab:u-creatinine','lab_order','Urine creatinine',NULL,NULL,NULL,'{"kind":"test","specimen":"urine"}'),
 ('lab:urea','lab_order','Blood urea','http://loinc.org','3094-0','Urea nitrogen [Mass/volume] in Serum or Plasma','{"kind":"test","specimen":"serum"}'),
 ('lab:esr','lab_order','ESR','http://loinc.org','4537-7','Erythrocyte sedimentation rate by Westergren method','{"kind":"test","specimen":"blood"}'),
 ('lab:urine-re','lab_order','Urine routine examination','http://loinc.org','24357-6','Urinalysis macro (dipstick) panel - Urine','{"kind":"panel","specimen":"urine"}'),
 ('lab:ecg','lab_order','ECG',NULL,NULL,NULL,'{"kind":"test","specimen":null}'),
 ('lab:xray-ls','lab_order','X-ray lumbosacral spine',NULL,NULL,NULL,'{"kind":"test","specimen":null}')
ON CONFLICT DO NOTHING;

-- class-A (verified) aliases; class-B generated aliases are produced at load time and never
-- promoted to A automatically; class-C doctor aliases arrive from adjudication (Phase D).
INSERT INTO kb_alias (concept_id, alias, alias_norm, alias_class) VALUES
 ('drug:telma-40','Telma 40','telma 40','A'), ('drug:telma-40','Telma-40','telma 40','A'),
 ('drug:telma-20','Telma 20','telma 20','A'),
 ('drug:telmikind-40','Telmikind 40','telmikind 40','A'),
 ('drug:metformin-500','Metformin 500','metformin 500','A'), ('drug:metformin-500','Metformin 500 mg','metformin 500 mg','A'),
 ('drug:glycomet-500','Glycomet 500','glycomet 500','A'),
 ('drug:amlodipine-5','Amlodipine 5','amlodipine 5','A'), ('drug:amlodipine-5','Amlong 5','amlong 5','A'),
 ('drug:atorvastatin-10','Atorvastatin 10','atorvastatin 10','A'), ('drug:atorvastatin-10','Atorva 10','atorva 10','A'),
 ('drug:ecosprin-75','Ecosprin 75','ecosprin 75','A'),
 ('drug:sorbitrate-5','Sorbitrate 5','sorbitrate 5','A'),
 ('drug:clopidogrel-75','Clopidogrel 75','clopidogrel 75','A'), ('drug:clopidogrel-75','Clopitab 75','clopitab 75','A'),
 ('drug:pantocid-40','Pantocid 40','pantocid 40','A'),
 ('drug:pan-40','Pan 40','pan 40','A'), ('drug:pan-40','Pan-40','pan 40','A'),
 ('drug:paracetamol-650','Paracetamol 650','paracetamol 650','A'), ('drug:paracetamol-650','Dolo 650','dolo 650','A'),
 ('drug:aceclofenac-100','Aceclofenac 100','aceclofenac 100','A'), ('drug:aceclofenac-100','Hifenac','hifenac','A'),
 ('drug:thiocolchicoside-4','Thiocolchicoside 4','thiocolchicoside 4','A'), ('drug:thiocolchicoside-4','Myoril 4','myoril 4','A'),
 ('drug:pregabalin-75','Pregabalin 75','pregabalin 75','A'), ('drug:pregabalin-75','Pregalin 75','pregalin 75','A'),
 ('drug:etoricoxib-90','Etoricoxib 90','etoricoxib 90','A'), ('drug:etoricoxib-90','Etoshine 90','etoshine 90','A'),
 ('drug:montek-lc','Montek LC','montek lc','A'),
 ('drug:azithromycin-500','Azithromycin 500','azithromycin 500','A'), ('drug:azithromycin-500','Azee 500','azee 500','A'),
 ('drug:thyronorm-50','Thyronorm 50','thyronorm 50','A'),
 ('lab:cbc','CBC','cbc','A'), ('lab:cbc','Complete blood count','complete blood count','A'), ('lab:cbc','CBP','cbp','A'),
 ('lab:kft','KFT','kft','A'), ('lab:kft','RFT','rft','A'), ('lab:kft','Kidney function test','kidney function test','A'), ('lab:kft','Renal function test','renal function test','A'),
 ('lab:lft','LFT','lft','A'), ('lab:lft','Liver function test','liver function test','A'),
 ('lab:lipid','Lipid profile','lipid profile','A'), ('lab:lipid','Lipid','lipid','A'), ('lab:lipid','FLP','flp','A'),
 ('lab:hba1c','HbA1c','hba1c','A'), ('lab:hba1c','A1c','a1c','A'), ('lab:hba1c','Glycated haemoglobin','glycated haemoglobin','A'),
 ('lab:fbs','FBS','fbs','A'), ('lab:fbs','Fasting blood sugar','fasting blood sugar','A'),
 ('lab:ppbs','PPBS','ppbs','A'), ('lab:ppbs','PP blood sugar','pp blood sugar','A'),
 ('lab:tsh','TSH','tsh','A'),
 ('lab:s-creatinine','S. Creatinine','s creatinine','A'), ('lab:s-creatinine','Sr. Creatinine','sr creatinine','A'),
 ('lab:s-creatinine','Serum creatinine','serum creatinine','A'), ('lab:s-creatinine','S.Creat','s creat','A'),
 ('lab:s-creatinine','Sr Creat','sr creat','A'), ('lab:s-creatinine','SCr','scr','A'),
 ('lab:u-creatinine','Urine creatinine','urine creatinine','A'),
 ('lab:urea','Urea','urea','A'), ('lab:urea','Blood urea','blood urea','A'), ('lab:urea','B. Urea','b urea','A'),
 ('lab:esr','ESR','esr','A'),
 ('lab:urine-re','Urine R/E','urine r e','A'), ('lab:urine-re','Urine routine','urine routine','A'),
 ('lab:ecg','ECG','ecg','A'), ('lab:ecg','EKG','ekg','A'),
 ('lab:xray-ls','X-ray LS spine','x ray ls spine','A')
ON CONFLICT DO NOTHING;

-- =====================================================================
-- D. NORMALIZED PRESCRIPTION TABLES (E17-S1) - a 3NF projection of the facts, each row
--    linked to its fact (and through it to provenance / observations)
-- =====================================================================
CREATE TABLE IF NOT EXISTS rx_prescription (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id       uuid NOT NULL UNIQUE REFERENCES source_document(id),
  patient_id        uuid REFERENCES patient_identity(id),
  encounter_id      uuid REFERENCES encounter(id) ON DELETE SET NULL,
  practitioner_id   uuid REFERENCES cn_practitioner(id),
  doc_type          text,
  encounter_date    timestamptz,
  follow_up_text    text,
  synced_at         timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rx_medication_order (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  line_no         int,
  drug_text       text NOT NULL,
  concept_id      text REFERENCES kb_concept(id),
  code_system     text, code text,
  strength_num    numeric, strength_unit text, form text,
  dose_num        numeric, dose_unit text, route text,
  frequency_code  text, frequency_per_day numeric, duration_days int,
  prn             bool, instructions text
);
CREATE TABLE IF NOT EXISTS rx_investigation_order (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  order_text      text NOT NULL,
  concept_id      text REFERENCES kb_concept(id),
  code_system     text, code text,
  kind            text                   -- test|panel (panels are never expanded)
);
CREATE TABLE IF NOT EXISTS rx_diagnosis (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  diagnosis_text  text NOT NULL,
  code_system     text, code text
);
CREATE TABLE IF NOT EXISTS rx_complaint (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  complaint_text  text NOT NULL,
  code_system     text, code text
);
CREATE TABLE IF NOT EXISTS rx_vital (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  name            text NOT NULL, value_num numeric, value_text text, unit text,
  code_system     text, code text
);
CREATE TABLE IF NOT EXISTS rx_advice (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prescription_id uuid NOT NULL REFERENCES rx_prescription(id) ON DELETE CASCADE,
  fact_id         uuid NOT NULL REFERENCES clinical_fact(id) ON DELETE CASCADE,
  advice_text     text NOT NULL
);
CREATE OR REPLACE VIEW v_rx_governed_medication AS
  SELECT m.*, f.review_state, rp.document_id, rp.patient_id
    FROM rx_medication_order m
    JOIN rx_prescription rp ON rp.id = m.prescription_id
    JOIN clinical_fact f ON f.id = m.fact_id
   WHERE f.review_state IN ('auto_accepted','clinician_confirmed','corrected') AND f.is_current;

-- =====================================================================
-- E. FHIR OUTBOX + BLOB INDEX (E17-S3..S5)
-- =====================================================================
CREATE TABLE IF NOT EXISTS fhir_outbox (
  id              bigserial PRIMARY KEY,
  patient_id      uuid NOT NULL,
  document_id     uuid,
  reason          text NOT NULL,                 -- validated|review_decision|manual
  status          text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','processing','done','dead')),
  attempts        smallint NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
  next_attempt_at timestamptz NOT NULL DEFAULT now(),
  last_error      text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_outbox_due ON fhir_outbox (status, next_attempt_at);
-- one pending job per document is enough: later events coalesce into it
CREATE UNIQUE INDEX IF NOT EXISTS ux_outbox_pending ON fhir_outbox (document_id) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS fhir_bundle_blob (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id        uuid NOT NULL,
  document_id       uuid,
  artifact_type     text NOT NULL,
  version           int NOT NULL,
  object_key        text NOT NULL,
  sha256            char(64) NOT NULL,
  byte_size         int NOT NULL,
  bundle_status     text NOT NULL,                -- draft|ready_to_share
  validation_status text NOT NULL,
  validator         text,
  governed_facts    int NOT NULL DEFAULT 0,
  held_facts        int NOT NULL DEFAULT 0,
  created_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (document_id, artifact_type, version)
);
CREATE INDEX IF NOT EXISTS ix_blob_patient ON fhir_bundle_blob (patient_id, created_at DESC);
DROP TRIGGER IF EXISTS trg_blob_immutable ON fhir_bundle_blob;
CREATE TRIGGER trg_blob_immutable BEFORE UPDATE OR DELETE ON fhir_bundle_blob
  FOR EACH ROW EXECUTE FUNCTION cdi_evidence_is_immutable();

-- =====================================================================
-- F. FILE LISTENER LIFECYCLE (E16)
-- =====================================================================
CREATE TABLE IF NOT EXISTS listener_file (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  connector       text NOT NULL,                  -- local|onedrive|sharepoint
  remote_id       text NOT NULL,                  -- path (local) or driveItem id (graph)
  name            text NOT NULL,
  etag            text NOT NULL,                  -- version marker: mtime+size (local) / eTag (graph)
  byte_size       bigint,
  sha256          char(64),
  state           text NOT NULL DEFAULT 'seen'
                    CHECK (state IN ('seen','processing','completed','error','quarantine')),
  attempts        smallint NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
  error_class     text,                           -- transient|data|code
  last_error      text,
  lease_owner     text,
  lease_until     timestamptz,
  next_attempt_at timestamptz,
  document_id     uuid REFERENCES source_document(id),
  stable_polls    smallint NOT NULL DEFAULT 0,
  history         jsonb NOT NULL DEFAULT '[]',
  first_seen_at   timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (connector, remote_id, etag)
);
CREATE INDEX IF NOT EXISTS ix_listener_state ON listener_file (state, next_attempt_at);

-- =====================================================================
-- G. DOWNSTREAM SCREEN DISPATCH (E20) - every target disabled until approved
-- =====================================================================
CREATE TABLE IF NOT EXISTS dispatch_target (
  id           text PRIMARY KEY,
  name         text NOT NULL,
  channel      text NOT NULL CHECK (channel IN ('rest','fhir','ui')),
  base_url     text,
  auth_ref     text,                              -- name of a secret, never the secret
  enabled      bool NOT NULL DEFAULT false,
  approved_by  text,
  approved_at  timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  CHECK (NOT enabled OR (approved_by IS NOT NULL AND approved_at IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS dispatch_mapping (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_id   text NOT NULL REFERENCES dispatch_target(id),
  screen      text NOT NULL,
  version     int NOT NULL,
  mapping     jsonb NOT NULL,
  active      bool NOT NULL DEFAULT false,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (target_id, screen, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mapping_active ON dispatch_mapping (target_id, screen) WHERE active;
CREATE TABLE IF NOT EXISTS dispatch_log (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_id     text NOT NULL REFERENCES dispatch_target(id),
  screen        text NOT NULL,
  mapping_id    uuid REFERENCES dispatch_mapping(id),
  dispatch_key  text NOT NULL UNIQUE,              -- target+screen+record+version: idempotency
  patient_id    uuid,
  document_id   uuid,
  payload       jsonb NOT NULL,
  status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','sent','acknowledged','failed','dead')),
  attempts      smallint NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
  response      jsonb,
  last_error    text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
""")


def downgrade() -> None:
    _exec("""
    DROP TABLE IF EXISTS dispatch_log, dispatch_mapping, dispatch_target, listener_file,
      fhir_bundle_blob, fhir_outbox, rx_advice, rx_vital, rx_complaint, rx_diagnosis,
      rx_investigation_order, rx_medication_order, rx_prescription, kb_alias, kb_concept,
      kb_indication_group, verified_fact, interpretation_candidate CASCADE;
    DROP VIEW IF EXISTS v_rx_governed_medication, v_ocr_observation_current;
    DROP TABLE IF EXISTS ocr_observation CASCADE;
    DROP FUNCTION IF EXISTS cdi_evidence_is_immutable() CASCADE;
    ALTER TABLE ocr_block DROP COLUMN IF EXISTS observation_ids, DROP COLUMN IF EXISTS recognition;
    ALTER TABLE clinical_fact DROP COLUMN IF EXISTS evidence_state, DROP COLUMN IF EXISTS field_policy,
      DROP COLUMN IF EXISTS decision_trace;
    ALTER TABLE source_document DROP COLUMN IF EXISTS practitioner_id,
      DROP COLUMN IF EXISTS practitioner_link_method, DROP COLUMN IF EXISTS practitioner_link_confidence,
      DROP COLUMN IF EXISTS practitioner_evidence;
    DELETE FROM cn_practitioner WHERE is_sample;
    DROP INDEX IF EXISTS ux_cn_pract_reg;
    ALTER TABLE cn_practitioner DROP COLUMN IF EXISTS designation, DROP COLUMN IF EXISTS name_aliases,
      DROP COLUMN IF EXISTS council, DROP COLUMN IF EXISTS clinic, DROP COLUMN IF EXISTS is_sample;
    """)
