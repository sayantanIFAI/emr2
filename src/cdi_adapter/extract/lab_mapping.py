"""Many written names -> ONE standard lab test (the mapping table).

A doctor writes the same test many ways ("creatinine", "creatine", "sr creatinine", "S. Creat", "RFT"). This table says
which standard test each written name stands for, and the screen shows the whole table so a person can see, add to and
switch off any row. It is consulted FIRST, before the Indian national list and the curated abbreviation table.

* The rows live in ``lab_test_alias`` (migration 0010). ``SEED`` below is what the application puts there on first use
  (``ensure_seed``); a row a person switched off stays off (rows are never deleted, only ``enabled`` = false).
* With no database (a unit test, a first start) the seed alone answers, so a lookup never fails.
* A name is matched whole, after lower-casing and dropping punctuation, and then again without a leading specimen word
  ("sr", "s", "serum", "plasma", "blood": "S. Creatinine" = "creatinine"). "urine creatinine" is NOT cut: it is a
  different test.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ..logging import get_logger
from .indian_codes import norm

log = get_logger(__name__)

# (written name, the standard test, its LOINC code or None, note). Codes only where they are certain.
_CREAT_NOTE = "the owner asked for these to map to creatinine"
SEED: list[tuple[str, str, str | None, str]] = [
    ("creatinine", "Creatinine", "2160-0", ""),
    ("creatine", "Creatinine", "2160-0", "a common misspelling of creatinine; " + _CREAT_NOTE),
    ("sr creatinine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("sr creatine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("serum creatinine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("serum creatine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("s creatinine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("s creatine", "Creatinine", "2160-0", _CREAT_NOTE),
    ("creat", "Creatinine", "2160-0", _CREAT_NOTE),
    ("s creat", "Creatinine", "2160-0", _CREAT_NOTE),
    ("sr creat", "Creatinine", "2160-0", _CREAT_NOTE),
    ("rft", "Creatinine", "2160-0",
     "RFT is a PANEL (urea, creatinine, electrolytes); mapped to creatinine because the owner asked. Switch off here to keep it a panel."),
    ("renal function test", "Creatinine", "2160-0", "a panel; mapped to creatinine because the owner asked"),
    ("renal function tests", "Creatinine", "2160-0", "a panel; mapped to creatinine because the owner asked"),
    ("kft", "Creatinine", "2160-0", "a panel; mapped to creatinine because the owner asked"),
    ("kidney function test", "Creatinine", "2160-0", "a panel; mapped to creatinine because the owner asked"),
    ("hba1c", "HbA1c", "4548-4", ""),
    ("glycosylated hemoglobin", "HbA1c", "4548-4", ""),
    ("glycated hemoglobin", "HbA1c", "4548-4", ""),
    ("glycosylated haemoglobin", "HbA1c", "4548-4", ""),
    ("glycated haemoglobin", "HbA1c", "4548-4", ""),
    ("a1c", "HbA1c", "4548-4", ""),
    ("tsh", "TSH", "3016-3", ""),
    ("thyroid stimulating hormone", "TSH", "3016-3", ""),
    ("fbs", "Fasting blood sugar", "1558-6", ""),
    ("fbg", "Fasting blood sugar", "1558-6", ""),
    ("fasting blood sugar", "Fasting blood sugar", "1558-6", ""),
    ("fasting blood glucose", "Fasting blood sugar", "1558-6", ""),
    ("fasting glucose", "Fasting blood sugar", "1558-6", ""),
    ("ppbs", "Post-prandial blood sugar", None, ""),
    ("post prandial blood sugar", "Post-prandial blood sugar", None, ""),
    ("postprandial blood sugar", "Post-prandial blood sugar", None, ""),
    ("pp2bs", "Post-prandial blood sugar", None, ""),
    # the common tests and every way doctors write them (a LOINC code only where it is certain; the rest are names only)
    ("cbc", "Complete blood count", '58410-2', ""),
    ("complete blood count", "Complete blood count", '58410-2', ""),
    ("complete blood picture", "Complete blood count", '58410-2', ""),
    ("cbp", "Complete blood count", '58410-2', ""),
    ("hemogram", "Complete blood count", '58410-2', ""),
    ("haemogram", "Complete blood count", '58410-2', ""),
    ("full blood count", "Complete blood count", '58410-2', ""),
    ("fbc", "Complete blood count", '58410-2', ""),
    ("blood counts", "Complete blood count", '58410-2', ""),
    ("crp", "C-reactive protein", '1988-5', ""),
    ("c reactive protein", "C-reactive protein", '1988-5', ""),
    ("s crp", "C-reactive protein", '1988-5', ""),
    ("sr crp", "C-reactive protein", '1988-5', ""),
    ("lft", "Liver function test", '24325-3', ""),
    ("liver function test", "Liver function test", '24325-3', ""),
    ("liver function tests", "Liver function test", '24325-3', ""),
    ("liver function", "Liver function test", '24325-3', ""),
    ("hepatic function panel", "Liver function test", '24325-3', ""),
    ("lft profile", "Liver function test", '24325-3', ""),
    ("liver profile", "Liver function test", '24325-3', ""),
    ("lipid profile", "Lipid profile", None, ""),
    ("lipid", "Lipid profile", None, ""),
    ("lipids", "Lipid profile", None, ""),
    ("lipid panel", "Lipid profile", None, ""),
    ("fasting lipid profile", "Lipid profile", None, ""),
    ("flp", "Lipid profile", None, ""),
    ("ft4", "Free T4", '3024-7', ""),
    ("free t4", "Free T4", '3024-7', ""),
    ("free thyroxine", "Free T4", '3024-7', ""),
    ("ft3", "Free T3", '3051-0', ""),
    ("free t3", "Free T3", '3051-0', ""),
    ("free triiodothyronine", "Free T3", '3051-0', ""),
    ("esr", "ESR", '4537-7', ""),
    ("erythrocyte sedimentation rate", "ESR", '4537-7', ""),
    ("hb", "Haemoglobin", '718-7', ""),
    ("hgb", "Haemoglobin", '718-7', ""),
    ("hemoglobin", "Haemoglobin", '718-7', ""),
    ("haemoglobin", "Haemoglobin", '718-7', ""),
    ("urea", "Urea", '3091-6', ""),
    ("s urea", "Urea", '3091-6', ""),
    ("blood urea", "Urea", '3091-6', ""),
    ("serum urea", "Urea", '3091-6', ""),
    ("uric acid", "Uric acid", '3084-1', ""),
    ("s uric acid", "Uric acid", '3084-1', ""),
    ("serum uric acid", "Uric acid", '3084-1', ""),
    ("sgot", "AST (SGOT)", '1920-8', ""),
    ("ast", "AST (SGOT)", '1920-8', ""),
    ("s sgot", "AST (SGOT)", '1920-8', ""),
    ("aspartate aminotransferase", "AST (SGOT)", '1920-8', ""),
    ("sgpt", "ALT (SGPT)", '1742-6', ""),
    ("alt", "ALT (SGPT)", '1742-6', ""),
    ("s sgpt", "ALT (SGPT)", '1742-6', ""),
    ("alanine aminotransferase", "ALT (SGPT)", '1742-6', ""),
    ("b12", "Vitamin B12", '2132-9', ""),
    ("vit b12", "Vitamin B12", '2132-9', ""),
    ("vitamin b12", "Vitamin B12", '2132-9', ""),
    ("s b12", "Vitamin B12", '2132-9', ""),
    ("cyanocobalamin level", "Vitamin B12", '2132-9', ""),
    ("vit d", "Vitamin D", None, ""),
    ("vitamin d", "Vitamin D", None, ""),
    ("25 oh vitamin d", "Vitamin D", None, ""),
    ("25 hydroxy vitamin d", "Vitamin D", None, ""),
    ("vit d3 level", "Vitamin D", None, ""),
    ("ferritin", "Ferritin", '2276-4', ""),
    ("s ferritin", "Ferritin", '2276-4', ""),
    ("serum ferritin", "Ferritin", '2276-4', ""),
    ("lipase", "Lipase", '3040-3', ""),
    ("s lipase", "Lipase", '3040-3', ""),
    ("serum lipase", "Lipase", '3040-3', ""),
    ("amylase", "Amylase", '1798-8', ""),
    ("s amylase", "Amylase", '1798-8', ""),
    ("serum amylase", "Amylase", '1798-8', ""),
    ("calcium", "Calcium", '17861-6', ""),
    ("s calcium", "Calcium", '17861-6', ""),
    ("serum calcium", "Calcium", '17861-6', ""),
    ("triglycerides", "Triglycerides", '2571-8', ""),
    ("triglyceride", "Triglycerides", '2571-8', ""),
    ("tg", "Triglycerides", '2571-8', ""),
    ("s triglycerides", "Triglycerides", '2571-8', ""),
    ("total cholesterol", "Total cholesterol", '2093-3', ""),
    ("cholesterol", "Total cholesterol", '2093-3', ""),
    ("s cholesterol", "Total cholesterol", '2093-3', ""),
    ("hdl", "HDL cholesterol", '2085-9', ""),
    ("hdl cholesterol", "HDL cholesterol", '2085-9', ""),
    ("fructosamine", "Fructosamine", None, ""),
    ("s fructosamine", "Fructosamine", None, ""),
    ("serum fructosamine", "Fructosamine", None, ""),
    ("urine routine", "Urine routine examination", None, ""),
    ("urine r e", "Urine routine examination", None, ""),
    ("urine re", "Urine routine examination", None, ""),
    ("urine routine examination", "Urine routine examination", None, ""),
    ("urine examination", "Urine routine examination", None, ""),
    ("urine culture", "Urine culture", None, ""),
    ("urine c s", "Urine culture", None, ""),
    ("urine culture and sensitivity", "Urine culture", None, ""),
    ("stool routine", "Stool routine examination", None, ""),
    ("stool r e", "Stool routine examination", None, ""),
    ("stool re", "Stool routine examination", None, ""),
    ("stool examination", "Stool routine examination", None, ""),
    ("d dimer", "D-dimer", None, ""),
    ("ddimer", "D-dimer", None, ""),
    ("d dimer test", "D-dimer", None, ""),
    ("psa", "PSA", None, ""),
    ("prostate specific antigen", "PSA", None, ""),
    # blood sugar as doctors write it: F / FS / FBS = fasting, PP / PPS / PPBS = post-prandial; PT, APTT, BT, hepatitis and HIV screens
    ("blood sugar f", "Fasting blood sugar", '1558-6', ""),
    ("blood sugar fs", "Fasting blood sugar", '1558-6', ""),
    ("blood sugar fbs", "Fasting blood sugar", '1558-6', ""),
    ("blood sugar fasting", "Fasting blood sugar", '1558-6', ""),
    ("bs f", "Fasting blood sugar", '1558-6', ""),
    ("bs fs", "Fasting blood sugar", '1558-6', ""),
    ("sugar f", "Fasting blood sugar", '1558-6', ""),
    ("sugar fs", "Fasting blood sugar", '1558-6', ""),
    ("sugar fasting", "Fasting blood sugar", '1558-6', ""),
    ("pp", "Post-prandial blood sugar", None, ""),
    ("pps", "Post-prandial blood sugar", None, ""),
    ("blood sugar pp", "Post-prandial blood sugar", None, ""),
    ("blood sugar pps", "Post-prandial blood sugar", None, ""),
    ("blood sugar ppbs", "Post-prandial blood sugar", None, ""),
    ("blood sugar post prandial", "Post-prandial blood sugar", None, ""),
    ("bs pp", "Post-prandial blood sugar", None, ""),
    ("bs pps", "Post-prandial blood sugar", None, ""),
    ("sugar pp", "Post-prandial blood sugar", None, ""),
    ("sugar pps", "Post-prandial blood sugar", None, ""),
    ("pp sugar", "Post-prandial blood sugar", None, ""),
    ("blood sugar", "Blood glucose", None, ""),
    ("blood glucose", "Blood glucose", None, ""),
    ("rbs", "Random blood sugar", None, ""),
    ("random blood sugar", "Random blood sugar", None, ""),
    ("random blood glucose", "Random blood sugar", None, ""),
    ("pt", "Prothrombin time", '5902-2', ""),
    ("prothrombin time", "Prothrombin time", '5902-2', ""),
    ("inr", "INR", '6301-6', ""),
    ("pt inr", "INR", '6301-6', ""),
    ("aptt", "aPTT", None, ""),
    ("apttt", "aPTT", None, ""),
    ("activated partial thromboplastin time", "aPTT", None, ""),
    ("bt", "Bleeding time", None, ""),
    ("bleeding time", "Bleeding time", None, ""),
    ("clotting time", "Clotting time", None, ""),
    ("coagulation time", "Clotting time", None, ""),
    ("hbsag", "Hepatitis B surface antigen", None, ""),
    ("hbs ag", "Hepatitis B surface antigen", None, ""),
    ("hepatitis b surface antigen", "Hepatitis B surface antigen", None, ""),
    ("australia antigen", "Hepatitis B surface antigen", None, ""),
    ("hiv", "HIV antibody", None, ""),
    ("hiv 1 2", "HIV antibody", None, ""),
    ("hiv antibody", "HIV antibody", None, ""),
    ("hcv", "Hepatitis C antibody", None, ""),
    ("anti hcv", "Hepatitis C antibody", None, ""),
    ("hepatitis c antibody", "Hepatitis C antibody", None, ""),
    ("ecg", "ECG", None, ""),
    # sodium and potassium, chest ECG, fever profile (MEASURED on real prescriptions: "Na+ & K+ Level Test", "Chest ECG", "Blood for fever profile")
    ("na k", "Sodium and potassium (electrolytes)", None, ""),
    ("na k level", "Sodium and potassium (electrolytes)", None, ""),
    ("na k level test", "Sodium and potassium (electrolytes)", None, ""),
    ("sodium potassium", "Sodium and potassium (electrolytes)", None, ""),
    ("electrolytes", "Sodium and potassium (electrolytes)", None, ""),
    ("serum electrolytes", "Sodium and potassium (electrolytes)", None, ""),
    ("s electrolytes", "Sodium and potassium (electrolytes)", None, ""),
    ("chest ecg", "ECG", None, ""),
    ("ecg chest", "ECG", None, ""),
    ("12 lead ecg", "ECG", None, ""),
    ("fever profile", "Fever profile", None, "a panel; what it holds is the lab's"),
    ("blood for fever profile", "Fever profile", None, "a panel; what it holds is the lab's"),
    ("fever panel", "Fever profile", None, "a panel; what it holds is the lab's"),
    # printed checklist names (MEASURED on a real pad: FPG, 2hr PPG, ACR were not in the lists at all)
    ("fpg", "Fasting blood sugar", "1558-6", ""),
    ("ppg", "Post-prandial blood sugar", None, ""),
    ("2hr ppg", "Post-prandial blood sugar", None, ""),
    ("2 hr ppg", "Post-prandial blood sugar", None, ""),
    ("2hrs ppg", "Post-prandial blood sugar", None, ""),
    ("ppg 2hr", "Post-prandial blood sugar", None, ""),
    ("ppg 2 hr", "Post-prandial blood sugar", None, ""),
    ("acr", "Urine albumin/creatinine ratio", None, ""),
    ("urine acr", "Urine albumin/creatinine ratio", None, ""),
    ("urine albumin creatinine ratio", "Urine albumin/creatinine ratio", None, ""),
    # investigations written under "Adv" (MEASURED on a real page: "Digital OPG, FBS, BT, CT" was read by the model 6 of 6 times and still missed)
    ("opg", "OPG (orthopantomogram)", None, "a dental panoramic X-ray"),
    ("digital opg", "OPG (orthopantomogram)", None, "a dental panoramic X-ray"),
    ("dental opg", "OPG (orthopantomogram)", None, "a dental panoramic X-ray"),
    ("orthopantomogram", "OPG (orthopantomogram)", None, "a dental panoramic X-ray"),
    ("ct", "CT (clotting time or CT scan)", None, "ambiguous: clotting time (with BT) or a CT scan; either is an order when it is listed, check which"),
    ("ekg", "ECG", None, ""),
    ("electrocardiogram", "ECG", None, ""),
    # the owner's investigations: EEG, MRI, CT, USG, echo and X-ray are tests of this system (printed on the pad's checklist, MEASURED on a real Sonoscan page:
    # "EEG" and "MRI Scan of Brain" struck through; "2D echo" missed on another day)
    ("ige", "Total IgE (immunoglobulin E)", None, ""),
    ("total ige", "Total IgE (immunoglobulin E)", None, ""),
    ("serum ige", "Total IgE (immunoglobulin E)", None, ""),
    # procedures the owner counts as investigations (a handwritten "Laryngoscopy" was missed)
    ("laryngoscopy", "Laryngoscopy", None, ""),
    ("laryngoscope", "Laryngoscopy", None, ""),
    ("video laryngoscopy", "Video laryngoscopy", None, ""),
    ("nasal endoscopy", "Nasal endoscopy", None, ""),
    ("dne", "Diagnostic nasal endoscopy", None, ""),
    ("diagnostic nasal endoscopy", "Diagnostic nasal endoscopy", None, ""),
    ("endoscopy", "Endoscopy", None, ""),
    ("upper gi endoscopy", "Upper GI endoscopy", None, ""),
    ("colonoscopy", "Colonoscopy", None, ""),
    ("bronchoscopy", "Bronchoscopy", None, ""),
    ("cystoscopy", "Cystoscopy", None, ""),
    ("hysteroscopy", "Hysteroscopy", None, ""),
    ("audiometry", "Audiometry", None, ""),
    ("pta", "Pure tone audiometry", None, ""),
    ("pure tone audiometry", "Pure tone audiometry", None, ""),
    ("tympanometry", "Tympanometry", None, ""),
    ("spirometry", "Spirometry", None, ""),
    ("pft", "Pulmonary function test", None, ""),
    ("pulmonary function test", "Pulmonary function test", None, ""),
    ("tmt", "Treadmill test (TMT)", None, ""),
    ("treadmill test", "Treadmill test (TMT)", None, ""),
    ("holter", "Holter monitoring", None, ""),
    ("emg", "EMG", None, ""),
    ("ncv", "Nerve conduction velocity (NCV)", None, ""),
    ("fnac", "FNAC", None, ""),
    ("x ray pns", "X-ray PNS", None, ""),
    ("ct pns", "CT PNS", None, ""),
    # handwritten forms of everyday tests (MEASURED on a real page: "Ca2+" and "Vit D3" were not placed)
    ("ca2", "Calcium", '17861-6', ""),
    ("total calcium", "Calcium", '17861-6', ""),
    ("vit d3", "Vitamin D", None, ""),
    ("vitamin d3", "Vitamin D", None, ""),
    ("eeg", "EEG", None, "electroencephalogram"),
    ("electroencephalogram", "EEG", None, ""),
    ("2d echo", "2D Echo", None, "echocardiography"),
    ("2d echocardiography", "2D Echo", None, ""),
    ("2d echo with doppler", "2D Echo with Doppler", None, ""),
    ("echo", "Echocardiography", None, ""),
    ("echocardiography", "Echocardiography", None, ""),
    ("echocardiography with doppler", "Echocardiography with Doppler", None, ""),
    ("mri", "MRI", None, "which part: check the page"),
    ("mri brain", "MRI brain", None, ""),
    ("mri of brain", "MRI brain", None, ""),
    ("mri scan of brain", "MRI brain", None, ""),
    ("mri scan brain", "MRI brain", None, ""),
    ("mri head", "MRI brain", None, ""),
    ("mri spine", "MRI spine", None, ""),
    ("mri ls spine", "MRI LS spine", None, ""),
    ("mri lumbosacral spine", "MRI LS spine", None, ""),
    ("mri knee", "MRI knee", None, ""),
    ("ct scan", "CT scan", None, "which part: check the page"),
    ("ct brain", "CT scan brain", None, ""),
    ("ct scan brain", "CT scan brain", None, ""),
    ("ct scan of brain", "CT scan brain", None, ""),
    ("ct head", "CT scan brain", None, ""),
    ("ct chest", "CT scan chest", None, ""),
    ("ct scan chest", "CT scan chest", None, ""),
    ("ct abdomen", "CT scan abdomen", None, ""),
    ("ct scan abdomen", "CT scan abdomen", None, ""),
    ("usg", "USG", None, "which part: check the page"),
    ("usg abdomen", "USG abdomen", None, ""),
    ("usg whole abdomen", "USG whole abdomen", None, ""),
    ("usg of whole abdomen", "USG whole abdomen", None, ""),
    # "USG W/A" is the doctor's short form of USG whole abdomen (owner, 24 Oct 2026: written "Adv (USG W/A)" and nothing reached the lab tests); "VSG" is
    # the misreading of USG measured on a real page. Only the whole-abdomen short form is known: no other letter pair is guessed.
    ("usg w a", "USG whole abdomen", None, "W/A written for whole abdomen"),
    ("usg wa", "USG whole abdomen", None, "W/A written for whole abdomen"),
    ("usg w abd", "USG whole abdomen", None, "W/Abd written for whole abdomen"),
    ("usg whole abd", "USG whole abdomen", None, ""),
    ("vsg w a", "USG whole abdomen", None, "W/A written for whole abdomen; the U was read as V"),
    ("vsg whole abdomen", "USG whole abdomen", None, "the U was read as V"),
    # TC / DC: the total and the differential leucocyte count (owner, 24 Oct 2026: the page printed "tcDC" and the lists had no such name, so the model's "TCOE" was kept)
    ("tc dc", "Total count and differential count", None, "TC / DC: total and differential leucocyte count"),
    ("tcdc", "Total count and differential count", None, "TC / DC: total and differential leucocyte count"),
    ("tlc dlc", "Total count and differential count", None, "TLC / DLC: total and differential leucocyte count"),
    ("tlcdlc", "Total count and differential count", None, "TLC / DLC: total and differential leucocyte count"),
    # TVS is the doctor's short form of transvaginal ultrasound; Echo-2D / Echo 2D are 2D Echo (the lists knew only "2D Echo": "Echo-2D" was dropped as imaging
    # the lists do not place). Owner, 24 Oct 2026: "TVS, Echo-2D missed".
    ("tvs", "USG TVS", None, "TVS: transvaginal ultrasound"),
    ("t v s", "USG TVS", None, "TVS: transvaginal ultrasound"),
    ("usg tvs", "USG TVS", None, ""),
    ("tvs usg", "USG TVS", None, ""),
    ("tvs scan", "USG TVS", None, ""),
    ("transvaginal sonography", "USG TVS", None, ""),
    ("transvaginal usg", "USG TVS", None, ""),
    ("usg transvaginal", "USG TVS", None, ""),
    ("echo 2d", "2D Echo", None, "echocardiography"),
    ("echo 2 d", "2D Echo", None, "echocardiography"),
    # TORCH is one panel (Toxoplasma, Rubella, cytomegalovirus, herpes simplex): the national list has the four parts but no panel, so it is named, with no code
    ("torch", "TORCH panel", None, "Toxoplasma, Rubella, Cytomegalovirus, Herpes simplex"),
    ("torch panel", "TORCH panel", None, ""),
    ("torch profile", "TORCH panel", None, ""),
    ("torch screen", "TORCH panel", None, ""),
    ("torch screening", "TORCH panel", None, ""),
    ("torch test", "TORCH panel", None, ""),
    ("torch igg igm", "TORCH panel", None, ""),
    # ABO and Rh typing is ONE test (a blood group). The national list holds "ABO group" (883-9) and "Rh" (10331-7) apart, so the pair has no single code here.
    ("abo", "ABO and Rh blood group", None, "ABO group 883-9 and Rh 10331-7 in the national list"),
    ("abo rh", "ABO and Rh blood group", None, ""),
    ("abo and rh", "ABO and Rh blood group", None, ""),
    ("abo rh typing", "ABO and Rh blood group", None, ""),
    ("abo and rh typing", "ABO and Rh blood group", None, ""),
    ("abo rh group", "ABO and Rh blood group", None, ""),
    ("abo rh blood group", "ABO and Rh blood group", None, ""),
    ("abo typing", "ABO and Rh blood group", None, ""),
    ("blood group", "ABO and Rh blood group", None, "a blood group is ABO and Rh"),
    ("blood grouping", "ABO and Rh blood group", None, ""),
    ("blood group and rh", "ABO and Rh blood group", None, ""),
    ("blood group rh typing", "ABO and Rh blood group", None, ""),
    # karyotype: the national list has it for amniotic fluid only; the written verb form and the "both parents" qualifier were not matched
    ("karyotyping", "Karyotype", None, "the national list holds Karyotype for amniotic fluid only"),
    ("karyotyping both parents", "Karyotype", None, ""),
    ("karyotype both parents", "Karyotype", None, ""),
    ("karyotype blood", "Karyotype", None, ""),
    ("blood karyotype", "Karyotype", None, ""),
    ("blood karyotyping", "Karyotype", None, ""),
    ("whole abdomen", "USG whole abdomen", None, "written without USG: an abdomen written alone in the investigations is the ultrasound"),
    ("whole abd", "USG whole abdomen", None, "written without USG: an abdomen written alone in the investigations is the ultrasound"),
    ("lower abdomen", "USG lower abdomen", None, "written without USG: an abdomen written alone in the investigations is the ultrasound"),
    ("lower abd", "USG lower abdomen", None, "written without USG: an abdomen written alone in the investigations is the ultrasound"),
    ("upper abdomen", "USG upper abdomen", None, "written without USG: an abdomen written alone in the investigations is the ultrasound"),
    ("usg lower abdomen", "USG lower abdomen", None, ""),
    ("usg kub", "USG KUB", None, ""),
    ("ultrasound abdomen", "USG abdomen", None, ""),
    ("x ray", "X-ray", None, "which part: check the page"),
    ("xray", "X-ray", None, "which part: check the page"),
    ("x ray chest", "X-ray chest", None, ""),
    ("chest x ray", "X-ray chest", None, ""),
    ("cxr", "X-ray chest", None, ""),
    ("x ray chest pa", "X-ray chest PA", None, ""),
    ("x ray chest pa view", "X-ray chest PA", None, ""),
    ("x ray chest pa ap view", "X-ray chest PA/AP view", None, ""),
    ("x ray nasopharynx", "X-ray nasopharynx", None, ""),
]

_LEADING = ("bl for ", "blood for ", "blood test for ", "bld for ", "test for ", "sr ", "s ", "serum ", "plasma ", "blood ")      # "Bl for TORCH": the lead-in is not part of the name


@dataclass(frozen=True)
class Mapped:
    alias: str
    canonical: str
    loinc: str | None
    note: str
    source: str                     # "seed" | "user"


_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "rows": None}
_TTL_S = 30.0


def _seed_rows() -> dict[str, Mapped]:
    return {norm(a): Mapped(a, c, l, n, "seed") for a, c, l, n in SEED}


def invalidate() -> None:
    with _lock:
        _cache["rows"] = None
        _cache["at"] = 0.0


def _load() -> dict[str, Mapped]:
    """The enabled rows (database, with the seed underneath), cached for a short time."""
    now = time.time()
    with _lock:
        if _cache["rows"] is not None and now - _cache["at"] < _TTL_S:
            return _cache["rows"]
    rows = _seed_rows()
    try:
        from ..config import settings

        if not settings.lab_mapping_db:
            raise RuntimeError("database not used")
        from ..db import session_scope

        with session_scope() as sess:
            got = sess.execute(text(
                "SELECT alias_key, alias, canonical, loinc, coalesce(note,'') AS note, source, enabled FROM lab_test_alias"
            )).mappings().all()
        for r in got:
            if r["enabled"]:
                rows[r["alias_key"]] = Mapped(r["alias"], r["canonical"], r["loinc"], r["note"], r["source"])
            else:
                rows.pop(r["alias_key"], None)               # switched off: the seed does not bring it back
    except Exception as exc:  # noqa: BLE001 - no database / no table yet: the seed alone answers
        log.info("lab_mapping_seed_only", error=str(exc)[:120])
    with _lock:
        _cache["rows"], _cache["at"] = rows, now
    return rows


def _flat(rows: dict[str, Mapped]) -> dict[str, Mapped]:
    """The rows keyed by their name with the spaces taken out (rebuilt only when the rows were)."""
    with _lock:
        if _cache.get("flat_for") is not rows:
            _cache["flat"] = {key.replace(" ", ""): m for key, m in rows.items() if " " in key and len(key.replace(" ", "")) >= 6}
            _cache["flat_for"] = rows
        return _cache["flat"]


_MODALITY = re.compile(r"(?i)^\W*(x[\s.\-]*rays?|xrays?|mri|ct(?:\s*scan)?|usg|ultra\s?sound)(?![a-z])\W*")
_MODALITY_NAME = (("mri", "MRI"), ("ct", "CT scan"), ("usg", "USG"), ("ultra", "USG"))
_BODY_PART = re.compile(r"(?i)(?<![a-z])(?:pelvis|hips?|spine|knees?|shoulders?|chest|skull|brain|head|neck|ankles?|foot|feet|hands?|wrists?|elbows?|femur|humerus|forearm|"
                        r"leg|thigh|sinus(?:es)?|pns|mastoids?|orbits?|face|jaw|tmj|kub|lungs?|clavicle|scapula|ribs?|sacrum|coccyx|joints?|cervical|lumbar|"
                        r"lumbosacral|dorsal|thoracic|ls|cs|(?:[a-z]\.?\s?){1,2}spine)(?![a-z])")


_NEAR_BODY = ("chest", "abdomen", "pelvis", "spine", "knee", "shoulder", "head", "brain", "neck", "hip", "kub")


def _sim(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def _near_modality(tok: str) -> str | None:
    """The modality a short word is a near-miss of ("usa", "usq", "vsg" for USG; "xy", "xg", "xry" for X-ray). Only used when a body part follows."""
    t = re.sub(r"[^a-z]", "", tok.casefold())
    if 2 <= len(t) <= 5 and t[0] == "x" and _sim(t, "xray") >= 0.6:
        return "X-ray"
    if 3 <= len(t) <= 4 and _sim(t, "usg") >= 0.6 and t not in ("us",):
        return "USG"
    return None


def _near_body(tok: str) -> str | None:
    t = re.sub(r"[^a-z]", "", tok.casefold())
    if len(t) < 3:
        return None
    for b in _NEAR_BODY:
        if _sim(t, b) >= 0.6 or (t[:2] == b[:2] and _sim(t, b) >= 0.45):
            return b
    return None


def _imaging_near_miss(text: str) -> Mapped | None:
    """"USA where aldoros" is USG whole abdomen and "xy chd Panies (visit)" is X-ray chest PA: a modality written one or two letters off, then a body part
    written badly. The doctor's "whole" after USG is the whole abdomen (owner's rule). Both parts must be there: a stray "usa" alone is nothing."""
    toks = re.findall(r"[A-Za-z]+", text)
    if len(toks) < 2:
        return None
    mod = _near_modality(toks[0])
    if mod is None:
        return None
    for i, t in enumerate(toks[1:4], start=1):
        if mod == "USG" and _sim(re.sub(r"[^a-z]", "", t.casefold()), "whole") >= 0.6:
            return Mapped(text, "USG whole abdomen", None, "read as '" + text + "' (USG whole abdomen)", "seed")
        body = _near_body(t)
        if body:
            views = " PA" if any(re.sub(r"[^a-z]", "", x.casefold()).startswith("pa") for x in toks[i + 1:i + 3]) else ""
            return Mapped(text, f"{mod} {body}{views}", None, "read as '" + text + "' (" + mod + " " + body + ")", "seed")
    return None


_USG_WORDS = frozenset(("usg", "uss", "usgs", "ultrasound", "ultrasonography", "sonography", "sono", "usgt", "vsg", "vsgt", "ugs", "usq", "usc", "uso", "usa", "isg", "ysg"))      # + misreadings seen on real pages


def _usg_whole_abdomen(text: str) -> Mapped | None:
    """USG whole abdomen, however it is written, in either order: any form of USG (USG, U.S.G., USS, ultrasound, sonography, or a near-miss such as USA / VSG /
    VSGT / UGS) AND the word "whole" (or W/A, WA, W/Abd, "whole" misspelt) in the same phrase. Per the owner, "USG whole" means the whole abdomen, a bare
    "whole abdomen" is the ultrasound, and "W/A" is whole abdomen (24 Oct 2026: "USG WHOLE ABDOMEN or USG W/A must be extracted correctly").
    Not applied when the phrase names another part (pelvis, KUB, lower / upper) or says "w/ abdomen" (with abdomen: the abdomen, not the whole)."""
    t = text.casefold()
    t = re.sub(r"(?<![a-z])([a-z])\.(?=[a-z]\.?(?![a-z]))", r"\1", t)             # u.s.g. -> usg, w.a -> wa
    t = t.replace(".", "")
    toks = re.findall(r"[a-z]+", t)
    if len(toks) < 1 or any(x in ("pelvis", "kub", "lower", "upper", "pelvic", "kidney", "chest", "thorax") for x in toks):
        return None
    glued = "".join(toks)
    mod = any(x in _USG_WORDS or x.startswith("usg") or (3 <= len(x) <= 4 and _sim(x, "usg") >= 0.6 and x != "us") for x in toks) \
        or ("ultra" in toks and "sound" in toks) or glued.startswith(("usgw", "usgwa"))
    whole = False
    for i, x in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if x in ("wa", "wabd") or _sim(x, "whole") >= 0.6 or x.startswith("whole") or (x == "w" and nxt in ("a", "abd")) or x.startswith("usgwa") or x.startswith("usgw") and "abd" in x:
            whole = True
    if "w" in toks and "abdomen" in toks and not whole:
        return None                                                       # "w/ abdomen": with abdomen
    if mod and whole:
        return Mapped(text, "USG whole abdomen", None, "read as '" + text + "' (USG whole abdomen)", "seed")
    if whole and not mod and re.search(r"(?<![a-z])(?:wa|whole)\W*(?:abd|abdomen)?\W*(?:usg|uss)(?![a-z])", t):
        return Mapped(text, "USG whole abdomen", None, "read as '" + text + "' (USG whole abdomen)", "seed")
    return None


def imaging(name: str | None) -> Mapped | None:
    """An imaging order written as MODALITY + BODY PART ("X-ray of pelvis with S/I hip AP", "X-ray LS spine AP / lat", "MRI of C spine"): ONE test, named as the
    doctor wrote it with the modality made standard. The lists hold only a few fixed names, so every other part was thrown out as "imaging the lists do not
    place" or reduced to the bare "MRI" (owner, 24 Oct 2026). It needs a body part: a bare modality stays the generic test and nothing is guessed."""
    text = str(name or "").strip()
    usg = _usg_whole_abdomen(text)
    if usg is not None:
        return usg
    m = _MODALITY.match(text)
    if not m:
        return _imaging_near_miss(text)
    rest = re.sub(r"(?i)^(?:of|for|scan\s+of|scan)\b\s*", "", text[m.end():]).strip(" .-:")
    if not _BODY_PART.search(rest) or len(re.sub(r"[^A-Za-z]", "", rest)) < 3:
        return None
    key = re.sub(r"[^a-z]", "", m.group(1).casefold())
    modality = "X-ray" if key.startswith("x") else next((n for k, n in _MODALITY_NAME if key.startswith(k)), "MRI")
    rest_one_line = re.sub(r"(?<![A-Za-z])(?:ls|cs|ap|pa|si|sc|kub|pns|tmj)(?![A-Za-z])", lambda mm: mm.group(0).upper(), re.sub(r"\s+", " ", rest), flags=re.I)
    canonical = f"{modality} {rest_one_line}"
    return Mapped(text, canonical, None, "modality and body part as written", "seed")


def lookup(name: str | None) -> Mapped | None:
    """The standard test this written name stands for, or None."""
    k = norm(name)
    if not k:
        return None
    rows = _load()
    hit = rows.get(k)
    if hit is not None:
        return hit
    img = imaging(name)
    if img is not None:
        return img
    for lead in _LEADING:
        if k.startswith(lead):
            hit = rows.get(k[len(lead):])
            if hit is not None:
                return hit
    flat = k.replace(" ", "")
    if len(flat) >= 6:                                      # "LIPIDPROFILE" / "FreeT4": the readers drop the spaces
        return _flat(rows).get(flat)
    return None


def fit(written: str | None) -> Mapped | None:
    """The mapping row a reading with ``?`` for the letters it could not read fits ("vitam?? D" -> "vitamin d"). At least four letters
    must be known and at most 40% of the characters may be ``?``; when the fit is more than one row for different standard tests, or
    none, the answer is None (nothing is guessed)."""
    key = re.sub(r"[^a-z0-9? ]", "", (written or "").casefold())
    key = re.sub(r"\s+", " ", key).strip()
    if "?" not in key or sum(c.isalpha() for c in key) < 4 or key.count("?") > 0.4 * len(key.replace(" ", "")):
        return None
    pat = re.compile(re.escape(key).replace("\\?", ".").replace("\\ ", " "))
    hits = {k: m for k, m in _load().items() if pat.fullmatch(k)}
    if not hits or len({m.canonical for m in hits.values()}) != 1:
        return None
    return hits[sorted(hits)[0]]


def ensure_seed() -> int:
    """Put the seed rows into ``lab_test_alias`` (only the ones that are not there); returns how many were added."""
    from ..db import session_scope

    added = 0
    with session_scope() as sess:
        for a, c, l, n in SEED:
            r = sess.execute(text(
                "INSERT INTO lab_test_alias (alias_key, alias, canonical, loinc, note, source) "
                "VALUES (:k, :a, :c, :l, :n, 'seed') ON CONFLICT (alias_key) DO NOTHING RETURNING 1"),
                {"k": norm(a), "a": a, "c": c, "l": l, "n": n}).first()
            added += 1 if r else 0
    invalidate()
    return added


def table(sess: Any) -> list[dict[str, Any]]:
    """Every row, for the screen (switched-off rows included, so they can be switched back on)."""
    rows = sess.execute(text(
        "SELECT alias_key, alias, canonical, loinc, coalesce(note,'') AS note, source, enabled, updated_at "
        "FROM lab_test_alias ORDER BY canonical, alias")).mappings().all()
    return [{**dict(r), "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None} for r in rows]


def upsert(sess: Any, alias: str, canonical: str, loinc: str | None, note: str | None) -> dict[str, Any]:
    key = norm(alias)
    canonical = " ".join((canonical or "").split())
    if not key or not canonical:
        raise ValueError("Both the written name and the standard test are needed.")
    if len(key) > 80 or len(canonical) > 120:
        raise ValueError("That name is too long.")
    sess.execute(text(
        "INSERT INTO lab_test_alias (alias_key, alias, canonical, loinc, note, source, enabled) "
        "VALUES (:k, :a, :c, :l, :n, 'user', true) "
        "ON CONFLICT (alias_key) DO UPDATE SET alias=:a, canonical=:c, loinc=:l, note=:n, enabled=true, updated_at=now()"),
        {"k": key, "a": " ".join(alias.split()), "c": canonical, "l": (loinc or "").strip() or None, "n": (note or "").strip() or None})
    invalidate()
    return {"alias_key": key, "alias": alias.strip(), "canonical": canonical}


def set_enabled(sess: Any, alias_key: str, enabled: bool) -> bool:
    r = sess.execute(text("UPDATE lab_test_alias SET enabled=:e, updated_at=now() WHERE alias_key=:k RETURNING 1"),
                     {"e": enabled, "k": alias_key}).first()
    invalidate()
    return bool(r)
