# Reading doctors' handwriting: what is measured, what was decided

Status date: 2026-10-07. Everything below that says MEASURED was measured on **three real prescription photos from three
different doctors** (two human clinics, one veterinary clinic). Three pages is a small sample: it shows direction, not a rate.

## Decision 1: Qwen is the only handwriting reader

| MEASURED on the three pages | |
|---|---|
| lines a second line-recognition model and Qwen both read | 249 |
| exact agreement | 3% |
| close agreement (similarity 0.8 or more) | 11% |

On real handwriting that second model (trained on clean English sentences) wrote fluent unrelated English
("spouses" for the Apollo logo, "campaigned" for "( 1PM 2PM )", "umbarcina mother's best" for "umbilicus noticed few").
With agreement that low, "the readers disagree" stops meaning "this line is risky": it sends everything to review.

* It has been **removed**: from the code, the settings, the model registry, the deploy scripts and the pod's model
  download. Nothing for it is downloaded or run.
* Qwen2.5-VL-7B is the one handwriting reader. Every handwriting line is single-reader, and the
  validation policy (`validate/policy.py`) sends single-reader lines for medicines, conditions, vitals and tests to review.
  That is safe and it is also why nothing is auto-accepted yet.

## Decision 2: do not send non-text to a reader (`recognition/nontext.py`)

"Qwen returns nothing on 45% of lines" was read at first as unreadable handwriting. Looking at the crops, almost all of the
empty ones were **not text**: bedspread fabric, the edge of the paper, a hand, stray strokes, ruled lines. Qwen answering
"?" there is correct. A region is now set aside (kept with its measurements and the reason, never read) when it has almost
no thin pen/print strokes, or its background is not the page's paper colour (compared by colour only, so a shadow does not
matter), or it is too small to carry a word. A line RapidOCR read as print with high confidence is trusted whatever the colour.

MEASURED effect on handwriting regions sent to the readers: 85 -> 60 (Apollo clinic), 70 -> 45 (Apollo Sugar), 74 -> 64 (vet).
The limits (`CDI_NONTEXT_*`) are PLACEHOLDERS from these three photos.

## What the photos tell us about the input

* The photos are small: 615x895 to 780x1040 px, 65-95 KB. Handwriting is 10-25 px tall. No model can recover strokes that
  were not captured; a 14 px crop upscaled to 32 px is a bigger blur, not more detail.
* One of the three is a **veterinary** prescription (Moitri Veterinary Clinic: pet name, species, weight in kg). It is out of
  scope for a human EMR; medicines read from it are not meaningful for a human patient.

## Decision 3: medicines are a choice among reference names, not free text (`extract/medicine_resolve.py`)

* Advice listed as a medicine ("steam inhalation", "gargle with warm water", "plenty of fluids") is filed as advice.
* A medicine name the lists do not place exactly but that is close to reference names is put to the model as a numbered
  CHOICE with a "none" option; only an offered candidate that is textually close to the reading can come out. What the model
  chose is kept apart from what was written (`reference_name` in the result) and is **always `needs_check`**.
* First measurement on the real pages is MIXED: some picks are right (Coraxin -> Covaxin), several are wrong or unverifiable
  (Zincolit -> Zincort where the page says Zincovit, which the medicine list does not contain; Bonpaz -> Bonipraz where the
  page says Sompraz). The reference list's coverage and the quality of the free reading bound this method. It is a
  suggestion for a person, not a decision, until a labelled set measures it.

## What would actually move accuracy (in order)

1. A labelled set of real lines (the owner expects 100,000+ lines from about 150 doctors): `eval/` scores it, the
   correction loop exports corrections, `recognition/scoring.py` scores labelled lines.
2. A LoRA fine-tune of Qwen2.5-VL-7B (Apache-2.0) on those lines, split by doctor so a new doctor is a fair test.
3. Reading each line against the reference lists by likelihood (rank the candidates by how well the model's own probabilities
   fit the crop) instead of asking for a number. Not built yet.
4. Better capture: a flat page, closer, steady (UP-S2 will tell the person at the counter).
