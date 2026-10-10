"""human corrections + per-doctor lexicon (the correction loop)

* ``correction``: append-only (database trigger): the original prediction, what the engines read,
  what a person corrected it to, who, which doctor, which crop. It is both the knowledge-update
  source and the training-data source for the next offline fine-tuning cycle.
* ``doctor_lexicon``: how one doctor writes a thing and what a reviewer says it means, with counts.
  Once a mapping is confirmed often enough and resolves to exactly one concept it is promoted to a
  class-C ``kb_alias`` for that doctor (used immediately by the alias cascade, no retraining).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-06
"""
from __future__ import annotations

from alembic import op


def _exec(sql: str) -> None:
    with op.get_bind().connection.dbapi_connection.cursor() as cur:
        cur.execute(sql)


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    _exec(r"""
CREATE TABLE IF NOT EXISTS correction (
  id                uuid PRIMARY KEY,
  document_id       uuid NOT NULL REFERENCES source_document(id),
  fact_id           uuid,                          -- no FK on purpose: re-extraction replaces facts, the correction must survive
  doctor_id         uuid REFERENCES cn_practitioner(id),   -- NULL = doctor unknown (never guessed)
  field_type        text NOT NULL,                 -- the fact type: investigation_order | condition | advice | ...
  original_value    text,                          -- what the system held before the correction
  qwen_value        text,                          -- what the reader read (NULL when not read)
  corrected_value   text NOT NULL,
  confidence        numeric(5,4),
  prediction_status text,                          -- accepted | needs_review (before the correction)
  crop_hash         char(64),
  crop_ref          jsonb NOT NULL DEFAULT '{}',   -- observation ids, page id, bbox: enough to re-cut the crop
  reviewer_id       text NOT NULL,
  model_stack       jsonb NOT NULL DEFAULT '{}',
  created_at        timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_correction_doctor ON correction (doctor_id, field_type, created_at);
CREATE INDEX IF NOT EXISTS ix_correction_document ON correction (document_id);
DROP TRIGGER IF EXISTS trg_correction_immutable ON correction;
CREATE TRIGGER trg_correction_immutable BEFORE UPDATE OR DELETE ON correction
  FOR EACH ROW EXECUTE FUNCTION cdi_evidence_is_immutable();

CREATE TABLE IF NOT EXISTS doctor_lexicon (
  id              bigserial PRIMARY KEY,
  practitioner_id uuid NOT NULL REFERENCES cn_practitioner(id) ON DELETE CASCADE,
  field_type      text NOT NULL,
  raw_norm        text NOT NULL,                   -- how this doctor writes it, normalised
  raw_example     text NOT NULL,
  canonical       text NOT NULL,                   -- what a reviewer says it means
  count           int NOT NULL DEFAULT 0,
  verified_count  int NOT NULL DEFAULT 0,
  concept_id      text REFERENCES kb_concept(id),  -- set once promoted to a class-C alias; NULL = not promoted
  first_seen      timestamptz NOT NULL,
  last_seen       timestamptz NOT NULL,
  UNIQUE (practitioner_id, field_type, raw_norm, canonical)
);
CREATE INDEX IF NOT EXISTS ix_doctor_lexicon_doctor ON doctor_lexicon (practitioner_id, field_type);
""")


def downgrade() -> None:
    _exec("""
    DROP TABLE IF EXISTS doctor_lexicon CASCADE;
    DROP TABLE IF EXISTS correction CASCADE;
    """)
