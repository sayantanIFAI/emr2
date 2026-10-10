"""Tables of the correction loop as SQLAlchemy Core objects (migration 0006 is the source of truth in
PostgreSQL; these mirror it so the same SQL can be run and tested on SQLite, and a test keeps the two
in step). ``kb_concept`` / ``kb_alias`` are mirrored only as far as the promotion step needs them."""
from __future__ import annotations

from sqlalchemy import (
                        JSON,
                        BigInteger,
                        Boolean,
                        Column,
                        DateTime,
                        Integer,
                        MetaData,
                        Numeric,
                        String,
                        Table,
                        Text,
                        UniqueConstraint,
                        Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()
_JSON = JSON().with_variant(JSONB(), "postgresql")
_ID = Integer().with_variant(BigInteger(), "postgresql")

correction = Table(
    "correction", metadata,
    Column("id", Uuid(as_uuid=False), primary_key=True),
    Column("document_id", Uuid(as_uuid=False), nullable=False),
    Column("fact_id", Uuid(as_uuid=False)),
    Column("doctor_id", Uuid(as_uuid=False)),
    Column("field_type", Text, nullable=False),
    Column("original_value", Text),
    Column("qwen_value", Text),
    Column("corrected_value", Text, nullable=False),
    Column("confidence", Numeric(5, 4)),
    Column("prediction_status", Text),
    Column("crop_hash", String(64)),
    Column("crop_ref", _JSON, nullable=False),
    Column("reviewer_id", Text, nullable=False),
    Column("model_stack", _JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

doctor_lexicon = Table(
    "doctor_lexicon", metadata,
    Column("id", _ID, primary_key=True, autoincrement=True),
    Column("practitioner_id", Uuid(as_uuid=False), nullable=False),
    Column("field_type", Text, nullable=False),
    Column("raw_norm", Text, nullable=False),
    Column("raw_example", Text, nullable=False),
    Column("canonical", Text, nullable=False),
    Column("count", Integer, nullable=False, default=0),
    Column("verified_count", Integer, nullable=False, default=0),
    Column("concept_id", Text),
    Column("first_seen", DateTime(timezone=True), nullable=False),
    Column("last_seen", DateTime(timezone=True), nullable=False),
    UniqueConstraint("practitioner_id", "field_type", "raw_norm", "canonical"),
)

kb_concept = Table(
    "kb_concept", metadata,
    Column("id", Text, primary_key=True),
    Column("domain", Text, nullable=False),
    Column("canonical_name", Text, nullable=False),
    Column("active", Boolean, nullable=False, default=True),
)

kb_alias = Table(
    "kb_alias", metadata,
    Column("id", _ID, primary_key=True, autoincrement=True),
    Column("concept_id", Text, nullable=False),
    Column("alias", Text, nullable=False),
    Column("alias_norm", Text, nullable=False),
    Column("alias_class", String(1), nullable=False),
    Column("practitioner_id", Uuid(as_uuid=False)),
    Column("source", Text, nullable=False, default="seed"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("concept_id", "alias_norm", "alias_class", "practitioner_id"),
)
