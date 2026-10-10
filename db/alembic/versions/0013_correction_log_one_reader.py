"""0013: the correction log keeps what Qwen read, not a second engine's reading: drop the unused column.

Only the column is dropped (a schema change; the immutable-row trigger guards rows, not columns). A database created from 0006 on never has it."""
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE correction DROP COLUMN IF EXISTS trocr_value")


def downgrade() -> None:
    pass        # the column held no data that is read anywhere; it is not brought back
