"""0014: the department the front desk named when uploading (``source_document.department_hint``): used only as a prior when a handwritten investigation cannot be read."""
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE source_document ADD COLUMN IF NOT EXISTS department_hint text")


def downgrade() -> None:
    op.execute("ALTER TABLE source_document DROP COLUMN IF EXISTS department_hint")
