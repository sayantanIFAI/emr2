"""0012: one flat table for the whole extraction (intake, patient, organisation, doctor, booking, lab test): ``prescription_flat``.

The same statements are run by ``output.flat_table.ensure_table`` the first time a row is written, so a host that has not migrated still works."""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from cdi_adapter.output.flat_table import DDL
    for stmt in [x for x in DDL.split(";\n") if x.strip()]:
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS prescription_flat")
