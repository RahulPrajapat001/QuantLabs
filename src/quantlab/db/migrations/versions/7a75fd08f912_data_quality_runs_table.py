"""data quality runs table

Revision ID: 7a75fd08f912
Revises: 4f33604d2756
Create Date: 2026-07-26 12:17:00.434592

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a75fd08f912'
down_revision: str | None = '4f33604d2756'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('data_quality_runs',
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('rows_examined', sa.Integer(), nullable=False),
    sa.Column('checks_run', sa.JSON(), nullable=False),
    sa.Column('error_count', sa.Integer(), nullable=False),
    sa.Column('warning_count', sa.Integer(), nullable=False),
    sa.Column('info_count', sa.Integer(), nullable=False),
    sa.Column('scope', sa.String(length=256), nullable=True),
    sa.PrimaryKeyConstraint('run_id')
    )
    with op.batch_alter_table('data_quality_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_data_quality_runs_created_at'), ['created_at'], unique=False)



def downgrade() -> None:
    # NOTE: dropping this loses the record of which validation runs happened,
    # including the clean ones. `--fail-on-regression` then treats the next run as
    # a fresh baseline and cannot catch a clean -> dirty regression.
    with op.batch_alter_table('data_quality_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_data_quality_runs_created_at'))

    op.drop_table('data_quality_runs')
