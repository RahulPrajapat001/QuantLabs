"""data quality findings table

Revision ID: 4f33604d2756
Revises: 7c1a4e9d2b30
Create Date: 2026-07-25 20:57:52.636428

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f33604d2756'
down_revision: str | None = '7c1a4e9d2b30'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('data_quality_findings',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('check', sa.String(length=64), nullable=False),
    sa.Column('severity', sa.String(length=8), nullable=False),
    sa.Column('security_id', sa.String(length=32), nullable=True),
    sa.Column('date', sa.Date(), nullable=True),
    sa.Column('detail', sa.String(length=512), nullable=False),
    sa.Column('remedy', sa.String(length=512), nullable=False),
    sa.Column('rows_examined', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['security_id'], ['securities.security_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('data_quality_findings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_data_quality_findings_check'), ['check'], unique=False)
        batch_op.create_index(batch_op.f('ix_data_quality_findings_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_data_quality_findings_date'), ['date'], unique=False)
        batch_op.create_index(batch_op.f('ix_data_quality_findings_run_id'), ['run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_data_quality_findings_security_id'), ['security_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_data_quality_findings_severity'), ['severity'], unique=False)
        batch_op.create_index('ix_dq_findings_run_security', ['run_id', 'security_id'], unique=False)


def downgrade() -> None:
    # NOTE: this drops the record of every defect ever found, including the
    # history that makes a recurring bad print distinguishable from a new one.
    # Findings are recomputable from `quantlab validate`, but their timestamps
    # are not.
    with op.batch_alter_table('data_quality_findings', schema=None) as batch_op:
        batch_op.drop_index('ix_dq_findings_run_security')
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_severity'))
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_security_id'))
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_run_id'))
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_date'))
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_created_at'))
        batch_op.drop_index(batch_op.f('ix_data_quality_findings_check'))

    op.drop_table('data_quality_findings')
