"""research log table

Revision ID: 7c1a4e9d2b30
Revises: 44bbf26487c0
Create Date: 2026-07-25 10:12:41.180322

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c1a4e9d2b30'
down_revision: str | None = '44bbf26487c0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('research_log',
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('git_commit_hash', sa.String(length=40), nullable=True),
    sa.Column('git_dirty', sa.Boolean(), nullable=True),
    sa.Column('strategy_name', sa.String(length=128), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('universe', sa.String(length=64), nullable=False),
    sa.Column('index_name', sa.String(length=32), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=False),
    sa.Column('split', sa.String(length=32), nullable=False),
    sa.Column('cost_model', sa.String(length=64), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=False),
    sa.Column('sharpe', sa.Float(), nullable=True),
    sa.Column('n_periods', sa.Integer(), nullable=True),
    sa.Column('returns_skew', sa.Float(), nullable=True),
    sa.Column('returns_kurtosis', sa.Float(), nullable=True),
    sa.Column('periods_per_year', sa.Integer(), server_default='252', nullable=False),
    sa.Column('holdout_touched', sa.Boolean(), nullable=False),
    sa.Column('hypothesis_id', sa.String(length=32), nullable=True),
    sa.Column('hypothesis_hash', sa.String(length=64), nullable=True),
    sa.Column('exploratory', sa.Boolean(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('error', sa.String(length=2048), nullable=True),
    sa.Column('notes', sa.String(length=1024), nullable=True),
    sa.PrimaryKeyConstraint('run_id')
    )
    with op.batch_alter_table('research_log', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_research_log_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_research_log_hypothesis_id'), ['hypothesis_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_research_log_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_research_log_strategy_name'), ['strategy_name'], unique=False)


def downgrade() -> None:
    # NOTE: dropping this table destroys the trials record that every reported
    # Sharpe in the project is deflated against. It is here for schema symmetry;
    # in practice the research log is append-only and never rolled back.
    with op.batch_alter_table('research_log', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_research_log_strategy_name'))
        batch_op.drop_index(batch_op.f('ix_research_log_status'))
        batch_op.drop_index(batch_op.f('ix_research_log_hypothesis_id'))
        batch_op.drop_index(batch_op.f('ix_research_log_created_at'))

    op.drop_table('research_log')
