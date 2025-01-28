"""add_model_tracking_columns

Revision ID: 232a6a55afaa
Revises: 
Create Date: 2025-01-28 09:34:08.556423

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '232a6a55afaa'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns
    op.add_column('measurements', sa.Column('model_used', sa.String(), nullable=True))
    op.add_column('measurements', sa.Column('created_at', sa.DateTime(), nullable=True))

    # Update existing rows with current timestamp
    op.execute("""
        UPDATE measurements 
        SET created_at = CURRENT_TIMESTAMP 
        WHERE created_at IS NULL
    """)


def downgrade() -> None:
    # Remove the columns if we need to roll back
    op.drop_column('measurements', 'model_used')
    op.drop_column('measurements', 'created_at')
