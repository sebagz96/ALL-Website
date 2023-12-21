"""Initial migration

Revision ID: 88ceb7d85c7e
Revises: 
Create Date: 2023-09-29 03:08:36.039922

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '88ceb7d85c7e'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('result',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('patient_first_name', sa.String(length=100), nullable=True),
    sa.Column('patient_last_name', sa.String(length=100), nullable=True),
    sa.Column('image_path', sa.String(length=200), nullable=True),
    sa.Column('classification_result', sa.String(length=50), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('result')
    # ### end Alembic commands ###
