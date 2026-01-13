"""remove_unique_constraint_from_file_path

Revision ID: 328afecc7200
Revises: 94d6210f7c39
Create Date: 2026-01-13 11:35:17.698173

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '328afecc7200'
down_revision: Union[str, None] = '94d6210f7c39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Remove UNIQUE constraint from files.file_path to allow multi-project analysis.

    Same physical file can now be analyzed by multiple projects/deployments.
    For SQLite, we recreate the table without the UNIQUE constraint.
    """
    # Create new table without UNIQUE constraint
    op.execute("""
        CREATE TABLE files_new (
            id VARCHAR(36) NOT NULL,
            deployment_id VARCHAR(36) NOT NULL,
            file_path TEXT NOT NULL,
            file_type VARCHAR(10) NOT NULL,
            file_format VARCHAR(10),
            size_bytes INTEGER,
            width_px INTEGER,
            height_px INTEGER,
            timestamp DATETIME NOT NULL,
            exif_data JSON,
            duration_seconds FLOAT,
            created_at DATETIME NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(deployment_id) REFERENCES deployments (id) ON DELETE CASCADE
        )
    """)

    # Copy data from old table
    op.execute("""
        INSERT INTO files_new
        SELECT id, deployment_id, file_path, file_type, file_format, size_bytes,
               width_px, height_px, timestamp, exif_data, duration_seconds, created_at
        FROM files
    """)

    # Drop old table
    op.execute("DROP TABLE files")

    # Rename new table
    op.execute("ALTER TABLE files_new RENAME TO files")

    # Recreate indexes
    op.execute("CREATE INDEX idx_files_deployment ON files (deployment_id)")
    op.execute("CREATE INDEX idx_files_timestamp ON files (timestamp)")


def downgrade() -> None:
    """
    Re-add UNIQUE constraint to files.file_path.
    """
    # Create table with UNIQUE constraint
    op.execute("""
        CREATE TABLE files_new (
            id VARCHAR(36) NOT NULL,
            deployment_id VARCHAR(36) NOT NULL,
            file_path TEXT NOT NULL,
            file_type VARCHAR(10) NOT NULL,
            file_format VARCHAR(10),
            size_bytes INTEGER,
            width_px INTEGER,
            height_px INTEGER,
            timestamp DATETIME NOT NULL,
            exif_data JSON,
            duration_seconds FLOAT,
            created_at DATETIME NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(deployment_id) REFERENCES deployments (id) ON DELETE CASCADE,
            UNIQUE (file_path)
        )
    """)

    # Copy data
    op.execute("""
        INSERT INTO files_new
        SELECT id, deployment_id, file_path, file_type, file_format, size_bytes,
               width_px, height_px, timestamp, exif_data, duration_seconds, created_at
        FROM files
    """)

    # Drop and rename
    op.execute("DROP TABLE files")
    op.execute("ALTER TABLE files_new RENAME TO files")

    # Recreate indexes
    op.execute("CREATE INDEX idx_files_deployment ON files (deployment_id)")
    op.execute("CREATE INDEX idx_files_timestamp ON files (timestamp)")
