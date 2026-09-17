"""Durable queue leases, active-job uniqueness and detection integrity.

Revision ID: 0002_reliable_processing
Revises: 0001_initial
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_reliable_processing"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "processing_jobs", sa.Column("attempts", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column("processing_jobs", sa.Column("lease_token", sa.Uuid(), nullable=True))
    op.add_column(
        "processing_jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
    )
    # Stop legacy workers before upgrading. Their tasks have no durable lease.
    op.execute(
        "UPDATE processing_jobs SET status='pending', started_at=NULL WHERE status='processing'"
    )
    op.execute("""WITH duplicates AS (
        SELECT id, row_number() OVER (PARTITION BY image_id ORDER BY created_at DESC, id DESC) AS n
        FROM processing_jobs WHERE status='pending'
    ) UPDATE processing_jobs SET status='failed', finished_at=now(),
      error_message='Superseded duplicate job during queue migration'
      WHERE id IN (SELECT id FROM duplicates WHERE n > 1)""")
    op.create_index(
        "uq_processing_jobs_active_image",
        "processing_jobs",
        ["image_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )
    op.create_index("ix_processing_jobs_lease_expires_at", "processing_jobs", ["lease_expires_at"])
    op.create_index("ix_detections_model", "detections", ["model"])
    op.execute(
        "UPDATE detections SET license_plate=NULL WHERE NOT plate_readable OR trim(license_plate)=''"
    )
    op.execute("UPDATE detections SET plate_readable=false WHERE license_plate IS NULL")
    op.create_check_constraint(
        "ck_detection_bbox",
        "detections",
        "bbox_x1 >= 0 AND bbox_y1 >= 0 AND bbox_x2 <= 1 AND bbox_y2 <= 1 AND bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1",
    )
    op.create_check_constraint(
        "ck_detection_confidence",
        "detections",
        "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
    )
    op.create_check_constraint(
        "ck_detection_plate",
        "detections",
        "(plate_readable AND license_plate IS NOT NULL) OR (NOT plate_readable AND license_plate IS NULL)",
    )


def downgrade():
    for name in ("ck_detection_bbox", "ck_detection_confidence", "ck_detection_plate"):
        op.drop_constraint(name, "detections", type_="check")
    op.drop_index("ix_detections_model", "detections")
    op.drop_index("ix_processing_jobs_lease_expires_at", "processing_jobs")
    op.drop_index("uq_processing_jobs_active_image", "processing_jobs")
    for name in ("lease_expires_at", "lease_token", "attempts"):
        op.drop_column("processing_jobs", name)
