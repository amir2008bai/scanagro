"""Record which stage decided the machine type, and how sure it was.

`vehicle_type` on its own does not say whether it came from a reference match, a badge
reading, or COCO's coarse guess — and those deserve very different trust. Both columns are
nullable with no server default, so rows written before this migration stay valid and
providers that do not produce the information simply leave it null.

Revision ID: 0004_vehicle_type_provenance
Revises: 0003_plate_geometry
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_vehicle_type_provenance"
down_revision = "0003_plate_geometry"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "detections", sa.Column("vehicle_type_source", sa.String(length=40), nullable=True)
    )
    op.add_column("detections", sa.Column("vehicle_type_confidence", sa.Float(), nullable=True))
    op.create_check_constraint(
        "ck_detection_vehicle_type_confidence",
        "detections",
        "vehicle_type_confidence IS NULL OR "
        "(vehicle_type_confidence >= 0 AND vehicle_type_confidence <= 1)",
    )
    op.create_index("ix_detections_vehicle_type_source", "detections", ["vehicle_type_source"])

    # Existing rows were all typed by the detector's COCO class; say so rather than
    # leaving it ambiguous, and carry over the score that decision was made on.
    op.execute(
        "UPDATE detections SET vehicle_type_source = 'detector', "
        "vehicle_type_confidence = vehicle_confidence "
        "WHERE vehicle_type_source IS NULL"
    )


def downgrade():
    op.drop_index("ix_detections_vehicle_type_source", "detections")
    op.drop_constraint("ck_detection_vehicle_type_confidence", "detections", type_="check")
    op.drop_column("detections", "vehicle_type_confidence")
    op.drop_column("detections", "vehicle_type_source")
