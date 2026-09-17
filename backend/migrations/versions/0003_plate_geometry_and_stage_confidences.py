"""Separate plate geometry and per-stage confidences on detections.

Every column added here is nullable with no server default, so the migration is additive
and rows written by the previous version stay valid. `confidence` keeps its existing
meaning (the vehicle score) for API compatibility; `vehicle_confidence` is backfilled
from it so the new column is immediately usable for existing data.

Revision ID: 0003_plate_geometry
Revises: 0002_reliable_processing
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_plate_geometry"
down_revision = "0002_reliable_processing"
branch_labels = None
depends_on = None

_FLOAT_COLUMNS = (
    "vehicle_confidence",
    "manufacturer_confidence",
    "plate_bbox_x1",
    "plate_bbox_y1",
    "plate_bbox_x2",
    "plate_bbox_y2",
    "plate_detection_confidence",
    "plate_ocr_confidence",
    "plate_perspective_skew",
)

_STRING_COLUMNS = (
    ("source_label", 60),
    ("plate_text_raw", 64),
    ("plate_format", 40),
    ("plate_region", 60),
    ("plate_ocr_engine", 40),
    ("plate_ocr_variant", 40),
    ("plate_status", 60),
    ("plate_crop_filename", 255),
    ("plate_rectified_crop_filename", 255),
)


def upgrade():
    for name in _FLOAT_COLUMNS:
        op.add_column("detections", sa.Column(name, sa.Float(), nullable=True))
    for name, length in _STRING_COLUMNS:
        op.add_column("detections", sa.Column(name, sa.String(length=length), nullable=True))
    op.add_column("detections", sa.Column("plate_quad", postgresql.JSONB(), nullable=True))
    op.add_column("detections", sa.Column("plate_rectified", sa.Boolean(), nullable=True))

    # Existing rows already carry the vehicle score in `confidence`.
    op.execute(
        "UPDATE detections SET vehicle_confidence = confidence "
        "WHERE vehicle_confidence IS NULL AND confidence IS NOT NULL"
    )

    op.create_check_constraint(
        "ck_detection_plate_bbox",
        "detections",
        "(plate_bbox_x1 IS NULL AND plate_bbox_y1 IS NULL "
        " AND plate_bbox_x2 IS NULL AND plate_bbox_y2 IS NULL) OR "
        "(plate_bbox_x1 >= 0 AND plate_bbox_y1 >= 0 "
        " AND plate_bbox_x2 <= 1 AND plate_bbox_y2 <= 1 "
        " AND plate_bbox_x2 > plate_bbox_x1 AND plate_bbox_y2 > plate_bbox_y1)",
    )
    op.create_check_constraint(
        "ck_detection_stage_confidences",
        "detections",
        "(plate_detection_confidence IS NULL OR "
        " (plate_detection_confidence >= 0 AND plate_detection_confidence <= 1)) AND "
        "(plate_ocr_confidence IS NULL OR "
        " (plate_ocr_confidence >= 0 AND plate_ocr_confidence <= 1)) AND "
        "(vehicle_confidence IS NULL OR "
        " (vehicle_confidence >= 0 AND vehicle_confidence <= 1))",
    )
    op.create_index("ix_detections_plate_text_raw", "detections", ["plate_text_raw"])
    op.create_index("ix_detections_plate_format", "detections", ["plate_format"])


def downgrade():
    op.drop_index("ix_detections_plate_format", "detections")
    op.drop_index("ix_detections_plate_text_raw", "detections")
    op.drop_constraint("ck_detection_stage_confidences", "detections", type_="check")
    op.drop_constraint("ck_detection_plate_bbox", "detections", type_="check")
    op.drop_column("detections", "plate_rectified")
    op.drop_column("detections", "plate_quad")
    for name, _length in _STRING_COLUMNS:
        op.drop_column("detections", name)
    for name in _FLOAT_COLUMNS:
        op.drop_column("detections", name)
