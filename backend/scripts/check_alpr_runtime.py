"""Smoke test REAL local PaddleOCR and SR. Can run with docker --network none."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.alpr_processor import ALPRConfig, ALPRProcessor  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dir", type=Path, default=Path("/data/models/alpr"))
args = parser.parse_args()
image = np.full((220, 700, 3), 30, np.uint8)
cv2.rectangle(image, (90, 50), (590, 170), (240, 240, 240), -1)
cv2.putText(image, "123ABC10", (125, 133), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (5, 5, 5), 3)
processor = ALPRProcessor(
    ocr_model_dir=args.dir / "en_PP-OCRv4_mobile_rec",
    sr_model_path=args.dir / "FSRCNN_x2.pb",
    config=ALPRConfig(sr_blur_threshold=1_000_000),  # Exercise the real SR branch.
)
result = processor.process_image(image)
print(json.dumps({"plate_text": result["plate_text"], "confidence": result["confidence"],
                  "png_bytes_base64": len(result["warp_image_base64"])}))
assert result["plate_text"] == "123ABC10", "Synthetic plate OCR smoke check failed"
