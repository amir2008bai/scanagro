"""Orchestrates the five recognition stages into one ``VisionImageResult``.

    frame -> vehicles -> plates -> quad/rectify -> OCR -> attributes -> detections

Each stage keeps its own confidence. The number a caller sees against the vehicle is the
detector's score for the vehicle; the number against the plate box is the plate
detector's score; the number against the text is measured from the OCR model's own
per-character probabilities. They are never collapsed into a single figure, because a
crisp detection of an illegible plate and a blurry detection of a legible one are
different situations and the consumer has to be able to tell them apart.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.vision import NormalizedBBox, PlateObservation, VisionDetection, VisionImageResult
from app.services.vision.local import geometry, runtime
from app.services.vision.local.attributes import AttributeReader
from app.services.vision.local.plate_detect import PlateBox, PlateDetector
from app.services.vision.local.plate_ocr import PlateReader, PlateReading
from app.services.vision.local.reference_gallery import ReferenceGallery
from app.services.vision.local.vehicle import VehicleBox, VehicleDetector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    model_dir: Path
    device: str = "cpu"
    onnx_threads: int = 0
    max_parallel_images: int = 1
    vehicle_input_size: int = 960
    vehicle_threshold: float = 0.35
    plate_model: str = "yolo-v9-s-608-license-plate-end2end"
    plate_threshold: float = 0.25
    plate_tile: int = 608
    plate_tile_overlap: float = 0.35
    plate_frame_sweep: bool = True
    ocr_model: str = "cct-s-v2-global-model"
    use_ppocr: bool = True
    alpr_enabled: bool = False
    alpr_ocr_dir: Path | None = None
    alpr_sr_path: Path | None = None
    ocr_accept_confidence: float = 0.55
    ocr_accept_confidence_unformatted: float = 0.88
    read_attributes: bool = True
    cyrillic_attributes: bool = True
    reference_gallery: bool = True
    reference_dir: Path | None = None
    reference_min_similarity: float = 0.62
    reference_min_margin: float = 0.04
    reference_top_k: int = 5
    #: Fraction of the frame height at the top to ignore when accepting plate boxes.
    #: These cameras burn a timestamp there and the plate detector reliably fires on it.
    overlay_band_top: float = 0.045
    overlay_band_bottom: float = 0.0
    #: Keep an unassociated plate only if it reads cleanly enough to be worth reporting.
    orphan_plate_min_confidence: float = 0.60
    save_plate_crops: bool = True
    max_detections: int = 100


class LocalRecognitionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vehicle_detector = VehicleDetector(
            config.model_dir, config.device, config.onnx_threads, config.vehicle_input_size
        )
        self.plate_detector = PlateDetector(
            config.plate_model,
            config.model_dir,
            config.plate_threshold,
            config.plate_tile,
            config.plate_tile_overlap,
            device=config.device,
            threads=config.onnx_threads,
        )
        self.plate_reader = PlateReader(
            config.model_dir,
            config.ocr_model,
            config.use_ppocr,
            config.device,
            config.ocr_accept_confidence,
            config.ocr_accept_confidence_unformatted,
        )
        if config.alpr_enabled:
            from app.services.vision.local.alpr_reader import ALPRPlateReader

            if config.alpr_ocr_dir is None:
                raise ValueError("VISION_ALPR_OCR_DIR is required when ALPR is enabled")
            self.plate_reader = ALPRPlateReader(
                self.plate_reader, config.alpr_ocr_dir, config.alpr_sr_path
            )
        self.attribute_reader = AttributeReader(
            config.model_dir, config.read_attributes, cyrillic=config.cyrillic_attributes
        )
        self.reference_gallery = None
        if config.reference_gallery and config.reference_dir is not None:
            self.reference_gallery = ReferenceGallery(
                config.reference_dir,
                config.model_dir,
                device=config.device,
                threads=config.onnx_threads,
                min_similarity=config.reference_min_similarity,
                min_margin=config.reference_min_margin,
                top_k=config.reference_top_k,
            )

    def warm_up(self) -> list[str]:
        """Build every session up front so the first request is not the slow one."""
        blank = np.zeros((64, 128, 3), dtype=np.uint8)
        self.vehicle_detector.detect(np.zeros((320, 320, 3), dtype=np.uint8), threshold=0.99)
        self.plate_detector._scan(blank, 0, 0, "warmup")
        self.plate_reader.read(blank)
        if self.config.read_attributes:
            self.attribute_reader.read(blank, "truck")
        if self.reference_gallery is not None and self.reference_gallery.available:
            self.reference_gallery.classify(blank)
        return runtime.loaded_keys()

    # ----- helpers ---------------------------------------------------------------

    def _in_overlay_band(self, plate: PlateBox, height: int) -> bool:
        cy = plate.centre[1]
        if self.config.overlay_band_top and cy < self.config.overlay_band_top * height:
            return True
        if (
            self.config.overlay_band_bottom
            and cy > (1.0 - self.config.overlay_band_bottom) * height
        ):
            return True
        return False

    @staticmethod
    def _assign(plates: list[PlateBox], vehicles: list[VehicleBox]) -> dict[int, list[PlateBox]]:
        """Attach each plate to the vehicle it belongs to.

        The plate centre normally falls inside its own vehicle's box. When boxes overlap
        (a tractor in front of its trailer) the *smallest* containing box wins, because
        that is the more specific object. Plates whose centre falls in no box stay
        unassigned rather than being attached to whatever is nearest.
        """
        assignment: dict[int, list[PlateBox]] = {index: [] for index in range(len(vehicles))}
        assignment[-1] = []
        for plate in plates:
            cx, cy = plate.centre
            containing = [
                (vehicle.area, index)
                for index, vehicle in enumerate(vehicles)
                if vehicle.x1 <= cx <= vehicle.x2 and vehicle.y1 <= cy <= vehicle.y2
            ]
            if containing:
                assignment[min(containing)[1]].append(plate)
            else:
                assignment[-1].append(plate)
        return assignment

    def _resolve_type(self, crop, vehicle: VehicleBox, attributes):
        """Settle on a machine type from three sources of different specificity.

        Precedence, strongest first:

        1. **Reference gallery.** It looks at the whole machine and can name things COCO
           cannot, so when it is willing to answer it wins.
        2. **Badge text.** `KAMAZ` on a grille implies a lorry. Coarser than the gallery
           (it cannot tell a lorry from its trailer) but it is direct evidence.
        3. **The detector's COCO class**, mapped onto our vocabulary. Always available,
           least specific, and wrong for every tractor.

        The chosen source is reported alongside the type, so a consumer can see whether
        `tractor` came from a reference match or from a guess about a COCO `truck`.
        """
        evidence: list[dict] = []
        if self.reference_gallery is not None and self.reference_gallery.available:
            try:
                match = self.reference_gallery.classify(crop)
            except Exception:  # a cold or corrupt index must not fail the whole image
                match = None
            if match is not None:
                entry = {
                    "source": "reference_gallery",
                    "vehicle_type": match.vehicle_type,
                    "similarity": match.similarity,
                    "margin": match.margin,
                    "status": match.reason,
                    "neighbours": match.neighbours[:3],
                }
                evidence.append(entry)
                if match.reason == "accepted":
                    # A web photo may depict a truck AND trailer. A narrow retrieval
                    # margin cannot override agreement between a vehicle detection
                    # and a confidently read manufacturer badge on that vehicle.
                    conflict = (
                        match.vehicle_type != vehicle.vehicle_type
                        and attributes.vehicle_type == vehicle.vehicle_type
                        and (attributes.manufacturer_confidence or 0.0) >= 0.90
                        and match.margin < 0.10
                    )
                    if conflict:
                        entry["status"] = "conflict_with_detector_and_badge"
                        return (
                            attributes.vehicle_type, "badge_text",
                            attributes.manufacturer_confidence, evidence,
                        )
                    return match.vehicle_type, "reference_gallery", match.similarity, evidence

        if attributes.vehicle_type:
            return (
                attributes.vehicle_type,
                "badge_text",
                attributes.manufacturer_confidence,
                evidence,
            )
        return vehicle.vehicle_type, "detector", round(vehicle.confidence, 4), evidence

    def _norm_box(
        self, x1: int, y1: int, x2: int, y2: int, width: int, height: int
    ) -> NormalizedBBox:
        return NormalizedBBox(
            x1=min(max(x1 / width, 0.0), 1.0),
            y1=min(max(y1 / height, 0.0), 1.0),
            x2=min(max(x2 / width, 0.0), 1.0),
            y2=min(max(y2 / height, 0.0), 1.0),
        )

    def _crop(self, image: np.ndarray, plate: PlateBox) -> tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        margin = max(3, int(0.10 * plate.height))
        x1, y1 = max(0, plate.x1 - margin), max(0, plate.y1 - margin)
        x2, y2 = min(width, plate.x2 + margin), min(height, plate.y2 + margin)
        return image[y1:y2, x1:x2], x1, y1

    def _plate_observation(
        self, image: np.ndarray, plate: PlateBox, width: int, height: int, crop_sink
    ) -> tuple[PlateObservation, PlateReading]:
        crop, ox, oy = self._crop(image, plate)
        reading = self.plate_reader.read(crop)
        quad = None
        if reading.quad is not None:
            quad = geometry.quad_to_normalised(reading.quad.quad, ox, oy, width, height)
        before_name = after_name = None
        if crop_sink is not None and crop.size:
            before_name, after_name = crop_sink(crop, reading.rectified_image)
        observation = PlateObservation(
            bbox=self._norm_box(plate.x1, plate.y1, plate.x2, plate.y2, width, height),
            quad=quad,
            detection_confidence=round(plate.confidence, 4),
            ocr_confidence=reading.ocr_confidence,
            text=reading.text,
            text_raw=reading.raw_text,
            readable=reading.readable,
            plate_format=reading.plate_format,
            region=reading.region,
            ocr_engine=reading.engine,
            ocr_variant=reading.variant,
            rectified=reading.rectified,
            perspective_skew=reading.skew,
            status=reading.reason,
            crop_filename=before_name,
            rectified_crop_filename=after_name,
            alternatives=[
                {
                    "engine": c.engine,
                    "variant": c.variant,
                    "text": c.compact,
                    "mean_confidence": round(c.mean_confidence, 4),
                    "min_confidence": round(c.min_confidence, 4),
                }
                for c in reading.candidates[:6]
            ],
            disagreement_positions=reading.disagreement_positions,
        )
        return observation, reading

    # ----- entry point -----------------------------------------------------------

    def analyze_array(
        self, image_bgr: np.ndarray, crop_sink=None
    ) -> tuple[VisionImageResult, dict[str, Any]]:
        started = time.perf_counter()
        height, width = image_bgr.shape[:2]
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        vehicles = self.vehicle_detector.detect(image_bgr, self.config.vehicle_threshold)
        timings["vehicle_detect"] = round(time.perf_counter() - mark, 3)

        mark = time.perf_counter()
        plates = self.plate_detector.detect(
            image_bgr, vehicles, frame_sweep=self.config.plate_frame_sweep
        )
        dropped_overlay = [p for p in plates if self._in_overlay_band(p, height)]
        plates = [p for p in plates if not self._in_overlay_band(p, height)]
        timings["plate_detect"] = round(time.perf_counter() - mark, 3)

        mark = time.perf_counter()
        assignment = self._assign(plates, vehicles)
        detections: list[VisionDetection] = []

        for index, vehicle in enumerate(vehicles):
            crop = image_bgr[vehicle.y1 : vehicle.y2, vehicle.x1 : vehicle.x2]
            attributes = self.attribute_reader.read(crop, vehicle.coco_label)
            reference_crop = (
                self.reference_gallery.vehicle_crop(image_bgr, vehicle)
                if self.reference_gallery is not None and self.reference_gallery.available
                else crop
            )
            vehicle_type, type_source, type_confidence, type_evidence = self._resolve_type(
                reference_crop, vehicle, attributes
            )
            observations: list[PlateObservation] = []
            for plate in sorted(assignment[index], key=lambda p: -p.confidence):
                observation, _ = self._plate_observation(image_bgr, plate, width, height, crop_sink)
                observations.append(observation)
            # The plate we report against the vehicle is the best-read one it carries.
            observations.sort(key=lambda o: (o.readable, o.ocr_confidence or 0.0), reverse=True)
            primary = observations[0] if observations else None
            detections.append(
                VisionDetection(
                    vehicle_type=vehicle_type,
                    manufacturer=attributes.manufacturer,
                    model=attributes.model,
                    license_plate=primary.text if primary and primary.readable else None,
                    plate_readable=bool(primary and primary.readable),
                    confidence=round(vehicle.confidence, 4),
                    vehicle_confidence=round(vehicle.confidence, 4),
                    manufacturer_confidence=attributes.manufacturer_confidence,
                    bbox=self._norm_box(
                        vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2, width, height
                    ),
                    plate=primary,
                    extra_plates=observations[1:3],
                    source_label=vehicle.coco_label,
                    vehicle_type_source=type_source,
                    vehicle_type_confidence=type_confidence,
                    evidence=(type_evidence + attributes.evidence)[:8],
                    notes=None,
                )
            )

        for plate in assignment[-1]:
            observation, reading = self._plate_observation(
                image_bgr, plate, width, height, crop_sink
            )
            if (
                not reading.readable
                or (observation.ocr_confidence or 0.0) < self.config.orphan_plate_min_confidence
            ):
                continue
            # A readable plate with no vehicle box around it: reported, but the box is the
            # plate's own and the type is explicitly unknown. We do not invent a vehicle.
            detections.append(
                VisionDetection(
                    vehicle_type="unidentified_vehicle",
                    license_plate=observation.text,
                    plate_readable=True,
                    confidence=None,
                    vehicle_confidence=None,
                    bbox=observation.bbox,
                    plate=observation,
                    source_label=None,
                    notes="Plate read but no vehicle box contained it; bbox is the plate itself.",
                )
            )
        timings["plate_read"] = round(time.perf_counter() - mark, 3)

        raw = {
            "provider": "local",
            "pipeline": (
                "rtdetr_v2_r18 + yolov9_plate + paddle_alpr/cct/ppocr"
                if self.config.alpr_enabled else "rtdetr_v2_r18 + yolov9_plate + cct/ppocr"
            ),
            "image_size": [width, height],
            "timings_seconds": timings,
            "total_seconds": round(time.perf_counter() - started, 3),
            "vehicles_detected": len(vehicles),
            "plates_detected": len(plates),
            "plates_dropped_overlay_band": len(dropped_overlay),
            "unassigned_plates": len(assignment[-1]),
            "device": self.config.device,
        }
        return VisionImageResult(detections=detections[: self.config.max_detections]), raw

    def analyze_path(self, path: Path, crop_sink=None) -> tuple[VisionImageResult, dict[str, Any]]:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        return self.analyze_array(image, crop_sink)
