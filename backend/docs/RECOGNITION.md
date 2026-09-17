# Recognition pipeline

What was chosen, why, what it costs, and what it still gets wrong.

The service keeps the original architecture. Recognition is reached through the existing
`VisionProvider` adapter, so `mock` and `openai_compatible` continue to work unchanged and
`local` is a third option rather than a replacement.

---

## 1. The problem this dataset poses

The supplied footage is a fisheye CCTV camera over a weighbridge at an agricultural
depot. Every frame is 2688x1520. That sounds generous until you measure a plate: they run
about **60-140 px wide**, roughly **0.2 % of the frame area**.

This single fact determines the design. The plate detector takes a 608x608 input, so
feeding it a whole frame scales a 100 px plate down to about 23 px and it disappears.
Measured on these 25 frames with the same YOLOv9-s 608 detector throughout:

| Pass | Frames with a plate detected |
| --- | --- |
| Whole frame, resized once to 608x608 | **1 / 25** |
| Overlapping 608 px tiles at native resolution | 19 / 25 |
| Vehicle stage first, then tiles over each vehicle crop | **15 / 15** frames where a plate is legible |

The first row is what a straightforward `fast-alpr` call does, and it is why this needed
more than wiring an existing framework together.

Other properties that shaped the design:

| Property | Consequence |
| --- | --- |
| Strong barrel distortion near the frame edges | Plates are not merely rotated; edges are genuinely non-parallel |
| Plates at 20-45° to the camera | Rotation alone leaves the glyphs sheared — a homography is required |
| Agricultural machinery (Kirovets, MTZ) | COCO has no tractor class; type must be handled explicitly |
| Burnt-in timestamp along the top edge | The plate detector fires on it reliably; it must be excluded |
| Grille mouldings reading `KAMAZ` | The plate detector fires on those too |
| Kazakh plates, Latin + digits | A Latin-alphabet OCR model is appropriate; Cyrillic is not needed for plates |

---

## 2. Components evaluated

Every candidate below was actually installed and run on the 25 frames unless the table
says otherwise.

| Component | Licence (code) | Licence (weights) | Verdict |
| --- | --- | --- | --- |
| **RT-DETRv2 R18, ONNX** — [onnx-community/rtdetr_v2_r18vd-ONNX](https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX) | Apache-2.0 | Apache-2.0 (from `PekingU/rtdetr_v2_r18vd`) | **Adopted** for vehicle detection |
| **open-image-models** — [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) | MIT | MIT | **Adopted** for plate detection (YOLOv9-s 608 end2end) |
| **fast-plate-ocr** — [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | MIT | MIT | **Adopted** as the fast OCR engine (`cct-s-v2-global-model`) |
| **RapidOCR / PP-OCRv4** — [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR), weights from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | Apache-2.0 | **Adopted** as the accurate OCR engine and the badge reader |
| **fast-alpr** — [ankandrew/fast-alpr](https://github.com/ankandrew/fast-alpr) | MIT | n/a (orchestrator) | **Components adopted, wrapper not.** It is a thin orchestrator over the two libraries above; we drive them directly because we need tiling, our own geometry stage and per-stage confidences that its single-call API does not expose. |
| **Nomeroff Net** — [ria-com/nomeroff-net](https://github.com/ria-com/nomeroff-net) | **GPL-3.0** | per-model | **Rejected on licence.** It has Kazakh-specific OCR models and its own keypoint stage, which is genuinely attractive here. GPL-3.0 is viral for a backend service that would link it, so adopting it would relicense this project. Revisit only if the deployment can be GPL. |
| **Ultralytics YOLOv8/11** | **AGPL-3.0** | AGPL-3.0 | **Rejected on licence.** AGPL reaches network-served software, which is exactly what this is. RT-DETRv2 gives comparable COCO detection under Apache-2.0. |
| **PaddleOCR (full `paddlepaddle`)** | Apache-2.0 | Apache-2.0 | **Not adopted directly.** Same weights as RapidOCR, but the `paddlepaddle` runtime is a much heavier dependency than ONNX Runtime for no accuracy gain. RapidOCR ships the PP-OCRv4 ONNX weights inside its wheel. |
| **OpenALPR** — [openalpr/openalpr](https://github.com/openalpr/openalpr) | AGPL-3.0 | n/a | **Rejected.** The open library's last stable release is from 2018 and it predates the detector architectures that make small plates findable at all; the maintained product is commercial. Licence is AGPL on top of that. |
| **DINOv2-S, ONNX** — [onnx-community/dinov2-small-ONNX](https://huggingface.co/onnx-community/dinov2-small-ONNX) | Apache-2.0 | Apache-2.0 (from `facebook/dinov2-small`) | **Adopted** as the encoder for reference-photo retrieval. Self-supervised, so a new machine type needs photos, not retraining |
| **OpenCV perspective transform** | Apache-2.0 | n/a | **Adopted** for the geometry stage |

Everything runs locally from open weights. No paid API, no key, no external endpoint.

### Why two OCR engines

They fail differently, and the difference is large enough to be worth the cost. Measured
on the same crops from these frames:

| Plate | `fast-plate-ocr` (CCT) | RapidOCR (PP-OCRv4) | True |
| --- | --- | --- | --- |
| frame 3 | `592LBA00` / `592LBA70` | `592LBA10` (0.98) | `592 LBA 10` |
| frame 12 | `153AX10` / `153JX10` | `153AJX10` (0.96) | `153 AJX 10` |
| frame 25 | `384C8A10` | `384CBA10` (0.99) | `384 CBA 10` |
| frame 11 | `302BL10` | `302BLL10` (0.91) | `302 BL? 10` |
| frame 18 | `041AHF10` (1.00) | `041AHF10` (1.00) | `041 AHF 10` |
| frame 19 | `927DPA10` (1.00) | `927DPA10` (0.93) | `927 DPA 10` |

PP-OCR is more accurate on this footage; the CCT model is ~100x faster and supplies a
per-character probability, which is what the ranking uses as a real OCR confidence. Both
are run, all readings are kept, and the best-scoring one wins. Set `VISION_USE_PPOCR=false`
to drop to the fast engine alone at a measurable accuracy cost.

RapidOCR also earns its place a second time: it reads the lettering on the machines
themselves, which is where the manufacturer comes from.

---

## 3. Stages

```
frame
  → 1. vehicles      RT-DETRv2 @ 960², COCO classes mapped to machinery
  → 1b. type         DINOv2 retrieval against a gallery of reference photos
  → 2. plates        YOLOv9 tiled at native resolution over each vehicle crop
  → 3. geometry      glyph-row edge fit → four corners → homography
  → 4. OCR           two engines × four renderings, ranked
  → 5. attributes    badge text → manufacturer
  → detections
```

### 1. Vehicles

RT-DETRv2 is NMS-free and accepts a dynamic input size. Running at 960² instead of 640²
costs about 0.2 s per frame and fixes the distant vehicles: frames 2, 11 and 13 are
classified `train` at 640² and correctly `truck` at 960².

COCO has no class for agricultural machinery. A Kirovets or an MTZ comes back as `truck`,
and elongated grain trailers often as `train`; the detector alone gets the type right on
6 of 24 frames. Rather than pretend otherwise, the mapping is explicit in `vehicle.py`, the
**raw COCO label is preserved** in `source_label`, and the type is settled by a separate
stage that matches the crop against reference photographs — 20 of 24 with refusals instead
of guesses. That stage has its own document: [REFERENCE_GALLERY.md](REFERENCE_GALLERY.md).

Overlapping and nested boxes are suppressed: RT-DETR emits several queries per object, and
a tractor with a trailer also produces a box for the pair.

### 2. Plates

The detector is run over overlapping 608 px tiles of each vehicle crop **at native
resolution**, then deduplicated. A full-frame sweep runs only as a fallback — when the
vehicle stage found nothing, or when the vehicle crops yielded no plate — because it costs
a second full tiling pass.

Two guards suppress the known false positives:

* the **top 4.5 % band** is excluded (`VISION_OVERLAY_BAND_TOP`), because these cameras burn
  a timestamp there and the detector fires on it every time. Set it to `0` for footage
  without an overlay;
* everything else — grille badges, headlights, reflective blanks — is rejected downstream
  by the plate grammar, since `KAMAZ` and `WED1355507` are not possible plates.

Each plate is attached to the vehicle whose box contains its centre; when boxes overlap,
the **smallest** containing box wins, as that is the more specific object. A plate inside no
box stays unassigned and is only reported if it reads cleanly, as an
`unidentified_vehicle` whose bbox is the plate's own — no vehicle box is invented.

### 3. Geometry — the part that matters

A slanted plate is not a rotated plate. Under perspective its top and bottom edges stop
being parallel, so rotating it leaves the glyphs sheared and unevenly scaled.

Two estimators, in order:

1. **Glyph-row fit (preferred).** Segment the dark characters on the bright field, keep the
   dominant height cluster, then fit one line through the glyph tops and another through
   the glyph bottoms. Those two lines converge exactly as the plate's own edges do. Close
   the quadrilateral past the outermost glyphs to take in the border, the `KZ` band and the
   region box.
2. **Bright-field fit (fallback).** Isolate the plate field as a connected component, sample
   its boundary column by column and row by row, and fit the four edges.

The glyph-row method was added after the bright-field method alone proved unreliable on
real crops: it was latching onto bumper and shadow merged with the plate, producing quads
with an aspect near **1.3-1.9** for plates that are physically **4.7:1**. With the glyph-row
fit the recovered aspect lands at **3.7-4.5**, and the rectified crops are genuinely
fronto-parallel. The measured aspect is then used as independent evidence: a layout whose
physical shape cannot produce that plate is rejected.

`QuadEstimate.skew` reports how far the quad departs from a parallelogram, in plate
heights. Near zero means rotation would have sufficed. On this dataset it runs **0.02-1.4**,
so the frames really do contain perspective, not just rotation.

The estimate is validated before use (position, area, plausible aspect). **If validation
fails, no warp is applied** — an unreliable homography is worse than none.

### 4. OCR and the acceptance rule

Four renderings are built per crop: `raw`, `rotated` (the rotation-only null hypothesis,
kept so the homography can be measured against it), `rectified`, and `rectified_clahe`.
The CCT model reads all four; PP-OCR, being ~100x slower, reads the untouched crop plus the
best geometric correction, plus the CLAHE variant only when the crop is measurably flat.

Candidates are ranked by

```
score = (0.65·min_char_prob + 0.35·mean_char_prob) × (0.45 + 0.55·structural_plausibility)
```

`min_char_prob` is the weakest character in the read, which is what catches a single
hallucinated glyph inside an otherwise confident string.

**A read is only asserted when it matches a known Kazakh layout.** This is the rule that
keeps the output honest:

* `160ALV10`, `592LBA10`, `041AHF10` match `kz_civil_2012` → accepted, written to
  `license_plate`, `plate_readable = true`.
* `597AQ10` is outside this implementation's supported modern private-car layout.
  Two-letter corporate layouts also exist: rejection is a limitation of this grammar,
  not proof of a missing letter. The supplied labels need independent verification
  before that format is enabled. It is reported in `plate_text_raw` with
  `plate_status = "no_matching_plate_format"` and **`plate_readable = false`**.
* `WED1355507` (the timestamp) and `TOO2026` (yard signage) are rejected the same way.

Nothing is ever repaired. The grammar classifies and inserts spaces; it never substitutes,
pads or corrects a character, and there is a test asserting that (`test_plate_format.py`).
Upscaling is plain bicubic interpolation — no generative enhancement, which could not be
used as evidence that a stroke was present in the original.

Region codes are validated against the issued list (01-20). A read ending in a code that
was never issued is not a Kazakh plate.

### 5. Manufacturer and model

There is no open dataset of KAMAZ/MAZ/MTZ badges to classify against, and guessing a brand
from silhouette is exactly the confident invention the brief rules out. So this stage reads
the lettering manufacturers put on their own machines, and reports a brand **only when the
text is there**. Every answer carries its evidence: the string read, its confidence, and
the brand it matched.

One trap is handled explicitly. `MAZ` is a substring of `KAMAZ`; on frame 3 PP-OCR reads
only `MA3` off a KAMAZ grille whose first letters fall outside the crop. OCR gives no word
boundary, so a bare `MAZ` **cannot** be asserted — the rule is marked `ambiguous_with`, the
result is `manufacturer: null`, and the raw token is kept in the evidence with
`rejected: "ambiguous_brand_token"`. This costs recall on genuine MAZ trucks. Reporting a
KAMAZ as a MAZ would be worse.

A **model** is reported only when its designation is itself legible next to a known brand.
On these 25 frames that never happens, so `model` is `null` throughout. That is the correct
answer, not a gap.

---

## 4. Results on the 25 supplied frames

Reproduce with:

```bash
python eval/evaluate.py --photos ../photos --out eval/results
```

Labels are in `eval/ground_truth.json`: each frame was inspected at full resolution and
each plate region cropped and magnified before labelling. They are one person's reading of
the pixels, not an official record, and they were written before the pipeline's answers
were compared against them.

Frames are split three ways, and the split is the honest part of the measurement:

* **readable (7)** — every character legible to a human. This is the denominator for
  exact-match accuracy.
* **partial (8)** — a plate is clearly present, some characters legible, at least one not.
  Not scored for accuracy; used to check that the system correctly declines.
* **none legible (10)** — no plate could be read from the frame by the labeller. Not a
  claim that the machine has no plate.

Current results are in `eval/results/summary.json` and `eval/results/per_frame.json`, with
the per-frame table reproduced in `RESULTS.md`.

**Read this denominator carefully.** Seven fully legible plates from one camera at one site
cannot establish general accuracy. They establish behaviour on this camera and they catch
regressions. Nothing here was trained or fine-tuned on these frames; thresholds were chosen
while looking at them, which is itself a form of fitting, so treat the numbers as a sanity
check rather than a held-out result.

### Perspective A/B

```bash
python eval/perspective_report.py --photos ../photos --out eval/perspective
```

writes `crops/<frame>_<n>_before.png` and `_after.png` for every plate, plus
`comparison.csv` with every engine × rendering reading and its confidence, and a tally of
which rendering produced the winning read.

---

## 5. Known limitations

* **Small sample.** 25 frames, one camera, one site, one season, mostly one region code.
* **Thresholds were tuned while looking at these frames.** They are not validated on held-out
  data.
* **Tractors and self-propelled machinery.** COCO cannot name them, so the type comes from
  the reference gallery instead — see [REFERENCE_GALLERY.md](REFERENCE_GALLERY.md), and note
  that the shipped gallery was seeded from these same frames. Frames 5, 6, 7, 15, 16, 23 and
  24 contain tractors and none yields a legible plate, so the *plate* path is effectively
  untested on machinery plates.
* **Two-row machinery plates are provisional.** The row order in `kz_machinery_square` is
  inferred from frame 4, not from a published specification. Those layouts are flagged
  `provisional` and held to a higher confidence bar. Verify against the real standard before
  relying on the grouping.
* **Cyrillic badge OCR is now available.** `VISION_CYRILLIC_ATTRIBUTES=true` enables
  pinned PP-OCRv3 weights alongside the existing reader. Small/stylised `КИРОВЕЦ`
  lettering on these frames remains unreadable. `СТАЙЕР` is decorative text and is
  retained as evidence, never mapped to manufacturer STEYR. See `ASTRA6_REVIEW.md`.
* **The overlay guard is camera-specific.** `VISION_OVERLAY_BAND_TOP` assumes the timestamp
  is along the top. Re-check it for a different camera.
* **No re-identification.** Frames 2 and 13 are the same vehicle, as are 8 and 20. The
  service treats every image independently.
* **`confidence` is a model score, not a calibrated probability.** None of these numbers have
  been calibrated against outcome frequency.
* **Background vehicles are reported.** The brief asks for all visible machinery, so parked
  cars at the edge of frame 25 are included. Raise `VISION_VEHICLE_THRESHOLD` to suppress
  them.

---

## 6. Hardware and cost

Measured on a laptop CPU (no GPU used), `VISION_MAX_PARALLEL_IMAGES=1`:

| | |
| --- | --- |
| Per frame, 2688x1520, both OCR engines | 5-19 s, mean 8-9 s across runs |
| Per frame with `VISION_USE_PPOCR=false` | ~2-4 s, at a measurable accuracy cost |
| Model load (once per process) | ~5 s |
| Resident memory, steady state | ~0.6-1.0 GB |
| Weights on disk | ~115 MB total |

**Minimum:** 2 CPU cores, 2 GB RAM for the worker, ~1 GB disk for weights and cache. No GPU
is required and none is assumed.

**GPU is optional.** `VISION_DEVICE=cuda` requires `onnxruntime-gpu` with a matching CUDA
runtime instead of the pinned `onnxruntime`; the code raises a clear error if the provider
is unavailable rather than silently falling back. This was **not** benchmarked on GPU —
treat any speedup as unverified.

Scale by adding worker processes, not by raising `VISION_MAX_PARALLEL_IMAGES` alone: the
ONNX sessions are shared per process, but per-image tensors are not, and those are what
drive peak memory.

---

## 7. Attribution

| Project | Licence | Used for |
| --- | --- | --- |
| [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) / [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd) | Apache-2.0 | Vehicle detection weights |
| [onnx-community/rtdetr_v2_r18vd-ONNX](https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX) | Apache-2.0 | ONNX export |
| [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) | MIT | Plate detection |
| [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | MIT | Plate OCR (CCT) |
| [ankandrew/fast-alpr](https://github.com/ankandrew/fast-alpr) | MIT | Reference design for the detector + OCR pairing |
| [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | Apache-2.0 | ONNX PP-OCR runtime |
| [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | PP-OCRv4 weights |
| [OpenCV](https://opencv.org/) | Apache-2.0 | Geometry and image operations |
| [ONNX Runtime](https://onnxruntime.ai/) | MIT | Inference |

### Weight provenance

Every ONNX artefact the pipeline loads is pinned in
`app/services/vision/local/model_registry.py` with its exact download URL, SHA-256 and
licence, fetched by `scripts/fetch_models.py`, and **loaded by explicit path**:

| File | Size | Licence | Source |
| --- | --- | --- | --- |
| `rtdetr_v2_r18vd.onnx` | 81 MB | Apache-2.0 | onnx-community |
| `rtdetr_v2_r18vd.config.json` | 6 KB | Apache-2.0 | onnx-community |
| `yolo-v9-s-608-license-plates-end2end.onnx` | 29 MB | MIT | open-image-models releases |
| `cct_s_v2_global.onnx` | 5 MB | MIT | cnn-ocr-lp releases |
| `cct_s_v2_global_plate_config.yaml` | 2 KB | MIT | cnn-ocr-lp releases |
| `dinov2_small.onnx` | 89 MB | Apache-2.0 | onnx-community |

PP-OCRv4's weights (16 MB, Apache-2.0) ship inside the `rapidocr-onnxruntime` wheel and
need no fetching.

Loading by explicit path is deliberate. Left to themselves, `open-image-models` and
`fast-plate-ocr` download into `~/.cache` on first use: unwritable in a container with a
read-only root and no home directory, invisible to any integrity check, and impossible to
pin to a version. A checksum mismatch is a hard failure, not a warning — substituted or
truncated weights must not silently degrade recognition.

To build a fully offline image, uncomment the `RUN python scripts/fetch_models.py` line in
the `Dockerfile`. To verify an existing model directory without re-downloading:

```bash
python scripts/fetch_models.py --check-only
```
