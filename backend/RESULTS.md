# Results on the 25 supplied frames

> Historical Opus baseline below. The September 17 Astra follow-up and freshly measured
> before/after results are in [ASTRA6_REVIEW.md](ASTRA6_REVIEW.md).

Produced by `eval/evaluate.py` against `eval/ground_truth.json`, on CPU, with
`VISION_PROVIDER=local` and the defaults in `.env.example`.

```bash
python eval/evaluate.py --photos ../photos --out eval/results
```

Raw output: `eval/results/per_frame.json`, `eval/results/summary.json`.

---

## Read the denominator first

The 25 frames come from **one fisheye CCTV camera at one weighbridge**. They were labelled
by hand: each frame was opened at full resolution and each plate region cropped and
magnified before a label was written. The labels are one person's reading of the pixels,
not an official record.

Frames are split three ways, and the split is the honest part of this measurement:

| Category | Count | Meaning |
| --- | --- | --- |
| **readable** | 7 | Every character legible to a human. **This is the accuracy denominator.** |
| **partial** | 8 | A plate is clearly present; some characters legible, at least one not. Used to check that the system declines rather than guesses. |
| **none legible** | 10 | No plate could be read from the frame by the labeller. Not a claim that the machine has no plate. |

**Seven fully legible plates cannot establish general accuracy.** They establish behaviour
on this camera and they catch regressions. Nothing was trained or fine-tuned on these
frames, but the thresholds were chosen while looking at them, which is itself a mild form
of fitting. Treat the figures below as a sanity check, not a held-out result.

---

## Summary

| Metric | Result |
| --- | --- |
| Plate localised, on the 15 frames where a plate is visible | **15 / 15** |
| Exact plate match, on the 7 fully readable frames | **7 / 7** |
| Character errors on those 7 | **0** |
| Correctly declined, on the 8 partial frames | **8 / 8** |
| False accepts, on the 10 illegible frames | **0 / 10** |
| Manufacturer correct, where a brand is present | 3 / 10 |
| Manufacturer **wrong** | **0** |
| Machine type — detector alone (COCO) | 6 / 24 |
| Machine type — reference gallery | **20 / 24**, 1 wrong, 3 abstained |
| Accepted via the perspective-corrected crop | 2 of 7 |
| Mean seconds per frame (CPU, both OCR engines) | 9.24 |

The two numbers that matter most are the last two in the abstention rows. A plate reader
that never says "I don't know" is worse than useless for a weighbridge, because a
plausible-looking wrong plate propagates into the weight record silently.

---

## Per frame

| # | Category | Label | Reported | Raw OCR | Status | Engine | Rendering | Mfr | Veh | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | readable | 160 ALV 10 | **160 ALV 10** | 160ALV10 | accepted | ppocr | raw | KAMAZ | 1 | 13.9 |
| 2 | partial | 691 A?? 10 | — | E8TAS10 | no_matching_plate_format | — | — | — | 1 | 6.1 |
| 3 | readable | 592 LBA 10 | **592 LBA 10** | 592LBA10 | accepted | ppocr | **rectified** | — | 1 | 10.6 |
| 4 | partial | AAH ? 10 98 | — | AAH1098 | no_matching_plate_format | — | — | — | 1 | 7.0 |
| 5 | none legible | — | — | TOO2026 | below_confidence_threshold | — | — | — | 1 | 8.1 |
| 6 | none legible | — | — | 0 | no_matching_plate_format | — | — | — | 2 | 5.6 |
| 7 | none legible | — | — | — | no_plate_box | — | — | — | 2 | 7.7 |
| 8 | partial | 597 A?? 10 | — | 597AQ10 | no_matching_plate_format | — | — | — | 1 | 7.0 |
| 9 | partial | AFE ? P523 | — | AFEP523 | no_matching_plate_format | — | — | — | 2 | 6.0 |
| 10 | readable | 124 AIX 10 | **124 AIX 10** | 124AIX10 | accepted | fast_plate_ocr | rotated | — | 1 | 6.3 |
| 11 | partial | 302 BL? 10 | — | 302BL10 | no_matching_plate_format | — | — | — | 1 | 5.5 |
| 12 | readable | 153 AJX 10 | **153 AJX 10** | 153AJX10 | accepted | ppocr | **rectified** | MAN | 2 | 9.5 |
| 13 | partial | 691 A?? 10 | — | 681AG10 | no_matching_plate_format | — | — | — | 1 | 6.9 |
| 14 | none legible | — | — | 522505Z | no_matching_plate_format | — | — | — | 1 | 6.7 |
| 15 | none legible | — | — | 9 | no_matching_plate_format | — | — | — | 1 | 5.8 |
| 16 | none legible | — | — | — | no_plate_box | — | — | — | 3 | 10.7 |
| 17 | partial | ?45 EUA 10 | — | 45EUA10 | no_matching_plate_format | — | — | — | 1 | 9.5 |
| 18 | readable | 041 AHF 10 | **041 AHF 10** | 041AHF10 | accepted | fast_plate_ocr | raw | — | 1 | 6.9 |
| 19 | readable | 927 DPA 10 | **927 DPA 10** | 927DPA10 | accepted | fast_plate_ocr | raw | — | 1 | 8.0 |
| 20 | partial | 597 A?? 10 | — | 597AQ10 | no_matching_plate_format | — | — | — | 2 | 8.3 |
| 21 | none legible | — | — | — | no_plate_box | — | — | — | 1 | 12.4 |
| 22 | none legible | — | — | WED1355507 | no_matching_plate_format | — | — | — | 2 | 18.9 |
| 23 | none legible | — | — | 435 | no_matching_plate_format | — | — | — | 1 | 6.9 |
| 24 | none legible | — | — | — | no_plate_box | — | — | — | 2 | 17.9 |
| 25 | readable | 384 CBA 10 | **384 CBA 10** | 384CBA10 | accepted | ppocr | raw | KAMAZ | 5 | 19.0 |

---

## What the failures actually are

Nothing here is a silent error. Every non-answer carries a reason.

**Frames 8, 11, 17, 20 — the interesting ones.** The OCR reads are almost certainly close
to correct (`597AQ10`, `302BL10`, `45EUA10`), and a less careful system would emit them.
They are rejected because they are **structurally impossible** for a Kazakh plate: two
letters where the 2012 layout needs three, two digits where it needs three. We cannot tell
from the pixels whether a character is worn off, occluded, or simply missed by the model —
so the read goes into `plate_text_raw` with `status = no_matching_plate_format` and
`license_plate` stays empty. An operator can confirm it through
`PATCH /api/v1/detections/{id}`.

This is a deliberate trade. It costs recall. It buys the guarantee that anything in
`license_plate` is structurally a real plate.

**Frames 2, 13 — genuinely ambiguous.** The same distant vehicle in both. The engines
return `E8TAS10` and `681AG10`, which disagree with each other and with the label
(`691 A?? 10`). Correctly rejected.

**Frames 4, 9 — two-row plates.** `AAH1098` and `AFEP523` are read correctly as token
sequences, but the row order of Kazakh square machinery plates is not confirmed against a
published specification, so those layouts are marked `provisional` and held to a higher
confidence bar. Neither cleared it. Verify the real standard before relying on this path.

**Frames 5, 22 — false positives, caught.** The plate detector fires on the yard banner
(`TOO` / `OLZHAAGRO 2026`) and on the burnt-in CCTV timestamp (`WED1355507`). Both are
rejected, by the confidence threshold and the grammar respectively. This is why the
grammar exists.

**Frames 7, 16, 21, 24 — no plate box at all.** Tractors and distant machinery whose plates
are not visible from the camera angle. The labeller could not read them either.

**Manufacturer 3/10.** Reported only where a Latin badge is legible: KAMAZ on frames 1 and
25, MAN on frame 12. Frame 3 is also a KAMAZ, but a tight crop reads only `MA3` — the tail
of the badge — and `MAZ` is a substring of `KAMAZ`, so it is **deliberately not asserted**
(`rejected: ambiguous_brand_token` in the evidence). Frames 5, 15 and 23 are Kirovets
tractors labelled in Cyrillic (`КИРОВЕЦ`), which the bundled Latin OCR model does not read.

**Zero wrong manufacturers** is the point of that row. `model` is `null` on every frame,
because no model designation is legible on any of them. That is the correct answer.

---

## Perspective correction

```bash
python eval/perspective_report.py --photos ../photos --out eval/perspective
```

Writes `eval/perspective/crops/<frame>_<n>_before.jpg` and `_after.jpg` for every detected
plate, plus `comparison.csv` with every engine × rendering reading and its confidence.

Across all **25 plate candidates** the detector found, the rendering that produced the
best-scoring read was:

| Rendering | Plates won | Share |
| --- | --- | --- |
| `raw` — untouched crop | 16 | 64 % |
| `rectified` — perspective corrected | 5 | 20 % |
| `rectified_clahe` — corrected + contrast | 2 | 8 % |
| `rotated` — rotation only | 2 | 8 % |

**Perspective correction won 7 of 25 (28 %); rotation alone won 2 (8 %).** By engine:
PP-OCR 15, fast-plate-ocr 10.

Two of the seven *accepted* plates (frames 3 and 12) came from the rectified crop, one
(frame 10) from the rotation-only variant, and four from the untouched crop.

The honest reading of that table: the raw crop is right most of the time, and a system
that always rectified would be worse than one that never did on nearly two thirds of
plates. That is exactly why every rendering is tried and ranked by measured confidence
instead of one being applied unconditionally — and why the contrast pass is a candidate
rather than a default.

Recovered plate aspect after rectification runs **3.6–7.2** against a physical
520 × 110 mm plate (4.7:1), and `perspective_skew` — the departure from a parallelogram,
in plate heights — runs **0.0–1.13**. The frames contain real perspective, not just
rotation. The glyph-row estimator supplied the quad for 92 of 158 readings; the rest fell
back to the bright-field fit, whose aspect is not trusted for gating.

Worth recording: an earlier bright-field quad estimator produced aspects of **1.3–1.9** for
those same plates, i.e. it was latching onto bumper and shadow rather than the plate. The
glyph-row fit replaced it. The measured aspect is only used as evidence when it comes from
that fit (`QuadEstimate.trusted_aspect`); the fallback estimator is good enough to warp by
but not good enough to reject a read with.

---

## Machine type

COCO has no class for agricultural machinery, so the detector alone calls almost
everything `truck` and is right on **6 of 24** frames. A reference gallery — reference
photos embedded with DINOv2 and matched by cosine similarity — raises that to **20 of 24**
with 1 wrong answer and 3 refusals.

```bash
python eval/evaluate_reference.py --photos ../photos --out eval/reference_results
```

| Gallery | Correct | Wrong | Abstained |
| --- | --- | --- | --- |
| Detector alone | 6 / 24 | — | — |
| Site frames (24 references) | **20 / 24** | 1 | 3 |
| Site frames + web (72 references) | 20 / 24 | 1 | 3 |
| Web photos only (48 references) | 6 / 24 | 4 | 14 |

Scoring is leave-one-frame-out: a frame is never compared against a reference cut from
itself. The last row is worth reading twice — 48 openly licensed catalogue photos classify
this CCTV footage no better than COCO, and add nothing on top of site crops.

All four failures are rear views, where a light truck's cargo body and a trailer are
genuinely alike; the system refused on three of them rather than guessing. Full breakdown,
including how to add your own references, is in
[docs/REFERENCE_GALLERY.md](docs/REFERENCE_GALLERY.md).

The same caveat as everywhere else applies, and harder: the gallery was seeded from these
very frames. Leave-one-out stops that being circular, but it says nothing about a machine
the gallery has never seen.

---

## Reproducing

```bash
python scripts/fetch_models.py
python scripts/build_reference_gallery.py --seed-from reference_seed
python eval/evaluate.py --photos ../photos --out eval/results
python eval/evaluate_reference.py --photos ../photos --out eval/reference_results
python eval/perspective_report.py --photos ../photos --out eval/perspective
```

Results vary slightly with ONNX Runtime version and thread count; the accepted plate
strings did not vary across runs on this machine.
