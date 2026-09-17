# Machine type from reference photos

> Update, 2026-09-17: packaged references now include 30% horizontal and 10% vertical
> padding per side. The same policy is saved in the index manifest and used for queries.
> Fresh leave-one-machine-out result: **21/24 correct, 0 wrong, 3 abstained**.
> Frame 14 is now correct; frame 13 abstains instead of incorrectly saying trailer.
> The historical tight-crop experiment below remains for comparison.
> See [the follow-up report](../ASTRA6_REVIEW.md) for migration instructions and limitations.

How the service decides that something is a tractor when the detector can only say
"truck", what that is measured at, and what it still cannot do.

---

## 1. Why this stage exists

The vehicle detector is RT-DETRv2 trained on COCO. COCO has 80 classes and **not one of
them is agricultural machinery**. On this dataset that produces:

| What is in the frame | What COCO calls it |
| --- | --- |
| Kirovets K-700 | `truck` |
| MTZ / Belarus tractor | `truck`, sometimes `train` |
| Grain trailer | `truck` or `train` |
| GAZ light lorry | `truck` |
| KAMAZ | `truck` |

Measured: the detector alone gets the machine type right on **6 of 24** frames, because it
answers `truck` to almost everything.

Retraining a detector would need a labelled dataset of Kazakh agricultural machinery that
nobody has. What a depot *does* have is photographs of its own machines. So this stage
keeps the detector for **where** the machine is, and answers **what it is** by comparing
the crop to reference photos.

---

## 2. How it works

```
reference photos ──embed once──► index.npz  (DINOv2-S, 768-d, L2-normalised)
                                     │
vehicle crop ─────embed───────► cosine similarity ──► weighted top-k vote
                                     │
                          similarity floor + margin check
                                     │
                        accepted type   or   no answer
```

**Encoder: DINOv2-S** (ViT-S/14, Apache-2.0, 88 MB), pinned by checksum in the model
registry. Self-supervised, so its features were never fitted to a fixed label set: adding
a machine type means dropping photos into a folder and rebuilding the index, not
retraining anything. The embedding is the CLS token concatenated with the mean of the
patch tokens, which is the pairing DINOv2's own retrieval evaluations use.

**Voting** is over the top 5 neighbours, weighted by similarity cubed, so one very close
reference outweighs several mediocre ones but a single outlier cannot carry a decision.

**Two refusal conditions**, and they are the point of the design:

* `below_similarity_floor` — nothing in the gallery is close enough
  (`VISION_REFERENCE_MIN_SIMILARITY`, default 0.62);
* `classes_too_close` — the best two classes are within
  `VISION_REFERENCE_MIN_MARGIN` (default 0.04) of each other.

On a refusal the detector's coarse label stands and `vehicle_type_source` says `detector`,
so a consumer can always tell a matched type from a fallback.

**Precedence** when several stages have an opinion:

1. reference gallery — looks at the whole machine, most specific;
2. badge text — `KAMAZ` on a grille implies a lorry, but cannot tell a lorry from its
   trailer;
3. the detector's COCO class — always available, least specific, wrong for every tractor.

Every answer carries its evidence: which references matched, their similarity, and which
source directory they came from. A wrong answer can be traced to the photo that caused it,
and that photo can be deleted.

---

## 3. Measured on the 25 supplied frames

```bash
python eval/evaluate_reference.py --photos ../photos --out eval/reference_results
```

Labels are in `eval/reference_labels.json` — the main machine in each frame, typed by hand
from its crop. Frame 9 is deliberately unlabelled: it shows a rusty tipper body from
behind and could honestly be a truck or a trailer.

Scoring uses **leave-one-frame-out**: when frame N is scored, the reference cut from frame
N is removed from the index. A stricter **leave-one-machine-out** additionally drops
frames showing the same physical machine (frames 8 and 20 share a plate; 2/13 and 5/15 are
visual judgements). Both figures came out identical here.

| Gallery | Correct | Wrong | Abstained | Accuracy when it answers |
| --- | --- | --- | --- | --- |
| Detector alone (COCO) | 6 / 24 | — | — | — |
| **Site frames (24 refs)** | **20 / 24** | 1 | 3 | **0.95** |
| Site frames + web (72 refs) | 20 / 24 | 1 | 3 | 0.95 |
| **Web photos only (48 refs)** | **6 / 24** | 4 | 14 | 0.60 |

### What the web photos are worth

The last row is the finding worth keeping. **48 openly licensed catalogue photos from
Wikimedia Commons classify this footage no better than COCO does**, and adding them to a
gallery of site crops changes nothing. Catalogue shots are taken side-on in good light; a
fisheye CCTV camera looks down at an angle through dust and glare, and DINOv2 embeddings
are sensitive to that difference.

They are kept anyway, in their own `web/` source directory, for one reason: they cover
`combine_harvester` and `wheel_loader`, which never appear in the 25 frames. When such a
machine first drives onto the weighbridge the gallery will at least have something to
compare against. Replace them with site photos as soon as there are any.

`--exclude-source web` and `--exclude-source site_frames` reproduce the comparison.

### Where it fails

All four failures are **rear views**:

| Frame | Label | Result | Why |
| --- | --- | --- | --- |
| 4 | trailer | abstained | nearest neighbours split trailer / truck_light |
| 8 | truck_light | abstained | same |
| 13 | truck_light | **wrong** (trailer) | closest reference is a trailer rear at 0.77 |
| 14 | trailer | abstained | same |

From behind, a GAZ flatbed loaded with grain and a grain trailer are genuinely similar
objects — the distinguishing feature is a cab that the crop does not contain. The system
declined on three of the four rather than guessing, which is the designed behaviour. The
fix is more rear-view references, which is a depot archive, not a code change.

### Read the denominator

24 frames, one camera, one site, and the gallery was seeded from those same frames.
Leave-one-out keeps that from being circular, but it does not tell you the accuracy on a
machine the gallery has never seen. **It says the retrieval mechanism works on this
footage.** Only an archive of other machines answers the general question.

---

## 4. Adding your own references

```
data/reference/
  tractor/
    site_archive/kirovets_front_2026-09-16.jpg
    site_archive/kirovets_rear_2026-09-16.jpg
    web/...
  truck/
  trailer/
  combine_harvester/
```

The directory under a class is the *source tag*; it is carried into the index so the
evaluation can include or exclude whole sources.

```bash
# from your own footage, given a label file
python scripts/seed_reference_from_frames.py --photos /path/to/frames \
    --labels my_labels.json --source site_archive

# openly licensed photos, with attribution written to SOURCES.md
python scripts/fetch_web_references.py --dir data/reference

# rebuild after any change
python scripts/build_reference_gallery.py
python scripts/build_reference_gallery.py --report
```

What actually helps, in order:

1. **Photos from this camera.** Nothing else comes close — 20/24 against 6/24.
2. **Both views of each machine.** Every failure above is a rear view. Front and rear
   look nothing alike to an image encoder; they need separate references.
3. **Five to fifteen per class.** The build script flags any class under four.
4. Photos from other cameras at the same site, then anything else.

A gallery with no images is a valid state: the stage disables itself and the detector's
label stands.

---

## 5. What this cannot do

**Brands.** Measured on these frames, a MAZ cab sits 0.91 from another MAZ *and* 0.91 from
a KAMAZ. Whole-machine silhouettes do not separate manufacturers at this resolution and
angle. Brand stays with the badge-OCR stage, which reads the lettering and abstains when
the text is ambiguous. Do not add brand-level directories to the gallery expecting them to
work; they will produce confident wrong answers.

**Models.** Further still out of reach for the same reason.

**Anything not in the gallery.** An unfamiliar machine produces a refusal, not a nearest
guess — by design. Watch `vehicle_type_source` in the exports: a rising share of
`detector` means the gallery is going stale and needs new photos.

---

## 6. Cost

| | |
| --- | --- |
| Encoder load, once per process | ~1 s |
| Per vehicle crop | ~40-60 ms on CPU |
| Index for 72 references | 230 KB |
| Added to per-frame time | well under 0.2 s |

`VISION_REFERENCE_GALLERY=false` disables the stage entirely.

## 7. Attribution

DINOv2 — [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), Apache-2.0,
ONNX export by [onnx-community](https://huggingface.co/onnx-community/dinov2-small-ONNX).

Web reference photos come from Wikimedia Commons under CC0, public domain, CC BY or
CC BY-SA. Per-file author, licence and source page are in `reference_seed/SOURCES.md`,
which must travel with the images.
