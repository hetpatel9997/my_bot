# Weed detector v2: a big model teaches the small one (design)

Date: 2026-11-25. Status: draft for the owner's approval. Size: project. Builds on
`2026-11-24-weed-detector-design.md` (v1); everything there still holds unless this document changes it.

## Why
v1 (YOLO11n, trained on Colab 2026-11-24/25, best round 82 of 97; `~/weedbot_data/models/weed_detector_v1/`)
scored well on its public test photos: weed precision 0.73 / recall 0.84 / mAP50 0.86, keep precision 0.74 /
recall 0.53 / mAP50 0.63. But it **fails on the owner's backyard**: on IMG_0229/0230 (top-down, small
red-stemmed spotted-spurge sprigs between white marble stones) it only draws one box over the whole picture
(keep 0.21-0.76 or weed 0.12-0.38), also on 512 / 768 px close-up tiles (every box covers its whole tile).

Causes (box statistics from the VM sample of the same sources):
1. No weeds among stones in the data: CottonWeedDet12 is close-ups of single weeds on brown farm soil
   (median box 24 % of the image), grass-weeds is dock in grassland.
2. 27 % of the Open Images `keep` boxes cover more than 40 % of the picture: the model learned "a busy
   textured picture = keep".
3. Almost every training picture has a box (7 of ~2,500 empty): the model never learned "stones = nothing".

Test that decides this design: **OWLv2** (`google/owlv2-base-patch16-ensemble`, an open-vocabulary detector
that finds things from a text description, no training) run on quarter tiles of the two photos with the
query "a plant with reddish stems and small leaves" / "a small weed plant" found most spurge clumps and the
larger green weed, and boxed no bare stones (`backyard_test/IMG_0230_owlv2.jpg`). It is far too slow for the
robot (16-44 s per quarter photo on the VM CPU) but good enough to **label training photos automatically**.

## What the owner decided (2026-11-25)
- **A now:** the big model labels thousands of public photos of the kinds of weeds and plants in the yard,
  plus "nothing here" photos of stones and lawn; the small model is trained again on them.
- **B later:** the robot's own photos, AND real-time detection on the robot while it drives, "so everything
  merges together": one pipeline where the live detector, the robot's recorded frames and the big-model
  labelling feed each other (see Phase B below; not built in this plan).
- Unchanged from v1: no photos or labelling by the owner; free Google Colab; look-only before any spraying.

## Goals
- G1: a v2 model that passes the backyard check below and does not lose the public test scores.
- G2: the labelling step ("teacher") is a reusable module that labels any folder of images, so Phase B can
  label the robot's own frames with the same code.

## Non-goals
- Anything on the robot (detector node, real-time, CoreML, pixel -> ground): Phase B.
- Weed species classes (still one `weed` class), segmentation outlines.
- Synthetic pictures (cut-out weeds pasted onto stone backgrounds): only if v2 fails the backyard check.

## Anti-goals (failure even if it "works")
- Training on, or tuning thresholds on, the owner's two photos: they are the only real test. They and
  anything derived from them stay out of git (public repo).
- Lawn grass labelled `weed` (the spray is lawn-safe broadleaf; grass is background).
- A model that boxes whole scenes again (the v1 failure): checked explicitly.
- The owner having to review or label images.

## Approaches considered
1. **Teacher-labelled public photos** (chosen, owner): OWLv2 draws the boxes on public photos whose
   species is already known (so the box class comes from the species list, the teacher only says WHERE).
2. Robot-collected photos labelled by the teacher: best match to the yard, needs the robot driving -> B.
3. The big model itself on the robot: works on these photos but ~1 min or more per stop on a Pi -> no.
Inside 1, synthetic composites were considered and deferred (more moving parts; only if the check fails).

## Design

### Data (one dataset, three classes of source)
| Source | ~Images | Boxes from | Class | Licence |
|---|---|---|---|---|
| `cottonweed` (v1 part on Drive, reused as is) | 5,648 | humans | weed | CC BY-NC 4.0 |
| `grassweeds` (v1 part, reused) | 2,486 | humans | weed | CC BY 4.0 |
| `openimages` (v1 part, reused, **filtered at merge**: an image with any box > 40 % of its area is dropped, not just the box) | ~2,300 | humans | keep | CC BY 2.0 / CC BY 4.0 |
| NEW `inat_weed`: iNaturalist research-grade observations of yard weeds, first photo, `large` (1024 px) | ~8,000 | teacher | weed | per photo: CC0 / CC BY / CC BY-NC |
| NEW `inat_keep`: iNaturalist observations of common garden flowers and shrubs (cultivated ones allowed: gardens are cultivated) | ~3,000 | teacher | keep | same |
| NEW `background`: Open Images validation+test photos with a human tag Gravel, Pebble, Rock, Soil, Mulch, Lawn, Grass, Cobblestone, Walkway or Concrete and NO plant tag (~2,000 such photos exist, counted 2026-11-25) | ~1,500-2,000 | none (empty label file) | - | CC BY 2.0 |

- Weed species (broadleaf only, common in gravel, paving and lawns; Claude's call, the builder may add):
  spotted spurge (*Euphorbia maculata*, taxon 58893, ~29,000 open-licence research observations) and
  prostrate spurge, common purslane (58991), common dandelion (47602), white clover (55745), broadleaf and
  narrowleaf plantain, creeping woodsorrel, prostrate knotweed, common chickweed, henbit, black medic,
  carpetweed. Spurge gets the largest share (it is what the rock beds have). No grasses or sedges.
- Keep species (generic garden plants; the owner's exact shrubs are unknown): roses, lavender, hydrangea,
  boxwood, marigold, petunia, geranium / pelargonium, hibiscus, salvia, daylily, hosta, juniper,
  azalea / rhododendron. The Open Images weed-tag exclusion of v1 stays.
- Photos come from the iNaturalist API (observations by taxon, `photo_license` cc0,cc-by,cc-by-nc, one
  request per second at most, plain `User-Agent` naming the project, no personal data) and the open-data
  bucket `inaturalist-open-data.s3.amazonaws.com/photos/<id>/large.jpg` (checked: HTTP 200, ~60 KB).
  Each image's licence, observer and observation URL go into the manifest (v1 manifest format + those).
- Splits: per observation id with v1's `stable_split` (80/10/10); a teacher-labelled test split measures
  agreement with the teacher, so it is reported separately from the human-labelled test sources.

### Teacher (the big model that draws the boxes)
- OWLv2 base patch16 ensemble on Colab's T4 GPU (estimate: well under 1 s per image -> ~1-2 h for ~13,000).
- Per image: text queries = the class's plant queries + distractor queries (e.g. "grass", "a stone",
  "soil", "a human hand", "a shoe"). Each box takes its best-scoring query; it becomes a label only if that
  query is a plant query and its score is at least the threshold; boxes overlapping a better box are merged
  (NMS); tiny boxes (< 0.3 % of the image) are dropped.
- Image rules: a weed / keep photo where the teacher finds NO plant box is dropped (the plant is there,
  the teacher missed it: it must not become a "nothing here" example); a photo with a box over 60 % of the
  image is dropped (too close-up, and the v1 failure mode); a background photo where the teacher finds any
  plant box above a (lower) threshold is dropped instead of being used as "nothing here".
- Thresholds and prompt wording are set from a **sample check on the VM before any Colab time**: the
  teacher labels ~30 photos of each new source on the CPU, Claude looks at the boxes drawn on them (contact
  sheets) and fixes prompts / thresholds; the values are written into the code with the reason.
- Teacher labels are cached per image (resumable: a stopped Colab session continues).
- The teacher is its own module with one entry point "label these images with these queries", reused in B.

### Training
- Fine-tune from v1 `best.pt` (it already knows plants; the corrected data removes the whole-picture habit),
  ~50 epochs, imgsz 640, patience 15, v1's augmentation settings, run `runs/v2` on Drive (v1 untouched).
- Expected ~18,000-19,000 training images -> ~5 min per epoch on a T4, ~4-5 h: likely two Colab days
  (free GPU limit seen: ~4.5 h, then "usage limit" for hours). Resumable as in v1; the step-2b
  finish-without-GPU notebook works for v2 too.
- Owner steps stay the same shape: one dataset notebook (needs a **GPU** this time, for the teacher),
  one training notebook, optional finish notebook; result `MyDrive/weedbot/weed_detector_v2.zip`.
  Drive space needed beyond v1's: ~3 GB.

### Checks before v2 goes on the robot (Claude, on the VM)
1. **Backyard check.** Reference: the teacher's boxes on quarter tiles of IMG_0229/0230, each kept or
   removed by Claude by eye, stored in `~/weedbot_data/backyard_photos/reference/` (never in git). v2 runs on
   640-px views of the photos (quarter tiles and the whole photo). PASS = on the quarter tiles at least 70 %
   of the reference weed clumps are hit (a v2 `weed` box, confidence >= 0.25, whose centre lies inside the
   reference box or IoU >= 0.3), at most 2 boxes per photo on bare stones, and no box covering more than
   50 % of a view.
2. **Public test (no big loss):** weed recall on the human-labelled test splits not more than 0.05 below v1
   per source (cottonweed 0.81, grassweeds 0.71); keep precision >= 0.70; keep recall reported (v1 0.53).
   Teacher-labelled test splits reported separately.
3. **Pi version:** the NCNN export and `best.pt`, both fed the same 640 x 640 picture, give the same boxes
   (IoU >= 0.9, confidence within 0.05). (v1 showed different scores, but its two runs were fed different
   shapes, so that comparison was not fair.)
If check 1 fails: v2 is not used; next step decided with the owner (synthetic composites, or B first).
Passing does not mean spraying: look-only first, as in v1.

### Phase B (later plans; owner's direction 2026-11-25)
- Real-time: the detector runs live on the robot's camera stream (`/phone/image/compressed`, ~10 Hz)
  while it drives, either on the Pi (NCNN) or on the iPhone 16 Pro itself (Core ML export: the phone's
  neural engine would run YOLO11n at camera rate). Which one is B's design decision.
- Merged loop: the same stream is recorded, the teacher from this plan labels it on Colab, the model is
  retrained on public data + the yard's own frames; the phone's LiDAR depth helps turn a box into a ground
  point; automatic keep zones and the spray rule from the v1 design stay.

### Decisions from the builder check (2026-11-25; Claude's calls, the owner can overrule)
A fresh reviewer read this design as the builder and found nothing important left open. Its closest calls,
settled here:
- Backyard check, strict reading: the 70 % hit rate must hold on EACH photo (pooled number reported too);
  "boxes on bare stones" counts both classes at confidence >= 0.25; the 50 % rule applies to the quarter
  tiles and to the whole photo; the check runs ONCE, on the final model, after the reference boxes are frozen
  (running it on in-between checkpoints would be tuning on the owner's photos).
- A failed public-test check (2) is handled like a failed backyard check: report to the owner, no retraining
  inside this plan. A failed Pi check (3) is an export bug and gets fixed.
- `inat_keep` takes observations of any quality grade (cultivated garden plants are "casual" on
  iNaturalist); `inat_weed` stays research grade.
- Teacher prompts: in weed photos the keep queries ("a flower", "a shrub") act as distractors, in keep
  photos the weed queries do, so a flower in a weed photo does not become `weed` and the other way round; a
  keep box covers the whole plant, not single blossoms. Confirmed or changed in the VM sample check.

## Constraints
- VM: 4 cores, ~4.8 GB RAM, no GPU, ~17 GB free disk: teacher only on small samples, no full dataset.
- Colab free tier: T4 for a few hours a day, then blocked for hours; everything resumable.
- iNaturalist: only CC0, CC BY and CC BY-NC photos (no share-alike, no ND, no "all rights reserved"); max ~1 API request/s, <= 10,000 results
  per query (page through by id); the model stays non-commercial (CottonWeedDet12 and CC BY-NC photos).
- Public repo: the owner's photos, the reference boxes and downloaded data never go into git.

## Left to the builder
Exact species ids and per-species counts, prompt wording and thresholds (from the sample check, recorded
with reasons), notebook layout, caching format, NMS details, hyperparameters beyond those named.

## Known risks
- Teacher mistakes become label noise; a missed plant teaches "background". Mitigations: drop images with
  no box, the sample check, human-labelled sources stay in the mix.
- Viewpoint: iNaturalist photos are mixed angles and often close-ups; the owner's photos are top-down from
  standing height; the robot camera is 0.19 m high, tilted 35 deg down. B closes this gap.
- Keep plants are generic, not the owner's own shrubs (unchanged from v1; keep zones + look-only).
- Colab GPU limits stretch the work over several days.
