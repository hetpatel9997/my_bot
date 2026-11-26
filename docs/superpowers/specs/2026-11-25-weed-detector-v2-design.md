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
| NEW `inat_weed`: iNaturalist research-grade observations of yard weeds, first photo, `large` (1024 px) | ~12,000 downloaded, ~3,300 after the teacher | teacher | weed | per photo: CC0 / CC BY / CC BY-NC |
| NEW `inat_keep`: iNaturalist observations of common garden flowers and shrubs (cultivated ones allowed: gardens are cultivated) | ~6,000 downloaded, ~1,100 after the teacher | teacher | keep | same |
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
- OWLv2 base patch16 ensemble on Colab's T4 GPU (estimate: well under 1 s per image -> ~1.5-3 h for ~20,000 images).
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
- Expected ~16,000 training images -> ~5 min per epoch on a T4, ~4-5 h: likely two Colab days
  (free GPU limit seen: ~4.5 h, then "usage limit" for hours). Resumable as in v1; the step-2b
  finish-without-GPU notebook works for v2 too.
- Owner steps stay the same shape: one dataset notebook (needs a **GPU** this time, for the teacher),
  one training notebook, optional finish notebook; result `MyDrive/weedbot/weed_detector_v2.zip`.
  Drive space needed beyond v1's: ~5.3 GB.

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
- Download counts raised (weed x1.5, keep x2) after the teacher sample check measured 28 % / 18 % yield (controller ruling, 2026-11-25).

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

## Teacher settings from the sample check (2026-11-25)
Done on the VM CPU with `training/weed_detector/teacher_check.py`, on public photos only (the owner's photos
were not used or opened). Sample: `inat_weed` 39 = 3 research-grade photos x 13 taxa, `inat_keep` 39 = 3
cultivated (`captive=true`, any grade) photos x 13 taxa (iNaturalist returns the oldest open-licence
observations first), `background` 30 = the first 30 Open Images ground photos by image id. Every photo on
every contact sheet was judged by eye (`~/weedbot_data/teacher_check/sheets/`, pass 1 in `pass1/`).
- Pass 1: the starting values, real teacher, all 108 photos. OWLv2's image features were saved in the same
  run. A query's score on a box does not depend on the other queries (only which query a box is credited to
  does), so other prompt / threshold sets were then scored offline from those features. That reproduced the
  pass-1 cache to 4e-6 on all 2,278 detections, so nothing was guessed.
- Pass 2: the final values, real teacher again (new calls for weed + background; keep came from the cache,
  since only its threshold changed). The verdicts matched the offline prediction photo for photo.

Final values (`teacher.py`; NMS_IOU 0.5, MIN_BOX_AREA 0.003, MAX_BOX_AREA 0.6 unchanged):
| kind | plant queries | distractors | threshold |
|---|---|---|---|
| weed | a weed, a small plant, a plant with reddish stems and small leaves | grass, a stone, soil, a human hand, a shoe, a flower, a shrub | 0.15 |
| keep | a flower, a shrub, a flowering plant, a bush | grass, a stone, soil, a human hand, a shoe, a weed, a flower pot | 0.25 |
| background | a weed, a small plant, a flower, a shrub, trees, vegetation | grass, a stone, soil, gravel | 0.15 |

Results. Right = every box sits on a plant of the photo's class. Wrong = some box sits mainly on something
else. Missed = a clearly visible plant of the class has no box (the rest of a mat or carpet does not count).
| source | pass | kept / photos | dropped | boxes | right / wrong | missed |
|---|---|---|---|---|---|---|
| inat_weed | 1 | 10 / 39 | close-up 22, no plant 7 | 21 | 9 / 1 | in 3 |
| inat_weed | 2 | 11 / 39 | close-up 15, no plant 13 | 34 | 10 / 0, + 1 loose | in 2 |
| inat_keep | 1 | 7 / 39 | close-up 27, no plant 5 | 14 | 6 / 1 | - |
| inat_keep | 2 | 7 / 39 | close-up 27, no plant 5 | 8 | 7 / 0 | - |
| background | 1 | 23 / 30 | plant found 7 | - | 4 of 23 kept show a plant | - |
| background | 2 | 19 / 30 | plant found 11 | - | 1 of 19 kept shows a plant | - |
Pass 1 wrong: weed, a box on a black pot while the weed mat below was unboxed; keep, a second box on stems and a
person's legs. Pass 2, weed: the loose box is a strip at the bottom of a black-medic photo (medic leaves with grass
blades, bark and soil); the missed plants are the main knotweed body (credited to the distractor "a shrub") and a
few henbit plants. No kept weed box sits on grass alone. Two boxes include grass blades around the weed: a ribwort
plantain rosette in a lawn, and the strip above. The one background photo with a plant is a grassy airstrip with a
distant tree line. The background targets are met: <= 1 in 10 kept photos shows a plant; the 80 % right target is
met for weed (91 %, 100 % if the loose box counts) and keep (100 %).

Changes (also commented in `teacher.py`):
- weed: "a plant with small leaves" -> "a plant with reddish stems and small leaves". The old query put
  whole-picture boxes on any foliage (18 of the 22 close-up drops) and drew the one wrong weed box. The new query
  scores best on the spurge photos.
- keep threshold 0.15 -> 0.25: the boxes scoring 0.15-0.24 were extra blossoms or the wrong stems/legs box; each
  kept photo keeps its main box (0.28-0.74).
- background: + "trees", "vegetation", threshold 0.12 -> 0.15. They drop the forest, scrub-hillside and
  cliff-top photos. "vegetation" scores 0.12-0.14 on bare rock and soil, hence 0.15. Every photo dropped before
  is still dropped.
- Builder-check decision changed: a `keep` box is usually a blossom or flower cluster, not the whole plant.
  "a flower" as a keep distractor (to force whole-plant boxes) gave worse boxes (part of a bush, a pot with
  leaves) and 4 kept photos instead of 7. Whole shrubs mostly fill the iNaturalist photo and are dropped as
  close-ups.
Tried and not taken: "a small weed plant" (more close-ups, 8-9 kept), fewer distractors (more close-ups, 6-9 kept),
"a blade of grass" / "a tuft of grass" as distractors (they took the narrow plantain rosettes), weed threshold
0.20 (9 of 9 right, but it removes 15 boxes scoring 0.15-0.20, 14 of them correct: more unboxed plants), MAX_BOX_AREA 0.65 (only 1 of the 15
close-up drops lies between 0.6 and 0.65).

Open points (not blocking, for the owner / the Colab plan):
- Yield is low: weed 28 %, keep 18 %, background 63 %. At these rates the planned ~8,000 / ~3,000 / ~2,000
  photos give roughly 2,200 / 540 / 1,270 labelled images. Spurge: 1 of 6 kept (the others are close-ups or
  unboxed mats). Most drops are the close-up rule doing its job, but it also drops some photos with a correct box
  (a single spotted spurge boxed at 0.90 covering 62 %). If more images are wanted, ask for more photos per taxon
  (keep especially) rather than loosening the rules.
- Keep foliage outside the blossom box stays unlabelled (daylily, marigold), so the model sees keep leaves as
  background. Keep zones and look-only still cover this.
- The sample is small (11 + 7 + 19 kept photos judged): one photo moves a rate by 5-14 points.
