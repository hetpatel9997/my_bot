# Weed detector that learns by itself (design)

Date: 2026-11-24. Status: draft for the owner's approval. Size: project (sub-project 3 "detector",
with the on-robot parts of sub-project 2 sketched for later).

## Why
The WeedBot must find weeds by itself through the iPhone camera on the robot and spray them with the
arm, and never spray plants the owner wants to keep. The backyard has two kinds of ground: beds of
loose white/grey marble stones (2-5 cm) where small low weeds grow between the stones (reddish stems,
tiny oval leaves, very likely spotted spurge, Euphorbia maculata; also a larger green broadleaf weed),
and a lawn. The owner's two real photos (`~/weedbot_data/backyard_photos/originals/IMG_0229.JPG`,
`IMG_0230.JPG`, 1536 x 2048, top-down) show the rock beds; 136 other "photos" in the same folder were
failed exports (135-byte Apple plist files, not images).

## What the owner decided (2026-11-24)
- The robot works in the rock beds AND on the lawn.
- There are plants in those areas that must NOT be sprayed.
- The owner will NOT take or label photos: "it should know by itself".
- Train on public photos of common weeds, about 10,000.
- Chose both of: (1) weed detector + no-spray zones, (2) a detector with a "keep" class. Combined here:
  one model with two classes AND automatic no-spray zones built from its "keep" sightings.
- Training on a free Google Colab GPU (the VM has no GPU). The owner can press Run again when a free
  session stops (2-4 times over a day or two) and has >= 8-10 GB free in Google Drive.
- Plants to keep: **shrubs/bushes and flowers**, growing **mixed in with the weeds** inside the rock beds
  and the lawn (not only at the edges).
- Spray: a **lawn-safe broadleaf** weed killer (spares grass; harms broadleaf plants, i.e. also the
  owner's flowers and shrubs).

## Goals
- G1: a first model from public data only: classes `weed` and `keep`, ~10-12k labelled images.
- G2 (later, on the robot): it adapts to the owner's yard by itself from frames the robot records.
- G3 (later): it protects plants by itself (automatic no-spray zones) and starts look-only.

## Non-goals
- Telling weed species apart (one `weed` class; species only used to pick data).
- Pixel-exact outlines (boxes are enough to aim: the arm aims at the box centre on the ground).
- Running on anything but the Pi 4 CPU (no accelerator) in the end.

## Anti-goals (failure even if it "works")
- Spraying a plant the owner wants to keep, or spraying on a weak guess.
- The owner having to photograph, label or review images to make it work.
- Automatic self-training that reinforces its own mistakes (only very confident labels feed back).

## Approaches considered
1. Weed-only detector + no-spray zones the owner draws: safest protection, but needs the owner to draw.
2. One detector with `weed` and `keep`: protection depends on the model knowing the owner's plants.
3. Find every plant, then classify weed / keep: two models on the Pi (slower); "every plant" fails on a lawn.
**Chosen (owner): 1 + 2** - one two-class detector; its confident `keep` sightings become no-spray zones
automatically on the robot's map; owner-drawn zones stay possible but optional.

## Design

### Decisions made from the answers (builder check, 2026-11-24)
- Owner's: keep = shrubs + flowers, mixed among weeds; broadleaf spray; repeated Colab runs OK.
- Claude's calls (owner can overrule):
  - Grass-type weeds (e.g. goosegrass in CottonWeedDet12, crabgrass anywhere) are **dropped**: lawn grass
    is background, a broadleaf spray does not kill them, and labelling them `weed` would teach the model
    that grass is a weed.
  - Broadleaf lawn weeds (dandelion, clover, plantain) **are** `weed`; open sets of them are added where a
    licence allows (Weed-AI CC BY 4.0, open Roboflow sets), the builder picks.
  - Purslane / spurge stay `weed` (no succulents to protect).
  - Because keep plants grow among the weeds, `keep` needs both recall (a missed flower can be sprayed)
    and precision (false keeps hide weeds): both are reported; the spray rule keeps a margin around every
    `keep` box, and no-spray zones are a second layer only.
  - Pass mark: v1 goes onto the robot in **look-only** mode whatever its scores (looking cannot hurt);
    spraying waits for the look-only review. This plan has one dataset build + one training run, no
    retraining loop.
  - Colab is split into two resumable notebooks: (1) dataset build, one source at a time, result stored
    as one zip (~3-5 GB) on Drive; (2) training, checkpoints on Drive, resumes after a stop.
  - Detection boxes (not segmentation outlines) with classes `weed` / `keep`: this replaces the older
    CLAUDE.md plan "YOLO segmentation, classes weed_broadleaf / keep_plant".

### Phase A - first model from public data (this plan)
**Classes:** 0 = `weed`, 1 = `keep`. Background = everything else (stones, soil, lawn grass).

**Sources** (each downloaded, converted, shrunk and its raw files deleted before the next, because the
two big ones are ~28 GB each and Colab's free disk is ~80 GB):

| Source | Images used | Mapping | Licence |
|---|---|---|---|
| CottonWeedDet12 (Zenodo 7535814: one file `CottonWeedDet12.7z`, 28.59 GB, must be downloaded whole; YOLO txt included) | all ~5,648 | all 12 weed species -> `weed` (includes spotted spurge 956 boxes, purslane 992) | CC BY-NC 4.0, confirmed on Zenodo (non-commercial: OK for this hobby robot) |
| RumexWeeds (DTU/figshare 17040518: `RumexWeeds_v1.zip`, 30.21 GB, https://ndownloader.figshare.com/files/42426954; a zip can be read member by member over HTTP range requests, so only the frames used need downloading) | ~2,000 (sequences at 5 fps: keep every 3rd frame, split by sequence) | dock plants -> `weed`; grass = background | CC BY 4.0, confirmed via the figshare API |
| Mendeley weed-crop (mthv4ppwyw v2, 1,120 images, YOLO) | all | 5 weed species -> `weed`; 8 crop species -> `keep` | CC BY 4.0, confirmed via the Mendeley API |
| Open Images V7 via FiftyOne (boxable classes confirmed: `Flower`, `Houseplant`, `Plant`, `Rose`, `Lily`; outdoor garden shots preferred: shrubs come mostly from `Plant`) | ~3,000 | -> `keep`; drop images whose image-level labels include dandelion or other weeds | images CC BY 2.0, boxes CC BY 4.0 |
Target total ~11,000 images. Exact counts, any extra open sets (e.g. Weed-AI CC BY 4.0 garden/lawn
sets, dandelion sets) and dataset class names are the builder's call, written into a manifest.

**Preprocessing:** long side resized to 1024 px (JPEG q90), labels converted to YOLO txt with the two
class ids, per-source split train/val/test 80/10/10 (RumexWeeds by sequence, no near-duplicates across
splits), a `manifest.csv` (file, source, original class, licence) and `data.yaml`.

**Model:** Ultralytics YOLO11n detection (the Pi-friendly size), image size 640, ~100 epochs, default
augmentation plus stronger brightness/contrast (white stones are very bright), batch as the GPU allows.
Checkpoints and the processed dataset are kept on the owner's Google Drive so a dropped Colab session
resumes instead of restarting.

**Outputs (in one Drive folder):** `best.pt`, `best.onnx`, NCNN export for the Pi, metrics per source and
per class (precision, recall, mAP50), the confusion matrix, and the model's predictions drawn on the two
real backyard photos.

**Checks before the model goes on the robot:** test-split mAP50 for `weed` reported per source; `keep`
precision reported (how often a keep box is really a keep plant); on IMG_0229/0230 the spurge plants are
found as `weed` and the stones are not (judged by eye, there are no labels for them).

**Delivery:** the owner downloads the Drive folder once (zip) to the Mac; Claude copies it to the VM
(`~/weedbot_data/models/weed_detector_v1/`) and tests it there on CPU.

### Phase B - on the robot (later plans; needs the robot driving with the phone)
- `weed_detector_node` on the Pi: NCNN model on `/phone/image/compressed`, publishes detections.
- Ground position: box centre -> ray from the phone camera (`/phone/camera_info` + TF `phone_optical`)
  -> ground plane -> point in `base_link` / `map`; the arm already aims at such points.
- **Look-only first:** the robot records what it WOULD spray (crop + position) and sprays nothing;
  spraying is switched on only after the owner and Claude have looked at those records together.
- **Automatic no-spray zones:** a `keep` seen at the same place on >= N runs becomes a circular zone
  (radius r) in the keep-out zones file the spray controller already obeys.
- **Self-training:** the robot saves camera frames automatically while it drives; the current model
  labels them; only very confident boxes are kept as labels; periodic retraining on Colab with the
  public data plus these frames.
- **Spray rule (on top of every existing spray check):** `weed` confidence >= tau_w, no `keep` box within
  d, not inside any zone.

## Constraints
- VM: no GPU, 4 cores, 4 GB RAM, 24 GB free disk -> no training or big datasets on the VM.
- Colab free: T4 GPU, ~80 GB disk, sessions end after hours or when idle -> sequential sources, Drive
  checkpoints, resumable training.
- Pi 4 CPU at the end -> nano model, NCNN export.
- Spray safety rules in CLAUDE.md stay above everything (never spray in sim or on the bench without the
  owner asking in that session, /safety_state clear, keep-out zones, pool >= 1 m).

## Left to the builder
Notebook layout, exact per-source counts and filters, the class-name lists inside each source,
hyperparameters beyond those above, threshold values (to be set later from validation and look-only runs),
and which extra open datasets to add if a source turns out unavailable.

## Known risks
- Domain gap: farm-field and grassland photos, not white stones or a home lawn; Phase B adapts it.
- `keep` from flowers / house plants / crops is not the owner's shrubs; the automatic zones and the
  look-only phase cover that.
- Lawn weeds are green on green and harder than weeds on white stones.
- Licences confirmed 2026-11-24: CottonWeedDet12 CC BY-NC 4.0, RumexWeeds CC BY 4.0, Mendeley CC BY 4.0,
  Open Images CC BY 2.0 / CC BY 4.0. Credit them in the model folder's README.

## Changes found while building (2026-11-24)
- RumexWeeds and the Mendeley set block scripted downloads (HTTP 403 from figshare / S3), so they are
  replaced: **Roboflow-100 grass-weeds** (HF `LibreYOLO/grass-weeds`, 2,486 images of broad-leaved dock in
  grassland, CC BY 4.0) -> `weed`, and **CropWeeds-YOLO** (HF `toolambr/crops_weeds`, 867 images, crop -> `keep`,
  weed -> `weed`, MIT).
- CottonWeedDet12 comes from the Hugging Face copy `Voxel51/CottonWeedDet12` (one labels file + one image per
  download), not the 28.6 GB 7z: no full download, no disk peak.
- Open Images `keep` pictures are filtered: no people, no group boxes (one box over a whole hedge would teach
  "patch of plants = keep"), no drawings, no boxes under 0.5 % of the image, and none that a human tagged as
  Dandelion / Clover / Spurge / Thistle / Weed (237 such pictures in the sample run).
- Code, tests and the two notebooks: `training/weed_detector/` (README = owner steps). A dry run of both
  notebooks on the VM (fake Drive, small limits, CPU) ran end to end on 2026-11-24.
- Independent review (2026-11-24) changes: CropWeeds-YOLO DROPPED (all 1,454 boxes are "crop" on pictures of
  field weeds: it would teach weeds as `keep`); `keep` now comes from Open Images only (~3,000 pictures),
  with more weed tags excluded (dock, plantago, wood sorrel, knotweed, lamb's quarters, amaranth);
  CottonWeedDet12 split by capture session (no near-duplicates across splits); step 2 refuses to run without
  a GPU, checks a finished run before resuming (Ultralytics 8.4 would silently start a new run), reports
  precision / recall PER CLASS on the whole test split (keep recall = plants protected) with a confusion
  matrix, and a failed NCNN export no longer loses the result zip; image cache on disk for speed.
