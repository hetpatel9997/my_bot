# Weed Detector Training (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two Colab notebooks that build a ~12k-image `weed`/`keep` dataset from open sources and train a YOLO11n detector, proven end to end on the VM at tiny scale before the owner spends Colab time.

**Architecture:** Plain Python package `training/weed_detector/` (converters per source -> one YOLO dataset; a resumable trainer). The notebooks only mount Drive, `git clone` the public repo, `pip install`, and call it. Unit tests run on the VM without a GPU; a smoke run downloads a few real images per source, builds a mini dataset and trains 1 CPU epoch.

**Tech Stack:** Python 3.10+, Pillow, huggingface_hub, fiftyone (Open Images), ultralytics (YOLO11n), pytest. Colab T4 + Google Drive.

**Spec:** `docs/superpowers/specs/2026-11-24-weed-detector-design.md`

## Global Constraints
- Classes: 0 `weed`, 1 `keep`. Goosegrass and any grass-type weed boxes are dropped (images kept, the grass becomes background).
- Sources (changed from the spec after recon 2026-11-24; the spec gets a note):
  - CottonWeedDet12 from Hugging Face `Voxel51/CottonWeedDet12` (`samples.json` + `data/*.jpg`, FiftyOne boxes `[x, y, w, h]` relative, top-left): all 12 species -> `weed` except `Goosegrass` (dropped). CC BY-NC 4.0.
  - `LibreYOLO/grass-weeds` (HF, YOLO, class `ridderzuring` = broad-leaved dock in grass) -> `weed`; own train/valid/test splits. CC BY 4.0. Replaces RumexWeeds (figshare blocks scripted downloads, HTTP 403).
  - `toolambr/crops_weeds` (HF, YOLO, classes `crop`, `weed`) -> `keep`, `weed`. MIT. Replaces Mendeley (S3 download 403).
  - Open Images V7 via FiftyOne, splits validation + test only (small annotation files), classes `Flower`, `Houseplant`, `Plant`, `Rose`, `Lily` -> `keep`; skip images whose labels include `Dandelion`; up to 3,000 images. CC BY 2.0 images / CC BY 4.0 boxes.
- Images resized to long side 1024 px, JPEG q90; splits 80/10/10 by a stable hash of the source file name where a source has no splits.
- Output layout `<out>/{train,val,test}/{images,labels}/<source>__<name>.jpg|.txt`, `<out>/manifest.csv` (file, split, source, orig_labels, licence), `<out>/data.yaml`.
- Build is resumable: a source with a `<out>/.done_<source>` marker is skipped; an existing output image is not redone.
- Train: `yolo11n.pt`, imgsz 640, epochs 100, patience 30, `hsv_v=0.5`, project on Drive; resume from `last.pt` when present; exports `onnx` and `ncnn`; per-source test metrics via one `val` per source subset.
- Nothing is pushed to GitHub before the owner OKs it (the repo is public); the notebooks clone the branch that holds the code.
- VM: no GPU, 4 GB RAM: never load more than a few images in memory; smoke data <= 50 images.

## Review Focus
- Box conversion off by half a box (FiftyOne top-left vs YOLO centre) -> test with a known box and a drawn image checked by eye.
- A whole image of goosegrass becoming an all-background sample is fine, but a weed box left with class 0 when it is goosegrass is not -> test.
- Same image in two splits (data leak) -> split test per source.
- Notebook stops half-way and is re-run -> resumable markers test.
- Open Images `Plant` boxes that are trees/indoor clutter or dandelions -> filter test + smoke images checked by eye.

---

### Task 1: Converters + builder (`training/weed_detector/sources.py`, `build.py`) with unit tests
**Files:** create `training/weed_detector/__init__.py`, `sources.py`, `build.py`, `requirements.txt`; `test/weed_detector/test_sources.py`.
**Interfaces (produces):**
- `Item = namedtuple('Item', 'source name src split boxes licence orig')` where `src` is a local path or URL, `boxes` = list of `(cls:int, cx, cy, w, h)` relative.
- `fo_box_to_yolo(x, y, w, h) -> (cx, cy, w, h)`; `stable_split(name, seed='weedbot') -> 'train'|'val'|'test'` (80/10/10).
- `cottonweed_items(samples_json_path, image_root_or_url) -> list[Item]`; `yolo_dir_items(root, source, class_map: dict[str,int|None], licence) -> list[Item]` (None = drop the box); `openimages_items(fo_dataset, max_images) -> list[Item]`.
- `build(items, out_dir, long_side=1024, quality=90) -> dict(counts)`; `write_yaml(out_dir)`.
- [ ] Tests first: box conversion; goosegrass dropped; class maps for each source; split stability and 80/10/10 within 5 % on 1,000 names; resumable build (second call writes nothing); manifest rows; image resized so long side = 1024 and labels unchanged (relative).
- [ ] Implement; `pytest test/weed_detector -q` -> all pass.

### Task 2: Trainer (`training/weed_detector/train.py`) with a CPU smoke test
**Interfaces:** `train(data_yaml, project_dir, epochs=100, imgsz=640, device=None) -> Path(best.pt)` (resumes from `project_dir/weights/last.pt`); `evaluate_per_source(best_pt, data_dir) -> dict[source] -> {map50_weed, map50_keep, p, r}`; `export(best_pt) -> list[Path]` (onnx, ncnn).
- [ ] Smoke on the VM in a venv (`~/weedbot_data/venv_yolo`): download <= 15 images per source, build a mini dataset, check 6 images with boxes drawn (by eye), train 1 epoch imgsz 320 on CPU, evaluate, export ONNX. Expected: runs end to end, files exist.

### Task 3: Colab notebooks + owner instructions
**Files:** `training/weed_detector/01_build_dataset.ipynb`, `02_train.ipynb`, `training/weed_detector/README.md`.
- [ ] Notebook 1: mount Drive -> clone repo branch -> pip install -> build each source into `/content/weeds_ds` (CottonWeedDet12 streamed: download one image, resize, delete) -> zip to `MyDrive/weedbot/weeds_ds_v1.zip` -> print counts per source/class/split.
- [ ] Notebook 2: mount Drive -> unzip to `/content` -> train (resume) with project `MyDrive/weedbot/runs/v1` -> per-source metrics -> export -> zip `MyDrive/weedbot/weed_detector_v1.zip`.
- [ ] Both validated with `python3 -m json.tool` and by running their code cells' Python (minus Drive/Colab parts) on the VM smoke data.
- [ ] README: what to click, how long, what to send back.

### Task 4: Review + docs
- [ ] Independent review of the branch; fix; commit on branch `weed-detector`; CLAUDE.md "Weed detector" section (how to run, sources + licences, status); ask the owner before pushing the branch to GitHub (needed for Colab to clone it).
