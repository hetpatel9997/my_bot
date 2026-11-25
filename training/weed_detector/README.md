# WeedBot weed detector: training (Phase A)

A small YOLO11n detector with two classes, `weed` (spray) and `keep` (the owner's flowers and shrubs:
never spray), trained only on public photos. Design: `docs/superpowers/specs/2026-11-24-weed-detector-design.md`,
plan: `docs/superpowers/plans/2026-11-24-weed-detector-training.md`.

## What you (the owner) do - Google Colab, free
Both notebooks are in this folder (also copied to the VM Desktop, `WeedBot_training/`). Use Chrome on the VM.

**Step 1 - collect the photos (`01_build_dataset.ipynb`)**
1. Open https://colab.research.google.com, sign in with your Google account.
2. File -> Upload notebook -> pick `01_build_dataset.ipynb`.
3. Runtime -> Change runtime type -> **CPU** -> Save (no GPU needed here).
4. Runtime -> **Run all**. Allow Google Drive access when asked.
5. Wait until the last cell prints `DONE: all 3 parts are on Drive` (about 30-90 minutes). Keep the laptop
   awake (lid open: a closed lid pauses the VM and the Colab tab). If the session stops, press Run all again:
   finished parts are skipped. An old `cropsweeds.zip` from an earlier version is simply ignored.

**Step 2 - train (`02_train.ipynb`)**
1. File -> Upload notebook -> pick `02_train.ipynb`.
2. Runtime -> Change runtime type -> **T4 GPU** -> Save.
3. Runtime -> **Run all**, allow Drive access.
4. Training takes a few hours. If Colab stops it ("runtime disconnected" or GPU limit), come back later,
   open the notebook and press **Run all** again: it continues from the last saved epoch.
5. When it prints `ALL DONE`, open Google Drive -> `weedbot` -> download `weed_detector_v1.zip`
   (in Chrome on the VM it lands in `~/Downloads`). Tell Claude.

## What gets made
- Dataset (~11,000 images, long side 1024 px): CottonWeedDet12 (12 US weed species incl. spotted spurge and
  purslane; goosegrass removed; split by photo session) -> weed; Roboflow-100 grass-weeds (dock in grass) ->
  weed; Open Images V7 Flower / Houseplant / Plant / Rose / Lily -> keep (pictures with people, group boxes,
  drawings, tiny boxes or a human weed tag such as dandelion / clover / spurge / dock / plantain left out).
  CropWeeds-YOLO was dropped after review: all its boxes say "crop" on pictures of field weeds.
- `weed_detector_v1.zip`: `best.pt`, `best.onnx`, `best_ncnn_model/` (for the Pi), `results.csv`, plots,
  `metrics_per_source.json` (test scores per class incl. keep recall, and per source), `test_confusion_matrix.png`,
  manifests (where each image came from), README.
- Licences: CottonWeedDet12 CC BY-NC 4.0 (non-commercial), grass-weeds CC BY 4.0, Open Images CC BY 2.0 images /
  CC BY 4.0 boxes.

## For developers (VM)
- Code: `sources.py` (per-source converters), `build.py` (resize + labels + manifest, resumable),
  `colab.py` (Drive parts, merge, downloads), `train.py` (train / resume / per-source eval / export).
- After any change: `python3 training/weed_detector/make_notebooks.py` (regenerates both notebooks).
- Tests: `python3 -m pytest test/weed_detector -q`. End-to-end check on the VM (internet, CPU):
  `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python test/weed_detector/dryrun_notebooks.py`.
- The training venv `~/weedbot_data/venv_yolo` (torch CPU, ultralytics, fiftyone) is outside ROS on purpose;
  run it with `env -u PYTHONPATH` so ROS packages do not mix in.
