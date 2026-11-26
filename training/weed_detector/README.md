# WeedBot weed detector: training (Phase A)

A small YOLO11n detector with two classes, `weed` (spray) and `keep` (the owner's flowers and shrubs:
never spray), trained only on public photos. v1 design: `docs/superpowers/specs/2026-11-24-weed-detector-design.md`,
plan: `docs/superpowers/plans/2026-11-24-weed-detector-training.md`. v2 design:
`docs/superpowers/specs/2026-11-25-weed-detector-v2-design.md`, plan: `docs/superpowers/plans/2026-11-25-weed-detector-v2.md`.

## What you (the owner) do - Google Colab, free
All notebooks are in this folder (also copied to the VM Desktop, `WeedBot_training/`). Use Chrome on the VM.
Each one works the same way: open https://colab.research.google.com, sign in, File -> Upload notebook ->
pick the notebook, choose the runtime type it says, then Runtime -> **Run all** and allow Google Drive access.
Keep the laptop awake the whole time (lid open: a closed lid pauses the VM and the Colab tab).

### Version 2 (now): steps 3, 4 and 4b
Why: v1 draws one box over the whole picture on your rock-bed photos. For v2 a big model (OWLv2, called the
"teacher") draws the boxes on ~20,000 more public photos: weeds, garden plants, and "nothing here" photos of
gravel, soil and lawn. Then the small model learns again from all of them, starting from v1.

**Before you start**
- Keep ALL the v1 files in Google Drive (`MyDrive/weedbot/`: `dataset_parts/` and `runs/v1/`). Version 2
  reuses v1's photos and starts from v1's model. Do not delete anything in `MyDrive/weedbot/` until Claude
  says v2 is finished.
- Google Drive needs about **5.3 GB more** free space (keep 6 GB free to be safe). Check it at
  https://drive.google.com -> Storage (bottom left). A free Google account has 15 GB in all.

**Step 3 - collect the new photos (`03_v2_collect_photos.ipynb`, CPU)**
1. File -> Upload notebook -> pick `03_v2_collect_photos.ipynb`.
2. Runtime -> Change runtime type -> **CPU** -> Save (no GPU needed here; it saves your free GPU time for step 4).
3. Runtime -> **Run all**. Allow Google Drive access when asked.
4. Wait until the last line says `DONE: all 3 photo parts are on Drive, go to step 4` (about 1-3 hours).
5. If the session stops, or the last line says `Not all parts finished (... of 3): press Run all again`, press
   **Run all** again: finished parts are skipped (it prints `already on Drive, skipped`).

**Step 4 - label the photos and train v2 (`04_v2_label_and_train.ipynb`, T4 GPU)**
1. File -> Upload notebook -> pick `04_v2_label_and_train.ipynb`.
2. Runtime -> Change runtime type -> **T4 GPU** -> Save.
3. Runtime -> **Run all**, allow Drive access.
4. It does two jobs, one after the other. First the teacher draws the boxes ("labelling", about 3-4.5 hours
   of T4 time), then the small model trains (about 4-5 hours; one "round" is one pass over all the photos, 50
   rounds at most). That is more GPU time than free Colab gives in one day, so **plan for two to three days**
   (the first day is mostly labelling).
   The install cell near the top may print red warnings about other packages ("dependency conflicts"): that is
   expected, ignore them. If Colab asks you to restart the session, click Restart session, then **Run all** again.
5. **When Colab stops** ("runtime disconnected", or a message about GPU usage limits): come back later
   (often the next day), open `04_v2_label_and_train.ipynb` again and press **Run all**. Nothing is lost:
   finished labelling is kept on Drive (it prints `inat_weed: already labelled on Drive, skipped`, and the same
   for `inat_keep` and `background`; a part that Colab cut off while saving it prints `... is damaged ...:
   labelling again from the saved teacher boxes` and is made again in minutes), and training continues from its
   last round. After each restart it first unpacks and prepares the photos for about 10-20 minutes before
   training goes on: that is normal, it is not stuck. Do this as often as needed.
   If Colab says the GPU is not available, try again later (do not switch to CPU).
6. When it prints `ALL DONE -> MyDrive/weedbot/weed_detector_v2.zip`, open Google Drive -> `weedbot` ->
   download `weed_detector_v2.zip` (in Chrome on the VM it lands in `~/Downloads`). Then see "After training".

If the first cell stops with `v1 must be trained first; keep the v1 files on Drive`, the v1 model is
missing from Drive (`runs/v1/weights/best.pt`): tell Claude. If it stops with `... is missing: run step 3
(03_v2_collect_photos) until it prints DONE`, do step 3 first.

**Step 4b - only if step 4 stopped near the end and Colab says "Cannot connect to GPU backend"
(`04b_v2_finish_without_gpu.ipynb`, CPU)**
1. File -> Upload notebook -> pick `04b_v2_finish_without_gpu.ipynb`.
2. Runtime -> Change runtime type -> **CPU** -> Save.
3. Runtime -> **Run all**, allow Drive access. About 30-60 minutes. It does not train: it takes the best model
   step 4 already saved, tests it and makes the Pi versions. The saved training run is not changed.
4. When it prints `ALL DONE -> MyDrive/weedbot/weed_detector_v2.zip`, download it as in step 4. Then see
   "After training": this zip is the final model, so do not run step 4 again afterwards.

**After training - send the zip to Claude**
1. Send Claude `weed_detector_v2.zip`. **The first zip you send is the final v2 model**, also when it came from
   step 4b. Do not run step 4 or 4b again after that: step 4 could finish the last rounds and write a new zip
   over it, but the backyard check below runs only once, so the model must be fixed before it runs.
2. Claude's job, on the VM (nothing for you to do): unzip it to `~/weedbot_data/models/weed_detector_v2/` (the
   check needs `best.pt` and `best_ncnn_model/` there), then, from `~/dev_ws/src/my_bot/training`, run the
   backyard check ONCE: `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m weed_detector.backyard_check
   check ~/weedbot_data/models/weed_detector_v2 ~/weedbot_data/backyard_photos/reference/reference.json`.
3. Claude tells you the result of the three checks (v2 design, "Checks before v2 goes on the robot"):
   - check 1, your two backyard photos: the command above, PASS needed on both photos;
   - check 2, public test photos, from `metrics_per_source.json` in the zip: weed recall at most 0.05 below v1
     (`per_source_test` -> `cottonweed` -> `recall` at least 0.764, v1 0.814; `grassweeds` at least 0.663,
     v1 0.713) and keep precision at least 0.70, read from **`per_source_test` -> `openimages` -> `precision`**
     (v1 0.703): the only keep number measured on human-drawn boxes alone. The overall
     `overall_test` -> `per_class` -> `keep` -> `precision` (v1 0.742) also counts the teacher-labelled test
     photos, so it is only reported, as is keep recall (v1 0.53);
   - check 3, the Pi version gives the same boxes as `best.pt`: in the backyard check's report.
   If check 1 fails, v2 is not used; Claude and you decide the next step. Passing does not mean spraying: the
   robot only looks at first.

### Version 1 (done 2026-11-25): steps 1, 2 and 2b
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

**Step 2b - only if step 2 stopped near the end and Colab says "Cannot connect to GPU backend"
(`02b_finish_without_gpu.ipynb`)**
1. File -> Upload notebook -> pick `02b_finish_without_gpu.ipynb`.
2. Runtime -> Change runtime type -> **CPU** -> Save.
3. Runtime -> **Run all**, allow Drive access. About 30-60 minutes; it does not train, it takes the best model
   step 2 already saved, tests it and makes the Pi versions. The saved training run is not changed, so step 2
   can still finish the last rounds later when a GPU is free (that writes a new zip).
4. When it prints `ALL DONE`, download `weed_detector_v1.zip` as in step 2 and tell Claude.

## What gets made
- v1 dataset (~11,000 images, long side 1024 px): CottonWeedDet12 (12 US weed species incl. spotted spurge and
  purslane; goosegrass removed; split by photo session) -> weed; Roboflow-100 grass-weeds (dock in grass) ->
  weed; Open Images V7 Flower / Houseplant / Plant / Rose / Lily -> keep (pictures with people, group boxes,
  drawings, tiny boxes or a human weed tag such as dandelion / clover / spurge / dock / plantain left out).
  CropWeeds-YOLO was dropped after review: all its boxes say "crop" on pictures of field weeds.
- v2 dataset (~16,000 training images): the three v1 parts (Open Images pictures with any box over 40 % of the
  picture are dropped at the merge) plus `inat_weed` (iNaturalist research-grade photos of 13 yard weeds, the
  largest share spotted spurge) and `inat_keep` (13 garden flowers and shrubs), both boxed by the teacher, and `background` (Open Images
  ground photos without any plant tag, empty label files = "nothing here"). The teacher drops a photo when it
  finds no plant, when a box covers more than 60 % of it (a close-up), or (ground photos) when it finds a plant.
  Its prompts and thresholds come from a sample check on the VM (v2 design doc, "Teacher settings").
- `weed_detector_v1.zip` / `weed_detector_v2.zip`: `best.pt`, `best.onnx`, `best_ncnn_model/` (for the Pi),
  `results.csv`, plots, `metrics_per_source.json` (test scores per class incl. keep recall, and per source),
  `test_confusion_matrix.png`, manifests (where each image came from), README.txt. In v2 the scores of the
  teacher-labelled sources (`inat_weed`, `inat_keep`, `background`) measure agreement with the teacher; they
  are listed per source, apart from the human-labelled ones.
- Licences: CottonWeedDet12 CC BY-NC 4.0 (non-commercial), grass-weeds CC BY 4.0, Open Images CC BY 2.0 images /
  CC BY 4.0 boxes; v2 adds iNaturalist photos under each photo's own licence (CC0, CC BY or CC BY-NC; observer and
  URL per photo in `manifest_inat_weed.csv` / `manifest_inat_keep.csv`) and boxes made by OWLv2
  (`google/owlv2-base-patch16-ensemble`, Apache 2.0). The models are for non-commercial use.

## For developers (VM)
- Code: `sources.py` (per-source converters, iNaturalist taxa and counts), `build.py` (resize + labels +
  manifest, resumable), `colab.py` (Drive parts, `label_part`, merge, `drop_big_box_images`, downloads),
  `train.py` (train / resume / per-source eval / export / `final_weights` for a stopped run), `teacher.py`
  (OWLv2 teacher: `Owlv2Teacher`, `label_images` with its JSONL cache of raw detections, so thresholds and box
  rules can change without running the model again; prompts and thresholds with the reasons from the sample check).
- `teacher_check.py` (VM only, never on the owner's photos): the teacher labels a small public sample on the CPU
  and draws contact sheets to judge its boxes; that is how `teacher.py`'s prompts and thresholds were set. From
  `training/`: `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m weed_detector.teacher_check ~/weedbot_data/teacher_check`.
- `backyard_check.py` (VM only): the v2 acceptance check on the owner's two photos against frozen reference
  boxes (`~/weedbot_data/backyard_photos/reference/`, never in git). `reference OUT_DIR` made the reference (done,
  frozen: it refuses to run again); `check MODEL_DIR REFERENCE_JSON` runs ONCE, on the final v2 model (running
  it on in-between models would be tuning on the owner's photos). Same venv and folder as above,
  `-m weed_detector.backyard_check ...`.
- `teacher_check.py` and `backyard_check.py` are not pasted into the notebooks (not in `make_notebooks.py` MODULES).
- Versions: notebook 04 installs transformers 4.57.6 only when another version is there (the teacher's prompts
  and thresholds were tuned with it; 5.x pads photos black instead of grey, which moves the scores a little;
  `teacher.py` calls the image processor's post-processing, which exists in 4.57 and 5.x, so a missed pin does
  not crash). 04 and 04b pin ultralytics 8.4.218 (v1's version; `train.run_finished` depends on 8.4 behaviour).
- After any change: `python3 training/weed_detector/make_notebooks.py` (regenerates all six notebooks; the v1
  notebooks' cells stay the same apart from the pasted modules).
- Tests: `python3 -m pytest test/weed_detector -q` (and the same with the venv's python). End-to-end check on the
  VM (internet, CPU): `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python test/weed_detector/dryrun_notebooks.py [steps]`
  with steps `1 2 2b` (v1; 2b fakes a run stopped after round 2 of 3 and finishes it with the step 2b notebook) and
  `3 4 4b 4r` (v2; 3 first deletes the dry run's own v2 outputs; the real teacher on the CPU, ~36 photos, takes
  15-25 minutes; run 4 twice to see the skips; 4b and 4r use a v2 run stopped after round 2 of 3).
- The training venv `~/weedbot_data/venv_yolo` (torch CPU, ultralytics, fiftyone, transformers) is outside ROS on
  purpose; run it with `env -u PYTHONPATH` so ROS packages do not mix in.
