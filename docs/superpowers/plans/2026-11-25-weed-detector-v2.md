# Weed detector v2 (teacher-labelled data) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything the owner needs to make a v2 weed/keep detector on Colab: iNaturalist and Open Images
"nothing here" photo sources, an OWLv2 teacher that boxes them, v2 merge/train/export notebooks, and a VM
tool for the backyard check, all proven by unit tests and a VM dry run.

**Architecture:** Extends `training/weed_detector/` (v1). New sources produce Items without boxes; their
images are built into `<source>_images.zip` parts on Drive by a CPU notebook; a GPU notebook runs the teacher
(`teacher.py`, OWLv2) over each part, writes YOLO labels, stores `<source>.zip`, merges with the reused v1
parts, fine-tunes v1 into run `v2` and packs `weed_detector_v2.zip`. `backyard_check.py` (VM only) scores a
model against frozen reference boxes on the owner's two photos.

**Tech Stack:** Python 3.10 (VM) / Colab Python 3.12+, Ultralytics YOLO11n 8.4, Hugging Face transformers
(OWLv2), PIL, urllib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-weed-detector-v2-design.md` (builds on
`docs/superpowers/specs/2026-11-24-weed-detector-design.md`).

## Global Constraints
- Branch `weed-detector` in `~/dev_ws/src/my_bot`. Commit ONLY the files your task names (`git add <paths>`);
  the working tree holds the owner's uncommitted work (CLAUDE.md, CMakeLists.txt, config/*, ...): never add,
  stash, reset or commit it.
- The owner's photos (`~/weedbot_data/backyard_photos/`), the reference boxes, downloaded data and models live
  under `~/weedbot_data/` and NEVER go into git (public repo).
- Heavy imports (torch, transformers, ultralytics, fiftyone, huggingface_hub) only inside functions: the unit
  tests also run under the system python, where those packages are missing (tests use `pytest.importorskip`).
- Modules import each other relatively (`from . import sources as S`); every module in
  `training/weed_detector/make_notebooks.py` MODULES is pasted into the notebooks with `%%writefile`.
- VM runs of the venv: `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python ...` (has torch CPU,
  ultralytics 8.4.218, fiftyone, transformers 4.57.6). VM: 4 cores, ~4.8 GB RAM, no GPU.
- Tests: `cd ~/dev_ws/src/my_bot && python3 -m pytest test/weed_detector -q` AND
  `env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m pytest test/weed_detector -q`; both must pass
  (the system python may skip torch tests). v2 tests go in `test/weed_detector/test_v2.py`.
- Network: iNaturalist API at most 1 request per second, header `User-Agent: weedbot-dataset-builder`, no
  personal data in requests. Photo licences: only `cc0`, `cc-by`, `cc-by-nc`.
- Classes: `WEED = 0`, `KEEP = 1` (sources.py); "background" images have empty label files.
- Commit message trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. An iNaturalist observation whose FIRST photo has no licence or a licence outside the three (the API filter
   matches ANY photo of the observation) must be skipped, not crash and not be used -> Task 1 test.
2. A taxon with fewer open photos than asked for must return what exists and stop paging -> Task 1 test.
3. An Open Images "negative" tag (Confidence 0) for Plant must not exclude a background photo, and a positive
   plant / weed tag must -> Task 1 test.
4. Colab dying mid-labelling leaves a teacher cache whose last JSONL line is cut off: resume must ignore that
   line and redo only that image; a corrupt image must be counted and dropped, not stop the part -> Task 2 tests.
5. Re-running the GPU notebook after its labelling finished must not label again (parts on Drive are reused)
   and a missing images part must say which notebook to run -> Task 4 tests.

---

### Task 1: iNaturalist and background photo sources

**Files:**
- Modify: `training/weed_detector/sources.py` (constants + 3 functions, module docstring lists the new sources)
- Modify: `training/weed_detector/colab.py` (3 download functions)
- Create: `test/weed_detector/test_v2.py`

**Interfaces:**
- Produces (sources.py):
  - `INAT_WEED_TAXA: dict[str, tuple[int, int]]` name -> (taxon id, photos wanted), exactly:
    `'Euphorbia maculata': (58893, 2000), 'Euphorbia prostrata': (200053, 600), 'Portulaca oleracea': (58991, 490),
    'Taraxacum officinale': (47602, 490), 'Trifolium repens': (55745, 490), 'Plantago major': (58961, 490),
    'Plantago lanceolata': (53178, 490), 'Oxalis corniculata': (53168, 490), 'Polygonum aviculare': (1290817, 490),
    'Stellaria media': (1087592, 490), 'Lamium amplexicaule': (55459, 490), 'Medicago lupulina': (57056, 490),
    'Mollugo verticillata': (78090, 490)` (total 7,990; ids looked up 2026-11-25, 1290817 / 1087592 are complexes).
  - `INAT_KEEP_TAXA` same shape, 230 each: `Rosa 53438, Lavandula 61905, Hydrangea macrophylla 122034,
    Buxus sempervirens 133397, Tagetes 72389, Petunia 55406, Pelargonium 72281, Hibiscus × rosa-sinensis 1628098,
    Salvia 50644, Hemerocallis 82826, Hosta 82095, Juniperus 47574, Rhododendron 49487` (total 2,990).
  - `INAT_LICENCES = {'cc0': 'CC0', 'cc-by': 'CC BY', 'cc-by-nc': 'CC BY-NC'}`
  - `BG_MIDS` = Open Images tags Gravel /m/016b3m, Pebble /m/01tp0c, Rock /m/01cbzq, Soil /m/09dv7, Mulch /m/027t7q,
    Lawn /m/01v327, Grass /m/08t9c_, Cobblestone /m/02t2zd, Walkway /m/01jm2n, Concrete /m/01mxf (dict mid -> name).
  - `PLANT_MIDS` = Plant /m/05s2s, Flower /m/0c9ph5, Houseplant /m/03fp41, Tree /m/07j7r, Flowering plant /m/04sjm,
    Herb /m/03gq1, Groundcover /m/018ssc, Leaf /m/09t49, Flowerpot /m/0fm3zh, Garden /m/0bl0l, Moss /m/0184yy,
    Hedge /m/01xxn0, Vegetation /m/01fnns, plus every key of `OI_WEED_MIDS`.
  - `LICENCES['background'] = 'images CC BY 2.0 - Open Images V7 (Google), no boxes'`.
  - `inat_item(obs: dict, source: str) -> Item | None` (pure)
  - `oi_background_rows(csv_paths: list[str]) -> list[tuple[str, str, str]]` (image id, OI split, ';'-joined tag names)
  - `background_items(rows, max_images=2000) -> list[Item]`
- Produces (colab.py):
  - `inat_observations(taxon_id: int, n: int, quality_grade: str | None = 'research', captive: bool | None = None,
    get_json=None, sleep=time.sleep) -> list[dict]`
  - `inat_items_from_api(taxa: dict, source: str, quality_grade=..., captive=..., max_per_taxon: int | None = None,
    **kw) -> list[Item]`
  - `background_items_from_oi(tmp_dir: str, max_images=2000) -> list[Item]`

- [ ] **Step 1: Write the failing tests** in `test/weed_detector/test_v2.py` (fixture observation dicts shaped like
  the API: `{'id': 101, 'taxon': {'name': 'Euphorbia maculata'}, 'user': {'login': 'ann'},
  'photos': [{'id': 7, 'license_code': 'cc-by-nc', 'url': 'https://inaturalist-open-data.s3.amazonaws.com/photos/7/square.jpg'}]}`):
  - `test_inat_item_fields`: `it.name == '101_7.jpg'`, `it.src` ends `/photos/7/large.jpg`, `it.boxes == []`,
    `it.split == S.stable_split('inat101')`, `'CC BY-NC' in it.licence and 'ann' in it.licence and
    'https://www.inaturalist.org/observations/101' in it.licence`, `it.orig == 'Euphorbia maculata'`, `it.source == 'inat_weed'`.
  - `test_inat_item_rejects_bad_first_photo`: first photo licence `None`, `'cc-by-sa'`, `'cc-by-nd'`, or no photos
    -> `None` (even when a later photo is `cc0`).
  - `test_inat_observations_pages_by_id_and_stops`: fake `get_json(url)` serving two pages (2 obs, then 1 obs,
    then `[]`) records URLs; asserts the 2nd URL has `id_above=<last id of page 1>`, every URL has
    `photo_license=cc0,cc-by,cc-by-nc`, `order_by=id`, `order=asc`, `taxon_id=`; result length = 3 with n=10;
    fake `sleep` got >= 1.0 between calls; with n=2 only one call is made.
  - `test_inat_items_from_api_max_per_taxon`: `max_per_taxon=1` caps each taxon at 1 (fake get_json).
  - `test_oi_background_rows_filter` (tmp CSVs named `validation-annotations-human-imagelabels.csv` /
    `test-...`, columns `ImageID,Source,LabelName,Confidence`): Gravel=1 only -> kept with split `validation`;
    Gravel=1 + Plant=1 -> out; Gravel=1 + Plant=0 -> kept; Lawn=1 + Dandelion (/m/0fr19)=1 -> out; Gravel=0 -> out;
    a `test-` file row -> split `test`.
  - `test_background_items`: src `https://open-images-dataset.s3.amazonaws.com/validation/<id>.jpg`, boxes `[]`,
    `source == 'background'`, split = `S.stable_split(<id>)`, `max_images` caps, output order is stable (sorted by id).
- [ ] **Step 2: Run** `python3 -m pytest test/weed_detector/test_v2.py -q` -> FAIL (names missing).
- [ ] **Step 3: Implement** the sources.py names above, then the colab.py functions. `inat_observations`
  builds `https://api.inaturalist.org/v1/observations?taxon_id=..&photo_license=cc0,cc-by,cc-by-nc&per_page=200&order_by=id&order=asc`
  (+ `quality_grade` / `captive=true` when given, + `id_above` after the first page), keeps only observations
  for which `inat_item` is not None, stops at n or an empty page; the default `get_json` uses urllib with the
  User-Agent and the same retry rules as `build.fetch`. `inat_items_from_api` maps taxa -> Items (n = min(wanted,
  max_per_taxon)). `background_items_from_oi` downloads `S.OI_IMAGELABEL_URLS` into tmp_dir (skip if present) and
  returns `S.background_items(S.oi_background_rows(paths), max_images)`.
- [ ] **Step 4: Run** both test commands (Global Constraints) -> all pass.
- [ ] **Step 5: One live smoke call** (venv): `inat_observations(58893, 3)` returns 3 dicts whose `inat_item(...)`
  srcs download with `build.fetch` (HTTP 200, JPEG). Paste the 3 srcs in your report.
- [ ] **Step 6: Commit** `sources.py colab.py test/weed_detector/test_v2.py`: "training: iNaturalist and Open Images background sources (v2)".

### Task 2: The teacher (OWLv2 boxes for any folder of images)

**Files:**
- Create: `training/weed_detector/teacher.py`
- Modify: `training/weed_detector/make_notebooks.py` (add `'teacher.py'` to MODULES only)
- Test: `test/weed_detector/test_v2.py`

**Interfaces:**
- Consumes: `sources.WEED`, `sources.KEEP`.
- Produces (teacher.py):
  - `TEACHER_MODEL = 'google/owlv2-base-patch16-ensemble'`
  - `QUERIES: dict[str, dict[str, list[str]]]` kinds `weed` / `keep` / `background`, each `{'plant': [...], 'distract': [...]}`;
    initial values (Task 3 may change them): weed plant `['a weed', 'a small plant', 'a plant with small leaves']`,
    distract `['grass', 'a stone', 'soil', 'a human hand', 'a shoe', 'a flower', 'a shrub']`; keep plant
    `['a flower', 'a shrub', 'a flowering plant', 'a bush']`, distract `['grass', 'a stone', 'soil', 'a human hand',
    'a shoe', 'a weed', 'a flower pot']`; background plant `['a weed', 'a small plant', 'a flower', 'a shrub']`,
    distract `['grass', 'a stone', 'soil', 'gravel']`.
  - `THRESHOLDS = {'weed': 0.15, 'keep': 0.15, 'background': 0.12}`, `NMS_IOU = 0.5`, `MIN_BOX_AREA = 0.003`,
    `MAX_BOX_AREA = 0.6`, `RAW_MIN_SCORE = 0.05` (detections cached down to this score, so thresholds can change later
    without the GPU).
  - `Det = namedtuple('Det', 'query score x0 y0 x1 y1')` (query = the query text; coordinates relative 0..1).
  - `queries_for(kind) -> list[str]` (plant + distract, fixed order)
  - `pixel_box_to_rel(box_px, width, height) -> tuple` (clip to 0..1)
  - `select_boxes(dets, kind) -> list[tuple]` -> `(cls, cx, cy, w, h)`; cls = WEED for `weed`, KEEP for `keep`; `[]` for `background`.
  - `image_verdict(kind, boxes, dets) -> str | None` drop reason or None.
  - `class Owlv2Teacher: __init__(self, model_id=TEACHER_MODEL, device=None)`; `detect(self, image, queries, min_score=RAW_MIN_SCORE) -> list[Det]`
  - `label_images(image_paths, kind, teacher, cache_path, log_every=200) -> dict`
    returns `{'labelled': n, 'boxes': n, 'from_cache': n, 'dropped': {reason: n}}`

- [ ] **Step 1: Write the failing tests** (no model; a `FakeTeacher` with `detect(image, queries, min_score)` returning
  given Dets and counting calls):
  - `test_select_boxes_threshold_query_nms_area`: a weed-query det at 0.16 kept; at 0.14 dropped; a det whose query is a
    distractor (`'grass'`, score 0.9) dropped; two plant dets IoU 0.8 -> only the higher score kept; a det with area
    0.002 dropped; output tuple for box (0.1,0.2,0.3,0.6) == `(0, 0.2, 0.4, 0.2, 0.4)` (approx).
  - `test_image_verdicts`: weed with no boxes -> `'teacher found no plant'`; weed with a box of area 0.61 ->
    `'close-up'`; keep with one 0.3 box -> None; background with a plant det at score 0.12 -> `'plant found'`;
    background with only a `'grass'` det at 0.9 -> None.
  - `test_pixel_box_to_rel_non_square`: (100, 50, 300, 250) on 400 x 200 -> (0.25, 0.25, 0.75, 1.0) (clipped).
  - `test_label_images_writes_drops_and_resumes` (tmp dataset `train/images/*.jpg` + empty `train/labels/*.txt`):
    image A gets a weed box -> its label file has one line starting `0 `; image B gets none -> A kept, B's image AND
    label deleted, `dropped == {'teacher found no plant': 1}`; second call with a teacher whose detect raises ->
    same result, `from_cache == 1`, no exception.
  - `test_label_images_ignores_cut_last_cache_line`: cache file whose last line is `{"stem": "A", "w": 1` (no newline)
    -> A is re-detected (fake teacher called once) and the run completes.
  - `test_label_images_queries_change_invalidates_cache`: a cache line written with other queries is not reused.
  - `test_label_images_corrupt_image`: a .jpg containing `b'not a jpeg'` -> dropped as `'unreadable'`, others labelled.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement teacher.py.** Cache = one JSONL file, one line per image
  `{"stem", "w", "h", "queries": [...], "dets": [[query, score, x0, y0, x1, y1], ...]}`, appended and flushed every
  `log_every` images and at the end (a cut last line is skipped on load). `label_images` finds the label path by
  swapping the `images` folder for `labels` (build.py layout), overwrites the label file with `select_boxes`, deletes
  image + label on a verdict, prints progress with images per second. `Owlv2Teacher.detect`: `Owlv2Processor` +
  `Owlv2ForObjectDetection` from transformers, eval mode, `torch.no_grad()`, device = cuda if available; OWLv2 pads
  the image to a square at the bottom/right, so call `post_process_object_detection(outputs, threshold=min_score,
  target_sizes=[(side, side)])` with `side = max(w, h)`, then `pixel_box_to_rel(box, w, h)`; each box's label index is
  its best query. Add `'teacher.py'` to MODULES in make_notebooks.py (after `'build.py'`).
- [ ] **Step 4: Run** both test commands -> pass.
- [ ] **Step 5: Live check** (venv, CPU): `Owlv2Teacher().detect(<IMG_0230 lower-left quarter tile>,
  queries_for('weed'))` returns Dets, several with query `'a weed'` or `'a small plant'` and score >= 0.15; report the
  count and seconds per call. (Reading the owner's photo here is a smoke test only: change NO constant because of it.)
- [ ] **Step 6: Commit** `teacher.py make_notebooks.py test_v2.py`: "training: OWLv2 teacher labels image folders (v2)".

### Task 3: Teacher sample check on the VM (sets the prompts and thresholds)

**Files:**
- Create: `training/weed_detector/teacher_check.py` (VM tool: fetch sample, run teacher, contact sheets, summary)
- Modify: `training/weed_detector/teacher.py` (QUERIES / THRESHOLDS only, each changed value gets a comment with the reason)
- Modify: `docs/superpowers/specs/2026-11-25-weed-detector-v2-design.md` (append section "Teacher settings from the sample check (2026-11-25)")

**Interfaces:**
- Consumes: Task 1 `inat_items_from_api`, `background_items_from_oi`; `build.build`; Task 2 everything.
- Produces: `teacher_check.py` CLI `python -m weed_detector.teacher_check OUT_DIR [--per-taxon 3] [--background 30]`
  (run from `training/`), writing `OUT_DIR/{inat_weed,inat_keep,background}/...` (build layout), the caches
  `OUT_DIR/teacher_<source>.jsonl`, contact sheets `OUT_DIR/sheets/<source>_<n>.jpg` (12 thumbnails per sheet, boxes
  drawn: kept = green, below threshold = grey, distractor-best = red, text = query + score, dropped images marked),
  and `OUT_DIR/summary.json` (per source: images, kept, dropped by reason, boxes per kept image).

- [ ] **Step 1: Build the tool** (no unit test: it only wires tested pieces; keep it short). Use `OUT_DIR =
  ~/weedbot_data/teacher_check`. Sample = 3 photos per taxon from both iNat lists (`max_per_taxon=3`) + 30 background.
- [ ] **Step 2: Run it** in the venv (CPU, ~100 images x ~20-40 s: run it in the background and wait).
- [ ] **Step 3: Look at every contact sheet** (Read the jpgs). For each source write down: images where the kept
  boxes are right / wrong / missing a plant, background images kept although a plant is visible.
- [ ] **Step 4: Adjust** QUERIES / THRESHOLDS in teacher.py if needed and re-run (threshold-only changes re-use the
  caches: no new teacher calls). Targets: >= 80 % of KEPT weed / keep images have boxes that sit on the right plants
  (grass never boxed as weed); at most 1 in 10 KEPT background images shows a plant; dropping many images is fine
  (there are thousands more). NEVER use the owner's photos for this.
- [ ] **Step 5: Record** in the design doc section: final queries, thresholds, the sample numbers (kept / dropped by
  reason, right / wrong counts) and one line per change with its reason.
- [ ] **Step 6: Run both test commands** -> pass (tests must not assert the tuned values except through the constants).
- [ ] **Step 7: Commit** `teacher_check.py teacher.py <design doc>`: "training: teacher sample check sets prompts and thresholds".

### Task 4: Parts on Drive: images parts, teacher labelling, v2 merge filters

**Files:**
- Modify: `training/weed_detector/colab.py`
- Test: `test/weed_detector/test_v2.py`

**Interfaces:**
- Consumes: Task 2 `label_images`, `QUERIES`; existing `build_part`, `merge_parts`, `repair_after_merge`, `summary`.
- Produces (colab.py):
  - `V2_SOURCES = ('cottonweed', 'grassweeds', 'openimages', 'inat_weed', 'inat_keep', 'background')`
  - `TEACHER_SOURCES = {'inat_weed': 'weed', 'inat_keep': 'keep', 'background': 'background'}` (source -> teacher kind)
  - `OI_MAX_BOX_AREA = 0.4`
  - `build_part(source, items_fn, work_dir, drive_dir, zip_name=None, **build_kw)` (new keyword; default `zip_name = source`;
    v1 behaviour unchanged)
  - `label_part(source, work_dir, drive_dir, teacher_fn) -> dict` (`teacher_fn()` builds the teacher; called only if
    some image is not in the cache)
  - `drop_big_box_images(out_dir, source='openimages', max_area=OI_MAX_BOX_AREA) -> int`
  - `_rewrite_manifests(out_dir)` (factored out of `repair_after_merge`, which now calls it)

- [ ] **Step 1: Write the failing tests:**
  - `test_build_part_zip_name`: `zip_name='inat_weed_images'` -> `dataset_parts/inat_weed_images.zip` exists,
    `inat_weed.zip` does not.
  - `test_label_part_from_images_zip` (fake teacher): images zip with 2 images -> `dataset_parts/inat_weed.zip` holds the
    kept image, its non-empty label and `manifest_inat_weed.csv` whose rows match the kept images with the right
    `n_boxes`; the cache `dataset_parts/teacher_inat_weed.jsonl` is on the (fake) Drive.
  - `test_label_part_skips_when_labelled_exists`: labelled zip present -> returns `{'skipped': True}`, teacher_fn never
    called.
  - `test_label_part_missing_images_part`: raises RuntimeError whose message contains `03_v2_collect_photos`.
  - `test_drop_big_box_images`: openimages image with a 0.5-area box -> image + label removed; one with a 0.3 box kept;
    a cottonweed image with a 0.5 box untouched; manifests rewritten (dropped row gone); returns 1.
  - `test_repair_after_merge` (existing) still passes.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement.** `label_part`: if `<source>.zip` exists -> skip; unzip `<source>_images.zip` into
  `work_dir/<source>`; `label_images` over all `*/images/*.jpg` with kind `TEACHER_SOURCES[source]` and cache
  `dataset_parts/teacher_<source>.jsonl` on Drive; rewrite `manifest_<source>.csv` (n_boxes from the label files, rows of
  deleted images removed); zip to `<source>.zip` via `.part` + `os.replace` (like build_part); print counts.
- [ ] **Step 4: Run** both test commands -> pass.
- [ ] **Step 5: Commit** `colab.py test_v2.py`: "training: v2 parts - images parts, teacher labelling, big-box filter".

### Task 5: Backyard check tool and frozen reference (VM only)

**Files:**
- Create: `training/weed_detector/backyard_check.py`
- Test: `test/weed_detector/test_v2.py`

**Interfaces:**
- Consumes: Task 2 `Owlv2Teacher`, `queries_for`.
- Produces (backyard_check.py; boxes are `(x0, y0, x1, y1)` in full-photo pixels):
  - `quarter_tiles(img) -> list[tuple[int, int, PIL.Image]]` (x, y offset, tile)
  - `letterbox_square(img, size=640) -> PIL.Image` (scaled, padded grey 114 at bottom/right)
  - `is_hit(pred, ref) -> bool` (pred centre inside ref, or IoU >= 0.3)
  - `backyard_verdict(per_photo: dict) -> dict` where `per_photo[name] = {'ref': [...], 'weed': [(box, conf)],
    'all': [(box, conf, cls)], 'views': [(view_w, view_h, [(box, conf, cls)])]}`; returns per photo
    `hit_rate`, `hits`, `refs`, `stone_boxes` (any class, conf >= 0.25, centre in no ref box and IoU < 0.1 with all
    refs), `oversize` (boxes > 50 % of their view), `pass`; plus pooled hit rate and overall `pass`
    (strict: every photo >= 0.70, <= 2 stone boxes per photo, no oversize box).
  - `fidelity(pt_boxes, ncnn_boxes) -> dict` (each pt box conf >= 0.25 has an ncnn box of the same class, IoU >= 0.9,
    |conf diff| <= 0.05, and the reverse; `pass` bool)
  - CLI: `reference OUT_DIR` (teacher on quarter tiles of IMG_0229 / IMG_0230 with `queries_for('weed')`,
    threshold 0.10 -> `OUT_DIR/candidates.json` + numbered drawings), `check MODEL_DIR REFERENCE_JSON` (runs `best.pt`
    on the 4 quarter tiles and the whole photo, each letterboxed to 640, conf 0.25; runs `best_ncnn_model` on the same
    inputs for `fidelity`; writes `MODEL_DIR/backyard_check/report.json` + drawings).

- [ ] **Step 1: Write the failing tests** for `is_hit` (centre inside, IoU 0.3, miss), `backyard_verdict` (one photo
  7/10 hits passes, 6/10 fails; 3 stone boxes fails; a box covering 0.51 of its view fails; a photo passing and one
  failing -> overall fail), `letterbox_square` (400 x 200 -> 640 x 640, content 640 x 320 at the top), `fidelity`
  (identical lists pass; a 0.06 conf gap fails; an extra ncnn box fails).
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** both test commands -> pass.
- [ ] **Step 5: Make the reference** (venv): `reference ~/weedbot_data/backyard_photos/reference`. Read every numbered
  drawing; write `~/weedbot_data/backyard_photos/reference/reference.json` = `{photo: [[x0, y0, x1, y1], ...]}` with only
  the candidates that sit on a weed (spurge sprigs or the green broadleaf weed); record its sha256 in your report.
  It is FROZEN from now on (the design forbids changing it after seeing any v2 output).
- [ ] **Step 6: Baseline**: `check ~/weedbot_data/models/weed_detector_v1 <reference.json>` -> the report must say
  FAIL (v1 is known to fail); paste hit rates, stone and oversize counts, and the fidelity result.
- [ ] **Step 7: Commit** ONLY `backyard_check.py test_v2.py` (never the reference or drawings):
  "training: backyard check tool (v2 acceptance)".

### Task 6: v2 notebooks, owner README, VM dry run

**Files:**
- Modify: `training/weed_detector/make_notebooks.py`, `training/weed_detector/train.py` (only if needed),
  `training/weed_detector/README.md`, `test/weed_detector/dryrun_notebooks.py`
- Create (generated): `training/weed_detector/03_v2_collect_photos.ipynb`, `04_v2_label_and_train.ipynb`,
  `04b_v2_finish_without_gpu.ipynb`

**Interfaces:**
- Consumes: all of Tasks 1, 2, 4; existing `T.train`, `T.final_weights`, `T.evaluate_per_source`, `T.evaluate_all`,
  `T.export`, the `MERGE` / `EVALUATE` / `package()` cell builders.
- Produces: notebooks; `package(weights_dir, note, run='v1', zip_name='weed_detector_v1')` (v1 cells unchanged in
  behaviour); dry-run steps `3 4 4b`.

- [ ] **Step 1: Generalise the cell builders** in make_notebooks.py: EVALUATE and package() take the run name and zip
  name; a v2 MERGE cell = `C.merge_parts(DRIVE, DS, sources=C.V2_SOURCES)`, `C.repair_after_merge(DS)`,
  `C.drop_big_box_images(DS)`, `C.summary(DS)`, with `DS = '/content/weeds_ds_v2'`. Regenerate and check the v1
  notebooks' code cells are unchanged except whitespace (diff as in the 2b change).
- [ ] **Step 2: NB `03_v2_collect_photos.ipynb` (CPU):** markdown with owner steps; mount; modules; `!pip -q install
  huggingface_hub`; three cells `C.build_part('inat_weed', lambda: C.inat_items_from_api(S.INAT_WEED_TAXA, 'inat_weed',
  quality_grade='research', max_per_taxon=None), WORK, DRIVE, zip_name='inat_weed_images', workers=8)`, `inat_keep` with
  `S.INAT_KEEP_TAXA`, `quality_grade=None, captive=True`, `background` with `C.background_items_from_oi(WORK + '/oi',
  max_images=2000)`; a DONE cell listing the three `_images` parts and their MB, `drive.flush_and_unmount()`.
- [ ] **Step 3: NB `04_v2_label_and_train.ipynb` (T4):** markdown; mount + GPU assert; modules; `!pip -q install
  ultralytics transformers`; a labelling cell that calls `C.label_part(s, '/content/work_v2', DRIVE, lambda:
  TE.Owlv2Teacher())` for each of `C.TEACHER_SOURCES` inside a function, then frees the GPU (`del`, `gc.collect()`,
  `torch.cuda.empty_cache()`); the v2 MERGE cell; `base, _ = T.final_weights(DRIVE + '/runs', 'v1', '/content/v1_final')`;
  `best = T.train(DS + '/data.yaml', DRIVE + '/runs', name='v2', base=str(base), epochs=50, patience=15, imgsz=640)`;
  EVALUATE (run v2); package(run 'v2', zip 'weed_detector_v2'). Re-running after a stop must skip finished labelling and
  resume training.
- [ ] **Step 4: NB `04b_v2_finish_without_gpu.ipynb` (CPU):** like `02b` but the v2 MERGE cell (needs the labelled parts),
  `T.final_weights(DRIVE + '/runs', 'v2', '/content/final')`, EVALUATE + package for v2. `02b` itself unchanged.
- [ ] **Step 5: README.md**: an owner section "Version 2" with the three notebooks in order (03 CPU, 04 T4 GPU, 04b only
  if 04 stops near the end with "Cannot connect to GPU backend"), what each prints at the end, Drive space (~3 GB more),
  and that v1 files on Drive must stay; developer notes for teacher.py, teacher_check.py, backyard_check.py.
- [ ] **Step 6: Dry run** (`dryrun_notebooks.py` steps `3 4 4b`, venv, CPU, internet): adapt like v1 (fake Drive
  `~/weedbot_data/dryrun/drive/weedbot`, which already holds the v1 dry-run parts and `runs/v1`); limits
  `max_per_taxon=1`, `max_images=10` for background; teacher on CPU; training `epochs=1, imgsz=320, batch=8,
  device='cpu', workers=0, fraction=0.05`; export at 320; 4b on a v2 run stopped after round 2 of 3 (as 2b's dry run).
  Expected: `03` prints the DONE line with 3 parts; `04` labels 3 parts, trains, evaluates, prints
  `ALL DONE -> MyDrive/weedbot/weed_detector_v2.zip`; a second `04` run prints the 3 parts as skipped; `4b` prints its
  ALL DONE. Also re-run steps `2b` to prove v1 still works.
- [ ] **Step 7: Copy** the three new notebooks and README (as HOW_TO.md) to `~/Desktop/WeedBot_training/`.
- [ ] **Step 8: Run both test commands** -> pass. **Commit** make_notebooks.py, train.py (if changed), README.md,
  dryrun_notebooks.py and all `.ipynb` in `training/weed_detector/`: "training: v2 notebooks (collect, label + train,
  finish without GPU)".
