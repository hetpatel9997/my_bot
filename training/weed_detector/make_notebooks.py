"""make_notebooks.py - write the self-contained Colab notebooks from the tested modules in this folder.

v1: 01_build_dataset, 02_train, 02b_finish_without_gpu. v2: 03_v2_collect_photos, 04_v2_label_and_train,
04b_v2_finish_without_gpu.
Why self-contained: the owner uploads them to Colab from the VM; nothing has to be pushed to GitHub first.
Run after changing sources.py / build.py / teacher.py / colab.py / train.py:
    python3 training/weed_detector/make_notebooks.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ['__init__.py', 'sources.py', 'build.py', 'teacher.py', 'colab.py', 'train.py']
GPU_NOTEBOOKS = ('02_train.ipynb', '04_v2_label_and_train.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


def fill(template, **subs):
    """Put each value in place of its UPPER_CASE key in `template`, in one pass (a value is never filled again)."""
    return re.sub('|'.join(rf'\b{k}\b' for k in subs), lambda m: subs[m.group(0)], template)


def module_cells():
    cells = []
    for m in MODULES:
        with open(os.path.join(HERE, m)) as f:
            body = f.read()
        cells.append(code(f'%%writefile /content/weed_detector/{m}\n{body}'))
    return [code('import os\nos.makedirs("/content/weed_detector", exist_ok=True)')] + cells


MOUNT = '''
from google.colab import drive
drive.mount('/content/drive')
import os, sys
DRIVE = '/content/drive/MyDrive/weedbot'
os.makedirs(DRIVE, exist_ok=True)
sys.path.insert(0, '/content')
'''

GPU_CHECK = '''
import torch
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU (if none is free, try later)'
print('GPU:', torch.cuda.get_device_name(0))
'''

NB1 = [
    md('''
# WeedBot weed detector - step 1 of 2: collect the training photos
Downloads ~11,000 public photos (weeds + plants to keep), converts them to two classes (`weed`, `keep`) and
stores the result in your Google Drive (`MyDrive/weedbot/dataset_parts/`, about 2.5 GB).

**How:** Runtime -> Change runtime type -> **CPU** is enough here (keeps your free GPU time for step 2),
then Runtime -> **Run all**. Allow Google Drive access when asked. Takes roughly 30-90 minutes.
Keep the laptop awake (lid open) while it runs. If it stops, just press **Run all** again: finished parts
are skipped (a part that was half done starts that part again).

Sources and licences: CottonWeedDet12 (CC BY-NC 4.0), Roboflow-100 grass-weeds (CC BY 4.0),
Open Images V7 (CC BY 2.0 images / CC BY 4.0 boxes).
'''),
    code(MOUNT),
] + module_cells() + [
    code('!pip -q install huggingface_hub fiftyone'),
    code('''
from weed_detector import colab as C, sources as S
WORK = '/content/work'
C.build_part('cottonweed', lambda: C.cottonweed_items_from_hf(WORK), WORK, DRIVE, workers=8)
'''),
    code('''
C.build_part('grassweeds', lambda: C.hf_yolo_items('LibreYOLO/grass-weeds', 'grassweeds', ['ridderzuring'],
             {'ridderzuring': S.WEED}, WORK + '/src_grassweeds'), WORK, DRIVE, workers=8)
'''),
    code('''
C.build_part('openimages', lambda: C.openimages_items_from_zoo(WORK + '/oi', max_images=3000), WORK, DRIVE, workers=8)
'''),
    code('''
done = [s for s in C.EXPECTED_SOURCES if os.path.exists(C.part_zip(DRIVE, s))]
for s in done:
    print(s, round(os.path.getsize(C.part_zip(DRIVE, s)) / 1e6), 'MB')
from google.colab import drive
drive.flush_and_unmount()          # make sure everything is really written to Google Drive
print(f'DONE: all {len(done)} parts are on Drive, go to step 2' if len(done) == len(C.EXPECTED_SOURCES)
      else f'Not all parts finished ({len(done)} of {len(C.EXPECTED_SOURCES)}): press Run all again')
'''),
]

MERGE = '''
from weed_detector import colab as C, train as T
DS = '/content/weeds_ds'
info = C.merge_parts(DRIVE, DS)        # only cottonweed, grassweeds, openimages (an old cropsweeds part is ignored)
print('repair of parts from an older step 1:', C.repair_after_merge(DS))   # no-op for current parts
info = C.summary(DS)
print(info)
'''

MERGE_V2 = '''
from weed_detector import colab as C, train as T
DS = '/content/weeds_ds_v2'
info = C.merge_parts(DRIVE, DS, sources=C.V2_SOURCES)   # v1's three parts + the three labelled v2 parts
print('repair of parts from an older step 1:', C.repair_after_merge(DS))   # no-op for current parts
print('Open Images pictures with a box over 40 % of the picture, removed:', C.drop_big_box_images(DS))
info = C.summary(DS)
print(info)
'''

EVALUATE = '''
import json
per_source = T.evaluate_per_source(best, DS, split='test', imgsz=640)
overall = T.evaluate_all(best, DS + '/data.yaml', DRIVE + '/runs/RUN_NAME', split='test', imgsz=640)
print(json.dumps({'overall_test': overall, 'per_source_test': per_source}, indent=1))
json.dump({'overall_test': overall, 'per_source_test': per_source, 'dataset': info},
          open(DRIVE + '/runs/RUN_NAME/metrics_per_source.json', 'w'), indent=1)
'''


def evaluate(run='v1'):
    """The test cell for the training run `run` (MyDrive/weedbot/runs/<run>)."""
    return code(fill(EVALUATE, RUN_NAME=run))


# README.txt of the result zip: (first line, sources and licence lines), as Python expressions for the cell.
README_V1 = (
    r"""'WeedBot weed/keep detector v1 (YOLO11n, classes 0 weed, 1 keep, imgsz 640).\n'""",
    r"""'Trained on: CottonWeedDet12 (CC BY-NC 4.0, Lu et al., Zenodo 10.5281/zenodo.7535814), Roboflow-100 grass-weeds '
    '(CC BY 4.0), Open Images V7 (CC BY 2.0 images, CC BY 4.0 boxes).\n'
    'Non-commercial use (CottonWeedDet12 licence).\n'""")
README_V2 = (
    r"""'WeedBot weed/keep detector v2 (YOLO11n fine-tuned from v1, classes 0 weed, 1 keep, imgsz 640).\n'""",
    r"""'Trained on:\n'
    '- CottonWeedDet12 (CC BY-NC 4.0, Lu et al., Zenodo 10.5281/zenodo.7535814)\n'
    '- Roboflow-100 grass-weeds (CC BY 4.0)\n'
    '- Open Images V7 (CC BY 2.0 images, CC BY 4.0 boxes): keep plants and the "nothing here" ground photos\n'
    '- iNaturalist photos: each photo has its own licence (CC0, CC BY or CC BY-NC); the observer and the URL of '
    'every photo are in manifest_inat_weed.csv / manifest_inat_keep.csv\n'
    'The boxes on the iNaturalist photos were drawn by OWLv2 (google/owlv2-base-patch16-ensemble, Apache 2.0), '
    'which also checked that the ground photos show no plant.\n'
    'Non-commercial use (CottonWeedDet12 and the CC BY-NC photos).\n'""")


def package(weights_dir, note, run='v1', zip_name='weed_detector_v1', readme=README_V1):
    """The export + zip cell. `weights_dir`: Python expression for the folder holding best.pt (exports land
    next to it); `note`: Python expression for an extra README line; `run`: the training run (runs/<run> on
    Drive); `zip_name`: the result, MyDrive/weedbot/<zip_name>.zip; `readme`: (first line, sources and licence
    lines) of its README.txt, as Python expressions."""
    title, sources = readme
    return code(fill('''
import glob, shutil
exported = T.export(best, imgsz=640, formats=('onnx', 'ncnn'))
print('exported:', exported)
out = '/content/ZIP_NAME'
os.makedirs(out, exist_ok=True)
run = DRIVE + '/runs/RUN_NAME'
W = WEIGHTS_DIR
for f in ['results.csv', 'confusion_matrix.png', 'results.png', 'metrics_per_source.json', 'args.yaml']:
    if os.path.exists(os.path.join(run, f)):
        shutil.copy(os.path.join(run, f), out)
for f in ['best.pt', 'best.onnx']:
    if os.path.exists(os.path.join(W, f)):
        shutil.copy(os.path.join(W, f), out)
if os.path.exists(os.path.join(run, 'test_eval', 'confusion_matrix.png')):
    shutil.copy(os.path.join(run, 'test_eval', 'confusion_matrix.png'), os.path.join(out, 'test_confusion_matrix.png'))
ncnn = os.path.join(W, 'best_ncnn_model')
if os.path.isdir(ncnn):
    shutil.copytree(ncnn, os.path.join(out, 'best_ncnn_model'), dirs_exist_ok=True)
for src in sorted(glob.glob(DS + '/manifest_*.csv')):
    shutil.copy(src, out)
open(os.path.join(out, 'README.txt'), 'w').write(
    README_TITLE + NOTE +
    README_SOURCES)
shutil.make_archive(DRIVE + '/ZIP_NAME', 'zip', out)
size = round(os.path.getsize(DRIVE + '/ZIP_NAME.zip') / 1e6, 1)
from google.colab import drive
drive.flush_and_unmount()          # make sure the zip is really written to Google Drive
print('ALL DONE -> MyDrive/weedbot/ZIP_NAME.zip', size, 'MB')
''', WEIGHTS_DIR=weights_dir, NOTE=note, RUN_NAME=run, ZIP_NAME=zip_name, README_TITLE=title, README_SOURCES=sources))


FINAL = '''
best, stop = T.final_weights(DRIVE + '/runs', 'RUN_NAME', '/content/final')
print(f"training stopped after round {stop['last_epoch']} of {stop['epochs']}; best model is from round {stop['best_epoch']}")
'''
STOPPED_NOTE = ("f\"Training stopped after round {stop['last_epoch']} of {stop['epochs']} (Colab GPU limit); \"\n    "
                "f\"best.pt is the best round so far ({stop['best_epoch']}).\\n\"")

NB2 = [
    md('''
# WeedBot weed detector - step 2 of 2: train
Trains the small YOLO11n detector on the photos from step 1 and saves the result to your Google Drive as
`MyDrive/weedbot/weed_detector_v1.zip`.

**How:** Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> **Run all**. Allow Google Drive access.
Training takes a few hours; keep the laptop awake. Free Colab sessions stop after a while: when that
happens, open this notebook again and press **Run all** - it continues from the last saved epoch
(checkpoints are on Drive). If Colab says the GPU is not available, try again later (do not switch to CPU).
When it prints **ALL DONE**, download `weed_detector_v1.zip` from Drive (MyDrive/weedbot) and tell Claude.
'''),
    code(MOUNT + GPU_CHECK),
] + module_cells() + [
    code('!pip -q install ultralytics'),
    code(MERGE),
    code('''
best = T.train(DS + '/data.yaml', DRIVE + '/runs', name='v1', epochs=100, imgsz=640)
print('best model:', best)
'''),
    evaluate('v1'),
    package("run + '/weights'", "''"),
]

NB2B = [
    md('''
# WeedBot weed detector - step 2b: finish WITHOUT a GPU
Use this only when step 2 stopped near the end (for example round 97 of 100) and Colab says
"Cannot connect to GPU backend". It takes the best model step 2 already saved on your Drive, tests it,
makes the Pi versions and writes `MyDrive/weedbot/weed_detector_v1.zip`, all on a normal CPU.
It does not train and does not change the saved training run: step 2 can still finish it later with a GPU.

**How:** Runtime -> Change runtime type -> **CPU** -> Save, then Runtime -> **Run all**. Allow Google Drive
access. Takes about 30-60 minutes; keep the laptop awake. When it prints **ALL DONE**, download
`weed_detector_v1.zip` from Drive (MyDrive/weedbot) and tell Claude.
'''),
    code(MOUNT + '''
assert os.path.exists(DRIVE + '/runs/v1/weights/best.pt'), 'No saved model yet: run step 2 (02_train) first'
'''),
] + module_cells() + [
    code('!pip -q install ultralytics'),
    code(MERGE),
    code(fill(FINAL, RUN_NAME='v1')),
    evaluate('v1'),
    package("'/content/final'", STOPPED_NOTE),
]

NB3 = [
    md('''
# WeedBot weed detector v2 - step 3: collect the new photos
Downloads ~20,000 public photos that have no boxes yet: weeds (~12,000) and garden plants (~6,000) from
iNaturalist, and ~2,000 "nothing here" ground photos (gravel, soil, lawn) from Open Images. They are stored in
your Google Drive (`MyDrive/weedbot/dataset_parts/`): about 4 GB of the ~5.3 GB that version 2 needs (keep 6 GB
free to be safe). Step 4 draws the boxes and trains.

**How:** Runtime -> Change runtime type -> **CPU** (no GPU needed here: it keeps your free GPU time for step 4),
then Runtime -> **Run all**. Allow Google Drive access when asked. Takes roughly 1-3 hours.
Keep the laptop awake (lid open) while it runs. If it stops, just press **Run all** again: finished parts
are skipped. Keep the v1 files on Drive: version 2 uses them.
When it prints **DONE: all 3 photo parts are on Drive**, go to step 4 (`04_v2_label_and_train.ipynb`).

Licences: iNaturalist photos CC0 / CC BY / CC BY-NC (each photo's licence, author and link are saved with it),
Open Images V7 CC BY 2.0.
'''),
    code(MOUNT),
] + module_cells() + [
    code('!pip -q install huggingface_hub'),
    code('''
from weed_detector import colab as C, sources as S
WORK = '/content/work'
C.build_part('inat_weed', lambda: C.inat_items_from_api(S.INAT_WEED_TAXA, 'inat_weed', quality_grade='research',
             max_per_taxon=None), WORK, DRIVE, zip_name='inat_weed_images', workers=8)
'''),
    code('''
C.build_part('inat_keep', lambda: C.inat_items_from_api(S.INAT_KEEP_TAXA, 'inat_keep', quality_grade=None, captive=True,
             max_per_taxon=None), WORK, DRIVE, zip_name='inat_keep_images', workers=8)
'''),
    code('''
C.build_part('background', lambda: C.background_items_from_oi(WORK + '/oi', max_images=2000), WORK, DRIVE,
             zip_name='background_images', workers=8)
'''),
    code('''
parts = [s + '_images' for s in C.TEACHER_SOURCES]
done = [p for p in parts if os.path.exists(C.part_zip(DRIVE, p))]
for p in done:
    print(p, round(os.path.getsize(C.part_zip(DRIVE, p)) / 1e6), 'MB')
from google.colab import drive
drive.flush_and_unmount()          # make sure everything is really written to Google Drive
print(f'DONE: all {len(done)} photo parts are on Drive, go to step 4' if len(done) == len(parts)
      else f'Not all parts finished ({len(done)} of {len(parts)}): press Run all again')
'''),
]

NB4 = [
    md('''
# WeedBot weed detector v2 - step 4: label the new photos, then train v2
First a big model (OWLv2, the "teacher") draws a box around the plant in each photo from step 3 (about 3-4.5
hours of T4 time). Then the small detector is trained again, starting from v1, on all the photos (about 4-5 hours).
The result goes to your Google Drive as `MyDrive/weedbot/weed_detector_v2.zip`.

**How:** Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> **Run all**. Allow Google Drive access.
Keep the laptop awake. This needs more GPU time than free Colab gives in one day: plan for 2-3 days (the first
day is mostly labelling).
When Colab stops, come back later, open this notebook again and press **Run all**: finished labelling is kept
on Drive (it prints `already labelled on Drive, skipped`) and training continues from its last round. After each
restart it first unpacks and prepares the photos for about 10-20 minutes before training goes on: that is
normal, it is not stuck.
The install cell may print red warnings about other packages ("dependency conflicts"): expected, ignore them.
If Colab asks you to restart the session, click Restart session, then Runtime -> **Run all** again. If it stops
with `transformers 4.57.6 is not active`, do what it says (Runtime > Restart session, then Runtime > Run all);
if it stops there again, tell Claude.
If Colab says the GPU is not available, try again later (do not switch to CPU).
When it prints **ALL DONE**, download `weed_detector_v2.zip` from Drive (MyDrive/weedbot) and send it to Claude.
The first zip you send Claude is the final v2 model: do not run step 4 or 4b again after that.
If it stopped near the end (for example round 47 of 50) and Colab says "Cannot connect to GPU backend",
use step 4b (`04b_v2_finish_without_gpu.ipynb`) instead.
'''),
    code(MOUNT + GPU_CHECK + '''
assert os.path.exists(DRIVE + '/runs/v1/weights/best.pt'), 'v1 must be trained first; keep the v1 files on Drive'
'''),
] + module_cells() + [
    code('''
!pip -q install ultralytics==8.4.218
# The teacher must be transformers 4.57.6: its prompts and thresholds were tuned with that version on the VM
# (5.x pads the photos black instead of grey, which moves the scores a little). 4.57.6 does not need torch, so
# Colab's torch stays. Nothing may import transformers before this cell.
import importlib.metadata


def version_of(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


torch_before = version_of('torch')
if version_of('transformers') != '4.57.6':
    !pip -q install transformers==4.57.6
assert version_of('torch') == torch_before, 'pip changed torch: Runtime -> Restart session, then Run all again'
from weed_detector import teacher as TE
transformers_version = TE.require_transformers()      # STOPS here unless 4.57.6 is installed and active
print('transformers', transformers_version, '| ultralytics', version_of('ultralytics'), '| torch', torch_before)
'''),
    code('''
import gc, json, torch
from weed_detector import colab as C, teacher as TE
print('teacher runs on transformers', TE.require_transformers())   # stops unless 4.57.6 (the tuned teacher)


def label_all():
    """Boxes for each photo part. label_part builds the teacher only when some photo is not cached yet and drops
    it when it returns: no name holds the teacher afterwards, so there is nothing left to del."""
    return {s: C.label_part(s, '/content/work_v2', DRIVE, lambda: TE.Owlv2Teacher()) for s in C.TEACHER_SOURCES}


labelled = label_all()          # a part already labelled on Drive is skipped
gc.collect()
torch.cuda.empty_cache()        # the teacher's GPU memory goes back for the training
print(json.dumps(labelled, indent=1))
'''),
    code(MERGE_V2),
    code('''
base, _ = T.final_weights(DRIVE + '/runs', 'v1', '/content/v1_final')     # v2 starts from v1's best model
best = T.train(DS + '/data.yaml', DRIVE + '/runs', name='v2', base=str(base), epochs=50, patience=15, imgsz=640)
print('best model:', best)
'''),
    evaluate('v2'),
    package("run + '/weights'", "''", run='v2', zip_name='weed_detector_v2', readme=README_V2),
]

NB4B = [
    md('''
# WeedBot weed detector v2 - step 4b: finish WITHOUT a GPU
Use this only when step 4 stopped near the end (for example round 47 of 50) and Colab says
"Cannot connect to GPU backend". It takes the best model step 4 already saved on your Drive, tests it,
makes the Pi versions and writes `MyDrive/weedbot/weed_detector_v2.zip`, all on a normal CPU.
It does not train and does not change the saved training run.

**How:** Runtime -> Change runtime type -> **CPU** -> Save, then Runtime -> **Run all**. Allow Google Drive
access. Takes about 30-60 minutes; keep the laptop awake. When it prints **ALL DONE**, download
`weed_detector_v2.zip` from Drive (MyDrive/weedbot) and send it to Claude. The first zip you send Claude is the
final v2 model, also when it comes from this notebook: do not run step 4 or 4b again after that.
'''),
    code(MOUNT + '''
assert os.path.exists(DRIVE + '/runs/v2/weights/best.pt'), 'No saved v2 model yet: run step 4 (04_v2_label_and_train) first'
'''),
] + module_cells() + [
    code('!pip -q install ultralytics==8.4.218'),
    code(MERGE_V2),
    code(fill(FINAL, RUN_NAME='v2')),
    evaluate('v2'),
    package("'/content/final'", STOPPED_NOTE, run='v2', zip_name='weed_detector_v2', readme=README_V2),
]


def write(cells, name):
    nb = {'cells': cells, 'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU' if name in GPU_NOTEBOOKS else 'None'},
          'nbformat': 4, 'nbformat_minor': 0}
    path = os.path.join(HERE, name)
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    return path


if __name__ == '__main__':
    print(write(NB1, '01_build_dataset.ipynb'))
    print(write(NB2, '02_train.ipynb'))
    print(write(NB2B, '02b_finish_without_gpu.ipynb'))
    print(write(NB3, '03_v2_collect_photos.ipynb'))
    print(write(NB4, '04_v2_label_and_train.ipynb'))
    print(write(NB4B, '04b_v2_finish_without_gpu.ipynb'))
