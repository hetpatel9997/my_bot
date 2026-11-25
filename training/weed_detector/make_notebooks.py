"""make_notebooks.py - write the two self-contained Colab notebooks from the tested modules in this folder.

Why self-contained: the owner uploads them to Colab from the VM; nothing has to be pushed to GitHub first.
Run after changing sources.py / build.py / colab.py / train.py:
    python3 training/weed_detector/make_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ['__init__.py', 'sources.py', 'build.py', 'colab.py', 'train.py']


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


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

NB1 = [
    md('''
# WeedBot weed detector - step 1 of 2: collect the training photos
Downloads ~12,000 public photos (weeds + plants to keep), converts them to two classes (`weed`, `keep`) and
stores the result in your Google Drive (`MyDrive/weedbot/dataset_parts/`, about 3 GB).

**How:** Runtime -> Change runtime type -> **CPU** is enough here (keeps your free GPU time for step 2),
then Runtime -> **Run all**. Allow Google Drive access when asked. Takes roughly 30-90 minutes.
If it stops, just press **Run all** again: finished parts are skipped.

Sources and licences: CottonWeedDet12 (CC BY-NC 4.0), Roboflow-100 grass-weeds (CC BY 4.0),
CropWeeds-YOLO (MIT), Open Images V7 (CC BY 2.0 images / CC BY 4.0 boxes).
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
C.build_part('cropsweeds', lambda: C.hf_yolo_items('toolambr/crops_weeds', 'cropsweeds', ['crop', 'weed'],
             {'crop': S.KEEP, 'weed': S.WEED}, WORK + '/src_cropsweeds'), WORK, DRIVE, workers=8)
'''),
    code('''
C.build_part('openimages', lambda: C.openimages_items_from_zoo(WORK + '/oi', max_images=3000), WORK, DRIVE, workers=8)
'''),
    code('''
import glob
parts = sorted(glob.glob(DRIVE + '/dataset_parts/*.zip'))
for p in parts:
    print(os.path.basename(p), round(os.path.getsize(p) / 1e6), 'MB')
print('DONE: all 4 parts are on Drive, go to step 2' if len(parts) == 4 else 'Not all parts finished: Run all again')
'''),
]

NB2 = [
    md('''
# WeedBot weed detector - step 2 of 2: train
Trains the small YOLO11n detector on the photos from step 1 and saves the result to your Google Drive as
`MyDrive/weedbot/weed_detector_v1.zip`.

**How:** Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> **Run all**. Allow Google Drive access.
Training takes a few hours. Free Colab sessions stop after a while: when that happens, open this notebook
again and press **Run all** - it continues from the last saved epoch (checkpoints are on Drive).
When it prints **ALL DONE**, download `weed_detector_v1.zip` from Drive (MyDrive/weedbot) and tell Claude.
'''),
    code(MOUNT + '''
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - set Runtime type to T4 GPU!')
'''),
] + module_cells() + [
    code('!pip -q install ultralytics'),
    code('''
from weed_detector import colab as C, train as T
DS = '/content/weeds_ds'
info = C.merge_parts(DRIVE, DS)
print(info)
assert len(os.listdir(DRIVE + '/dataset_parts')) >= 4, 'run step 1 (01_build_dataset) until all 4 parts are on Drive'
'''),
    code('''
best = T.train(DS + '/data.yaml', DRIVE + '/runs', name='v1', epochs=100, imgsz=640)
print('best model:', best)
'''),
    code('''
import json
metrics = T.evaluate_per_source(best, DS, split='test', imgsz=640)
print(json.dumps(metrics, indent=1))
json.dump({'per_source_test': metrics, 'dataset': info}, open(DRIVE + '/runs/v1/metrics_per_source.json', 'w'), indent=1)
'''),
    code('''
import shutil
exported = T.export(best, imgsz=640, formats=('onnx', 'ncnn'))
out = '/content/weed_detector_v1'
os.makedirs(out, exist_ok=True)
run = DRIVE + '/runs/v1'
for f in ['weights/best.pt', 'weights/best.onnx', 'results.csv', 'confusion_matrix.png', 'results.png',
          'metrics_per_source.json', 'args.yaml']:
    if os.path.exists(os.path.join(run, f)):
        shutil.copy(os.path.join(run, f), out)
ncnn = os.path.join(run, 'weights', 'best_ncnn_model')
if os.path.isdir(ncnn):
    shutil.copytree(ncnn, os.path.join(out, 'best_ncnn_model'), dirs_exist_ok=True)
import glob
for src in sorted(glob.glob(DS + '/manifest_*.csv')):
    shutil.copy(src, out)
open(os.path.join(out, 'README.txt'), 'w').write(
    'WeedBot weed/keep detector v1 (YOLO11n, classes 0 weed, 1 keep, imgsz 640).\\n'
    'Trained on: CottonWeedDet12 (CC BY-NC 4.0, Lu et al., Zenodo 10.5281/zenodo.7535814), Roboflow-100 grass-weeds '
    '(CC BY 4.0), CropWeeds-YOLO via toolambr/crops_weeds (MIT), Open Images V7 (CC BY 2.0 images, CC BY 4.0 boxes).\\n'
    'Non-commercial use (CottonWeedDet12 licence).\\n')
shutil.make_archive(DRIVE + '/weed_detector_v1', 'zip', out)
print('ALL DONE ->', DRIVE + '/weed_detector_v1.zip', round(os.path.getsize(DRIVE + '/weed_detector_v1.zip') / 1e6, 1), 'MB')
'''),
]


def write(cells, name):
    nb = {'cells': cells, 'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'accelerator': 'GPU' if '02' in name else 'None'},
          'nbformat': 4, 'nbformat_minor': 0}
    path = os.path.join(HERE, name)
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    return path


if __name__ == '__main__':
    print(write(NB1, '01_build_dataset.ipynb'))
    print(write(NB2, '02_train.ipynb'))
