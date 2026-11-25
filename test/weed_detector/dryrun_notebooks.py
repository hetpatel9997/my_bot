"""dryrun_notebooks.py - run the two Colab notebooks' code cells on the VM (CPU), with a local folder in place
of Google Drive, small source limits and a tiny training run. Proves the notebook glue before Colab time is
spent. Needs internet and the training venv:
  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python test/weed_detector/dryrun_notebooks.py
"""
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NB_DIR = os.path.join(REPO, 'training', 'weed_detector')
FAKE_DRIVE = os.path.expanduser('~/weedbot_data/dryrun/drive/weedbot')
WORK = os.path.expanduser('~/weedbot_data/dryrun/work')
DS = os.path.expanduser('~/weedbot_data/dryrun/weeds_ds')
OUT = os.path.expanduser('~/weedbot_data/dryrun/weed_detector_v1')


def cells(nb):
    with open(os.path.join(NB_DIR, nb)) as f:
        for c in json.load(f)['cells']:
            if c['cell_type'] == 'code':
                yield ''.join(c['source'])


def adapt(src):
    """Colab-only lines -> local equivalents (the logic itself is left untouched)."""
    if src.startswith('%%writefile') or src.startswith('!pip') or 'os.makedirs("/content/weed_detector"' in src:
        return None                                   # modules are imported from the repo instead
    src = src.replace("from google.colab import drive\ndrive.mount('/content/drive')\n", '')
    src = src.replace("from google.colab import drive\ndrive.flush_and_unmount()", "pass")
    src = src.replace("assert torch.cuda.is_available()", "assert True or torch.cuda.is_available()")
    src = src.replace("print('GPU:', torch.cuda.get_device_name(0))", "print('GPU: none (VM dry run)')")
    src = src.replace("'/content/drive/MyDrive/weedbot'", repr(FAKE_DRIVE))
    src = src.replace("sys.path.insert(0, '/content')", f"sys.path.insert(0, {os.path.join(REPO, 'training')!r})")
    src = src.replace("'/content/work'", repr(WORK)).replace("'/content/weeds_ds'", repr(DS))
    src = src.replace("'/content/weed_detector_v1'", repr(OUT))
    # small limits: 30 CottonWeedDet12 images, 20 Open Images pictures, a tiny CPU training
    src = src.replace("lambda: C.cottonweed_items_from_hf(WORK)", "lambda: C.cottonweed_items_from_hf(WORK)[:30]")
    src = src.replace("max_images=3000)", "max_images=20, max_samples_per_split=80)")
    src = src.replace("epochs=100, imgsz=640)", "epochs=1, imgsz=320, batch=8, device='cpu', workers=0, fraction=0.05)")
    src = src.replace("split='test', imgsz=640)", "split='test', imgsz=320, device='cpu')")
    src = src.replace("T.export(best, imgsz=640, formats=('onnx', 'ncnn'))", "T.export(best, imgsz=320, formats=('onnx', 'ncnn'))")
    return src


def run(nb):
    g = {'__name__': '__dryrun__'}
    for i, src in enumerate(cells(nb)):
        s = adapt(src)
        if s is None:
            continue
        print(f'\n===== {nb} cell {i} =====\n{s.strip()[:160]}')
        exec(compile(s, f'{nb}:cell{i}', 'exec'), g)


if __name__ == '__main__':
    os.makedirs(FAKE_DRIVE, exist_ok=True)
    run('01_build_dataset.ipynb')
    run('02_train.ipynb')
