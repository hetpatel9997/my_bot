"""train.py - train / resume / evaluate / export the WeedBot weed-and-keep detector (Ultralytics YOLO11n).

Why resumable: free Colab sessions stop after a few hours. Checkpoints live in `project_dir` (Google Drive
in Colab); calling train() again continues from `last.pt` instead of starting over.
"""
import csv
import json
import os
from pathlib import Path

from .sources import CLASS_NAMES

TRAIN_DEFAULTS = dict(epochs=100, imgsz=640, patience=30, hsv_v=0.5, hsv_s=0.8, degrees=10.0, flipud=0.5,
                      batch=-1, workers=2, cache='disk')
# flipud: the camera looks DOWN at the ground, so upside-down is fine. hsv_v / hsv_s above the defaults:
# bright white stones and sun. cache='disk': a Colab T4 has only 2 CPU cores; decoding 1024-px JPEGs for
# every mosaic would leave the GPU waiting.


def run_finished(last_pt):
    """True when last.pt is a finished run (Ultralytics strips the optimizer and sets epoch -1 at the end).
    Ultralytics 8.4 does NOT raise on resume=True for a finished run: it silently starts a NEW training
    (on its coco8 default), so this must be checked before resuming."""
    import torch
    ck = torch.load(str(last_pt), map_location='cpu', weights_only=False)
    return ck.get('epoch', -1) < 0 or ck.get('optimizer') is None


def run_dir(project_dir, name='v1'):
    return Path(project_dir) / name


def train(data_yaml, project_dir, name='v1', base='yolo11n.pt', **overrides):
    """Train from `base`, or resume from <project_dir>/<name>/weights/last.pt. Returns the best.pt path."""
    from ultralytics import YOLO
    rd = run_dir(project_dir, name)
    last, best = rd / 'weights' / 'last.pt', rd / 'weights' / 'best.pt'
    if last.exists():
        if run_finished(last):
            print('training already finished:', best)
        else:
            YOLO(str(last)).train(resume=True)
        return best
    args = dict(TRAIN_DEFAULTS)
    args.update(overrides)
    YOLO(base).train(data=str(data_yaml), project=str(project_dir), name=name, exist_ok=True, **args)
    return best


def final_weights(project_dir, name='v1', dest_dir='/content/final'):
    """For a run that Colab stopped near its end (GPU limit): copy <run>/weights/best.pt to <dest_dir>/best.pt
    and strip the optimizer from the COPY only, so it can be evaluated and exported without a GPU while the
    run itself stays resumable. Returns (path, {'best_epoch', 'last_epoch', 'epochs'}); epochs count from 1,
    best_epoch is None for a run that already finished."""
    import shutil
    import torch
    from ultralytics.utils.torch_utils import strip_optimizer
    wdir = run_dir(project_dir, name) / 'weights'
    src, last = wdir / 'best.pt', wdir / 'last.pt'
    if not src.exists():
        raise FileNotFoundError(f'{src} not found: step 2 has not saved a model yet')
    ck = torch.load(str(src), map_location='cpu', weights_only=False)
    epochs = int((ck.get('train_args') or {}).get('epochs', 0))
    best_epoch = ck['epoch'] + 1 if ck.get('epoch', -1) >= 0 else None
    last_epoch = epochs
    if last.exists():
        lk = torch.load(str(last), map_location='cpu', weights_only=False)
        if lk.get('epoch', -1) >= 0:
            last_epoch = lk['epoch'] + 1
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / 'best.pt'
    shutil.copy(src, out)
    strip_optimizer(str(out))
    return out, {'best_epoch': best_epoch, 'last_epoch': last_epoch, 'epochs': epochs}


def _source_lists(data_dir, split):
    """{source: [image paths]} for one split, from the `<source>__` file-name prefix."""
    out = {}
    idir = Path(data_dir) / split / 'images'
    for p in sorted(idir.glob('*.jpg')):
        out.setdefault(p.name.split('__')[0], []).append(str(p))
    return out


def evaluate_per_source(best_pt, data_dir, split='test', **val_kw):
    """mAP50 per class, precision and recall for each source's images in `split`."""
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    results = {}
    tmp = Path(data_dir) / '_eval'
    tmp.mkdir(exist_ok=True)
    for source, paths in _source_lists(data_dir, split).items():
        lst = tmp / f'{source}_{split}.txt'
        lst.write_text('\n'.join(paths) + '\n')
        yml = tmp / f'{source}_{split}.yaml'
        names = ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASS_NAMES))
        yml.write_text(f'path: {Path(data_dir).resolve()}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnames:\n{names}')
        m = model.val(data=str(yml), split='val', plots=False, verbose=False, **val_kw)
        ap50 = {CLASS_NAMES[int(c)]: round(float(v), 3) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
        results[source] = {'images': len(paths), 'map50': ap50, 'precision': round(float(m.box.mp), 3),
                           'recall': round(float(m.box.mr), 3)}
    return results


def evaluate_all(best_pt, data_yaml, project_dir, split='test', **val_kw):
    """Whole test split: precision, recall and mAP50 PER CLASS (keep recall = how many of the plants to
    protect are found) plus the confusion matrix plot, saved under <project_dir>/test_eval."""
    from ultralytics import YOLO
    m = YOLO(str(best_pt)).val(data=str(data_yaml), split=split, plots=True, project=str(project_dir),
                              name='test_eval', exist_ok=True, verbose=False, **val_kw)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[CLASS_NAMES[int(c)]] = {'precision': round(float(m.box.p[i]), 3), 'recall': round(float(m.box.r[i]), 3),
                                          'map50': round(float(m.box.ap50[i]), 3)}
    return {'per_class': per_class, 'map50_all': round(float(m.box.map50), 3),
            'confusion_matrix': str(Path(project_dir) / 'test_eval' / 'confusion_matrix.png')}


def export(best_pt, imgsz=640, formats=('onnx', 'ncnn')):
    """Export for the Pi: ONNX and NCNN (the Pi 4 runs NCNN fastest on its CPU). A failing format is
    reported, not raised: best.pt is what matters, and NCNN can also be made later on the VM."""
    from ultralytics import YOLO
    out = {}
    for f in formats:
        try:
            out[f] = YOLO(str(best_pt)).export(format=f, imgsz=imgsz)
        except Exception as e:
            print(f'export {f} failed: {e}')
            out[f] = None
    return out


def summary(project_dir, name='v1'):
    """Last row of results.csv (per-epoch metrics) as a dict, for a quick look."""
    p = run_dir(project_dir, name) / 'results.csv'
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p)))
    return {k.strip(): v for k, v in rows[-1].items()} if rows else {}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('data_yaml'); ap.add_argument('project_dir'); ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    b = train(a.data_yaml, a.project_dir, epochs=a.epochs)
    print(json.dumps(evaluate_per_source(b, os.path.dirname(a.data_yaml)), indent=1))
