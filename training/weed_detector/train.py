"""train.py - train / resume / evaluate / export the WeedBot weed-and-keep detector (Ultralytics YOLO11n).

Why resumable: free Colab sessions stop after a few hours. Checkpoints live in `project_dir` (Google Drive
in Colab); calling train() again continues from `last.pt` instead of starting over.
"""
import csv
import json
import os
from pathlib import Path

from .sources import CLASS_NAMES

TRAIN_DEFAULTS = dict(epochs=100, imgsz=640, patience=30, hsv_v=0.5, hsv_s=0.6, degrees=10.0, flipud=0.5,
                      batch=-1, workers=2)   # flipud: the camera looks DOWN at the ground, so upside-down is fine


def run_dir(project_dir, name='v1'):
    return Path(project_dir) / name


def train(data_yaml, project_dir, name='v1', base='yolo11n.pt', **overrides):
    """Train from `base`, or resume from <project_dir>/<name>/weights/last.pt. Returns the best.pt path."""
    from ultralytics import YOLO
    rd = run_dir(project_dir, name)
    last, best = rd / 'weights' / 'last.pt', rd / 'weights' / 'best.pt'
    if last.exists():
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:                 # "training to N epochs is finished, nothing to resume"
            print('resume:', e)
        return best
    args = dict(TRAIN_DEFAULTS)
    args.update(overrides)
    YOLO(base).train(data=str(data_yaml), project=str(project_dir), name=name, exist_ok=True, **args)
    return best


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


def export(best_pt, imgsz=640, formats=('onnx', 'ncnn')):
    """Export for the Pi: ONNX and NCNN (the Pi 4 runs NCNN fastest on its CPU)."""
    from ultralytics import YOLO
    return [YOLO(str(best_pt)).export(format=f, imgsz=imgsz) for f in formats]


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
