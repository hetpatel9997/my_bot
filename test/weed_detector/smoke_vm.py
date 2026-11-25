"""smoke_vm.py - tiny end-to-end run of the weed detector pipeline on the VM (CPU, internet):
download a few real images per source, build a mini dataset, draw the boxes for a visual check, and
(optionally) train 1 epoch. Not a pytest test (needs internet + the training venv).

Run (training venv, without ROS paths):
  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python test/weed_detector/smoke_vm.py [--train]
Output: ~/weedbot_data/smoke/ (ds/, check/*.jpg with boxes drawn, runs/)
"""
import argparse
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
from weed_detector import build as B  # noqa: E402
from weed_detector import sources as S  # noqa: E402

HOME = os.path.expanduser('~/weedbot_data/smoke')
HF = 'https://huggingface.co/datasets'


def mirror_hf_subset(repo, n, pattern_img, dest):
    """Download n image + label pairs of a YOLO dataset on Hugging Face, keeping its folder layout."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo, repo_type='dataset')
    imgs = sorted(f for f in files if pattern_img(f) and f.lower().endswith(('.jpg', '.jpeg', '.png')))
    random.Random(0).shuffle(imgs)
    for f in imgs[:n]:
        parts = f.split('/')
        parts[parts.index('images')] = 'labels'
        lab = os.path.splitext('/'.join(parts))[0] + '.txt'
        for rel in (f, lab):
            if rel in files:
                hf_hub_download(repo, rel, repo_type='dataset', local_dir=dest)
    return dest


def draw(ds, out_dir, per_source=2):
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for split in ('train', 'val', 'test'):
        idir = os.path.join(ds, split, 'images')
        for f in sorted(os.listdir(idir)) if os.path.isdir(idir) else []:
            src = f.split('__')[0]
            lab = os.path.join(ds, split, 'labels', os.path.splitext(f)[0] + '.txt')
            lines = open(lab).read().split('\n') if os.path.exists(lab) else []
            if done.get(src, 0) >= per_source or not any(l.strip() for l in lines):
                continue
            im = Image.open(os.path.join(idir, f)).convert('RGB')
            d = ImageDraw.Draw(im)
            for line in lines:
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                x0, y0 = (cx - w / 2) * im.width, (cy - h / 2) * im.height
                x1, y1 = (cx + w / 2) * im.width, (cy + h / 2) * im.height
                d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0) if c == S.WEED else (0, 120, 255), width=5)
                d.text((x0 + 4, y0 + 4), S.CLASS_NAMES[c], fill=(255, 255, 0))
            im.thumbnail((700, 700))
            im.save(os.path.join(out_dir, f))
            done[src] = done.get(src, 0) + 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    ds = os.path.join(HOME, 'ds')
    os.makedirs(HOME, exist_ok=True)
    # 1. CottonWeedDet12 straight from Hugging Face (labels file + a few images by URL)
    sj = os.path.join(HOME, 'cw_samples.json')
    if not os.path.exists(sj):
        urllib.request.urlretrieve(f'{HF}/Voxel51/CottonWeedDet12/resolve/main/samples.json', sj)
    cw = S.cottonweed_items(sj, f'{HF}/Voxel51/CottonWeedDet12/resolve/main')
    spurge = [it for it in cw if 'SpottedSpurge' in it.orig][:a.n // 2]
    goose = [it for it in cw if 'Goosegrass' in it.orig][:2]
    other = [it for it in cw if it not in spurge and it not in goose][:a.n - len(spurge) - len(goose)]
    print('cottonweed:', B.build_source('cottonweed', spurge + goose + other, ds, workers=4))
    # 2. grass-weeds and 3. crops_weeds: a few pairs mirrored locally, then the folder converter
    gw = mirror_hf_subset('LibreYOLO/grass-weeds', a.n, lambda f: '/images/' in f, os.path.join(HOME, 'src_grassweeds'))
    print('grassweeds:', B.build_source('grassweeds', S.yolo_dir_items(gw, 'grassweeds', ['ridderzuring'],
                                                                         {'ridderzuring': S.WEED}, S.LICENCES['grassweeds']), ds))
    cr = mirror_hf_subset('toolambr/crops_weeds', a.n, lambda f: f.startswith('images/'), os.path.join(HOME, 'src_cropsweeds'))
    print('cropsweeds:', B.build_source('cropsweeds', S.yolo_dir_items(cr, 'cropsweeds', ['crop', 'weed'],
                                                                         {'crop': S.KEEP, 'weed': S.WEED}, S.LICENCES['cropsweeds']), ds))
    B.write_yaml(ds)
    print('checked images with boxes:', draw(ds, os.path.join(HOME, 'check')))
    if a.train:
        from ultralytics import YOLO
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
        from weed_detector import train as T
        best = T.train(os.path.join(ds, 'data.yaml'), os.path.join(HOME, 'runs'), epochs=1, imgsz=320, batch=4,
                       device='cpu', workers=0)
        print('best:', best, os.path.exists(best))
        print('per source:', json.dumps(T.evaluate_per_source(best, ds, split='train', imgsz=320, device='cpu'), indent=1))
        print('exported:', T.export(best, imgsz=320, formats=('onnx',)))


if __name__ == '__main__':
    main()
