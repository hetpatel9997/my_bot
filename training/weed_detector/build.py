"""build.py - write Items (sources.py) as a YOLO dataset: resized images, label files, manifest, data.yaml.

Layout:  <out>/{train,val,test}/{images,labels}/<source>__<name>.jpg|.txt, <out>/manifest.csv,
<out>/data.yaml. Resumable: an image already written is not redone, and build_source() skips a source
whose `.done_<source>` marker exists (a Colab session can stop and be re-run).
"""
import csv
import io
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

from .sources import CLASS_NAMES

MANIFEST_FIELDS = ['file', 'split', 'source', 'n_boxes', 'orig_labels', 'licence']
MAX_FAILED_SHARE = 0.005     # a source counts as finished when at most 0.5 % of its images failed to download


def _retry_wait(err, attempt):
    """Seconds to wait before the next try: the server's Retry-After for 429 / 503, else 2, 4, 8 ... (max 60)."""
    if isinstance(err, urllib.error.HTTPError) and err.code in (429, 503):
        hdrs = err.headers or {}
        try:
            return min(120.0, float(hdrs.get('Retry-After', '')))
        except (TypeError, ValueError):
            m = re.search(r't=(\d+)', hdrs.get('RateLimit', '') or '')      # Hugging Face: RateLimit: ...;t=N
            return min(120.0, float(m.group(1)) + 1.0) if m else min(60.0, 5.0 * 2 ** attempt)
    return min(60.0, 2.0 * 2 ** attempt)


def fetch(src, retries=8, sleep=time.sleep):
    """Bytes of a local file or a URL. Retries with growing waits: Hugging Face answers 429 ("too many
    requests") to anonymous downloads in bursts, and big downloads sometimes drop."""
    if not src.startswith(('http://', 'https://')):
        with open(src, 'rb') as f:
            return f.read()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(src, headers={'User-Agent': 'weedbot-dataset-builder'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404, 410)):
                raise                                   # missing / forbidden: retrying cannot help
            sleep(_retry_wait(e, attempt))


def _out_paths(out_dir, item):
    stem = f'{item.source}__{os.path.splitext(item.name)[0]}'
    img = os.path.join(out_dir, item.split, 'images', stem + '.jpg')
    lab = os.path.join(out_dir, item.split, 'labels', stem + '.txt')
    return img, lab


def _write_one(item, out_dir, long_side, quality):
    img_path, lab_path = _out_paths(out_dir, item)
    if os.path.exists(img_path) and os.path.exists(lab_path):
        return None
    with Image.open(io.BytesIO(fetch(item.src))) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        scale = long_side / float(max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        os.makedirs(os.path.dirname(lab_path), exist_ok=True)
        im.save(img_path + '.part', 'JPEG', quality=quality)
    with open(lab_path + '.part', 'w') as f:
        for c, cx, cy, w, h in item.boxes:
            f.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
    os.replace(lab_path + '.part', lab_path)
    os.replace(img_path + '.part', img_path)          # the image last: it marks the pair as complete
    return {'file': os.path.relpath(img_path, out_dir), 'split': item.split, 'source': item.source,
            'n_boxes': len(item.boxes), 'orig_labels': item.orig, 'licence': item.licence}


def build(items, out_dir, long_side=1024, quality=90, workers=1, log_every=500):
    """Write every item not yet written. Returns {'written': n, 'skipped_existing': n, 'failed': n}."""
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, 'manifest.csv')
    new_file = not os.path.exists(manifest)
    counts = {'written': 0, 'skipped_existing': 0, 'failed': 0}
    with open(manifest, 'a', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()

        def job(it):
            try:
                return _write_one(it, out_dir, long_side, quality)
            except Exception as e:                      # one bad download must not stop 10,000 images
                return {'error': f'{it.source}/{it.name}: {e}'}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(job, items), 1):
                if row is None:
                    counts['skipped_existing'] += 1
                elif 'error' in row:
                    counts['failed'] += 1
                    print('FAILED', row['error'])
                else:
                    writer.writerow(row)
                    counts['written'] += 1
                if log_every and i % log_every == 0:
                    mf.flush()
                    print(f'{i}/{len(items)} {counts}', flush=True)
    return counts


def build_source(source, items, out_dir, **kw):
    """build() one source and leave a `.done_<source>` marker; a finished source is skipped next time."""
    marker = os.path.join(str(out_dir), f'.done_{source}')
    if os.path.exists(marker):
        return {'skipped': True}
    counts = build(items, out_dir, **kw)
    if counts['failed'] <= MAX_FAILED_SHARE * max(1, len(items)):
        with open(marker, 'w') as f:
            f.write(str(counts))
    return counts


def write_yaml(out_dir):
    """data.yaml for Ultralytics (paths relative to `path`)."""
    p = os.path.join(str(out_dir), 'data.yaml')
    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))
    with open(p, 'w') as f:
        f.write(f'path: {os.path.abspath(str(out_dir))}\ntrain: train/images\nval: val/images\ntest: test/images\n'
                f'names:\n{names}\n')
    from pathlib import Path
    return Path(p)
