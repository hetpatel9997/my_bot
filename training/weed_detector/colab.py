"""colab.py - the glue the two Colab notebooks run: one dataset part per source on Google Drive, merging the
parts, and the source downloads. Kept here (not only in the notebooks) so it can be tested on the VM.

Drive layout (MyDrive/weedbot/):  dataset_parts/<source>.zip   one finished source each (a stopped
session only redoes the unfinished source);  runs/v1/   training checkpoints;  weed_detector_v1.zip  result.
"""
import collections
import glob
import os
import shutil
import zipfile

from . import build as B
from . import sources as S

HF = 'https://huggingface.co/datasets'
EXPECTED_SOURCES = ('cottonweed', 'grassweeds', 'openimages')   # cropsweeds dropped (mislabelled, see sources.py)


def part_zip(drive_dir, source):
    return os.path.join(drive_dir, 'dataset_parts', f'{source}.zip')


def build_part(source, items_fn, work_dir, drive_dir, **build_kw):
    """Build one source into work_dir/<source> and store it as dataset_parts/<source>.zip on Drive.
    `items_fn()` is only called (downloads only start) if that zip does not exist yet."""
    target = part_zip(drive_dir, source)
    if os.path.exists(target):
        print(f'{source}: already on Drive, skipped')
        return {'skipped': True}
    out = os.path.join(work_dir, source)
    items = items_fn()
    counts = B.build_source(source, items, out, **build_kw)
    if not os.path.exists(os.path.join(out, f'.done_{source}')):
        print(f'{source}: {counts["failed"]} of {len(items)} images failed (more than 0.5 %); run this cell again to retry them')
        return counts
    man = os.path.join(out, 'manifest.csv')
    if os.path.exists(man):
        os.replace(man, os.path.join(out, f'manifest_{source}.csv'))   # one manifest per part, no clash on merge
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = shutil.make_archive(os.path.join(work_dir, f'{source}_part'), 'zip', out)
    shutil.copy(tmp, target + '.part')
    os.replace(target + '.part', target)
    print(f'{source}: {counts} -> {target}')
    return counts


def merge_parts(drive_dir, out_dir, sources=EXPECTED_SOURCES):
    """Unzip the finished parts of `sources` (other parts on Drive, e.g. an old cropsweeds.zip, are ignored)
    into one dataset folder and write data.yaml. Returns summary(out_dir)."""
    missing = [s for s in sources if not os.path.exists(part_zip(drive_dir, s))]
    if missing:
        raise RuntimeError(f'missing dataset parts {missing}: run step 1 (01_build_dataset) until it prints DONE')
    os.makedirs(out_dir, exist_ok=True)
    for s in sources:
        try:
            with zipfile.ZipFile(part_zip(drive_dir, s)) as f:
                f.extractall(out_dir)
        except zipfile.BadZipFile:
            raise RuntimeError(f'{part_zip(drive_dir, s)} is damaged: delete it in Google Drive and run step 1 again')
    B.write_yaml(out_dir)
    return summary(out_dir)


def repair_after_merge(out_dir, weed_ids=None, tmp_dir='/content/work_oi'):
    """Bring parts built by an OLDER step 1 up to date (a no-op for parts built by the current one):
    - cottonweed images move to the split of their capture session (no near-duplicates across splits);
    - openimages `keep` pictures that a human tagged with any weed in S.OI_WEED_MIDS are removed.
    `weed_ids` (Open Images ids) is downloaded when not given. Manifests are rewritten to match.
    Returns {'cottonweed_moved': n, 'openimages_removed': n}."""
    import csv
    moved = removed = 0
    for split in ('train', 'val', 'test'):
        for img in glob.glob(os.path.join(out_dir, split, 'images', 'cottonweed__*.jpg')):
            stem = os.path.splitext(os.path.basename(img))[0]
            session = '_'.join(stem.split('__', 1)[1].split('_')[:3])
            target = S.stable_split(session)
            if target == split:
                continue
            for kind, ext in (('images', '.jpg'), ('labels', '.txt')):
                src = os.path.join(out_dir, split, kind, stem + ext)
                dst_dir = os.path.join(out_dir, target, kind)
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src):
                    os.replace(src, os.path.join(dst_dir, stem + ext))
            moved += 1
    if weed_ids is None:
        import urllib.request
        os.makedirs(tmp_dir, exist_ok=True)
        csvs = []
        for url in S.OI_IMAGELABEL_URLS:
            path = os.path.join(tmp_dir, os.path.basename(url))
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)
            csvs.append(path)
        weed_ids = S.oi_weed_image_ids(csvs)
    for split in ('train', 'val', 'test'):
        for img in glob.glob(os.path.join(out_dir, split, 'images', 'openimages__*.jpg')):
            stem = os.path.splitext(os.path.basename(img))[0]
            if stem.split('__', 1)[1] in weed_ids:
                os.remove(img)
                lab = os.path.join(out_dir, split, 'labels', stem + '.txt')
                if os.path.exists(lab):
                    os.remove(lab)
                removed += 1
    # manifests: rewrite file / split to where the images are now; drop removed images
    where = {}
    for split in ('train', 'val', 'test'):
        for img in glob.glob(os.path.join(out_dir, split, 'images', '*.jpg')):
            where[os.path.basename(img)] = split
    for man in glob.glob(os.path.join(out_dir, 'manifest_*.csv')):
        with open(man) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        keep_rows = []
        for r in rows:
            base = os.path.basename(r['file'])
            if base in where:
                r['split'] = where[base]
                r['file'] = f"{where[base]}/images/{base}"
                keep_rows.append(r)
        with open(man, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep_rows)
    return {'cottonweed_moved': moved, 'openimages_removed': removed}


def summary(out_dir):
    """{split: {source: images}} and {class name: boxes} from the label files."""
    images = collections.defaultdict(collections.Counter)
    boxes = collections.Counter()
    for split in ('train', 'val', 'test'):
        for lab in glob.glob(os.path.join(out_dir, split, 'labels', '*.txt')):
            images[split][os.path.basename(lab).split('__')[0]] += 1
            with open(lab) as f:
                for line in f:
                    if line.strip():
                        boxes[S.CLASS_NAMES[int(line.split()[0])]] += 1
    return {'images': {k: dict(v) for k, v in images.items()}, 'boxes': dict(boxes)}


# ---- source downloads (need internet; used in the Colab notebook) ----

def cottonweed_items_from_hf(tmp_dir):
    """CottonWeedDet12 labels file from Hugging Face; images are fetched one by one while building."""
    import urllib.request
    os.makedirs(tmp_dir, exist_ok=True)
    sj = os.path.join(tmp_dir, 'cottonweed_samples.json')
    if not os.path.exists(sj):
        urllib.request.urlretrieve(f'{HF}/Voxel51/CottonWeedDet12/resolve/main/samples.json', sj)
    return S.cottonweed_items(sj, f'{HF}/Voxel51/CottonWeedDet12/resolve/main')


def hf_yolo_items(repo, source, names, class_map, dest):
    """Download a (small) YOLO dataset from Hugging Face and convert it."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type='dataset', local_dir=dest)
    return S.yolo_dir_items(dest, source, names, class_map, S.LICENCES[source])


def openimages_items_from_zoo(tmp_dir, max_images=3000, max_samples_per_split=6000):
    """Open Images V7 validation + test pictures with plant boxes (FiftyOne), minus weed-tagged pictures."""
    import urllib.request
    import fiftyone.zoo as foz
    os.makedirs(tmp_dir, exist_ok=True)
    csvs = []
    for url in S.OI_IMAGELABEL_URLS:
        p = os.path.join(tmp_dir, os.path.basename(url))
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
        csvs.append(p)
    exclude = S.oi_weed_image_ids(csvs)
    print(f'openimages: {len(exclude)} pictures tagged with a weed are left out')
    items = []
    for split in ('validation', 'test'):
        ds = foz.load_zoo_dataset('open-images-v7', split=split, label_types=['detections', 'classifications'],
                                  classes=sorted(S.OI_KEEP), max_samples=max_samples_per_split,
                                  dataset_name=f'weedbot_oi_{split}', overwrite=True)
        items += S.openimages_items(ds, max_images=max_images - len(items), exclude_ids=exclude)
        if len(items) >= max_images:
            break
    return items
