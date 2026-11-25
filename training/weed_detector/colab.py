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
    counts = B.build_source(source, items_fn(), out, **build_kw)
    if counts.get('failed'):
        print(f'{source}: {counts["failed"]} images failed; run this cell again to retry them')
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


def merge_parts(drive_dir, out_dir):
    """Unzip every finished part into one dataset folder and write data.yaml. Returns summary(out_dir)."""
    os.makedirs(out_dir, exist_ok=True)
    for z in sorted(glob.glob(os.path.join(drive_dir, 'dataset_parts', '*.zip'))):
        with zipfile.ZipFile(z) as f:
            f.extractall(out_dir)
    B.write_yaml(out_dir)
    return summary(out_dir)


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
