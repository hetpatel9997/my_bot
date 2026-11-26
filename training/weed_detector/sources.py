"""sources.py - open weed / plant datasets -> items for ONE two-class YOLO dataset (weed = 0, keep = 1).

Why: the WeedBot detector learns from public photos only (the owner takes no photos). Each source has its
own labels; this file maps them onto two classes. Design: docs/superpowers/specs/2026-11-24-weed-detector-design.md.

  cottonweed   CottonWeedDet12 via Hugging Face Voxel51/CottonWeedDet12 (samples.json, FiftyOne boxes):
               all 12 weed species -> weed, EXCEPT grass-type weeds (Goosegrass): their boxes are dropped,
               because lawn grass is background and a broadleaf spray does not kill grasses.
  grassweeds   LibreYOLO/grass-weeds (YOLO, class "ridderzuring" = broad-leaved dock in grassland) -> weed.
  (cropsweeds  toolambr/crops_weeds is NOT used: review 2026-11-24 found all 1,454 of its boxes are class
               "crop" on pictures of field weeds - it would teach weeds as `keep`.)
  openimages   Open Images V7 (FiftyOne), Flower / Houseplant / Plant / Rose / Lily boxes -> keep; images
               that show a dandelion (a weed with a flower) or only drawings are skipped.
  inat_weed    iNaturalist research-grade photos of weeds (INAT_WEED_TAXA), no boxes (labelled later).
  inat_keep    iNaturalist photos of garden plants (INAT_KEEP_TAXA), no boxes (labelled later).
  background   Open Images V7 "nothing here" ground photos (gravel, soil, lawn ... without any plant tag), no boxes.

An Item is one image: where to get it (`src`, a local path or URL), its split and its YOLO boxes
(class, cx, cy, w, h, all relative 0..1).
"""
import glob
import hashlib
import json
import os
from collections import namedtuple

WEED, KEEP = 0, 1
CLASS_NAMES = ['weed', 'keep']
GRASS_TYPE = {'goosegrass', 'crabgrass', 'barnyardgrass', 'johnsongrass', 'foxtail', 'nutsedge'}
OI_KEEP = {'Flower', 'Houseplant', 'Plant', 'Rose', 'Lily'}
OI_WEED_HINTS = {'Dandelion'}
# Image-level labels (full Open Images vocabulary, human-verified) of weeds: a picture showing any of these
# never becomes a `keep` example. FiftyOne only loads the boxable labels, so these come from the separate
# `*-annotations-human-imagelabels.csv` files, see oi_weed_image_ids().
OI_WEED_MIDS = {'/m/0fr19': 'Dandelion', '/m/02qpvtx': 'Common Dandelion', '/m/0c2pc': 'Clover',
                '/m/0grfz': 'Spurge', '/m/06tzj3': 'Thistle', '/m/08486': 'Weed', '/m/03218s': 'Dock',
                '/m/0dmw0': 'Plantago', '/m/03715w': 'Wood sorrel', '/m/03pjsn': 'Knotweed',
                '/m/03gtp5': "Lamb's quarters", '/m/0qqk': 'Amaranth'}
OI_IMAGELABEL_URLS = ['https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv',
                      'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv']
OI_PEOPLE = {'Person', 'Man', 'Woman', 'Boy', 'Girl', 'Human body', 'Human face'}
OI_MIN_AREA = 0.005       # keep boxes smaller than 0.5 % of the image are too small to teach anything

LICENCES = {
    'cottonweed': 'CC BY-NC 4.0 - CottonWeedDet12, Lu et al., Zenodo 10.5281/zenodo.7535814 (HF Voxel51/CottonWeedDet12)',
    'grassweeds': 'CC BY 4.0 - Roboflow 100 "grass-weeds" (HF LibreYOLO/grass-weeds)',
    'cropsweeds': 'MIT - CropWeeds-YOLO (Kaggle swish9/weeds-detection) via HF toolambr/crops_weeds',
    'openimages': 'images CC BY 2.0, boxes CC BY 4.0 - Open Images V7 (Google)',
    'background': 'images CC BY 2.0 - Open Images V7 (Google), no boxes',
}

# iNaturalist: name -> (taxon id, photos wanted). Ids looked up 2026-11-25; 1290817 / 1087592 are complexes.
INAT_WEED_TAXA = {
    'Euphorbia maculata': (58893, 3000), 'Euphorbia prostrata': (200053, 900), 'Portulaca oleracea': (58991, 735),
    'Taraxacum officinale': (47602, 735), 'Trifolium repens': (55745, 735), 'Plantago major': (58961, 735),
    'Plantago lanceolata': (53178, 735), 'Oxalis corniculata': (53168, 735), 'Polygonum aviculare': (1290817, 735),
    'Stellaria media': (1087592, 735), 'Lamium amplexicaule': (55459, 735), 'Medicago lupulina': (57056, 735),
    'Mollugo verticillata': (78090, 735)}
INAT_KEEP_TAXA = {
    'Rosa': (53438, 460), 'Lavandula': (61905, 460), 'Hydrangea macrophylla': (122034, 460),
    'Buxus sempervirens': (133397, 460), 'Tagetes': (72389, 460), 'Petunia': (55406, 460),
    'Pelargonium': (72281, 460), 'Hibiscus \u00d7 rosa-sinensis': (1628098, 460), 'Salvia': (50644, 460),
    'Hemerocallis': (82826, 460), 'Hosta': (82095, 460), 'Juniperus': (47574, 460),
    'Rhododendron': (49487, 460)}
INAT_LICENCES = {'cc0': 'CC0', 'cc-by': 'CC BY', 'cc-by-nc': 'CC BY-NC'}   # the only photo licences used

# Open Images tags of bare ground (the "nothing here" photos) and of anything plant-like (those photos are dropped).
BG_MIDS = {'/m/016b3m': 'Gravel', '/m/01tp0c': 'Pebble', '/m/01cbzq': 'Rock', '/m/09dv7': 'Soil',
           '/m/027t7q': 'Mulch', '/m/01v327': 'Lawn', '/m/08t9c_': 'Grass', '/m/02t2zd': 'Cobblestone',
           '/m/01jm2n': 'Walkway', '/m/01mxf': 'Concrete'}
PLANT_MIDS = {'/m/05s2s': 'Plant', '/m/0c9ph5': 'Flower', '/m/03fp41': 'Houseplant', '/m/07j7r': 'Tree',
              '/m/04sjm': 'Flowering plant', '/m/03gq1': 'Herb', '/m/018ssc': 'Groundcover', '/m/09t49': 'Leaf',
              '/m/0fm3zh': 'Flowerpot', '/m/0bl0l': 'Garden', '/m/0184yy': 'Moss', '/m/01xxn0': 'Hedge',
              '/m/01fnns': 'Vegetation', **OI_WEED_MIDS}

Item = namedtuple('Item', 'source name src split boxes licence orig')
SPLIT_DIRS = {'train': 'train', 'valid': 'val', 'val': 'val', 'test': 'test'}


def fo_box_to_yolo(x, y, w, h):
    """FiftyOne / Open Images box [x, y, w, h] (relative, top-left) -> YOLO (cx, cy, w, h), clamped to the image."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(1.0, x + w), min(1.0, y + h)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def stable_split(name, seed='weedbot'):
    """train / val / test (80 / 10 / 10) from a hash of the file name: the same image always lands in the same split."""
    h = int(hashlib.sha1(f'{seed}:{name}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 80 else ('val' if h < 90 else 'test')


def cottonweed_items(samples_json, image_root):
    """CottonWeedDet12 samples.json -> Items. `image_root` is a folder or the HF base URL that holds data/*.jpg."""
    with open(samples_json) as f:
        samples = json.load(f)['samples']
    items = []
    for s in samples:
        name = os.path.basename(s['filepath'])
        boxes, orig = [], []
        for det in ((s.get('ground_truth') or {}).get('detections') or []):
            orig.append(det['label'])
            if det['label'].lower() in GRASS_TYPE:
                continue
            boxes.append((WEED,) + fo_box_to_yolo(*det['bounding_box']))
        src = image_root.rstrip('/') + '/' + s['filepath']
        # split by capture session (date_device_initials): photos of one session look alike, so a whole
        # session goes to one split; otherwise near-duplicates in train and test make the test score too good
        session = '_'.join(os.path.splitext(name)[0].split('_')[:3])
        items.append(Item('cottonweed', name, src, stable_split(session), boxes, LICENCES['cottonweed'], ';'.join(orig)))
    return items


def _yolo_line_to_box(parts):
    """'cls cx cy w h' or a polygon 'cls x1 y1 x2 y2 ...' -> (cx, cy, w, h)."""
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        return tuple(vals)
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def yolo_dir_items(root, source, names, class_map, licence):
    """A YOLO dataset folder (<split>/images/... or images/<split>/...) -> Items.
    `names` = the source's class names by id; `class_map` = name -> WEED / KEEP / None (None drops the box)."""
    root = str(root)
    items = []
    for img in sorted(glob.glob(os.path.join(root, '**', '*'), recursive=True)):
        if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        rel = os.path.relpath(img, root).split(os.sep)
        if 'images' not in rel:
            continue
        split = next((SPLIT_DIRS[p] for p in rel if p in SPLIT_DIRS), None) or stable_split(rel[-1])
        lab_parts = list(rel)
        lab_parts[lab_parts.index('images')] = 'labels'
        lab = os.path.splitext(os.path.join(root, *lab_parts))[0] + '.txt'
        boxes, orig = [], []
        if os.path.exists(lab):
            with open(lab) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cname = names[int(float(parts[0]))]
                    orig.append(cname)
                    cls = class_map.get(cname)
                    if cls is not None:
                        boxes.append((cls,) + _yolo_line_to_box(parts))
        items.append(Item(source, rel[-1], img, split, boxes, licence, ';'.join(orig)))
    return items


def oi_weed_image_ids(csv_paths):
    """Open Images image ids that a human tagged with a weed label (OI_WEED_MIDS, confidence 1)."""
    import csv
    ids = set()
    for path in csv_paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r['LabelName'] in OI_WEED_MIDS and r['Confidence'].startswith('1'):
                    ids.add(r['ImageID'])
    return ids


def openimages_items(dataset, max_images=3000, det_field='detections', exclude_ids=frozenset()):
    """FiftyOne Open Images samples -> Items with KEEP boxes only. Skips drawings (IsDepiction), images
    tagged with a weed (dandelion) or with people, images whose keep box covers a GROUP of plants
    (IsGroupOf), tiny keep boxes, and images left without any keep box. FiftyOne's Open Images zoo stores the
    boxes in the field `detections` (image-level labels in `positive_labels`)."""
    items = []
    for s in dataset:
        positives = set()
        if s.has_field('positive_labels') and s['positive_labels'] is not None:
            positives = {c.label for c in s['positive_labels'].classifications}
        if positives & (OI_WEED_HINTS | OI_PEOPLE):
            continue                                    # a weed flower in view, or a people scene
        if os.path.splitext(os.path.basename(s.filepath))[0] in exclude_ids:
            continue                                    # tagged with a weed (dandelion, clover, spurge...)
        dets = s[det_field].detections if s.has_field(det_field) and s[det_field] is not None else []
        if any(getattr(d, 'IsDepiction', False) for d in dets):
            continue                                    # drawings, not real plants
        keep_dets = [d for d in dets if d.label in OI_KEEP]
        if any(getattr(d, 'IsGroupOf', False) for d in keep_dets):
            continue                                    # one box over a whole hedge / bed: would teach "patch = keep"
        boxes = [(KEEP,) + fo_box_to_yolo(*d.bounding_box) for d in keep_dets
                 if d.bounding_box[2] * d.bounding_box[3] >= OI_MIN_AREA]
        if not boxes:
            continue
        name = os.path.basename(s.filepath)
        items.append(Item('openimages', name, s.filepath, stable_split(name), boxes, LICENCES['openimages'],
                          ';'.join(sorted({d.label for d in dets}))))
        if len(items) >= max_images:
            break
    return items


def inat_item(obs, source):
    """An iNaturalist API observation -> Item without boxes, or None. Only the FIRST photo is used and only
    when its licence is in INAT_LICENCES (cc0 / cc-by / cc-by-nc)."""
    photos = obs.get('photos') or []
    if not photos or photos[0].get('license_code') not in INAT_LICENCES:
        return None
    photo = photos[0]
    oid = obs['id']
    user = (obs.get('user') or {}).get('login') or 'unknown'
    licence = (f"{INAT_LICENCES[photo['license_code']]} - photo by {user} - "
               f"https://www.inaturalist.org/observations/{oid}")
    src = photo['url'].replace('/square.', '/large.')
    return Item(source, f"{oid}_{photo['id']}.jpg", src, stable_split(f'inat{oid}'), [], licence,
                (obs.get('taxon') or {}).get('name', ''))


def oi_background_rows(csv_paths):
    """Open Images image-level label files -> (image id, OI split, ';'-joined tag names) of photos that a human
    tagged (confidence 1) with a ground tag (BG_MIDS) and with no plant tag (PLANT_MIDS, confidence 1).
    The OI split ('validation' / 'test') comes from the file name."""
    import csv
    rows = []
    for path in csv_paths:
        base = os.path.basename(path)
        split = 'validation' if base.startswith('validation') else 'test'
        bg, plant = {}, set()
        with open(path) as f:
            for r in csv.DictReader(f):
                if not r['Confidence'].startswith('1'):
                    continue
                if r['LabelName'] in BG_MIDS:
                    bg.setdefault(r['ImageID'], []).append(BG_MIDS[r['LabelName']])
                if r['LabelName'] in PLANT_MIDS:
                    plant.add(r['ImageID'])
        rows += [(i, split, ';'.join(sorted(set(tags)))) for i, tags in bg.items() if i not in plant]
    return rows


def background_items(rows, max_images=2000):
    """oi_background_rows -> Items without boxes, sorted by image id (stable), at most max_images."""
    items = []
    for image_id, split, tags in sorted(rows)[:max_images]:
        src = f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
        items.append(Item('background', f'{image_id}.jpg', src, stable_split(image_id), [],
                          LICENCES['background'], tags))
    return items
