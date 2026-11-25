"""teacher.py - OWLv2 (open-vocabulary detector) draws the boxes on images whose class is already known.

Weed photos -> `weed` boxes, garden-plant photos -> `keep` boxes, "nothing here" ground photos -> must show
no plant. Each box's class comes from the folder (kind), not from the model; OWLv2 only says WHERE a plant is.
Distractor queries (grass, stone, soil ...) compete with the plant queries so those things do not become boxes.

The raw detections (all queries, down to RAW_MIN_SCORE) go into a JSONL cache, so thresholds and the box
rules can change later without running the model again. Heavy imports (torch, transformers) stay inside
Owlv2Teacher so the unit tests run without them.
"""
import json
import os
import time
from collections import Counter, namedtuple

from . import sources as S

TEACHER_MODEL = 'google/owlv2-base-patch16-ensemble'
QUERIES = {
    'weed': {'plant': ['a weed', 'a small plant', 'a plant with small leaves'],
             'distract': ['grass', 'a stone', 'soil', 'a human hand', 'a shoe', 'a flower', 'a shrub']},
    'keep': {'plant': ['a flower', 'a shrub', 'a flowering plant', 'a bush'],
             'distract': ['grass', 'a stone', 'soil', 'a human hand', 'a shoe', 'a weed', 'a flower pot']},
    'background': {'plant': ['a weed', 'a small plant', 'a flower', 'a shrub'],
                   'distract': ['grass', 'a stone', 'soil', 'gravel']},
}
THRESHOLDS = {'weed': 0.15, 'keep': 0.15, 'background': 0.12}
NMS_IOU = 0.5
MIN_BOX_AREA = 0.003     # share of the image; smaller boxes teach nothing
MAX_BOX_AREA = 0.6       # a plant filling more than this is a close-up, not a plant in a scene
RAW_MIN_SCORE = 0.05     # detections are cached down to this score
CLASS_OF = {'weed': S.WEED, 'keep': S.KEEP}

Det = namedtuple('Det', 'query score x0 y0 x1 y1')


def queries_for(kind):
    q = QUERIES[kind]
    return list(q['plant']) + list(q['distract'])


def pixel_box_to_rel(box_px, width, height):
    x0, y0, x1, y1 = box_px
    clip = lambda v: min(1.0, max(0.0, v))      # noqa: E731
    return (clip(x0 / width), clip(y0 / height), clip(x1 / width), clip(y1 / height))


def _iou(a, b):
    iw = min(a.x1, b.x1) - max(a.x0, b.x0)
    ih = min(a.y1, b.y1) - max(a.y0, b.y0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


def select_boxes(dets, kind):
    """YOLO boxes (cls, cx, cy, w, h) from raw dets: plant queries only, score >= threshold, not tiny, NMS.
    Boxes above MAX_BOX_AREA are kept here; image_verdict() rejects the whole image as a close-up."""
    if kind not in CLASS_OF:
        return []
    plant = set(QUERIES[kind]['plant'])
    cand = [d for d in dets if d.query in plant and d.score >= THRESHOLDS[kind]
            and (d.x1 - d.x0) * (d.y1 - d.y0) >= MIN_BOX_AREA]
    cand.sort(key=lambda d: -d.score)
    kept = []
    for d in cand:
        if all(_iou(d, k) <= NMS_IOU for k in kept):
            kept.append(d)
    return [(CLASS_OF[kind], (d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2, d.x1 - d.x0, d.y1 - d.y0) for d in kept]


def image_verdict(kind, boxes, dets):
    """Why an image must be dropped from the dataset, or None."""
    if kind == 'background':
        plant = set(QUERIES['background']['plant'])
        if any(d.query in plant and d.score >= THRESHOLDS['background'] for d in dets):
            return 'plant found'
        return None
    if not boxes:
        return 'teacher found no plant'
    if any(b[3] * b[4] > MAX_BOX_AREA for b in boxes):
        return 'close-up'
    return None


class Owlv2Teacher:
    def __init__(self, model_id=TEACHER_MODEL, device=None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image, queries, min_score=RAW_MIN_SCORE):
        """Dets for ALL queries >= min_score; each box carries its best query."""
        torch = self.torch
        w, h = image.size
        inputs = self.processor(text=[list(queries)], images=image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        side = max(w, h)          # OWLv2 pads the image to a square at the bottom / right
        res = self.processor.post_process_object_detection(out, threshold=min_score, target_sizes=[(side, side)])[0]
        dets = []
        for box, score, lab in zip(res['boxes'].tolist(), res['scores'].tolist(), res['labels'].tolist()):
            dets.append(Det(queries[lab], float(score), *pixel_box_to_rel(box, w, h)))
        return dets


def _label_path(img_path):
    folder, name = os.path.split(img_path)
    root, images = os.path.split(folder)
    if images != 'images':
        raise ValueError(f'not in an images/ folder: {img_path}')
    return os.path.join(root, 'labels', os.path.splitext(name)[0] + '.txt')


def _load_cache(path, queries):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row['queries'] == queries:
                    cache[row['stem']] = (row['w'], row['h'], [Det(*d) for d in row['dets']])
            except (ValueError, KeyError, TypeError):
                continue                    # a cut or damaged line: that image is simply detected again
    return cache


def label_images(image_paths, kind, teacher, cache_path, log_every=200):
    """Write label files for images in <root>/<split>/images/ (build.py layout) from the teacher's boxes;
    delete image + label when image_verdict() says so. Paths that no longer exist are skipped.
    Returns {'labelled', 'boxes', 'from_cache', 'dropped': {reason: n}}."""
    from PIL import Image, ImageOps
    queries = queries_for(kind)
    cache = _load_cache(cache_path, queries)
    res = {'labelled': 0, 'boxes': 0, 'from_cache': 0, 'dropped': Counter()}
    pending = []
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    def flush():
        if not pending:
            return
        need_nl = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
        if need_nl:
            with open(cache_path, 'rb') as f:
                f.seek(-1, os.SEEK_END)
                need_nl = f.read(1) != b'\n'
        with open(cache_path, 'a') as f:
            if need_nl:
                f.write('\n')
            f.write(''.join(pending))
            f.flush()
        pending.clear()

    def drop(img, lab, reason):
        for p in (img, lab):
            if os.path.exists(p):
                os.remove(p)
        res['dropped'][reason] += 1

    t0, done = time.time(), 0
    for img in image_paths:
        if not os.path.exists(img):
            continue
        lab = _label_path(img)
        stem = os.path.splitext(os.path.basename(img))[0]
        if stem in cache:
            dets = cache[stem][2]
            res['from_cache'] += 1
        else:
            try:
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    w, h = im.size
                    dets = teacher.detect(im, queries, RAW_MIN_SCORE)
            except (OSError, ValueError, SyntaxError):
                drop(img, lab, 'unreadable')
                continue
            pending.append(json.dumps({'stem': stem, 'w': w, 'h': h, 'queries': queries,
                                       'dets': [list(d) for d in dets]}) + '\n')
        boxes = select_boxes(dets, kind)
        reason = image_verdict(kind, boxes, dets)
        if reason:
            drop(img, lab, reason)
        else:
            with open(lab, 'w') as f:
                for c, cx, cy, bw, bh in boxes:
                    f.write(f'{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            res['labelled'] += 1
            res['boxes'] += len(boxes)
        done += 1
        if log_every and done % log_every == 0:
            flush()
            print(f'{done}/{len(image_paths)} images, {done / max(1e-9, time.time() - t0):.2f} img/s, '
                  f'{dict(res["dropped"])}', flush=True)
    flush()
    res['dropped'] = dict(res['dropped'])
    return res
