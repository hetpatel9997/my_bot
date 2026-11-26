"""backyard_check.py - VM tool: the ONE backyard check of a trained model on the owner's two rock-bed photos.

Design doc 2026-11-25-weed-detector-v2, "Checks before v2 goes on the robot" (checks 1 and 3) and "Decisions from
the builder check". Two commands, run from training/ in the training venv:

  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m weed_detector.backyard_check reference OUT_DIR
      The teacher (OWLv2, teacher.py) on the 4 quarter tiles of IMG_0229 / IMG_0230 with queries_for('weed'),
      score >= 0.10, plant queries only -> OUT_DIR/candidates.json + OUT_DIR/drawings/<photo>_tile<i>.jpg
      (numbered boxes). Claude keeps, by eye, the candidates that sit on a weed and writes
      OUT_DIR/reference.json = {photo: [[x0, y0, x1, y1], ...]}, records its sha256 and makes it read-only.
      That file is FROZEN before any v2 output exists; this command refuses to run once it is there.
      Next to it, OUT_DIR/unlisted.json (same format, also frozen) holds tight boxes on clearly visible weeds the
      teacher found no candidate for (written by hand from the drawings).

  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m weed_detector.backyard_check check MODEL_DIR REFERENCE_JSON
      best.pt on 5 views per photo (4 quarter tiles + the whole photo), each letterboxed to 640 x 640 here, so
      best_ncnn_model (the Pi version) gets exactly the same pixels -> MODEL_DIR/backyard_check/report.json +
      <photo>_tiles.jpg / <photo>_whole.jpg. Refuses to run without a valid reference file; uses unlisted.json
      from the reference's folder when it exists.

Every box is (x0, y0, x1, y1) in full-photo pixels after EXIF rotation, unless it says "640 input".
PASS (check 1, strict reading): on EVERY photo at least 70 % of the reference boxes are hit by a quarter-tile
`weed` box (conf >= 0.25, centre inside the reference box or IoU >= 0.3), at most 2 boxes (any class, any view,
conf >= 0.25) on bare stones (centre in no reference box and IoU < 0.1 with all of them), and no box covering
more than 50 % of the view it was found in. A box whose centre lies in an unlisted zone (a real weed without a
reference box) is not on bare stone: it is neither a stone box nor a hit, counted as `on_unlisted`; the 50 % rule
still applies to it. Check 3: every box >= 0.25 of one model has a box of the other model
on the same view with the same class, IoU >= 0.9 and confidence within 0.05.
Both models run at conf 0.20 = 0.25 - 0.05, so a box at 0.26 whose partner scored 0.23 is matched; the verdict
uses only boxes >= 0.25, which are exactly those a 0.25 run returns (NMS handles boxes in score order, so the
extra low boxes never remove a higher one).
NEVER in git: the owner's photos, the reference, the drawings and the reports live under ~/weedbot_data/.
NOT for Colab: this module is not in make_notebooks.py MODULES.
"""
import argparse
import hashlib
import json
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import teacher as TE

PHOTOS = ('IMG_0229', 'IMG_0230')
PHOTOS_DIR = os.path.expanduser('~/weedbot_data/backyard_photos/originals')
SIZE = 640                  # model input side
GREY = (114, 114, 114)      # Ultralytics' letterbox fill
CONF = 0.25                 # a model box counts from this confidence on
HIT_IOU = 0.3
MIN_HIT_RATE = 0.70
STONE_IOU = 0.1
MAX_STONE_BOXES = 2
MAX_VIEW_SHARE = 0.5
FID_IOU = 0.9
FID_CONF = 0.05
PREDICT_CONF = CONF - FID_CONF
REF_MIN_SCORE = 0.10        # teacher candidates for the reference


# ---------------------------------------------------------------- pure functions (unit-tested, no model)
def quarter_tiles(img):
    """[(x, y, tile)]: top-left, top-right, bottom-left, bottom-right; together they cover the whole photo."""
    w, h = img.size
    xs, ys = w // 2, h // 2
    return [(x0, y0, img.crop((x0, y0, x1, y1)))
            for y0, y1 in ((0, ys), (ys, h)) for x0, x1 in ((0, xs), (xs, w))]


def _fit(w, h, size):
    s = size / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def letterbox_square(img, size=SIZE):
    """img scaled so its long side is `size`, at the top-left of a size x size grey (114) square."""
    out = Image.new('RGB', (size, size), GREY)
    out.paste(img.convert('RGB').resize(_fit(*img.size, size), Image.BILINEAR), (0, 0))
    return out


def from_letterbox(box, view, size=SIZE):
    """A box on the 640 input of view (ox, oy, w, h) -> full-photo pixels, cut at the view's edges."""
    ox, oy, w, h = view
    nw, nh = _fit(w, h, size)
    x0, y0, x1, y1 = box
    fx = lambda v: ox + min(w, max(0.0, v * w / nw))      # noqa: E731
    fy = lambda v: oy + min(h, max(0.0, v * h / nh))      # noqa: E731
    return (fx(x0), fy(y0), fx(x1), fy(y1))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def centre_in(a, b):
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def is_hit(pred, ref):
    return centre_in(pred, ref) or iou(pred, ref) >= HIT_IOU


def on_stone(box, refs):
    """No reference weed under this box: its centre in no reference box and IoU < 0.1 with all of them."""
    return not any(centre_in(box, r) or iou(box, r) >= STONE_IOU for r in refs)


def backyard_verdict(per_photo):
    """per_photo[name] = {'ref': [box], 'weed': [(box, conf)] (quarter tiles), 'all': [(box, conf, cls)] (every
    view), 'views': [(view_w, view_h, [(box, conf, cls)])], optional 'unlisted': [box]}. A photo without
    reference boxes cannot show a hit rate and does not pass."""
    photos, hits_all, refs_all = {}, 0, 0
    for name, p in per_photo.items():
        refs = [tuple(r) for r in p['ref']]
        zones = [tuple(u) for u in p.get('unlisted', [])]
        in_zone = lambda b: any(centre_in(b, u) for u in zones)                    # noqa: E731
        weed = [b for b, c in p['weed'] if c >= CONF and not in_zone(b)]
        ref_hit = [any(is_hit(b, r) for b in weed) for r in refs]
        stone = [[list(b), c, k] for b, c, k in p['all'] if c >= CONF and not in_zone(b) and on_stone(b, refs)]
        unlisted = [[list(b), c, k] for b, c, k in p['all'] if c >= CONF and in_zone(b)]
        over = [[list(b), c, k] for vw, vh, boxes in p['views'] for b, c, k in boxes
                if c >= CONF and area(b) / (vw * vh) > MAX_VIEW_SHARE]
        hits = sum(ref_hit)
        rate = hits / len(refs) if refs else None
        photos[name] = {'hits': hits, 'refs': len(refs), 'hit_rate': rate, 'stone_boxes': len(stone),
                        'oversize': len(over), 'on_unlisted': len(unlisted), 'ref_hit': ref_hit,
                        'stone_box_list': stone, 'oversize_list': over, 'on_unlisted_list': unlisted,
                        'pass': rate is not None and rate >= MIN_HIT_RATE and len(stone) <= MAX_STONE_BOXES
                        and not over}
        hits_all += hits
        refs_all += len(refs)
    return {'photos': photos, 'pooled_hit_rate': hits_all / refs_all if refs_all else None,
            'pass': bool(photos) and all(p['pass'] for p in photos.values())}


def _matches(a, b):
    """Every box >= CONF in a against the boxes of b: (unmatched, worst best-IoU, worst conf gap of that partner)."""
    unmatched, ious, gaps = [], [], []
    for box, conf, cls in a:
        if conf < CONF:
            continue
        same = [(iou(box, b2), abs(conf - c2)) for b2, c2, k2 in b if k2 == cls]
        if not any(i >= FID_IOU and d <= FID_CONF for i, d in same):
            unmatched.append([list(box), conf, cls])
        best = max(same, key=lambda t: (t[0], -t[1]), default=(0.0, conf))
        ious.append(best[0])
        gaps.append(best[1])
    return unmatched, ious, gaps


def fidelity(pt_boxes, ncnn_boxes):
    """Same view, same 640 input: [(box, conf, cls)] of best.pt vs of the NCNN export; both ways round."""
    un_pt, i1, g1 = _matches(pt_boxes, ncnn_boxes)
    un_ncnn, i2, g2 = _matches(ncnn_boxes, pt_boxes)
    ious, gaps = i1 + i2, g1 + g2
    return {'pass': not un_pt and not un_ncnn, 'unmatched_pt': un_pt, 'unmatched_ncnn': un_ncnn,
            'min_iou': min(ious) if ious else None, 'max_conf_diff': max(gaps) if gaps else None}


# ---------------------------------------------------------------- files, models, drawings (CLI)
def load_photo(photos_dir, name):
    with Image.open(os.path.join(photos_dir, name + '.JPG')) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def views_of(img):
    """[(label, (ox, oy, w, h), image)]: the 4 quarter tiles, then the whole photo."""
    views = [(f'tile{i}', (x, y) + t.size, t) for i, (x, y, t) in enumerate(quarter_tiles(img))]
    return views + [('whole', (0, 0) + img.size, img)]


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _label(draw, xy, text, colour, font):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle((l - 2, t - 2, r + 2, b + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour, font=font)


COLOURS = [(255, 40, 40), (255, 220, 0), (0, 230, 255), (255, 0, 255), (60, 255, 60), (255, 140, 0)]


def reference(out_dir, photos_dir=PHOTOS_DIR, teacher=None):
    """Teacher candidates for the reference boxes: candidates.json + numbered drawings (one per quarter tile)."""
    if os.path.exists(os.path.join(out_dir, 'reference.json')):
        raise SystemExit(f'{out_dir}/reference.json exists: the reference is frozen, candidates are not made again '
                         '(their numbers must stay those the frozen file was chosen from)')
    teacher = teacher or TE.Owlv2Teacher()
    queries, plant = TE.queries_for('weed'), set(TE.QUERIES['weed']['plant'])
    os.makedirs(os.path.join(out_dir, 'drawings'), exist_ok=True)
    cands, font = [], _font(26)
    for name in PHOTOS:
        img = load_photo(photos_dir, name)
        for i, (ox, oy, tile) in enumerate(quarter_tiles(img)):
            tw, th = tile.size
            dets = [d for d in teacher.detect(tile, queries, REF_MIN_SCORE)       # Det boxes: 0-1 of the tile
                    if d.query in plant and d.score >= REF_MIN_SCORE]
            canvas = tile.copy()
            draw = ImageDraw.Draw(canvas)
            for d in sorted(dets, key=lambda d: -d.score):
                c = {'n': len(cands) + 1, 'photo': name, 'tile': i, 'query': d.query, 'score': round(d.score, 3),
                     'box': [round(ox + d.x0 * tw, 1), round(oy + d.y0 * th, 1),
                             round(ox + d.x1 * tw, 1), round(oy + d.y1 * th, 1)]}
                cands.append(c)
                col = COLOURS[c['n'] % len(COLOURS)]
                draw.rectangle((d.x0 * tw, d.y0 * th, d.x1 * tw, d.y1 * th), outline=col, width=3)
                _label(draw, (d.x0 * tw + 3, d.y0 * th + 3), f"{c['n']} {d.score:.2f}", col, font)
            canvas.save(os.path.join(out_dir, 'drawings', f'{name}_tile{i}.jpg'), quality=90)
            print(f'{name} tile {i}: {len(dets)} candidates', flush=True)
    with open(os.path.join(out_dir, 'candidates.json'), 'w') as f:
        json.dump({'queries': queries, 'min_score': REF_MIN_SCORE, 'teacher': TE.TEACHER_MODEL,
                   'candidates': cands}, f, indent=1)
    return cands


def load_reference(path):
    """{photo: [box]} and the file's sha256; SystemExit unless it is a valid reference for both photos."""
    if not os.path.isfile(path):
        raise SystemExit(f'no reference file {path}: make and freeze the reference first (reference command); '
                         'the check never runs without it')
    return _load_boxes(path, 'reference')


def load_unlisted(reference_json):
    """unlisted.json next to the reference: ({photo: [box]}, sha256, path), or (None, None, None) without it."""
    path = os.path.join(os.path.dirname(os.path.abspath(reference_json)), 'unlisted.json')
    if not os.path.isfile(path):
        return None, None, None
    return (*_load_boxes(path, 'unlisted'), path)


def _load_boxes(path, what):
    with open(path, 'rb') as f:
        raw = f.read()
    is_box = lambda b: (isinstance(b, list) and len(b) == 4                          # noqa: E731
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b)
                        and b[0] < b[2] and b[1] < b[3])
    try:
        ref = json.loads(raw)
        ok = (isinstance(ref, dict) and sorted(ref) == sorted(PHOTOS)
              and all(isinstance(v, list) and all(is_box(b) for b in v) for v in ref.values()))
    except ValueError:
        ok = False
    if not ok:
        raise SystemExit(f'bad {what} file {path}: must be {{photo: [[x0, y0, x1, y1], ...]}} for exactly '
                         f'{", ".join(PHOTOS)}')
    return {k: [tuple(float(x) for x in b) for b in v] for k, v in ref.items()}, hashlib.sha256(raw).hexdigest()


def yolo_predictor(path):
    """img (640 x 640) -> [(box on the 640 input, conf, class name)] at PREDICT_CONF."""
    from ultralytics import YOLO
    model = YOLO(path, task='detect')

    def predict(img):
        r = model.predict(img, imgsz=SIZE, conf=PREDICT_CONF, device='cpu', verbose=False)[0]
        return [(tuple(b), float(c), r.names[int(k)])
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return predict


def _draw_check(img, refs, ref_hit, zones, boxes, v, path):
    """Reference boxes green (hit) / yellow (missed), unlisted zones cyan; weed red, keep blue; tags ' S' = on
    stones, ' U' = on an unlisted weed, ' BIG' = > 50 % of its view."""
    canvas = img.copy()
    draw, font = ImageDraw.Draw(canvas), _font(30)
    for r, hit in zip(refs, ref_hit):
        draw.rectangle(r, outline=(0, 255, 0) if hit else (255, 255, 0), width=5)
    for u in zones:
        draw.rectangle(u, outline=(0, 230, 255), width=3)
    stone_set, over_set = {tuple(b) for b, _, _ in v['stone_box_list']}, {tuple(b) for b, _, _ in v['oversize_list']}
    unl_set = {tuple(b) for b, _, _ in v['on_unlisted_list']}
    used = set()                                  # labels of boxes sharing a corner are stacked, not overdrawn
    for b, c, k in boxes:
        col = (255, 40, 40) if k == 'weed' else (40, 120, 255)
        big = tuple(b) in over_set
        draw.rectangle(b, outline=(255, 0, 255) if big else col, width=9 if big else 4)
        tag = (f'{k} {c:.2f}' + (' S' if tuple(b) in stone_set else '') + (' U' if tuple(b) in unl_set else '')
               + (' BIG' if big else ''))
        x, y = round(b[0] + 4), round(b[1] + 4)
        while any(abs(x - ux) < 40 and abs(y - uy) < 36 for ux, uy in used):
            y += 36
        used.add((x, y))
        _label(draw, (x, y), tag, col, font)
    canvas.save(path, quality=88)


def check(model_dir, reference_json, photos_dir=PHOTOS_DIR, predictors=None):
    ref, sha = load_reference(reference_json)
    unlisted, unlisted_sha, unlisted_path = load_unlisted(reference_json)
    if predictors is None:
        paths = {'pt': os.path.join(model_dir, 'best.pt'), 'ncnn': os.path.join(model_dir, 'best_ncnn_model')}
        for p in paths.values():
            if not os.path.exists(p):
                raise SystemExit(f'model file missing: {p}')
        predictors = {k: yolo_predictor(p) for k, p in paths.items()}
    out = os.path.join(model_dir, 'backyard_check')
    os.makedirs(out, exist_ok=True)
    per_photo, boxes, fid_views, drawn = {}, {}, {}, {}
    for name in PHOTOS:
        img = load_photo(photos_dir, name)
        entry = {'ref': ref[name], 'weed': [], 'all': [], 'views': [], 'unlisted': (unlisted or {}).get(name, [])}
        boxes[name], drawn[name] = {}, {'tiles': [], 'whole': [], 'img': img}
        for label, view, vimg in views_of(img):
            inp = letterbox_square(vimg)                      # the SAME 640 x 640 picture for both models
            raw = {'pt': predictors['pt'](inp), 'ncnn': predictors['ncnn'](inp)}
            fid_views[f'{name}/{label}'] = fidelity(raw['pt'], raw['ncnn'])
            mapped = {k: [(from_letterbox(b, view), c, cls) for b, c, cls in v] for k, v in raw.items()}
            used = [x for x in mapped['pt'] if x[1] >= CONF]
            entry['all'] += used
            entry['views'].append((view[2], view[3], used))
            if label != 'whole':
                entry['weed'] += [(b, c) for b, c, k in used if k == 'weed']
            drawn[name]['whole' if label == 'whole' else 'tiles'] += used
            boxes[name][label] = {'view': list(view), **{k: [[list(b), c, cls] for b, c, cls in v]
                                                         for k, v in mapped.items()}}
        per_photo[name] = entry
    verdict = backyard_verdict(per_photo)
    ious = [f['min_iou'] for f in fid_views.values() if f['min_iou'] is not None]
    gaps = [f['max_conf_diff'] for f in fid_views.values() if f['max_conf_diff'] is not None]
    fid = {'pass': all(f['pass'] for f in fid_views.values()), 'min_iou': min(ious) if ious else None,
           'max_conf_diff': max(gaps) if gaps else None,
           'failed_views': {k: f for k, f in fid_views.items() if not f['pass']}, 'views': fid_views}
    word = lambda ok: 'PASS' if ok else 'FAIL'                # noqa: E731
    report = {'backyard_check': word(verdict['pass']), 'pi_version_check': word(fid['pass']),
              'model_dir': os.path.abspath(model_dir), 'reference': os.path.abspath(reference_json),
              'reference_sha256': sha, 'unlisted': unlisted_path, 'unlisted_sha256': unlisted_sha,
              'verdict': verdict, 'fidelity': fid,
              'settings': {'imgsz': SIZE, 'conf': CONF, 'predict_conf': PREDICT_CONF, 'hit_iou': HIT_IOU,
                           'min_hit_rate': MIN_HIT_RATE, 'stone_iou': STONE_IOU,
                           'max_stone_boxes': MAX_STONE_BOXES, 'max_view_share': MAX_VIEW_SHARE,
                           'fidelity_iou': FID_IOU, 'fidelity_conf': FID_CONF},
              'boxes': boxes}
    with open(os.path.join(out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    for name, d in drawn.items():
        v = verdict['photos'][name]
        for kind in ('tiles', 'whole'):
            _draw_check(d['img'], ref[name], v['ref_hit'], per_photo[name]['unlisted'], d[kind], v,
                        os.path.join(out, f'{name}_{kind}.jpg'))
    return report


def _print_report(rep):
    print(f"reference {rep['reference']} sha256 {rep['reference_sha256']}")
    print(f"unlisted  {rep['unlisted'] or '(none)'} sha256 {rep['unlisted_sha256']}")
    for name, p in rep['verdict']['photos'].items():
        rate = 'n/a' if p['hit_rate'] is None else f"{p['hit_rate']:.2f}"
        print(f"{name}: hits {p['hits']}/{p['refs']} (hit rate {rate}), stone boxes {p['stone_boxes']}, "
              f"oversize {p['oversize']}, on unlisted weeds {p['on_unlisted']} -> {'PASS' if p['pass'] else 'FAIL'}")
    pooled = rep['verdict']['pooled_hit_rate']
    print(f"pooled hit rate {'n/a' if pooled is None else f'{pooled:.2f}'}")
    print(f"BACKYARD CHECK: {rep['backyard_check']}")
    f = rep['fidelity']
    print(f"PI VERSION CHECK (NCNN vs best.pt): {rep['pi_version_check']} "
          f"(lowest IoU {f['min_iou']}, largest confidence gap {f['max_conf_diff']}, "
          f"{len(f['failed_views'])} views differ)")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Backyard check (VM): frozen reference boxes, then one check per model.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('reference', help='teacher candidates on the quarter tiles -> candidates.json + drawings')
    r.add_argument('out_dir')
    c = sub.add_parser('check', help='best.pt + best_ncnn_model on the 640 views -> MODEL_DIR/backyard_check/')
    c.add_argument('model_dir')
    c.add_argument('reference_json')
    a = ap.parse_args(argv)
    if a.cmd == 'reference':
        cands = reference(os.path.expanduser(a.out_dir))
        print(f'{len(cands)} candidates -> {a.out_dir}/candidates.json')
    else:
        rep = check(os.path.expanduser(a.model_dir), os.path.expanduser(a.reference_json))
        _print_report(rep)
        print(f"report: {os.path.join(rep['model_dir'], 'backyard_check', 'report.json')}")


if __name__ == '__main__':
    main()
