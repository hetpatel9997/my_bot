"""teacher_check.py - VM tool: the teacher (teacher.py) labels a small public sample; contact sheets to judge it.

Why: the teacher's prompts and thresholds decide what the robot learns, so they are set by looking at its boxes
on ~100 public photos before any Colab time (design doc 2026-11-25-weed-detector-v2, "Teacher").
NEVER point it at the owner's photos: they are the final test. Run from training/ in the training venv:
  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python -m weed_detector.teacher_check ~/weedbot_data/teacher_check
Writes OUT_DIR/{inat_weed,inat_keep,background}/ (build layout), OUT_DIR/teacher_<source>.jsonl (teacher caches),
OUT_DIR/sheets/<source>_<n>.jpg (contact sheets) and OUT_DIR/summary.json. The photos are never deleted (the
pipeline's label_images works on a copy in OUT_DIR/work/), so re-runs reuse photos and caches: a threshold change
needs no teacher call, a query change does (the cache stores the query list).
Sheets: green = kept box (background: a plant box that drops the photo), grey = best plant boxes below the
threshold, red = best boxes whose top query is a distractor; text = query + score; a dropped photo has a red
border and its reason in the caption.
"""
import argparse
import json
import os
import shutil
from collections import Counter

from . import build as B
from . import colab as C
from . import sources as S
from . import teacher as TE

SOURCES = {'inat_weed': 'weed', 'inat_keep': 'keep', 'background': 'background'}    # source -> teacher kind
PER_SHEET, COLS, CELL, CAPTION = 12, 4, 480, 40
EXTRA = 3                   # grey / red boxes drawn per photo (the best ones after NMS)
OWNER_PHOTOS = os.path.expanduser('~/weedbot_data/backyard_photos')


class LazyTeacher:
    """OWLv2 is loaded only when a photo is not in the cache yet (threshold-only re-runs need no model)."""
    def __init__(self):
        self.teacher = None

    def detect(self, image, queries, min_score=TE.RAW_MIN_SCORE):
        if self.teacher is None:
            self.teacher = TE.Owlv2Teacher()
        return self.teacher.detect(image, queries, min_score)


def sample_items(out_dir, per_taxon, n_background):
    """{source: Items}; stored in items.json so re-runs make no API calls and see the same photos."""
    path = os.path.join(out_dir, 'items.json')
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved['args'] == [per_taxon, n_background]:
            return {s: [S.Item(*it) for it in its] for s, its in saved['items'].items()}
    items = {'inat_weed': C.inat_items_from_api(S.INAT_WEED_TAXA, 'inat_weed', quality_grade='research',
                                                max_per_taxon=per_taxon),
             'inat_keep': C.inat_items_from_api(S.INAT_KEEP_TAXA, 'inat_keep', quality_grade=None, captive=True,
                                                max_per_taxon=per_taxon),
             'background': C.background_items_from_oi(os.path.join(out_dir, 'oi'), n_background)}
    with open(path, 'w') as f:
        json.dump({'args': [per_taxon, n_background], 'items': items}, f)
    return items


def _best(dets, n):
    out = []
    for d in sorted(dets, key=lambda d: -d.score):
        if len(out) < n and all(TE._iou(d, k) <= TE.NMS_IOU for k in out):
            out.append(d)
    return out


def judge(kind, dets):
    """The pipeline's own rules (select_boxes, image_verdict) on one photo's cached Dets, plus what to draw."""
    boxes = TE.select_boxes(dets, kind)
    reason = TE.image_verdict(kind, boxes, dets)
    plant, th = set(TE.QUERIES[kind]['plant']), TE.THRESHOLDS[kind]
    if kind == 'background':
        green = _best([d for d in dets if d.query in plant and d.score >= th], 2 * EXTRA)
    else:
        chosen = {tuple(b[1:]) for b in boxes}
        green = [d for d in dets if d.query in plant and d.score >= th
                 and ((d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2, d.x1 - d.x0, d.y1 - d.y0) in chosen]
    return {'boxes': len(boxes), 'reason': reason, 'green': green,
            'grey': _best([d for d in dets if d.query in plant and d.score < th], EXTRA),
            'red': _best([d for d in dets if d.query not in plant], EXTRA)}


def draw_sheets(source, rows, sheet_dir):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font = ImageFont.load_default(size=15)
    paths = []
    for n in range(0, len(rows), PER_SHEET):
        chunk = rows[n:n + PER_SHEET]
        sheet = Image.new('RGB', (COLS * CELL, -(-len(chunk) // COLS) * (CELL + CAPTION)), 'white')
        dr = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            x, y = (i % COLS) * CELL, (i // COLS) * (CELL + CAPTION)
            with Image.open(r['path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((CELL, CELL))
            sheet.paste(im, (x, y))
            w, h = im.size
            for colour, width, dets in (('#c8c8c8', 2, r['grey']), ('#ff2020', 2, r['red']),
                                        ('#00e000', 3, r['green'])):
                for d in dets:
                    box = (x + d.x0 * w, y + d.y0 * h, x + d.x1 * w, y + d.y1 * h)
                    dr.rectangle(box, outline=colour, width=width)
                    dr.text((box[0] + 3, box[1] + 2), f'{d.query} {d.score:.2f}', fill=colour, font=font,
                            stroke_width=2, stroke_fill='black')
            if r['reason']:
                dr.rectangle((x, y, x + CELL - 1, y + CELL + CAPTION - 1), outline='#ff2020', width=5)
                status, colour = f"DROPPED: {r['reason']}", '#d00000'
            else:
                status, colour = (f"KEPT, {r['boxes']} box(es)" if r['boxes'] else 'KEPT (nothing here)'), '#008000'
            dr.text((x + 6, y + CELL + 2), f"#{r['index']} {r['orig'][:40]}", fill='black', font=font)
            dr.text((x + 6, y + CELL + 20), status, fill=colour, font=font)
        paths.append(os.path.join(sheet_dir, f'{source}_{n // PER_SHEET + 1}.jpg'))
        sheet.save(paths[-1], quality=90)
    return paths


def run(out_dir, per_taxon=3, n_background=30, teacher=None, workers=4):
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.realpath(out_dir).startswith(os.path.realpath(OWNER_PHOTOS)):
        raise SystemExit("refused: the owner's photos are the final test, never a teacher sample")
    os.makedirs(out_dir, exist_ok=True)
    teacher = teacher or LazyTeacher()
    items = sample_items(out_dir, per_taxon, n_background)
    sheet_dir = os.path.join(out_dir, 'sheets')
    shutil.rmtree(sheet_dir, ignore_errors=True)
    os.makedirs(sheet_dir)
    summary = {'queries': TE.QUERIES, 'thresholds': TE.THRESHOLDS, 'nms_iou': TE.NMS_IOU,
               'min_box_area': TE.MIN_BOX_AREA, 'max_box_area': TE.MAX_BOX_AREA, 'sources': {}}
    for source, kind in SOURCES.items():
        src_dir, work = os.path.join(out_dir, source), os.path.join(out_dir, 'work', source)
        built = B.build(items[source], src_dir, workers=workers, log_every=0)
        pairs = [(it, B._out_paths(src_dir, it)) for it in items[source]]
        pairs = [(it, img, lab) for it, (img, lab) in pairs if os.path.exists(img)]
        shutil.rmtree(work, ignore_errors=True)       # label_images deletes dropped photos: give it copies
        copies = []
        for _, img, lab in pairs:
            for p in (img, lab):
                dst = os.path.join(work, os.path.relpath(p, src_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(p, dst)
            copies.append(os.path.join(work, os.path.relpath(img, src_dir)))
        cache = os.path.join(out_dir, f'teacher_{source}.jsonl')
        res = TE.label_images(copies, kind, teacher, cache, log_every=10)
        cached = TE._load_cache(cache, TE.queries_for(kind))
        rows = []
        for i, (it, img, _) in enumerate(pairs, 1):
            stem = os.path.splitext(os.path.basename(img))[0]
            if stem in cached:                        # not cached = unreadable (counted in label_images)
                rows.append(dict(judge(kind, cached[stem][2]), index=i, path=img, stem=stem, orig=it.orig))
        kept = [r for r in rows if not r['reason']]
        summary['sources'][source] = {
            'images': len(rows), 'kept': len(kept),
            'dropped': dict(Counter(r['reason'] for r in rows if r['reason'])),
            'boxes_per_kept_image': dict(sorted(Counter(r['boxes'] for r in kept).items())),
            'mean_boxes_per_kept_image': round(sum(r['boxes'] for r in kept) / max(1, len(kept)), 2),
            'kept_index': [r['index'] for r in kept], 'label_images': res, 'download': built,
            'sheets': draw_sheets(source, rows, sheet_dir)}
        print(source, json.dumps({k: v for k, v in summary['sources'][source].items() if k != 'sheets'}), flush=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Teacher sample check (VM): label a public sample, draw contact sheets.')
    ap.add_argument('out_dir')
    ap.add_argument('--per-taxon', type=int, default=3, help='iNaturalist photos per taxon (both lists)')
    ap.add_argument('--background', type=int, default=30, help='Open Images "nothing here" photos')
    a = ap.parse_args(argv)
    run(a.out_dir, a.per_taxon, a.background)


if __name__ == '__main__':
    main()
