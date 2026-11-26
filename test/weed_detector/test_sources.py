"""Unit tests for training/weed_detector (no GPU, no internet): converters, splits, builder.

Run:  cd ~/dev_ws/src/my_bot && python3 -m pytest test/weed_detector -q
"""
import csv
import json
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
from weed_detector import build as B  # noqa: E402
from weed_detector import sources as S  # noqa: E402


def test_fo_box_to_yolo_centre():
    # FiftyOne boxes are [x, y, w, h] relative with (x, y) the TOP-LEFT corner
    assert S.fo_box_to_yolo(0.2, 0.4, 0.2, 0.1) == pytest.approx((0.3, 0.45, 0.2, 0.1))


def test_fo_box_clamped_to_image():
    cx, cy, w, h = S.fo_box_to_yolo(-0.1, 0.9, 0.3, 0.2)
    assert (cx, cy, w, h) == pytest.approx((0.1, 0.95, 0.2, 0.1))


def test_stable_split_is_stable_and_80_10_10():
    names = [f'img_{i}.jpg' for i in range(2000)]
    first = [S.stable_split(n) for n in names]
    assert first == [S.stable_split(n) for n in names]
    share = {s: first.count(s) / len(first) for s in ('train', 'val', 'test')}
    assert share['train'] == pytest.approx(0.8, abs=0.05)
    assert share['val'] == pytest.approx(0.1, abs=0.05)
    assert share['test'] == pytest.approx(0.1, abs=0.05)


def _cottonweed_json(tmp_path):
    samples = [
        {'filepath': 'data/a.jpg', 'ground_truth': {'_cls': 'Detections', 'detections': [
            {'label': 'SpottedSpurge', 'bounding_box': [0.1, 0.1, 0.2, 0.2]},
            {'label': 'Goosegrass', 'bounding_box': [0.5, 0.5, 0.1, 0.1]}]}},
        {'filepath': 'data/b.jpg', 'ground_truth': {'_cls': 'Detections', 'detections': [
            {'label': 'Goosegrass', 'bounding_box': [0.5, 0.5, 0.1, 0.1]}]}},
        {'filepath': 'data/c.jpg', 'ground_truth': None},
    ]
    p = tmp_path / 'samples.json'
    p.write_text(json.dumps({'samples': samples}))
    return p


def test_cottonweed_species_become_weed_and_goosegrass_is_dropped(tmp_path):
    items = {it.name: it for it in S.cottonweed_items(_cottonweed_json(tmp_path), 'https://hf/x')}
    assert items['a.jpg'].boxes == [pytest.approx((S.WEED, 0.2, 0.2, 0.2, 0.2))]
    assert items['a.jpg'].src == 'https://hf/x/data/a.jpg'
    assert 'Goosegrass' in items['a.jpg'].orig
    assert items['b.jpg'].boxes == []            # only goosegrass: kept as a background image
    assert items['c.jpg'].boxes == []            # no labels at all
    assert all(it.source == 'cottonweed' and 'NC' in it.licence for it in items.values())


def _yolo_tree(root, rel_img, lines):
    img = root / rel_img
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (40, 20), (90, 140, 60)).save(img)
    parts = list(img.relative_to(root).parts)
    parts[parts.index('images')] = 'labels'
    lab = root.joinpath(*parts).with_suffix('.txt')
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text('\n'.join(lines) + '\n')


def test_yolo_dir_items_maps_classes_and_splits(tmp_path):
    # grass-weeds layout: <split>/images, crops_weeds layout: images/<split>
    _yolo_tree(tmp_path / 'gw', 'train/images/r1.jpg', ['0 0.5 0.5 0.2 0.2'])
    _yolo_tree(tmp_path / 'gw', 'valid/images/r2.jpg', ['0 0.4 0.4 0.1 0.1'])
    _yolo_tree(tmp_path / 'cw', 'images/test/c1.jpeg', ['0 0.3 0.3 0.1 0.1', '1 0.6 0.6 0.2 0.2'])
    gw = S.yolo_dir_items(tmp_path / 'gw', 'grassweeds', ['ridderzuring'], {'ridderzuring': S.WEED}, 'CC BY 4.0')
    assert {(it.name, it.split) for it in gw} == {('r1.jpg', 'train'), ('r2.jpg', 'val')}
    assert all(it.boxes[0][0] == S.WEED for it in gw)
    cw = S.yolo_dir_items(tmp_path / 'cw', 'cropsweeds', ['crop', 'weed'], {'crop': S.KEEP, 'weed': S.WEED}, 'MIT')
    assert cw[0].split == 'test'
    assert [b[0] for b in cw[0].boxes] == [S.KEEP, S.WEED]


def test_yolo_polygon_line_becomes_its_bounding_box(tmp_path):
    _yolo_tree(tmp_path / 'p', 'train/images/p.jpg', ['0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6'])
    (it,) = S.yolo_dir_items(tmp_path / 'p', 'poly', ['x'], {'x': S.WEED}, 'CC BY 4.0')
    assert it.boxes[0] == pytest.approx((S.WEED, 0.3, 0.4, 0.4, 0.4))


def test_yolo_class_mapped_to_none_is_dropped(tmp_path):
    _yolo_tree(tmp_path / 'd', 'train/images/d.jpg', ['0 0.5 0.5 0.2 0.2', '1 0.2 0.2 0.1 0.1'])
    (it,) = S.yolo_dir_items(tmp_path / 'd', 'drop', ['grass', 'weed'], {'grass': None, 'weed': S.WEED}, 'MIT')
    assert [b[0] for b in it.boxes] == [S.WEED]


class _Det:
    def __init__(self, label, box, depiction=False, group=False):
        self.label, self.bounding_box, self.IsDepiction, self.IsGroupOf = label, box, depiction, group


class _Sample:
    def __init__(self, path, dets, positives=()):
        self.filepath = path
        self._d = {'detections': type('D', (), {'detections': dets})(),
                   'positive_labels': type('C', (), {'classifications': [type('L', (), {'label': p})() for p in positives]})()}

    def __getitem__(self, k):
        return self._d[k]

    def has_field(self, k):
        return k in self._d


def test_openimages_keep_plants_and_filters():
    ds = [
        _Sample('/oi/1.jpg', [_Det('Flower', [0.1, 0.1, 0.2, 0.2]), _Det('Person', [0.5, 0.5, 0.3, 0.3])]),
        _Sample('/oi/2.jpg', [_Det('Flower', [0.1, 0.1, 0.2, 0.2])], positives=['Dandelion']),   # weed flower: skip
        _Sample('/oi/3.jpg', [_Det('Plant', [0.1, 0.1, 0.2, 0.2], depiction=True)]),             # a drawing: skip
        _Sample('/oi/4.jpg', [_Det('Person', [0.1, 0.1, 0.2, 0.2])]),                            # no keep box: skip
    ]
    items = S.openimages_items(ds, max_images=10)
    assert [it.name for it in items] == ['1.jpg']
    assert items[0].boxes == [pytest.approx((S.KEEP, 0.2, 0.2, 0.2, 0.2))]


def test_build_resizes_writes_labels_manifest_and_is_resumable(tmp_path):
    src = tmp_path / 'big.jpg'
    Image.new('RGB', (3000, 2000), (120, 160, 80)).save(src)
    it = S.Item('cottonweed', 'big.jpg', str(src), 'train', [(S.WEED, 0.5, 0.5, 0.2, 0.1)], 'CC BY-NC 4.0', 'SpottedSpurge')
    out = tmp_path / 'ds'
    assert B.build([it], out)['written'] == 1
    img = out / 'train' / 'images' / 'cottonweed__big.jpg'
    with Image.open(img) as im:
        assert max(im.size) == 1024 and im.size == (1024, 683)
    assert (out / 'train' / 'labels' / 'cottonweed__big.txt').read_text().split() == ['0', '0.500000', '0.500000', '0.200000', '0.100000']
    rows = list(csv.DictReader(open(out / 'manifest.csv')))
    assert rows[0]['file'] == 'train/images/cottonweed__big.jpg' and rows[0]['source'] == 'cottonweed'
    assert B.build([it], out)['written'] == 0            # second run: nothing redone
    assert len(list(csv.DictReader(open(out / 'manifest.csv')))) == 1


def test_build_source_marker_skips_finished_source(tmp_path):
    src = tmp_path / 's.jpg'
    Image.new('RGB', (100, 50)).save(src)
    it = S.Item('grassweeds', 's.jpg', str(src), 'val', [], 'CC BY 4.0', '')
    assert B.build_source('grassweeds', [it], tmp_path / 'ds')['written'] == 1
    assert B.build_source('grassweeds', [it], tmp_path / 'ds') == {'skipped': True}


def test_write_yaml(tmp_path):
    p = B.write_yaml(tmp_path)
    text = p.read_text()
    assert 'train: train/images' in text and "0: weed" in text and "1: keep" in text


def test_openimages_skips_group_boxes_people_and_tiny_boxes():
    ds = [
        _Sample('/oi/g.jpg', [_Det('Plant', [0.0, 0.0, 1.0, 0.7], group=True)]),            # a hedge-wide group box
        _Sample('/oi/p.jpg', [_Det('Flower', [0.1, 0.1, 0.3, 0.3])], positives=['Person']),  # people scene
        _Sample('/oi/t.jpg', [_Det('Flower', [0.5, 0.5, 0.03, 0.03])]),                       # 0.09 % of the image
        _Sample('/oi/ok.jpg', [_Det('Rose', [0.2, 0.2, 0.4, 0.5])]),
    ]
    assert [it.name for it in S.openimages_items(ds, max_images=10)] == ['ok.jpg']


def test_oi_weed_image_ids_and_exclusion(tmp_path):
    csvp = tmp_path / 'labels.csv'
    csvp.write_text('ImageID,Source,LabelName,Confidence\n'
                    'aaa,verification,/m/0fr19,1\n'          # dandelion: excluded
                    'bbb,verification,/m/0c2pc,0\n'          # clover, verified ABSENT: not excluded
                    'ccc,verification,/m/0c9ph5,1\n')        # flower only
    ids = S.oi_weed_image_ids([csvp])
    assert ids == {'aaa'}
    ds = [_Sample('/oi/aaa.jpg', [_Det('Flower', [0.1, 0.1, 0.3, 0.3])]),
          _Sample('/oi/ccc.jpg', [_Det('Flower', [0.1, 0.1, 0.3, 0.3])])]
    assert [it.name for it in S.openimages_items(ds, exclude_ids=ids)] == ['ccc.jpg']


def test_build_part_zip_on_drive_merge_and_summary(tmp_path):
    from weed_detector import colab as C
    drive, work, merged = tmp_path / 'drive', tmp_path / 'work', tmp_path / 'merged'
    imgs = []
    for i, (split, boxes) in enumerate([('train', [(S.WEED, .5, .5, .2, .2)]), ('val', [(S.KEEP, .3, .3, .1, .1)])]):
        p = tmp_path / f'i{i}.jpg'
        Image.new('RGB', (64, 48)).save(p)
        imgs.append(S.Item('grassweeds', f'i{i}.jpg', str(p), split, boxes, 'CC BY 4.0', ''))
    calls = []
    def items_fn():
        calls.append(1)
        return imgs
    assert C.build_part('grassweeds', items_fn, str(work), str(drive))['written'] == 2
    assert (drive / 'dataset_parts' / 'grassweeds.zip').exists()
    assert C.build_part('grassweeds', items_fn, str(work), str(drive)) == {'skipped': True}
    assert len(calls) == 1                       # no download again once the part is on Drive
    summ = C.merge_parts(str(drive), str(merged), sources=('grassweeds',))
    assert summ == {'images': {'train': {'grassweeds': 1}, 'val': {'grassweeds': 1}}, 'boxes': {'weed': 1, 'keep': 1}}
    assert (merged / 'manifest_grassweeds.csv').exists() and (merged / 'data.yaml').exists()


def test_fetch_waits_and_retries_on_429(monkeypatch):
    import io as _io
    import urllib.error
    calls, waits = [], []

    class _Resp(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many Requests', {'Retry-After': '7'}, None)
        return _Resp(b'jpegbytes')
    monkeypatch.setattr(B.urllib.request, 'urlopen', fake_urlopen)
    assert B.fetch('https://hf/x.jpg', sleep=waits.append) == b'jpegbytes'
    assert waits == [7.0, 7.0] and len(calls) == 3


def test_merge_ignores_unknown_parts_and_reports_missing(tmp_path):
    from weed_detector import colab as C
    assert 'cropsweeds' not in C.EXPECTED_SOURCES
    with pytest.raises(RuntimeError, match='missing dataset parts'):
        C.merge_parts(str(tmp_path), str(tmp_path / 'out'))


def test_cottonweed_split_by_capture_session(tmp_path):
    samples = [{'filepath': f'data/20210628_iPhoneSE_YL_{i}.jpg', 'ground_truth': None} for i in range(40)]
    p = tmp_path / 's.json'
    p.write_text(json.dumps({'samples': samples}))
    splits = {it.split for it in S.cottonweed_items(p, '/x')}
    assert len(splits) == 1                      # one session -> one split, no near-duplicates across splits


def test_openimages_extra_weed_tags():
    for name in ('Dock', 'Plantago', 'Wood sorrel', 'Knotweed', "Lamb's quarters", 'Amaranth', 'Dandelion', 'Clover'):
        assert name in S.OI_WEED_MIDS.values()


def test_fetch_does_not_retry_missing_files(monkeypatch):
    import urllib.error
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, None)
    monkeypatch.setattr(B.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        B.fetch('https://hf/missing.jpg', sleep=lambda s: None)
    assert len(calls) == 1


def test_retry_wait_reads_hf_ratelimit_header():
    import urllib.error
    e = urllib.error.HTTPError('u', 429, 'x', {'RateLimit': '"api";r=0;t=42'}, None)
    assert B._retry_wait(e, 0) == 43.0


def test_source_finished_with_a_few_failed_images(tmp_path, monkeypatch):
    good = tmp_path / 'g.jpg'
    Image.new('RGB', (20, 20)).save(good)
    items = [S.Item('grassweeds', f'g{i}.jpg', str(good), 'train', [], 'x', '') for i in range(300)]
    items.append(S.Item('grassweeds', 'bad.jpg', str(tmp_path / 'nope.jpg'), 'train', [], 'x', ''))
    counts = B.build_source('grassweeds', items, tmp_path / 'ds')
    assert counts['failed'] == 1 and (tmp_path / 'ds' / '.done_grassweeds').exists()   # 1 of 301 < 0.5 %


def test_resume_guard_detects_finished_run(tmp_path):
    torch = pytest.importorskip('torch')
    from weed_detector import train as T
    torch.save({'epoch': -1, 'optimizer': None}, tmp_path / 'done.pt')
    torch.save({'epoch': 12, 'optimizer': {'state': {}}}, tmp_path / 'mid.pt')
    assert T.run_finished(tmp_path / 'done.pt') is True
    assert T.run_finished(tmp_path / 'mid.pt') is False


def test_repair_after_merge_resplits_cottonweed_and_drops_weed_tagged(tmp_path):
    from weed_detector import colab as C
    ds = tmp_path / 'ds'
    # an old-style cottonweed part: one session spread over train and test
    session = '20210628_iPhoneSE_YL'
    right = S.stable_split(session)
    wrong = 'test' if right != 'test' else 'train'
    for i, split in enumerate([right, wrong]):
        for kind, ext, data in (('images', '.jpg', None), ('labels', '.txt', '0 0.5 0.5 0.1 0.1\n')):
            d = ds / split / kind
            d.mkdir(parents=True, exist_ok=True)
            p = d / f'cottonweed__{session}_{i}{ext}'
            if data is None:
                Image.new('RGB', (8, 8)).save(p)
            else:
                p.write_text(data)
    # two openimages keep pictures, one of them tagged with a weed
    for oid in ('aaa', 'bbb'):
        (ds / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (ds / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8)).save(ds / 'train' / 'images' / f'openimages__{oid}.jpg')
        (ds / 'train' / 'labels' / f'openimages__{oid}.txt').write_text('1 0.5 0.5 0.2 0.2\n')
    (ds / 'manifest_openimages.csv').write_text('file,split,source,n_boxes,orig_labels,licence\n'
                                                 'train/images/openimages__aaa.jpg,train,openimages,1,Plant,x\n'
                                                 'train/images/openimages__bbb.jpg,train,openimages,1,Plant,x\n')
    out = C.repair_after_merge(str(ds), weed_ids={'aaa'})
    assert out == {'cottonweed_moved': 1, 'openimages_removed': 1}
    assert sorted(p.name for p in (ds / right / 'images').glob('cottonweed__*')) == [f'cottonweed__{session}_0.jpg', f'cottonweed__{session}_1.jpg']
    assert (ds / right / 'labels' / f'cottonweed__{session}_1.txt').exists()
    assert not (ds / 'train' / 'images' / 'openimages__aaa.jpg').exists()
    rows = list(csv.DictReader(open(ds / 'manifest_openimages.csv')))
    assert [r['file'] for r in rows] == ['train/images/openimages__bbb.jpg']
    assert C.repair_after_merge(str(ds), weed_ids={'aaa'}) == {'cottonweed_moved': 0, 'openimages_removed': 0}


def test_final_weights_strips_a_copy_and_leaves_the_run_resumable(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from ultralytics.nn.tasks import DetectionModel
    from weed_detector import train as T
    w = tmp_path / 'runs' / 'v1' / 'weights'
    w.mkdir(parents=True)
    ema = DetectionModel('yolo11n.yaml', nc=2, verbose=False).half()
    torch.save({'epoch': 40, 'best_fitness': 0.5, 'model': None, 'ema': ema, 'updates': 9,
                'optimizer': {'state': {}}, 'train_args': {'epochs': 100}}, w / 'best.pt')
    torch.save({'epoch': 96, 'best_fitness': 0.5, 'model': None, 'ema': ema, 'updates': 9,
                'optimizer': {'state': {}}, 'train_args': {'epochs': 100}}, w / 'last.pt')
    before = (w / 'best.pt').read_bytes()
    out, info = T.final_weights(tmp_path / 'runs', 'v1', tmp_path / 'final')
    assert info == {'best_epoch': 41, 'last_epoch': 97, 'epochs': 100}
    assert (w / 'best.pt').read_bytes() == before            # the run on Drive is untouched
    assert not T.run_finished(w / 'last.pt')                # ... and still resumes with a GPU later
    assert T.run_finished(out)                               # the copy is a finished, optimizer-free model
    assert out == tmp_path / 'final' / 'best.pt'
    with pytest.raises(FileNotFoundError):
        T.final_weights(tmp_path / 'nothing', 'v1', tmp_path / 'final2')
