"""Unit tests for the v2 sources (iNaturalist photos, Open Images background photos); no internet."""
import csv
import os
import sys
import urllib.parse

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
from weed_detector import colab as C  # noqa: E402
from weed_detector import sources as S  # noqa: E402


def obs(oid, lic='cc-by-nc', pid=7, name='Euphorbia maculata'):
    return {'id': oid, 'taxon': {'name': name}, 'user': {'login': 'ann'},
            'photos': [{'id': pid, 'license_code': lic,
                        'url': f'https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/square.jpg'}]}


def test_inat_taxa_totals():
    assert sum(n for _, n in S.INAT_WEED_TAXA.values()) == 11985
    assert sum(n for _, n in S.INAT_KEEP_TAXA.values()) == 5980


def test_inat_item_fields():
    it = S.inat_item(obs(101), 'inat_weed')
    assert it.name == '101_7.jpg'
    assert it.src.endswith('/photos/7/large.jpg')
    assert it.boxes == []
    assert it.split == S.stable_split('inat101')
    assert 'CC BY-NC' in it.licence and 'ann' in it.licence
    assert 'https://www.inaturalist.org/observations/101' in it.licence
    assert it.orig == 'Euphorbia maculata'
    assert it.source == 'inat_weed'


@pytest.mark.parametrize('lic', [None, 'cc-by-sa', 'cc-by-nd'])
def test_inat_item_rejects_bad_first_photo(lic):
    o = obs(5, lic=lic)
    o['photos'].append({'id': 8, 'license_code': 'cc0', 'url': 'https://x/photos/8/square.jpg'})
    assert S.inat_item(o, 'inat_weed') is None
    o['photos'] = []
    assert S.inat_item(o, 'inat_weed') is None


def test_inat_observations_pages_by_id_and_stops():
    pages = [{'results': [obs(1), obs(2)]}, {'results': [obs(3)]}, {'results': []}]
    urls, sleeps = [], []

    def get_json(url):
        urls.append(url)
        return pages[len(urls) - 1]

    res = C.inat_observations(58893, 10, get_json=get_json, sleep=sleeps.append)
    assert len(res) == 3
    q = [urllib.parse.parse_qs(urllib.parse.urlparse(u).query) for u in urls]
    assert 'id_above' not in q[0]
    assert q[1]['id_above'] == ['2']
    for p in q:
        assert p['photo_license'] == ['cc0,cc-by,cc-by-nc']
        assert p['order_by'] == ['id'] and p['order'] == ['asc']
        assert p['taxon_id'] == ['58893']
    assert sum(sleeps) >= 1.0 and len(sleeps) >= 1 and all(s >= 1.0 for s in sleeps)

    urls.clear()
    pages[:] = [{'results': [obs(1), obs(2)]}, {'results': [obs(3)]}]
    assert len(C.inat_observations(58893, 2, get_json=get_json, sleep=lambda s: None)) == 2
    assert len(urls) == 1


def test_inat_observations_stops_when_nothing_left():
    calls = []

    def get_json(url):
        calls.append(url)
        return {'total_results': 2, 'results': [obs(1), obs(2)]}

    assert len(C.inat_observations(1, 50, get_json=get_json, sleep=lambda s: None)) == 2
    assert len(calls) == 1


def test_inat_observations_skips_bad_licence_but_pages_on():
    pages = [{'results': [obs(1, lic='cc-by-sa'), obs(2)]}, {'results': []}]
    urls = []

    def get_json(url):
        urls.append(url)
        return pages[len(urls) - 1]

    res = C.inat_observations(1, 10, get_json=get_json, sleep=lambda s: None)
    assert [o['id'] for o in res] == [2]
    assert 'id_above=2' in urls[1]


def test_inat_items_from_api_max_per_taxon():
    nxt = iter(range(1, 1000))

    def get_json(url):
        return {'results': [obs(next(nxt)), obs(next(nxt))]}

    taxa = {'Euphorbia maculata': (58893, 5), 'Trifolium repens': (55745, 5)}
    items = C.inat_items_from_api(taxa, 'inat_weed', max_per_taxon=1, get_json=get_json, sleep=lambda s: None)
    assert len(items) == 2 and all(it.source == 'inat_weed' for it in items)


def _write_labels(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ImageID', 'Source', 'LabelName', 'Confidence'])
        for r in rows:
            w.writerow([r[0], 'verification', r[1], r[2]])


def test_oi_background_rows_filter(tmp_path):
    v = tmp_path / 'validation-annotations-human-imagelabels.csv'
    t = tmp_path / 'test-annotations-human-imagelabels.csv'
    _write_labels(v, [
        ('a', '/m/016b3m', 1),                       # gravel only: kept
        ('b', '/m/016b3m', 1), ('b', '/m/05s2s', 1),  # gravel + plant: out
        ('c', '/m/016b3m', 1), ('c', '/m/05s2s', 0),  # plant tag negative: kept
        ('d', '/m/01v327', 1), ('d', '/m/0fr19', 1),  # lawn + dandelion: out
        ('e', '/m/016b3m', 0),                       # gravel negative: out
    ])
    _write_labels(t, [('f', '/m/09dv7', 1)])
    rows = S.oi_background_rows([str(v), str(t)])
    got = {r[0]: r for r in rows}
    assert set(got) == {'a', 'c', 'f'}
    assert got['a'] == ('a', 'validation', 'Gravel')
    assert got['f'][1] == 'test' and got['f'][2] == 'Soil'


def test_background_items():
    rows = [('zz', 'validation', 'Gravel'), ('aa', 'test', 'Soil;Rock'), ('mm', 'validation', 'Lawn')]
    items = S.background_items(rows, max_images=2)
    assert [it.name for it in items] == ['aa.jpg', 'mm.jpg']
    it = items[0]
    assert it.src == 'https://open-images-dataset.s3.amazonaws.com/test/aa.jpg'
    assert it.boxes == [] and it.source == 'background'
    assert it.split == S.stable_split('aa')
    assert it.licence == S.LICENCES['background']
    assert len(S.background_items(rows)) == 3


def test_inat_items_from_api_paces_requests_across_taxa():
    events = []
    nxt = iter(range(1, 1000))

    def get_json(url):
        events.append('req')
        return {'results': [obs(next(nxt))]}

    taxa = {'a': (1, 1), 'b': (2, 1), 'c': (3, 2)}
    C.inat_items_from_api(taxa, 'inat_weed', get_json=get_json, sleep=lambda s: events.append(('sleep', s)))
    assert events.count('req') >= 4
    for prev, cur in zip(events, events[1:]):
        assert not (prev == 'req' and cur == 'req')
    assert all(e[1] >= 1.0 for e in events if e != 'req')


# ---------------------------------------------------------------- teacher (OWLv2 boxes; no model here)
import json  # noqa: E402

from PIL import Image  # noqa: E402

from weed_detector import teacher as TE  # noqa: E402


def wq():
    return TE.QUERIES['weed']['plant'][0]


def wd():
    return TE.QUERIES['weed']['distract'][0]


def det(q, score, box=(0.1, 0.2, 0.3, 0.6)):
    return TE.Det(q, score, *box)


def test_select_boxes_threshold_query_nms_area():
    th = TE.THRESHOLDS['weed']
    assert len(TE.select_boxes([det(wq(), th + 0.01)], 'weed')) == 1
    assert TE.select_boxes([det(wq(), th - 0.01)], 'weed') == []
    assert TE.select_boxes([det(wd(), 0.9)], 'weed') == []
    two = [det(wq(), th + 0.05, (0.1, 0.1, 0.5, 0.5)), det(wq(), th + 0.2, (0.11, 0.11, 0.5, 0.5))]
    out = TE.select_boxes(two, 'weed')
    assert len(out) == 1 and out[0][1] == pytest.approx(0.305)       # the higher-scoring one (x0 0.11)
    tiny = det(wq(), th + 0.1, (0.1, 0.1, 0.1 + 0.002 / 0.5, 0.6))    # area 0.002 < MIN_BOX_AREA
    assert TE.select_boxes([tiny], 'weed') == []
    c, cx, cy, w, h = TE.select_boxes([det(wq(), th + 0.1)], 'weed')[0]
    assert (c, cx, cy, w, h) == pytest.approx((S.WEED, 0.2, 0.4, 0.2, 0.4))
    kq = TE.QUERIES['keep']['plant'][0]
    assert TE.select_boxes([det(kq, TE.THRESHOLDS['keep'] + 0.01)], 'keep')[0][0] == S.KEEP
    assert TE.select_boxes([det(TE.QUERIES['background']['plant'][0], 0.9)], 'background') == []


def test_image_verdicts():
    th = TE.THRESHOLDS['weed']
    assert TE.image_verdict('weed', [], []) == 'teacher found no plant'
    big = (0, 0.5, 0.5, 1.0, TE.MAX_BOX_AREA + 0.01)
    assert TE.image_verdict('weed', [big], []) == 'close-up'
    ok = (1, 0.5, 0.5, 0.5, 0.6)
    assert TE.image_verdict('keep', [ok], []) is None
    bq = TE.QUERIES['background']['plant'][0]
    bd = TE.QUERIES['background']['distract'][0]
    assert TE.image_verdict('background', [], [det(bq, TE.THRESHOLDS['background'] + 0.01)]) == 'plant found'
    assert TE.image_verdict('background', [], [det(bq, TE.THRESHOLDS['background'] - 0.01)]) is None
    assert TE.image_verdict('background', [], [det(bd, 0.9)]) is None
    assert th > 0


def test_pixel_box_to_rel_non_square():
    assert TE.pixel_box_to_rel((100, 50, 300, 250), 400, 200) == pytest.approx((0.25, 0.25, 0.75, 1.0))


class _Listish(list):
    def tolist(self):
        return list(self)


class _FakeImageProcessor:
    def __init__(self):
        self.calls = []

    def post_process_object_detection(self, outputs, threshold=0.1, target_sizes=None):
        self.calls.append((outputs, threshold, target_sizes))
        return [{'boxes': _Listish([[100.0, 50.0, 300.0, 250.0]]), 'scores': _Listish([0.4]), 'labels': _Listish([1])}]


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    """Like transformers 5.x's Owlv2Processor: post_process_object_detection exists only on .image_processor."""
    def __init__(self):
        self.image_processor = _FakeImageProcessor()

    def __call__(self, text, images, return_tensors):
        return _FakeInputs(pixel_values='px')


class _FakeTorch:
    @staticmethod
    def no_grad():
        import contextlib
        return contextlib.nullcontext()


def test_teacher_detect_uses_the_image_processor_post_process():
    assert not hasattr(_FakeProcessor(), 'post_process_object_detection')
    t = TE.Owlv2Teacher.__new__(TE.Owlv2Teacher)          # no model: the fakes stand in for torch + transformers
    t.torch, t.device, t.processor = _FakeTorch(), 'cpu', _FakeProcessor()
    t.model = lambda **inputs: ('model output', inputs)
    dets = t.detect(Image.new('RGB', (400, 200)), ['a weed', 'a small plant'])
    assert dets == [TE.Det('a small plant', 0.4, 0.25, 0.25, 0.75, 1.0)]
    (outputs, threshold, sizes), = t.processor.image_processor.calls
    assert outputs == ('model output', {'pixel_values': 'px'})
    assert threshold == TE.RAW_MIN_SCORE and sizes == [(400, 400)]       # padded square: the long side


class NamedFake:
    """Returns Dets by the image's pixel width (each test image gets its own width)."""
    def __init__(self, by_name=None, raises=False):
        self.by_name, self.raises, self.calls = by_name or {}, raises, 0

    def detect(self, image, queries, min_score=TE.RAW_MIN_SCORE):
        if self.raises:
            raise RuntimeError('must not be called')
        self.calls += 1
        return list(self.by_name.get(image.width, []))


def make_set(tmp_path, widths):
    paths = []
    for stem, wid in widths.items():
        (tmp_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        p = tmp_path / 'train' / 'images' / f'{stem}.jpg'
        Image.new('RGB', (wid, 20), 'green').save(p)
        (tmp_path / 'train' / 'labels' / f'{stem}.txt').write_text('')
        paths.append(str(p))
    return paths


def good_det():
    return det(wq(), TE.THRESHOLDS['weed'] + 0.1)


def test_label_images_writes_drops_and_resumes(tmp_path):
    paths = make_set(tmp_path, {'A': 30, 'B': 40})
    cache = str(tmp_path / 'cache.jsonl')
    fake = NamedFake({30: [good_det()], 40: []})
    res = TE.label_images(paths, 'weed', fake, cache, log_every=1)
    assert res['labelled'] == 1 and res['boxes'] == 1 and res['from_cache'] == 0
    assert res['dropped'] == {'teacher found no plant': 1}
    assert (tmp_path / 'train' / 'labels' / 'A.txt').read_text().startswith('0 ')
    assert os.path.exists(paths[0])
    assert not os.path.exists(paths[1])
    assert not (tmp_path / 'train' / 'labels' / 'B.txt').exists()
    res2 = TE.label_images(paths, 'weed', NamedFake(raises=True), cache)
    assert res2['from_cache'] == 1 and res2['labelled'] == 1
    assert (tmp_path / 'train' / 'labels' / 'A.txt').read_text().startswith('0 ')


def test_label_images_ignores_cut_last_cache_line(tmp_path):
    paths = make_set(tmp_path, {'A': 30})
    cache = tmp_path / 'cache.jsonl'
    cache.write_text('{"stem": "A", "w": 1')
    fake = NamedFake({30: [good_det()]})
    res = TE.label_images(paths, 'weed', fake, str(cache))
    assert fake.calls == 1 and res['labelled'] == 1
    again = NamedFake(raises=True)
    assert TE.label_images(paths, 'weed', again, str(cache))['from_cache'] == 1


def test_label_images_queries_change_invalidates_cache(tmp_path):
    paths = make_set(tmp_path, {'A': 30})
    cache = tmp_path / 'cache.jsonl'
    cache.write_text(json.dumps({'stem': 'A', 'w': 30, 'h': 20, 'queries': ['something else'],
                                 'dets': [[wq(), 0.9, 0.1, 0.2, 0.3, 0.6]]}) + '\n')
    fake = NamedFake({30: [good_det()]})
    res = TE.label_images(paths, 'weed', fake, str(cache))
    assert fake.calls == 1 and res['from_cache'] == 0


def test_label_images_corrupt_image(tmp_path):
    paths = make_set(tmp_path, {'A': 30})
    bad = tmp_path / 'train' / 'images' / 'C.jpg'
    bad.write_bytes(b'not a jpeg')
    (tmp_path / 'train' / 'labels' / 'C.txt').write_text('')
    res = TE.label_images(paths + [str(bad)], 'weed', NamedFake({30: [good_det()]}), str(tmp_path / 'c.jsonl'))
    assert res['dropped'] == {'unreadable': 1} and res['labelled'] == 1
    assert not bad.exists()


def test_label_images_model_error_keeps_image(tmp_path):
    paths = make_set(tmp_path, {'A': 30})

    class Broken:
        def detect(self, image, queries, min_score):
            raise OSError('model weights file vanished')

    with pytest.raises(OSError):
        TE.label_images(paths, 'weed', Broken(), str(tmp_path / 'c.jsonl'))
    assert os.path.exists(paths[0])


# ---------------------------------------------------------------- Drive parts (fake Drive = a temp folder)
import zipfile  # noqa: E402


def zip_names(path):
    with zipfile.ZipFile(path) as z:
        return set(z.namelist())


import csv as _csv  # noqa: E402


def make_images_part(tmp_path, widths, source='inat_weed'):
    """Run build_part the way step 3 does: photos without boxes -> dataset_parts/<source>_images.zip."""
    drive, work = tmp_path / 'drive', tmp_path / 'work'
    items = []
    for i, wid in enumerate(widths):
        p = tmp_path / f'src{i}.jpg'
        Image.new('RGB', (wid, 20), 'green').save(p)
        items.append(S.Item(source, f'{i}.jpg', str(p), 'train', [], 'CC0', 'x'))
    C.build_part(source, lambda: items, str(work), str(drive), zip_name=f'{source}_images')
    return drive, work


def test_build_part_zip_name(tmp_path):
    drive, _ = make_images_part(tmp_path, [30])
    assert (drive / 'dataset_parts' / 'inat_weed_images.zip').exists()
    assert not (drive / 'dataset_parts' / 'inat_weed.zip').exists()
    with zipfile.ZipFile(drive / 'dataset_parts' / 'inat_weed_images.zip') as z:
        assert 'manifest_inat_weed.csv' in z.namelist()      # named after the source, not zip_name


def test_label_part_from_images_zip(tmp_path):
    drive, work = make_images_part(tmp_path, [30, 40])
    fake = NamedFake({30: [good_det()], 40: []})
    res = C.label_part('inat_weed', str(work), str(drive), lambda: fake)
    assert res['labelled'] == 1 and res['dropped'] == {'teacher found no plant': 1}
    assert (drive / 'dataset_parts' / 'teacher_inat_weed.jsonl').exists()
    out = tmp_path / 'check'
    with zipfile.ZipFile(drive / 'dataset_parts' / 'inat_weed.zip') as z:
        z.extractall(out)
    assert [p.name for p in (out / 'train' / 'images').iterdir()] == ['inat_weed__0.jpg']
    assert (out / 'train' / 'labels' / 'inat_weed__0.txt').read_text().startswith('0 ')
    assert len(list((out / 'train' / 'labels').iterdir())) == 1
    with open(out / 'manifest_inat_weed.csv') as f:
        rows = list(_csv.DictReader(f))
    assert [(r['file'], r['n_boxes']) for r in rows] == [('train/images/inat_weed__0.jpg', '1')]


def test_label_part_skips_when_labelled_exists(tmp_path):
    drive, work = make_images_part(tmp_path, [30])
    C.label_part('inat_weed', str(work), str(drive), lambda: NamedFake({30: [good_det()]}))

    def boom():
        raise AssertionError('teacher_fn must not be called')
    assert C.label_part('inat_weed', str(work), str(drive), boom) == {'skipped': True}


def test_label_part_missing_images_part(tmp_path):
    with pytest.raises(RuntimeError, match='03_v2_collect_photos'):
        C.label_part('inat_weed', str(tmp_path / 'w'), str(tmp_path / 'd'), lambda: None)


def test_label_part_damaged_images_zip(tmp_path):
    drive = tmp_path / 'd'
    (drive / 'dataset_parts').mkdir(parents=True)
    (drive / 'dataset_parts' / 'inat_weed_images.zip').write_bytes(b'junk')
    with pytest.raises(RuntimeError, match='damaged'):
        C.label_part('inat_weed', str(tmp_path / 'w'), str(drive), lambda: None)


def test_label_part_rerun_uses_cache_and_builds_teacher_only_when_needed(tmp_path):
    drive, work = make_images_part(tmp_path, [30, 40])
    C.label_part('inat_weed', str(work), str(drive), lambda: NamedFake({30: [good_det()], 40: [good_det()]}))
    (drive / 'dataset_parts' / 'inat_weed.zip').unlink()          # as if the run stopped before the zip
    built = []
    res = C.label_part('inat_weed', str(work), str(drive), lambda: built.append(1))
    assert built == [] and res['from_cache'] == 2 and res['labelled'] == 2


def test_label_part_cache_covers_part_of_the_photos(tmp_path):
    drive, work = make_images_part(tmp_path, [30, 40])
    first = NamedFake({30: [good_det()]})
    C.label_part('inat_weed', str(work), str(drive), lambda: first)
    cache = drive / 'dataset_parts' / 'teacher_inat_weed.jsonl'
    lines = cache.read_text().splitlines()
    cache.write_text(lines[0] + '\n')                              # the stop happened after the first photo
    (drive / 'dataset_parts' / 'inat_weed.zip').unlink()
    second = NamedFake({30: [good_det()], 40: [good_det()]})
    res = C.label_part('inat_weed', str(work), str(drive), lambda: second)
    assert second.calls == 1 and res['from_cache'] == 1 and res['labelled'] == 2


def _damage_zip(path, how):
    """Break a labelled zip the ways a Colab stop during the copy to Drive can."""
    if how == 'junk':
        path.write_bytes(b'junk')
    elif how == 'empty':
        path.write_bytes(b'')
    elif how == 'cut':
        data = path.read_bytes()
        path.write_bytes(data[:len(data) // 2])
    elif how == 'bad_crc':                         # the zip opens (central directory fine), a file's bytes are wrong
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
            z.writestr('train/labels/x.txt', 'hello weed')
        path.write_bytes(path.read_bytes().replace(b'hello weed', b'jello weed', 1))


@pytest.mark.parametrize('how', ['junk', 'empty', 'cut', 'bad_crc'])
def test_label_part_relabels_a_damaged_labelled_zip_from_the_cache(tmp_path, how):
    drive, work = make_images_part(tmp_path, [30, 40])
    C.label_part('inat_weed', str(work), str(drive), lambda: NamedFake({30: [good_det()], 40: [good_det()]}))
    target = drive / 'dataset_parts' / 'inat_weed.zip'
    _damage_zip(target, how)
    assert not C.zip_ok(str(target))

    def no_teacher():
        raise AssertionError('every photo is in the teacher cache: no teacher (no GPU) needed')
    res = C.label_part('inat_weed', str(work), str(drive), no_teacher)
    assert res['from_cache'] == 2 and res['labelled'] == 2
    assert C.zip_ok(str(target))
    assert {'train/images/inat_weed__0.jpg', 'train/images/inat_weed__1.jpg'} <= zip_names(target)


def test_zip_ok_accepts_a_good_zip(tmp_path):
    p = tmp_path / 'good.zip'
    with zipfile.ZipFile(p, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('train/labels/x.txt', '0 0.5 0.5 0.1 0.1\n')
    assert C.zip_ok(str(p))
    assert not C.zip_ok(str(tmp_path / 'missing.zip'))


def _parts(drive, names, broken=()):
    """Valid (empty) zips for `names` and junk zips for `broken` in drive/dataset_parts."""
    (drive / 'dataset_parts').mkdir(parents=True, exist_ok=True)
    for n in names:
        zipfile.ZipFile(drive / 'dataset_parts' / f'{n}.zip', 'w').close()
    for n in broken:
        (drive / 'dataset_parts' / f'{n}.zip').write_bytes(b'junk')


def _merge_error(drive, tmp_path, sources):
    with pytest.raises(RuntimeError) as e:
        C.merge_parts(str(drive), str(tmp_path / 'out'), sources=sources)
    return str(e.value)


def test_merge_missing_v1_part_names_step_1(tmp_path):
    drive = tmp_path / 'd'
    _parts(drive, [s for s in C.V2_SOURCES if s != 'grassweeds'])
    msg = _merge_error(drive, tmp_path, C.V2_SOURCES)
    assert 'missing dataset parts' in msg and 'grassweeds' in msg and '01_build_dataset' in msg
    assert 'not step 2' in msg                         # in v2, running 02_train again would change runs/v1
    assert '03_v2' not in msg and '04_v2' not in msg
    msg1 = _merge_error(tmp_path / 'empty', tmp_path, C.EXPECTED_SOURCES)      # v1's own merge (02 / 02b)
    assert '01_build_dataset' in msg1 and 'not step 2' not in msg1


def test_merge_missing_v2_part_names_step_4(tmp_path):
    drive = tmp_path / 'd'
    _parts(drive, [s for s in C.V2_SOURCES if s != 'inat_keep'] + ['inat_keep_images'])
    msg = _merge_error(drive, tmp_path, C.V2_SOURCES)
    assert 'missing dataset parts' in msg and 'inat_keep' in msg and '04_v2_label_and_train' in msg
    assert '01_build_dataset' not in msg and '03_v2' not in msg


def test_merge_missing_v2_part_without_its_photos_names_step_3(tmp_path):
    drive = tmp_path / 'd'
    _parts(drive, [s for s in C.V2_SOURCES if s != 'background'])
    msg = _merge_error(drive, tmp_path, C.V2_SOURCES)
    assert 'background' in msg and '03_v2_collect_photos' in msg and '04_v2_label_and_train' in msg
    assert '01_build_dataset' not in msg


def test_merge_damaged_v1_part_names_step_1(tmp_path):
    drive = tmp_path / 'd'
    _parts(drive, [s for s in C.V2_SOURCES if s != 'openimages'], broken=['openimages'])
    msg = _merge_error(drive, tmp_path, C.V2_SOURCES)
    assert 'openimages.zip is damaged' in msg and '01_build_dataset' in msg and 'not step 2' in msg
    assert '04_v2' not in msg


def test_merge_damaged_v2_part_names_step_4(tmp_path):
    drive = tmp_path / 'd'
    _parts(drive, [s for s in C.V2_SOURCES if s != 'inat_weed'] + ['inat_weed_images'], broken=['inat_weed'])
    msg = _merge_error(drive, tmp_path, C.V2_SOURCES)
    assert 'inat_weed.zip is damaged' in msg and '04_v2_label_and_train' in msg
    assert '01_build_dataset' not in msg


def test_drop_big_box_images(tmp_path):
    out = tmp_path / 'ds'
    (out / 'train' / 'images').mkdir(parents=True)
    (out / 'train' / 'labels').mkdir(parents=True)
    side = lambda a: a ** 0.5                                      # noqa: E731  square box of area a
    cases = {'openimages__big': (0, C.OI_MAX_BOX_AREA + 0.1), 'openimages__ok': (1, C.OI_MAX_BOX_AREA - 0.1),
             'cottonweed__big': (0, C.OI_MAX_BOX_AREA + 0.1)}
    rows = []
    for stem, (cls, area) in cases.items():
        Image.new('RGB', (8, 8)).save(out / 'train' / 'images' / f'{stem}.jpg')
        (out / 'train' / 'labels' / f'{stem}.txt').write_text(f'{cls} 0.5 0.5 {side(area):.4f} {side(area):.4f}\n')
        rows.append({'file': f'train/images/{stem}.jpg', 'split': 'train', 'source': stem.split('__')[0],
                     'n_boxes': 1, 'orig_labels': '', 'licence': 'x'})
    for src in ('openimages', 'cottonweed'):
        with open(out / f'manifest_{src}.csv', 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(r for r in rows if r['source'] == src)
    assert C.drop_big_box_images(str(out)) == 1
    assert not (out / 'train' / 'images' / 'openimages__big.jpg').exists()
    assert not (out / 'train' / 'labels' / 'openimages__big.txt').exists()
    assert (out / 'train' / 'images' / 'openimages__ok.jpg').exists()
    assert (out / 'train' / 'images' / 'cottonweed__big.jpg').exists()
    with open(out / 'manifest_openimages.csv') as f:
        assert [r['file'] for r in _csv.DictReader(f)] == ['train/images/openimages__ok.jpg']
    with open(out / 'manifest_cottonweed.csv') as f:
        assert len(list(_csv.DictReader(f))) == 1


def test_label_images_model_error_keeps_cached_labels(tmp_path):
    paths = make_set(tmp_path, {'A': 30, 'B': 31, 'C': 32})
    cache = tmp_path / 'cache.jsonl'

    class DiesOnThird(NamedFake):
        def detect(self, image, queries, min_score=TE.RAW_MIN_SCORE):
            if self.calls == 2:
                raise OSError('model died')
            return super().detect(image, queries, min_score)

    with pytest.raises(OSError):
        TE.label_images(paths, 'weed', DiesOnThird({30: [good_det()], 31: [good_det()]}), str(cache), log_every=1000)
    assert len(cache.read_text().splitlines()) == 2             # the two finished photos were saved
    assert os.path.exists(paths[2])
    again = NamedFake({32: [good_det()]})
    res = TE.label_images(paths, 'weed', again, str(cache), log_every=1000)
    assert again.calls == 1 and res['from_cache'] == 2 and res['labelled'] == 3


# ---------------------------------------------------------------- backyard check (no model, no owner photo here)
import hashlib  # noqa: E402

from weed_detector import backyard_check as BC  # noqa: E402


def test_is_hit_centre_inside_iou_and_miss():
    ref = (0, 0, 10, 10)
    assert BC.is_hit((4, 4, 6, 6), ref)                    # centre inside (IoU only 0.04)
    assert BC.is_hit((0, 0, 22, 10), ref)                  # centre (11, 5) outside, IoU 0.45
    assert BC.is_hit((0, 0, 100, 10), (0, 0, 30, 10))      # centre outside, IoU exactly 0.3
    assert not BC.is_hit((6, 0, 16, 10), ref)              # centre outside, IoU 0.25
    assert not BC.is_hit((20, 20, 30, 30), ref)


def photo(n_refs, n_hit, stone=0, views=None, conf=0.5):
    """n_refs reference boxes in a row, the first n_hit of them hit by a quarter-tile weed box, + stone boxes."""
    refs = [(i * 100, 0, i * 100 + 50, 50) for i in range(n_refs)]
    weed = [((r[0] + 20, 20, r[0] + 30, 30), conf) for r in refs[:n_hit]]
    all_ = [(b, c, 'weed') for b, c in weed] + [((i * 100, 500, i * 100 + 20, 520), conf, 'keep') for i in range(stone)]
    return {'ref': refs, 'weed': weed, 'all': all_, 'views': [(1000, 1000, all_)] if views is None else views}


def test_backyard_verdict_hit_rate_70_percent():
    v = BC.backyard_verdict({'A': photo(10, 7)})
    a = v['photos']['A']
    assert (a['hits'], a['refs'], a['hit_rate'], a['pass']) == (7, 10, 0.7, True)
    assert v['pass'] is True and v['pooled_hit_rate'] == 0.7
    v = BC.backyard_verdict({'A': photo(10, 6)})
    assert v['photos']['A']['hit_rate'] == 0.6 and v['photos']['A']['pass'] is False and v['pass'] is False


def test_backyard_verdict_low_confidence_boxes_do_not_count():
    v = BC.backyard_verdict({'A': photo(10, 10, stone=3, conf=0.24)})
    assert v['photos']['A']['hits'] == 0 and v['photos']['A']['stone_boxes'] == 0


def test_backyard_verdict_stone_boxes():
    assert BC.backyard_verdict({'A': photo(10, 10, stone=2)})['pass'] is True
    v = BC.backyard_verdict({'A': photo(10, 10, stone=3)})
    assert v['photos']['A']['stone_boxes'] == 3 and v['pass'] is False
    p = photo(1, 1)                                        # ref (0, 0, 50, 50); both boxes have their centre outside it
    p['all'] += [((40, 0, 90, 50), 0.5, 'weed'),           # IoU 0.11 with the ref: on the weed, not a stone box
                 ((45, 0, 95, 50), 0.5, 'weed')]           # IoU 0.05: a stone box
    assert BC.backyard_verdict({'A': p})['photos']['A']['stone_boxes'] == 1


def test_backyard_verdict_box_over_half_a_view():
    big = BC.backyard_verdict({'A': photo(10, 10, views=[(100, 100, [((0, 0, 51, 100), 0.5, 'weed')])])})
    assert big['photos']['A']['oversize'] == 1 and big['pass'] is False
    half = BC.backyard_verdict({'A': photo(10, 10, views=[(100, 100, [((0, 0, 50, 100), 0.5, 'weed')])])})
    assert half['photos']['A']['oversize'] == 0 and half['pass'] is True


def test_backyard_verdict_every_photo_must_pass():
    v = BC.backyard_verdict({'A': photo(10, 8), 'B': photo(10, 5)})
    assert v['photos']['A']['pass'] is True and v['photos']['B']['pass'] is False
    assert v['pass'] is False and v['pooled_hit_rate'] == pytest.approx(0.65)


def test_backyard_verdict_photo_without_reference_boxes_fails():
    v = BC.backyard_verdict({'A': photo(0, 0)})
    assert v['photos']['A']['hit_rate'] is None and v['photos']['A']['pass'] is False
    assert v['pooled_hit_rate'] is None and v['pass'] is False


def test_letterbox_square_pads_grey_at_bottom_right():
    red, green, grey = (200, 10, 10), (10, 200, 10), (114, 114, 114)
    lb = BC.letterbox_square(Image.new('RGB', (400, 200), red))
    assert lb.size == (640, 640)
    assert lb.getpixel((0, 0)) == red and lb.getpixel((639, 319)) == red          # content 640 x 320 at the top
    assert lb.getpixel((0, 320)) == grey and lb.getpixel((639, 639)) == grey
    tall = BC.letterbox_square(Image.new('RGB', (100, 200), green))
    assert tall.getpixel((319, 639)) == green and tall.getpixel((320, 0)) == grey  # content 320 x 640 at the left


def test_quarter_tiles_cover_the_photo():
    im = Image.new('RGB', (9, 7))
    im.putpixel((8, 6), (255, 0, 0))
    tiles = BC.quarter_tiles(im)
    assert [(x, y, t.size) for x, y, t in tiles] == [(0, 0, (4, 3)), (4, 0, (5, 3)), (0, 3, (4, 4)), (4, 3, (5, 4))]
    assert tiles[3][2].getpixel((4, 3)) == (255, 0, 0)


def test_from_letterbox_undoes_scale_and_offset():
    view = (768, 1024, 768, 1024)            # bottom-right quarter of a 1536 x 2048 photo, shown at 480 x 640
    assert BC.from_letterbox((0, 0, 480, 640), view) == (768, 1024, 1536, 2048)
    assert BC.from_letterbox((48, 64, 96, 128), view) == pytest.approx((844.8, 1126.4, 921.6, 1228.8))
    assert BC.from_letterbox((400, 600, 600, 660), view) == pytest.approx((1408, 1984, 1536, 2048))  # cut at the view


def fb(conf=0.6, cls='weed', box=(10, 10, 110, 110)):
    return (box, conf, cls)


def test_fidelity_identical_lists_pass():
    a = [fb(), fb(0.3, 'keep', (200, 200, 260, 280))]
    f = BC.fidelity(a, list(a))
    assert f['pass'] is True and f['min_iou'] == 1.0 and f['max_conf_diff'] == 0.0
    assert BC.fidelity([], [])['pass'] is True


def test_fidelity_confidence_gap():
    f = BC.fidelity([fb(0.60)], [fb(0.66)])
    assert f['pass'] is False and f['max_conf_diff'] == pytest.approx(0.06)
    assert BC.fidelity([fb(0.60)], [fb(0.64)])['pass'] is True
    assert BC.fidelity([fb(0.27)], [fb(0.23)])['pass'] is True       # the partner may sit just under 0.25


def test_fidelity_extra_box_class_and_iou():
    f = BC.fidelity([fb()], [fb(), fb(0.4, 'weed', (300, 300, 350, 350))])
    assert f['pass'] is False and len(f['unmatched_ncnn']) == 1 and not f['unmatched_pt']
    assert BC.fidelity([fb()], [fb(), fb(0.2, 'weed', (300, 300, 350, 350))])['pass'] is True   # under 0.25: ignored
    assert BC.fidelity([fb(cls='weed')], [fb(cls='keep')])['pass'] is False
    f = BC.fidelity([fb()], [fb(box=(16, 10, 116, 110))])                                       # IoU 0.89
    assert f['pass'] is False and f['min_iou'] == pytest.approx(0.8868, abs=1e-3)


def make_photos(tmp_path, size=(200, 100)):
    d = tmp_path / 'photos'
    d.mkdir()
    for n in BC.PHOTOS:
        Image.new('RGB', size, (90, 90, 90)).save(d / f'{n}.JPG')
    return d


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def detect(self, image, queries, min_score=TE.RAW_MIN_SCORE):
        self.calls.append((image.size, list(queries), min_score))
        plant, distract = TE.QUERIES['weed']['plant'], TE.QUERIES['weed']['distract']
        return [TE.Det(plant[2], 0.3, 0.1, 0.2, 0.5, 0.6), TE.Det(distract[1], 0.9, 0.0, 0.0, 1.0, 1.0),
                TE.Det(plant[0], 0.12, 0.6, 0.6, 0.8, 0.9), TE.Det(plant[1], 0.08, 0.0, 0.0, 0.5, 0.5)]


def test_reference_candidates_in_photo_pixels(tmp_path):
    out, teacher = tmp_path / 'ref', FakeTeacher()
    BC.reference(str(out), photos_dir=str(make_photos(tmp_path)), teacher=teacher)
    assert teacher.calls == [((100, 50), TE.queries_for('weed'), 0.10)] * 8
    cands = json.loads((out / 'candidates.json').read_text())['candidates']
    assert [c['n'] for c in cands] == list(range(1, 17))          # 2 plant boxes >= 0.10 per tile, distractor dropped
    first, t3 = cands[0], [c for c in cands if c['photo'] == 'IMG_0229' and c['tile'] == 3]
    assert first['photo'] == 'IMG_0229' and first['tile'] == 0 and first['score'] == 0.3
    assert first['box'] == pytest.approx([10, 10, 50, 30])        # Det boxes are relative to the tile
    assert t3[0]['box'] == pytest.approx([110, 60, 150, 80])      # + the tile offset (100, 50)
    assert t3[1]['box'] == pytest.approx([160, 80, 180, 95])
    for n in BC.PHOTOS:
        for i in range(4):
            assert (out / 'drawings' / f'{n}_tile{i}.jpg').exists()
    assert not (out / 'reference.json').exists()


def test_reference_refuses_when_frozen_reference_exists(tmp_path):
    out = tmp_path / 'ref'
    out.mkdir()
    (out / 'reference.json').write_text('{}')
    teacher = FakeTeacher()
    with pytest.raises(SystemExit, match='frozen'):
        BC.reference(str(out), photos_dir=str(make_photos(tmp_path)), teacher=teacher)
    assert not teacher.calls


def test_check_refuses_without_reference(tmp_path):
    seen = []
    with pytest.raises(SystemExit, match='reference'):
        BC.check(str(tmp_path / 'model'), str(tmp_path / 'missing.json'), photos_dir=str(make_photos(tmp_path)),
                 predictors={'pt': seen.append, 'ncnn': seen.append})
    with pytest.raises(SystemExit, match='reference'):
        BC.main(['check', str(tmp_path / 'model'), str(tmp_path / 'missing.json')])   # before any model is loaded
    assert not seen and not (tmp_path / 'model').exists()


def test_check_refuses_a_bad_reference(tmp_path):
    photos = str(make_photos(tmp_path))
    for bad in ({'IMG_0229': [[0, 0, 10, 10]]}, {'IMG_0229': [[0, 0, 10]], 'IMG_0230': []},
                {'IMG_0229': [[10, 0, 0, 10]], 'IMG_0230': []}, ['not', 'a', 'dict'],
                {'IMG_0229': [['0', '0', '9', '9']], 'IMG_0230': []}):
        p = tmp_path / 'reference.json'
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match='reference'):
            BC.check(str(tmp_path / 'model'), str(p), photos_dir=photos, predictors={'pt': None, 'ncnn': None})


class FakeYolo:
    """Same box (on the 640 x 640 input) for every view; records what it was fed."""
    def __init__(self, conf=0.8):
        self.conf, self.fed = conf, []

    def __call__(self, img):
        self.fed.append((img.size, hashlib.md5(img.tobytes()).hexdigest()))
        return [((64, 64, 128, 128), self.conf, 'weed')]


def test_check_maps_views_scores_and_writes_report(tmp_path):
    # 200 x 100 photo: tiles 100 x 50 and the whole photo are shown at 640 x 320 -> the box (64, 64, 128, 128) is
    # (10, 10, 20, 20) + tile offset in every tile and (20, 20, 40, 40) in the whole photo.
    # IMG_0229: the 3rd reference box is hit only by the whole-photo box, which does not count for the hit rate
    ref = {'IMG_0229': [[8, 8, 22, 22], [110, 60, 120, 70], [22, 22, 40, 40]],       # 2 of 3 hit, 2 stone boxes
           'IMG_0230': [[10, 10, 20, 20], [110, 10, 120, 20], [10, 60, 20, 70], [110, 60, 120, 70]]}  # 4/4, 1 stone
    rp = tmp_path / 'reference.json'
    rp.write_text(json.dumps(ref))
    model = tmp_path / 'model'
    pt, ncnn = FakeYolo(), FakeYolo()
    rep = BC.check(str(model), str(rp), photos_dir=str(make_photos(tmp_path)), predictors={'pt': pt, 'ncnn': ncnn})
    assert pt.fed == ncnn.fed and len(pt.fed) == 10 and all(s == (640, 640) for s, _ in pt.fed)
    assert json.loads((model / 'backyard_check' / 'report.json').read_text()) == json.loads(json.dumps(rep))
    assert rep['reference_sha256'] == hashlib.sha256(rp.read_bytes()).hexdigest()
    a, b = rep['verdict']['photos']['IMG_0229'], rep['verdict']['photos']['IMG_0230']
    assert (a['hits'], a['refs'], a['stone_boxes'], a['oversize'], a['pass']) == (2, 3, 2, 0, False)
    assert (b['hits'], b['refs'], b['stone_boxes'], b['oversize'], b['pass']) == (4, 4, 1, 0, True)
    assert rep['verdict']['pass'] is False and rep['verdict']['pooled_hit_rate'] == pytest.approx(6 / 7)
    assert rep['backyard_check'] == 'FAIL' and rep['pi_version_check'] == 'PASS'
    boxes = rep['boxes']['IMG_0229']
    assert boxes['tile3']['view'] == [100, 50, 100, 50] and boxes['whole']['view'] == [0, 0, 200, 100]
    assert boxes['tile3']['pt'][0][0] == pytest.approx([110, 60, 120, 70])
    assert boxes['whole']['pt'][0][0] == pytest.approx([20, 20, 40, 40])
    assert rep['fidelity']['pass'] is True
    for n in BC.PHOTOS:
        for kind in ('tiles', 'whole'):
            assert (model / 'backyard_check' / f'{n}_{kind}.jpg').exists()


def test_check_reports_a_pi_version_mismatch(tmp_path):
    rp = tmp_path / 'reference.json'
    rp.write_text(json.dumps({'IMG_0229': [[8, 8, 22, 22]], 'IMG_0230': [[8, 8, 22, 22]]}))
    rep = BC.check(str(tmp_path / 'model'), str(rp), photos_dir=str(make_photos(tmp_path)),
                   predictors={'pt': FakeYolo(0.8), 'ncnn': FakeYolo(0.7)})
    assert rep['fidelity']['pass'] is False and rep['fidelity']['max_conf_diff'] == pytest.approx(0.1)
    assert rep['pi_version_check'] == 'FAIL'
    assert len(rep['fidelity']['failed_views']) == 10


# ---------------------------------------------------------------- unlisted zones: real weeds the teacher missed
def test_backyard_verdict_box_in_unlisted_zone_is_neither_stone_nor_hit():
    p = photo(10, 7)                                       # refs (i*100, 0, i*100+50, 50); refs 7, 8, 9 are missed
    p['unlisted'] = [(950, 0, 1000, 50)]                   # a weed next to ref 9 with no reference box
    box = (920, 0, 990, 50)                                # centre (955, 25) in the zone; IoU 0.33 with ref 9
    p['weed'].append((box, 0.5))
    p['all'].append((box, 0.5, 'weed'))
    a = BC.backyard_verdict({'A': p})['photos']['A']
    assert (a['hits'], a['stone_boxes'], a['on_unlisted'], a['pass']) == (7, 0, 1, True)
    del p['unlisted']                                      # without the zone the same box is a hit
    assert BC.backyard_verdict({'A': p})['photos']['A']['hits'] == 8


def test_backyard_verdict_three_boxes_on_unlisted_zones_pass_the_stone_rule():
    p = photo(10, 10, stone=2)                             # 2 real stone boxes: still allowed
    p['unlisted'] = [(i * 100, 800, i * 100 + 40, 840) for i in range(3)]
    p['all'] += [((i * 100 + 5, 805, i * 100 + 35, 835), 0.4, 'keep') for i in range(3)]
    a = BC.backyard_verdict({'A': p})['photos']['A']
    assert (a['stone_boxes'], a['on_unlisted'], a['pass']) == (2, 3, True)


def test_backyard_verdict_box_only_overlapping_an_unlisted_zone_is_a_stone_box():
    p = photo(10, 10)
    p['unlisted'] = [(500, 500, 540, 540)]
    p['all'].append(((520, 500, 600, 540), 0.5, 'weed'))  # centre (560, 520) outside the zone
    a = BC.backyard_verdict({'A': p})['photos']['A']
    assert (a['stone_boxes'], a['on_unlisted']) == (1, 0)


def test_backyard_verdict_oversize_box_on_unlisted_zone_still_fails():
    big = ((500, 600, 551, 700), 0.5, 'weed')               # 0.51 of a 100 x 100 view, centre in the zone
    p = photo(10, 10, views=[(100, 100, [big])])
    p['unlisted'] = [(500, 600, 600, 700)]
    p['all'].append(big)
    a = BC.backyard_verdict({'A': p})['photos']['A']
    assert (a['on_unlisted'], a['oversize'], a['pass']) == (1, 1, False)


def test_backyard_verdict_without_unlisted_is_unchanged():
    cases = [photo(10, 7), photo(10, 6, stone=3), photo(10, 10, views=[(100, 100, [((0, 0, 51, 100), 0.5, 'weed')])])]
    for p in cases:
        before = BC.backyard_verdict({'A': p})
        assert before == BC.backyard_verdict({'A': dict(p, unlisted=[])})
        assert before['photos']['A']['on_unlisted'] == 0


def test_check_uses_unlisted_json_next_to_the_reference(tmp_path):
    # same scene as test_check_maps_views_scores_and_writes_report: IMG_0229's stone boxes are the tile-1 box
    # (110, 10, 120, 20) and the tile-2 box (10, 60, 20, 70); a zone around the first one excuses only that one
    ref = {'IMG_0229': [[8, 8, 22, 22], [110, 60, 120, 70], [22, 22, 40, 40]],
           'IMG_0230': [[10, 10, 20, 20], [110, 10, 120, 20], [10, 60, 20, 70], [110, 60, 120, 70]]}
    d = tmp_path / 'refdir'
    d.mkdir()
    rp, up = d / 'reference.json', d / 'unlisted.json'
    rp.write_text(json.dumps(ref))
    photos = str(make_photos(tmp_path))
    run = lambda m: BC.check(str(tmp_path / m), str(rp), photos_dir=photos,          # noqa: E731
                             predictors={'pt': FakeYolo(), 'ncnn': FakeYolo()})
    plain = run('m1')
    assert plain['unlisted'] is None and plain['unlisted_sha256'] is None
    assert plain['verdict']['photos']['IMG_0229']['stone_boxes'] == 2
    up.write_text(json.dumps({'IMG_0229': [[105, 5, 125, 25]], 'IMG_0230': []}))
    rep = run('m2')
    a = rep['verdict']['photos']['IMG_0229']
    assert (a['stone_boxes'], a['on_unlisted'], a['hits']) == (1, 1, 2)
    assert rep['unlisted'] == str(up) and rep['unlisted_sha256'] == hashlib.sha256(up.read_bytes()).hexdigest()
    assert rep['reference_sha256'] == hashlib.sha256(rp.read_bytes()).hexdigest()
    up.write_text(json.dumps({'IMG_0229': [[5, 5, 1, 1]], 'IMG_0230': []}))
    with pytest.raises(SystemExit, match='unlisted'):
        run('m3')
    rp.unlink()
    with pytest.raises(SystemExit, match='reference'):                               # unlisted alone: refused
        run('m4')
