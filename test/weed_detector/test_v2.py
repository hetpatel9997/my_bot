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
    assert sum(n for _, n in S.INAT_WEED_TAXA.values()) == 7990
    assert sum(n for _, n in S.INAT_KEEP_TAXA.values()) == 2990


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
