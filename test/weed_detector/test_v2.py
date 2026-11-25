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
