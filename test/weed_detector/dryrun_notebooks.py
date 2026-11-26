"""dryrun_notebooks.py - run the Colab notebooks' code cells on the VM (CPU), with a local folder in place
of Google Drive, small source limits and a tiny training run. Proves the notebook glue before Colab time is
spent. Needs internet and the training venv:
  env -u PYTHONPATH ~/weedbot_data/venv_yolo/bin/python test/weed_detector/dryrun_notebooks.py [steps]
steps (default: all, in this order):
  1 2 2b  v1: build the parts, train, then finish a run that Colab "stopped" after round 2 of 3 (a REAL
          interrupted training) with the finish-without-GPU notebook. 2b needs the parts from 1 and the
          dataset from 2.
  3       v2: FIRST deletes the dry run's own v2 outputs (on the fake Drive the inat_weed / inat_keep /
          background parts, the teacher caches, runs/v2 and weed_detector_v2.zip, plus their work folders;
          nothing of v1 and nothing outside ~/weedbot_data/dryrun), then collects 1 photo per iNaturalist
          taxon and 10 ground photos into the three *_images parts.
  4       v2: the real OWLv2 teacher on the CPU labels the three parts, merge, 1 round of fine-tuning from the
          v1 dry-run model, test, pack. Run it a second time: the parts print as skipped and the finished run
          goes straight to the test.
  4b      v2: a v2 run that really stopped after round 2 of 3 (on its own copy of the fake Drive) is finished
          with the finish-without-GPU notebook (needs 4).
  4r      v2: notebook 04 again on that stopped copy: labelling is skipped and training resumes at round 3
          (run it after 4b: it finishes the run).
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NB_DIR = os.path.join(REPO, 'training', 'weed_detector')
DRYRUN = os.path.expanduser('~/weedbot_data/dryrun')
FAKE_DRIVE = os.path.join(DRYRUN, 'drive', 'weedbot')
WORK = os.path.join(DRYRUN, 'work')
DS = os.path.join(DRYRUN, 'weeds_ds')
OUT = os.path.join(DRYRUN, 'weed_detector_v1')
STOPPED_DRIVE = os.path.join(DRYRUN, 'drive_stopped', 'weedbot')
STOPPED_OUT = os.path.join(DRYRUN, 'weed_detector_v1_stopped')
FINAL = os.path.join(DRYRUN, 'final')
WORK_OI = os.path.join(DRYRUN, 'work_oi')
# v2 (all of these are the dry run's own v2 outputs, see clean_v2)
WORK_V2 = os.path.join(DRYRUN, 'work_v2')
DS_V2 = os.path.join(DRYRUN, 'weeds_ds_v2')
OUT_V2 = os.path.join(DRYRUN, 'weed_detector_v2')
V1_FINAL = os.path.join(DRYRUN, 'v1_final')
FINAL_V2 = os.path.join(DRYRUN, 'final_v2')
STOPPED_DRIVE_V2 = os.path.join(DRYRUN, 'drive_stopped_v2', 'weedbot')
STOPPED_OUT_V2 = os.path.join(DRYRUN, 'weed_detector_v2_stopped')

sys.path.insert(0, os.path.join(REPO, 'training'))


def cells(nb):
    with open(os.path.join(NB_DIR, nb)) as f:
        for c in json.load(f)['cells']:
            if c['cell_type'] == 'code':
                yield ''.join(c['source'])


def adapt(src, drive=FAKE_DRIVE, out=OUT, final=FINAL):
    """Colab-only lines -> local equivalents (the logic itself is left untouched)."""
    if src.startswith('%%writefile') or src.startswith('!pip') or 'os.makedirs("/content/weed_detector"' in src:
        return None                                   # modules are imported from the repo instead
    src = src.replace("from google.colab import drive\ndrive.mount('/content/drive')\n", '')
    src = src.replace("from google.colab import drive\ndrive.flush_and_unmount()", "pass")
    src = src.replace("assert torch.cuda.is_available()", "assert True or torch.cuda.is_available()")
    src = src.replace("print('GPU:', torch.cuda.get_device_name(0))", "print('GPU: none (VM dry run)')")
    src = src.replace("'/content/drive/MyDrive/weedbot'", repr(drive))
    src = src.replace("sys.path.insert(0, '/content')", f"sys.path.insert(0, {os.path.join(REPO, 'training')!r})")
    src = src.replace("C.repair_after_merge(DS)", f"C.repair_after_merge(DS, tmp_dir={WORK_OI!r})")
    src = src.replace("'/content/work'", repr(WORK)).replace("'/content/weeds_ds'", repr(DS))
    src = src.replace("'/content/work_v2'", repr(WORK_V2)).replace("'/content/weeds_ds_v2'", repr(DS_V2))
    src = src.replace("'/content/weed_detector_v1'", repr(out)).replace("'/content/weed_detector_v2'", repr(out))
    src = src.replace("'/content/final'", repr(final)).replace("'/content/v1_final'", repr(V1_FINAL))
    # small limits: 30 CottonWeedDet12 images, 20 Open Images pictures, a tiny CPU training
    src = src.replace("lambda: C.cottonweed_items_from_hf(WORK)", "lambda: C.cottonweed_items_from_hf(WORK)[:30]")
    src = src.replace("max_images=3000)", "max_images=20, max_samples_per_split=80)")
    src = src.replace("epochs=100, imgsz=640)", "epochs=1, imgsz=320, batch=8, device='cpu', workers=0, fraction=0.05)")
    # v2: 1 photo per iNaturalist taxon (13 + 13), 10 ground photos, the same tiny CPU training
    src = src.replace("max_per_taxon=None)", "max_per_taxon=1)")
    src = src.replace("max_images=2000)", "max_images=10)")
    src = src.replace("epochs=50, patience=15, imgsz=640)",
                      "epochs=1, patience=15, imgsz=320, batch=8, device='cpu', workers=0, fraction=0.05)")
    src = src.replace("split='test', imgsz=640)", "split='test', imgsz=320, device='cpu')")
    src = src.replace("T.export(best, imgsz=640, formats=('onnx', 'ncnn'))", "T.export(best, imgsz=320, formats=('onnx', 'ncnn'))")
    return src


def run(nb, **kw):
    g = {'__name__': '__dryrun__'}
    for i, src in enumerate(cells(nb)):
        s = adapt(src, **kw)
        if s is None:
            continue
        print(f'\n===== {nb} cell {i} =====\n{s.strip()[:160]}')
        exec(compile(s, f'{nb}:cell{i}', 'exec'), g)


class _Stop(Exception):
    pass


def make_stopped_run(drive=STOPPED_DRIVE, ds=DS, name='v1', base='yolo11n.pt', copy=('dataset_parts',)):
    """A training run on `drive` (fresh, with copies of the fake Drive's `copy` folders) that really stopped
    after round 2 of 3 (like Colab's GPU limit)."""
    from ultralytics import YOLO
    shutil.rmtree(os.path.dirname(drive), ignore_errors=True)
    os.makedirs(drive)
    for sub in copy:
        shutil.copytree(os.path.join(FAKE_DRIVE, sub), os.path.join(drive, sub))

    def stop(trainer):
        if trainer.epoch == 2:
            raise _Stop()
    model = YOLO(base)
    model.add_callback('on_train_epoch_start', stop)
    try:
        model.train(data=os.path.join(ds, 'data.yaml'), project=os.path.join(drive, 'runs'), name=name,
                    exist_ok=True, epochs=3, imgsz=320, batch=8, device='cpu', workers=0, fraction=0.05)
    except _Stop:
        print('training stopped after round 2 of 3 (on purpose)')


def clean_v2():
    """Delete the dry run's OWN v2 outputs so the v2 steps start clean: v1's dry-run files stay, and nothing
    outside DRYRUN is ever touched."""
    import glob
    from weed_detector import colab as C
    parts = os.path.join(FAKE_DRIVE, 'dataset_parts')
    doomed = []
    for s in C.TEACHER_SOURCES:                        # inat_weed, inat_keep, background
        doomed += glob.glob(os.path.join(parts, s + '*'))       # <s>.zip, <s>_images.zip (+ a cut .part)
        doomed += [os.path.join(WORK, s)] + glob.glob(os.path.join(WORK, s + '_part*'))
    doomed += glob.glob(os.path.join(parts, 'teacher_*.jsonl'))
    doomed += [os.path.join(FAKE_DRIVE, 'runs', 'v2'), os.path.join(FAKE_DRIVE, 'weed_detector_v2.zip'),
               WORK_V2, DS_V2, OUT_V2, V1_FINAL, FINAL_V2, os.path.dirname(STOPPED_DRIVE_V2), STOPPED_OUT_V2]
    for p in doomed:
        if not os.path.realpath(p).startswith(DRYRUN + os.sep):
            raise RuntimeError(f'refusing to delete {p}: outside {DRYRUN}')
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
        else:
            continue
        print('dry run: removed', p)


if __name__ == '__main__':
    steps = sys.argv[1:] or ['1', '2', '2b', '3', '4', '4b', '4r']
    os.makedirs(FAKE_DRIVE, exist_ok=True)
    if '1' in steps:
        run('01_build_dataset.ipynb')
    if '2' in steps:
        run('02_train.ipynb')
    if '2b' in steps:
        make_stopped_run()
        shutil.rmtree(FINAL, ignore_errors=True)
        shutil.rmtree(STOPPED_OUT, ignore_errors=True)
        run('02b_finish_without_gpu.ipynb', drive=STOPPED_DRIVE, out=STOPPED_OUT)
    if '3' in steps:
        clean_v2()
        run('03_v2_collect_photos.ipynb')
    if '4' in steps:
        shutil.rmtree(OUT_V2, ignore_errors=True)
        run('04_v2_label_and_train.ipynb', out=OUT_V2)
    if '4b' in steps:
        from weed_detector import train as T
        base, _ = T.final_weights(os.path.join(FAKE_DRIVE, 'runs'), 'v1', V1_FINAL)    # as notebook 04 does
        make_stopped_run(STOPPED_DRIVE_V2, DS_V2, 'v2', str(base), copy=('dataset_parts', os.path.join('runs', 'v1')))
        shutil.rmtree(FINAL_V2, ignore_errors=True)
        shutil.rmtree(STOPPED_OUT_V2, ignore_errors=True)
        run('04b_v2_finish_without_gpu.ipynb', drive=STOPPED_DRIVE_V2, out=STOPPED_OUT_V2, final=FINAL_V2)
    if '4r' in steps:
        shutil.rmtree(STOPPED_OUT_V2, ignore_errors=True)
        run('04_v2_label_and_train.ipynb', drive=STOPPED_DRIVE_V2, out=STOPPED_OUT_V2)
