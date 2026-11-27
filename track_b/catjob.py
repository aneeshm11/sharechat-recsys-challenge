"""CatBoost (GPU) job with the same job-spec as xgbjob.py: python catjob.py job.json
The training matrix is materialised on the host (float32); evaluation / test parts are predicted chunk by chunk."""
import json, os, sys, time
job = json.load(open(sys.argv[1]))
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
CH = int(job.get('chunk', 100000)); T0 = time.time(); TAG = job['tag']; OUT = job['out_dir']
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] [{TAG}] " + ' '.join(str(x) for x in a), flush=True)


_mm = {}


def mm(path):
    if path not in _mm:
        _mm[path] = np.load(path, mmap_mode='r')
    return _mm[path]


def part_chunks(part):
    n = int(part['n']); srcs = []
    for s in part['srcs']:
        rows = None if s.get('rows') is None else np.load(s['rows'])
        contiguous = rows is None or (len(rows) > 0 and int(rows[-1]) - int(rows[0]) + 1 == len(rows))
        start = 0 if rows is None else int(rows[0])
        cols = None if s.get('cols') is None else np.asarray(s['cols'], np.int64)
        srcs.append((mm(s['path']), rows, contiguous, start, cols))
    for a in range(0, n, CH):
        b = min(n, a + CH); blocks = []
        for M, rows, contiguous, start, cols in srcs:
            X = np.asarray(M[start + a:start + b] if contiguous else M[rows[a:b]], dtype=np.float32)
            blocks.append(X if cols is None else X[:, cols])
        yield np.hstack(blocks) if len(blocks) > 1 else np.ascontiguousarray(blocks[0])


def materialise(parts, stride=1):
    n = sum(len(range(0, int(p['n']), stride)) for p in parts); X = None; o = 0
    for p in parts:
        off = 0
        for C in part_chunks(p):
            first = (-off) % stride; Cs = C[first::stride]; off += len(C)
            if X is None:
                X = np.empty((n, C.shape[1]), np.float32)
            X[o:o + len(Cs)] = Cs; o += len(Cs)
    assert o == n, (o, n)
    return X


def labels_of(parts):
    return np.concatenate([np.load(p['y']).astype(np.float32) for p in parts])


def predict_parts(m, parts, ntree):
    out = []
    for p in parts:
        for C in part_chunks(p):
            out.append(m.predict_proba(C, ntree_end=ntree)[:, 1].astype(np.float32))
    return np.concatenate(out)


P = dict(job['params'])
kw = dict(task_type='GPU', devices=str(job.get('gpu', 0)), loss_function='Logloss', eval_metric='AUC', learning_rate=P.get('learning_rate', 0.08), depth=P.get('depth', 8),
          l2_leaf_reg=P.get('l2_leaf_reg', 5.0), border_count=P.get('border_count', 128), bootstrap_type='Bernoulli', subsample=P.get('subsample', 0.8),
          random_seed=int(job.get('seed', 0)), allow_writing_files=False, metric_period=int(job.get('metric_period', 100)),
          verbose=int(job.get('metric_period', 100)) * max(1, int(job.get('log_period', 200)) // int(job.get('metric_period', 100))))
Xtr = materialise(job['train']); ytr = labels_of(job['train']); log('train matrix', Xtr.shape)
res = {'tag': TAG, 'auc': {}, 'n_train': int(len(ytr))}
ev = job.get('evals') or {}
es_set = job.get('es_set') or (list(ev)[-1] if ev else None)
if job.get('fixed_rounds'):
    m = CatBoostClassifier(iterations=int(job['fixed_rounds']), **kw); m.fit(Pool(Xtr, ytr)); rounds = int(job['fixed_rounds'])
else:
    st = int(job.get('es_stride', 4))                                   # early stopping on every 4th hold-out row keeps host memory low
    Xe = materialise(ev[es_set], st); ye = np.concatenate([np.load(p['y']).astype(np.float32)[::st] for p in ev[es_set]])
    m = CatBoostClassifier(iterations=int(job.get('max_rounds', 6000)), od_type='Iter', od_wait=int(job.get('es', 3)), use_best_model=False, **kw)
    m.fit(Pool(Xtr, ytr), eval_set=Pool(Xe, ye))
    rounds = int(m.get_best_iteration()) + 1; del Xe
del Xtr
res['rounds'] = rounds; res['train_time'] = round(time.time() - T0, 1)
for k, parts in ev.items():
    p = predict_parts(m, parts, rounds); np.save(f'{OUT}/{TAG}__{k}.npy', p); res['auc'][k] = float(roc_auc_score(labels_of(parts) > 0.5, p))
log('rounds', rounds, 'AUC', {k: round(v, 5) for k, v in res['auc'].items()})
for k, parts in (job.get('predict') or {}).items():
    p = predict_parts(m, parts, rounds); np.save(f'{OUT}/{TAG}__{k}.npy', p); log('predicted', k, len(p), 'mean %.5f' % float(p.mean()))
if job.get('save_model'):
    m.save_model(f'{OUT}/{TAG}.cbm')
res['time'] = round(time.time() - T0, 1)
json.dump(res, open(f'{OUT}/{TAG}__res.json', 'w'))
log('DONE', json.dumps(res))
