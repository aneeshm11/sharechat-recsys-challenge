"""Generic XGBoost job: python xgbjob.py job.json
Feature matrices are assembled chunk-by-chunk from .npy memmaps (never fully materialised on the host)."""
import json, os, sys, time
job = json.load(open(sys.argv[1]))
USE_GPU = job.get('device', 'cuda') == 'cuda'
if USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job.get('gpu', 0))
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
if USE_GPU:
    import cupy as cp
CH = int(job.get('chunk', 100000)); T0 = time.time(); TAG = job['tag']; OUT = job['out_dir']
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] [{TAG}] " + ' '.join(str(x) for x in a), flush=True)


_mm = {}


def mm(path):
    if path not in _mm:
        _mm[path] = np.load(path, mmap_mode='r')
    return _mm[path]


def load_rows(p):
    return None if p is None else np.load(p)


def part_chunks(part):
    n = int(part['n']); srcs = []
    for s in part['srcs']:
        rows = load_rows(s.get('rows'))
        contiguous = rows is None or (len(rows) > 0 and int(rows[-1]) - int(rows[0]) + 1 == len(rows))
        start = 0 if rows is None else int(rows[0])
        if rows is not None:
            assert len(rows) == n, (s['path'], len(rows), n)
        cols = None if s.get('cols') is None else np.asarray(s['cols'], np.int64)
        srcs.append((mm(s['path']), rows, contiguous, start, cols))
    y = np.load(part['y']).astype(np.float32) if part.get('y') else None
    if y is not None:
        assert len(y) == n
    for a in range(0, n, CH):
        b = min(n, a + CH); blocks = []
        for M, rows, contiguous, start, cols in srcs:
            X = M[start + a:start + b] if contiguous else M[rows[a:b]]
            X = np.asarray(X, dtype=np.float32)
            if cols is not None:
                X = X[:, cols]
            blocks.append(X)
        yield (np.hstack(blocks) if len(blocks) > 1 else np.ascontiguousarray(blocks[0])), (None if y is None else y[a:b])


def labels_of(parts):
    return np.concatenate([np.load(p['y']).astype(np.float32) for p in parts])


class It(xgb.DataIter):
    def __init__(self, parts):
        self.parts = parts; self.gen = None
        super().__init__()

    def reset(self):
        self.gen = (c for p in self.parts for c in part_chunks(p))

    def next(self, input_data):
        try:
            X, y = next(self.gen)
        except StopIteration:
            return False
        if USE_GPU:
            input_data(data=cp.asarray(X), label=cp.asarray(y))
        else:
            input_data(data=X, label=y)
        return True


def predict_parts(b, parts, rounds):
    out = []
    for p in parts:
        for X, _ in part_chunks(p):
            pr = b.inplace_predict(cp.asarray(X) if USE_GPU else X, iteration_range=(0, rounds))
            out.append(cp.asnumpy(pr) if USE_GPU else np.asarray(pr))
    return np.concatenate(out).astype(np.float32)


params = dict(job['params']); params['device'] = 'cuda' if USE_GPU else 'cpu'; params.setdefault('tree_method', 'hist')
params['seed'] = int(job.get('seed', 0))
if not USE_GPU:
    params['nthread'] = int(job.get('nthread', os.cpu_count()))
res = {'tag': TAG, 'auc': {}, 'n_train': int(sum(p['n'] for p in job['train']))}
dtr = xgb.QuantileDMatrix(It(job['train']), max_bin=int(job.get('max_bin', 256)))
log('train matrix', dtr.num_row(), 'x', dtr.num_col())
ev_names = list(job.get('evals', {}).keys())
es_set = job.get('es_set') or (ev_names[-1] if ev_names else None)
ev_names = [k for k in ev_names if k != es_set] + ([es_set] if es_set else [])          # early stopping uses the LAST set
dev = {k: xgb.QuantileDMatrix(It(job['evals'][k]), ref=dtr) for k in ev_names}
evals = [(dev[k], k) for k in ev_names]
period = int(job.get('log_period', 200))
if job.get('fixed_rounds'):
    rounds = int(job['fixed_rounds'])
    b = xgb.train(params, dtr, rounds, evals=evals, verbose_eval=period)
else:
    b = xgb.train(params, dtr, int(job.get('max_rounds', 5000)), evals=evals, early_stopping_rounds=int(job.get('es', 100)), verbose_eval=period)
    rounds = int(b.best_iteration) + 1
res['rounds'] = rounds; res['train_time'] = round(time.time() - T0, 1)
for k in ev_names:
    p = b.predict(dev[k], iteration_range=(0, rounds)).astype(np.float32)
    np.save(f'{OUT}/{TAG}__{k}.npy', p)
    yk = labels_of(job['evals'][k])
    res['auc'][k] = float(roc_auc_score(yk > 0.5, p))
log('rounds', rounds, 'AUC', {k: round(v, 5) for k, v in res['auc'].items()})
del dev, evals, dtr
for k, parts in (job.get('predict') or {}).items():
    p = predict_parts(b, parts, rounds); np.save(f'{OUT}/{TAG}__{k}.npy', p)
    log('predicted', k, len(p), 'mean %.5f' % float(p.mean()))
if job.get('save_model'):
    b.save_model(f'{OUT}/{TAG}.ubj')
fn = job.get('fnames')
if fn:
    sc = b.get_score(importance_type='total_gain')
    g = {fn[int(k[1:])]: float(v) for k, v in sc.items()}
    json.dump(g, open(f'{OUT}/{TAG}__gain.json', 'w'))
    top = sorted(g.items(), key=lambda kv: -kv[1])[:25]
    log('top gain: ' + ', '.join(f'{k}:{v:.0f}' for k, v in top))
res['time'] = round(time.time() - T0, 1)
json.dump(res, open(f'{OUT}/{TAG}__res.json', 'w'))
log('DONE', json.dumps(res))
