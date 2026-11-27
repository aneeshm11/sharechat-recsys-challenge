"""Staleness forensics.
H1: is the decay after the fit window concentrated in posts seen during fit (memorised post effects)?
H2 (oracle upper bound): how much would a FRESH post-level watch rate be worth?  Feature = leave-one-out mean label of the same post
    within +-W hours (true labels of other rows; an oracle - in the test period it would have to be estimated from the trace)."""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T0 = time.time(); T = '/tmp/sc_full1'; R = '/kaggle/working/run'; OUT = '/kaggle/working/run/fz2/pred'; os.makedirs(OUT, exist_ok=True)
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'
meta = pd.read_parquet(f'{B}/meta.parquet', columns=['split', 'ts', 'feature18', 'watched'])
NTR = int((meta.split == 0).sum()); H = 3600_000
ts = meta.ts.values[:NTR].astype(np.int64); post = meta.feature18.values[:NTR].astype(np.int64); y = meta.watched.values[:NTR].astype(np.float64)
th = ts / H; fit = th < 48; rows_val = np.flatnonzero(~fit); yv = y[rows_val] > 0.5; tv = th[rows_val]
seen = np.isin(post[rows_val], np.unique(post[fit]))
print('=== H1. hold-out AUC by hours since the end of the fit window, posts seen in fit vs new posts')
for nm, p in (('track_a', np.load(f'{R}/s1/out/pred/h_d5__val.npy')), ('track_b', np.load(f'{R}/pd/out/pred/ph_wd_lr030_s31__valfull.npy'))):
    for a, b in [(48, 51), (51, 54), (54, 60), (60, 66), (66, 72)]:
        m = (tv >= a) & (tv < b)
        print(f'  {nm} h{a}-{b}: rows {m.sum():6d} | seen {roc_auc_score(yv[m & seen], p[m & seen]):.4f} (share {(m & seen).sum() / m.sum():.3f}) | new {roc_auc_score(yv[m & ~seen], p[m & ~seen]):.4f}', flush=True)
# ---- oracle fresh post rate
o = np.lexsort((ts, post)); key = post[o] * 10 ** 10 + ts[o]; cy = np.r_[0, np.cumsum(y[o])]; N = NTR
F = np.empty((N, 8), np.float32); cols = []
for i, W in enumerate((1, 3, 12)):
    a = np.searchsorted(key, key - W * H, side='left'); b = np.searchsorted(key, key + W * H, side='right')
    s = cy[b] - cy[a] - y[o]; n = (b - a - 1).astype(np.float64)
    F[o, 2 * i] = np.where(n > 0, s / np.maximum(n, 1), np.nan); F[o, 2 * i + 1] = np.log1p(n); cols += [f'loo_rate_{W}h', f'loo_n_{W}h']
# past-only version (labels of the same post in the previous 6 h): what an online system could know
a = np.searchsorted(key, key - 6 * H, side='left'); b = np.searchsorted(key, key, side='left'); s = cy[b] - cy[a]; n = (b - a).astype(np.float64)
F[o, 6] = np.where(n > 0, s / np.maximum(n, 1), np.nan); F[o, 7] = np.log1p(n); cols += ['past6h_rate', 'past6h_n']
np.save(f'{T}/oracle_te.npy', F)
print('oracle features', cols, 'coverage (3h window has other rows): %.3f' % np.isfinite(F[:, 2]).mean(), flush=True)
P = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 50, 'colsample_bytree': 0.5, 'reg_lambda': 5.0}
EL = f'{T}/feat/EL_tr.npy'


def job(tag, ocols, gpu):
    def part(rp, yp, n):
        s = [{'path': EL, 'rows': rp, 'cols': None}] + ([{'path': f'{T}/oracle_te.npy', 'rows': rp, 'cols': ocols}] if ocols else [])
        return {'srcs': s, 'y': yp, 'n': n}
    j = {'tag': tag, 'params': P, 'train': [part(f'{T}/rows_fit.npy', f'{T}/y_fit.npy', int(fit.sum()))], 'evals': {'val': [part(f'{T}/rows_val.npy', f'{T}/y_val.npy', len(rows_val))]},
         'max_rounds': 4000, 'es': 100, 'out_dir': OUT, 'seed': 1, 'device': 'cuda', 'gpu': gpu, 'chunk': 100000, 'log_period': 500}
    jp = f'{OUT}/{tag}.json'; json.dump(j, open(jp, 'w'))
    return subprocess.Popen([sys.executable, f'{T}/xgbjob.py', jp], stdout=open(f'{OUT}/{tag}.log', 'w'), stderr=subprocess.STDOUT)


runs = [('base', None), ('oracle_1h', [0, 1]), ('oracle_3h', [2, 3]), ('oracle_12h', [4, 5]), ('oracle_all', [0, 1, 2, 3, 4, 5]), ('past6h', [6, 7])]
g = int(os.environ.get('SC_GPU', '1'))
for tag, oc in runs:
    job(tag, oc, g).wait()
    r = json.load(open(f'{OUT}/{tag}__res.json')); p = np.load(f'{OUT}/{tag}__val.npy')
    print(f'=== H2 {tag:11s} val AUC {r["auc"]["val"]:.5f} rounds {r["rounds"]} | by block: ' + ' '.join(f'h{a}-{b} {roc_auc_score(yv[(tv >= a) & (tv < b)], p[(tv >= a) & (tv < b)]):.4f}' for a, b in [(48, 54), (54, 60), (60, 66), (66, 72)]), flush=True)
print('done %.0fs' % (time.time() - T0))
