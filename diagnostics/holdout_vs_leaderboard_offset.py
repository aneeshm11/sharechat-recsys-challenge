"""Forensics of the hold-out -> leaderboard offset (about -0.006 for every model family).  No test labels are used."""
import numpy as np, pandas as pd, time
from sklearn.metrics import roc_auc_score
T0 = time.time()
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'; R = '/kaggle/working/run'
meta = pd.read_parquet(f'{B}/meta.parquet', columns=['split', 'ts', 'feature18', 'feature5', 'feature23', 'watched'])
NTR = int((meta.split == 0).sum()); H = 3600_000
ts = meta.ts.values.astype(np.int64); post = meta.feature18.values; user = meta.feature5.values; auth = meta.feature23.values; y = meta.watched.values[:NTR] > 0.5
th = ts / H; fit = th[:NTR] < 48; val = ~fit; rows_val = np.flatnonzero(val)
print('rows train', NTR, 'test', len(ts) - NTR, '| fit', fit.sum(), 'val', val.sum())


def exp_auc(p):
    """AUC the model itself expects if its probabilities are calibrated: P(score of a positive > score of a negative)."""
    o = np.argsort(p, kind='mergesort'); q = p[o].astype(np.float64)
    cneg = np.cumsum(1 - q) - (1 - q)                     # expected negatives strictly below (ties ignored: scores are continuous)
    return float((q * cneg).sum() / (q.sum() * (1 - q).sum()))


def load(run, tag, key):
    return np.load(f'{R}/{run}/out/pred/{tag}__{key}.npy').astype(np.float64)


M = {'track_a row-only (d5)': (load('s1', 'h_d5', 'val'), load('s1', 'p_d5_s31', 'test')),
     'track_b trace (dense, lr.03)': (load('pd', 'ph_wd_lr030_s31', 'valfull'), load('pd', 'pf_wd_lr030_s31', 'test'))}
yv = y[rows_val]; tv = th[rows_val]; tt = th[NTR:]
print('\n=== D1. actual hold-out AUC vs the AUC the model expects from its own probabilities')
for nm, (pv, pt) in M.items():
    print(f'{nm:30s} val: actual {roc_auc_score(yv, pv):.5f} self-expected {exp_auc(pv):.5f} mean p {pv.mean():.5f} (rate {yv.mean():.5f}) | test: self-expected {exp_auc(pt):.5f} mean p {pt.mean():.5f}')
    for a, b in [(72, 96), (96, 120), (120, 150)]:
        m = (tt >= a) & (tt < b); print(f'{"":30s}   test hours {a}-{b}: rows {m.sum():7d} self-expected {exp_auc(pt[m]):.5f} mean p {pt[m].mean():.5f}')
    for a, b in [(48, 54), (54, 60), (60, 66), (66, 72)]:
        m = (tv >= a) & (tv < b); print(f'{"":30s}   val hours {a}-{b}: rows {m.sum():7d} actual {roc_auc_score(yv[m], pv[m]):.5f} self-expected {exp_auc(pv[m]):.5f} rate {yv[m].mean():.4f} mean p {pv[m].mean():.4f}')

print('\n=== D2. entities seen / not seen in the training window')
for nm, ent in (('post', post), ('author', auth), ('user', user)):
    seen_fit = np.isin(ent[rows_val], np.unique(ent[:NTR][fit])); seen_tr = np.isin(ent[NTR:], np.unique(ent[:NTR]))
    print(f'{nm:7s} val rows on {nm}s unseen in fit: {1 - seen_fit.mean():.4f} | test rows on {nm}s unseen in train: {1 - seen_tr.mean():.4f} | by test day: ' + ' '.join(f'{a}-{b}h {1 - seen_tr[(tt >= a) & (tt < b)].mean():.3f}' for a, b in [(72, 96), (96, 120), (120, 150)]))
    for mn, (pv, pt) in M.items():
        print(f'        {mn:30s} val AUC seen {roc_auc_score(yv[seen_fit], pv[seen_fit]):.5f} (rate {yv[seen_fit].mean():.4f}) | unseen {roc_auc_score(yv[~seen_fit], pv[~seen_fit]):.5f} (rate {yv[~seen_fit].mean():.4f})'
              f' | self-expected on test: seen {exp_auc(pt[seen_tr]):.5f} unseen {exp_auc(pt[~seen_tr]):.5f}')
# number of training impressions of the post
cnt_fit = pd.Series(post[:NTR][fit]).value_counts(); cv = pd.Series(post[rows_val]).map(cnt_fit).fillna(0).values
cnt_tr = pd.Series(post[:NTR]).value_counts(); ct = pd.Series(post[NTR:]).map(cnt_tr).fillna(0).values
print('\nrows by number of training impressions of the post (val w.r.t. fit | test w.r.t. train), and val AUC per bucket')
for a, b in [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)]:
    mv = (cv >= a) & (cv < b); mt = (ct >= a) & (ct < b)
    print(f'  {a:>4.0f}-{b:<6.0f} val share {mv.mean():.4f} test share {mt.mean():.4f} | ' + ' | '.join(f'{mn[:8]} val AUC {roc_auc_score(yv[mv], pv[mv]):.4f}' for mn, (pv, pt) in M.items()))

print('\n=== D3. reweighting the hold-out to the test mix of post exposure buckets (what AUC would the hold-out give with the test composition?)')
edges = [0, 1, 2, 5, 10, 20, 50, 100, 300, 1e9]
bv = np.digitize(cv, edges) - 1; bt = np.digitize(ct, edges) - 1
w = np.array([(bt == k).mean() / max((bv == k).mean(), 1e-9) for k in range(len(edges) - 1)])[bv]
for mn, (pv, pt) in M.items():
    print(f'  {mn:30s} val AUC {roc_auc_score(yv, pv):.5f} -> reweighted to test post-exposure mix {roc_auc_score(yv, pv, sample_weight=w):.5f}')
print('done %.0fs' % (time.time() - T0))
