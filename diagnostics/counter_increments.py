"""Look at the post counters' increments over the 10-minute window after a row: which ones behave like watch / view counters,
and how does the label depend on (watch increments, view increments)?"""
import json, numpy as np, pandas as pd, time
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'; B = '/kaggle/input/datasets/cassinicraft/recsysbackup'
fut = json.load(open('/kaggle/input/datasets/cassinicraft/sharechat-nb-darkgray-models/fut_cols.json'))['fut_cols']; K = len(fut)
meta = pd.read_parquet(f'{B}/meta.parquet', columns=['split', 'watched']); NTR = int((meta.split == 0).sum()); y = meta.watched.values[:NTR] > 0.5
FH = np.load(f'{T}/blk/fh_full.npy', mmap_mode='r')                    # horizons 300, 600, 1200, 2400 ; K columns each ; then 4 gaps
sl = slice(0, NTR, 3); ys = y[sl]
D = np.asarray(FH[sl, K:2 * K], np.float64)                             # deltas at the first same-post row >= 600 s later
gap = np.expm1(np.asarray(FH[sl, 4 * K + 1], np.float64))
ok = np.isfinite(D[:, 0]); print('rows', len(ys), 'with a snapshot >= 10 min later: %.3f' % ok.mean(), '| median gap %.0fs' % np.nanmedian(gap))
rows = []
for j, c in enumerate(fut):
    d = D[ok, j]; m = np.isfinite(d)
    if m.sum() < 1000:
        continue
    d = d[m]; yy = ys[ok][m]
    rows.append((c, np.mean(d == 0), np.median(d), np.mean(d), np.mean(d < 0), np.mean(d == np.round(d)), roc_auc_score(yy, d) if d.std() > 0 else .5,
                 yy[d == 0].mean() if (d == 0).any() else np.nan, yy[d > 0].mean() if (d > 0).any() else np.nan))
df = pd.DataFrame(rows, columns=['col', 'zero', 'median', 'mean', 'neg', 'integer', 'auc', 'rate|d=0', 'rate|d>0'])
pd.set_option('display.width', 250)
print(df.sort_values('auc', ascending=False).head(30).round(4).to_string())
print('\ncounter-like columns (integer increments, never negative), sorted by mean increment:')
cnt = df[(df.integer > 0.98) & (df.neg < 0.01)].sort_values('mean', ascending=False)
print(cnt.round(4).to_string())
# joint behaviour of the strongest counter with candidate 'view' counters
best = df.sort_values('auc', ascending=False).iloc[0].col; jb = fut.index(best)
print('\nstrongest column:', best)
w = D[ok, jb]; yy = ys[ok]
for v in sorted(set(np.clip(w[np.isfinite(w)], -1, 6).astype(int))):
    m = np.clip(w, -1, 6).astype(int) == v
    print(f'  increment {v:2d}{"+" if v == 6 else " "}: rows {m.sum():7d} rate {yy[m].mean():.4f}')
for c in list(cnt.col.head(8)):
    jv = fut.index(c)
    if c == best:
        continue
    v = D[ok, jv]; m = np.isfinite(w) & np.isfinite(v) & (v > 0)
    r = np.clip(w[m], 0, None) / v[m]
    print(f'  ratio {best}/{c}: rows {m.sum()} AUC of ratio {roc_auc_score(yy[m], r):.4f} | AUC of {best} increment alone on same rows {roc_auc_score(yy[m], w[m]):.4f} | median {c} increment {np.median(v[m]):.0f}'
          f' | rate when {c} increment<=3: w=0 {yy[m][(v[m] <= 3) & (w[m] <= 0)].mean():.4f} (n {((v[m] <= 3) & (w[m] <= 0)).sum()}) w>=1 {yy[m][(v[m] <= 3) & (w[m] >= 1)].mean():.4f} (n {((v[m] <= 3) & (w[m] >= 1)).sum()})')
