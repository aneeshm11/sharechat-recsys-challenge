"""Where is the remaining error?  Hold-out (test-like density) AUC by the time until the post is next seen in the file."""
import numpy as np, sys
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'
pred_path = sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working/run/pd/out/pred/ph_wd_lr030_s31__valthin.npy'
p = np.load(pred_path).astype(np.float64)
G = [np.load(f'{T}/g_val_{k}.npy') for k in range(3)]; L = [np.load(f'{T}/l_val_{k}.npy') for k in range(3)]
y = np.concatenate([np.load(f'{T}/y_val_{k}.npy') for k in range(3)]) > 0.5
gap = np.concatenate([np.asarray(np.load(f'{T}/blk/cb{k}.npy', mmap_mode='r')[L[k], 52]) for k in range(3)]).astype(np.float64)   # tr_gap_next (seconds)
g = np.concatenate(G); nfit = int(np.load(f'{T}/rows_fit.npy')[-1]) + 1
row = np.load('/kaggle/working/run/s1/out/pred/h_d5__val.npy').astype(np.float64)[g - nfit]
assert len(p) == len(y) == len(gap) == len(row)
print(f'rows {len(y)}  rate {y.mean():.4f}  AUC trace model {roc_auc_score(y, p):.5f}  row-only model {roc_auc_score(y, row):.5f}')
edges = [0, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 21600, 1e12]
print(f'{"next snapshot after":>22s} {"share":>7s} {"cum":>6s} {"rate":>7s} {"AUC trace":>10s} {"AUC row":>9s} {"lift":>7s}')
cum = 0
for a, b in zip(edges[:-1], edges[1:]):
    m = (gap >= a) & (gap < b); cum += m.mean()
    print(f'{f"{a:.0f}-{b:.0f}s" if b < 1e11 else f">{a:.0f}s":>22s} {m.mean():7.3f} {cum:6.3f} {y[m].mean():7.4f} {roc_auc_score(y[m], p[m]):10.4f} {roc_auc_score(y[m], row[m]):9.4f} {roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], row[m]):7.4f}')
m = ~np.isfinite(gap)
print(f'{"never (in file)":>22s} {m.mean():7.3f} {"":6s} {y[m].mean():7.4f} {roc_auc_score(y[m], p[m]):10.4f} {roc_auc_score(y[m], row[m]):9.4f} {roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], row[m]):7.4f}')
# how much of the global pair error sits in pairs whose positive has no early snapshot
order = np.argsort(p); rk = np.empty(len(p)); rk[order] = np.arange(len(p))
pos = np.flatnonzero(y); neg_rk = np.sort(rk[~y]); nneg = len(neg_rk)
below = np.searchsorted(neg_rk, rk[pos])                      # negatives ranked below each positive
err_per_pos = 1 - below / nneg
print('\nshare of all pair-ordering error carried by positives, by time to the next snapshot')
tot = err_per_pos.sum()
for a, b in zip(edges[:-1], edges[1:]):
    m = (gap[pos] >= a) & (gap[pos] < b)
    print(f'{f"{a:.0f}-{b:.0f}s" if b < 1e11 else f">{a:.0f}s":>22s} positives {m.mean():6.3f}  error share {err_per_pos[m].sum() / tot:6.3f}  mean miss-rank {err_per_pos[m].mean():.4f}')
m = ~np.isfinite(gap[pos]); print(f'{"never":>22s} positives {m.mean():6.3f}  error share {err_per_pos[m].sum() / tot:6.3f}  mean miss-rank {err_per_pos[m].mean():.4f}')
