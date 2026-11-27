"""Are any post-embedding heads shared across posts (keyed by author / category instead of post)?  If so, the trace has denser snapshots."""
import numpy as np, pandas as pd, time
T0 = time.time()
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'
n = 500000
meta = pd.read_parquet(f'{B}/meta.parquet').iloc[:n]
ts = meta.ts.values.astype(np.int64); post = meta.feature18.values.astype(np.int64); auth = meta.feature23.values.astype(np.int64); user = meta.feature5.values.astype(np.int64)
E = np.array(np.load('/tmp/sc_full1/feat/E_tr.npy', mmap_mode='r')[:n])            # in-memory copy (same rows as meta: train file order)
cats = np.array(np.load(f'{B}/r2_cats.npy', mmap_mode='r')[:n])
print('rows', n, 'posts', len(np.unique(post)), 'authors', len(np.unique(auth)), 'users', len(np.unique(user)), 'cats', cats.shape, '%.0fs' % (time.time() - T0), flush=True)


def pair_stats(key, other, name, max_dt=30_000, same_other=False):
    o = np.lexsort((ts, key)); k = key[o]; t = ts[o]; ot = other[o]
    m = (k[1:] == k[:-1]) & ((t[1:] - t[:-1]) < max_dt) & ((ot[1:] == ot[:-1]) if same_other else (ot[1:] != ot[:-1]))
    i = o[:-1][m]; j = o[1:][m]
    if len(i) > 100000:
        s = np.sort(np.random.default_rng(0).choice(len(i), 100000, replace=False)); i, j = i[s], j[s]
    A, Bm = E[i], E[j]
    d = np.abs(A - Bm).max(2); nz = (np.abs(A).max(2) > 0) & (np.abs(Bm).max(2) > 0)
    eq = [float(((d[:, h] == 0) & nz[:, h]).sum() / max(nz[:, h].sum(), 1)) for h in range(14)]
    cosm = []
    for h in range(14):
        a, b = A[:, h], Bm[:, h]; c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12); cosm.append(float(np.median(c[nz[:, h]])) if nz[:, h].any() else np.nan)
    print(f'{name:46s} pairs {len(i):7d} | identical  U ' + ' '.join(f'{x:.2f}' for x in eq[:7]) + ' | P ' + ' '.join(f'{x:.2f}' for x in eq[7:]), flush=True)
    print(f'{"":46s}               | med.cos    U ' + ' '.join(f'{x:.2f}' for x in cosm[:7]) + ' | P ' + ' '.join(f'{x:.2f}' for x in cosm[7:]), flush=True)


pair_stats(post, user, 'same post, different user, <30s')
pair_stats(auth, post, 'same author, DIFFERENT post, <30s')
pair_stats(user, post, 'same user & timestamp, different post', max_dt=1)
pair_stats(np.random.default_rng(1).integers(0, 50, n), post, 'random pairs (control)', max_dt=10 ** 12)
for c in range(cats.shape[1]):
    card = len(np.unique(cats[:, c]))
    if 5 < card < 100000:
        pair_stats(cats[:, c].astype(np.int64), post, f'same cat col {c} (card {card}), diff post, <5s', max_dt=5000)
print('done %.0fs' % (time.time() - T0), flush=True)
