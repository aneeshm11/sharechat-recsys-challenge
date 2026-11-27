"""Are post-embedding updates applied in synchronised (wall-clock) batches?  If yes the update latency of a view is predictable."""
import numpy as np, pandas as pd, time
T0 = time.time()
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'
meta = pd.read_parquet(f'{B}/meta.parquet', columns=['split', 'ts', 'feature18'])
N = int((meta.split == 0).sum()); ts = meta.ts.values[:N].astype(np.int64); post = meta.feature18.values[:N].astype(np.int64)
Em = np.load('/tmp/sc_full1/feat/E_tr.npy', mmap_mode='r')
o = np.lexsort((ts, post)); po = post[o]; to = ts[o]
same = po[1:] == po[:-1]; dt = to[1:] - to[:-1]
for head in (6, 3, 4):
    P = np.array(Em[:, 7 + head])[o]
    chg = (P[1:] != P[:-1]).any(1)
    print(f'--- post head {head}: consecutive same-post snapshots: {same.sum()} | changed share by gap: ' + ' '.join(
        f'{a}-{b}s {chg[same & (dt >= a * 1000) & (dt < b * 1000)].mean():.3f}' for a, b in [(0, 1), (1, 5), (5, 15), (15, 60), (60, 300), (300, 1200)]), flush=True)
    sel = same & (dt <= 10_000) & (dt > 0)
    tt = to[1:][sel]; cc = chg[sel]
    print(f'    tight pairs (<=10 s): {sel.sum()}  changed {cc.mean():.4f}')
    # change rate per 10 s wall-clock bin
    b = tt // 10_000; nb = int(b.max()) + 1
    cnt = np.bincount(b, minlength=nb).astype(float); ch = np.bincount(b, weights=cc, minlength=nb)
    ok = cnt >= 30; r = np.where(ok, ch / np.maximum(cnt, 1), np.nan)
    print(f'    10 s bins with >=30 pairs: {ok.sum()} | change rate quantiles (1,10,50,90,99,max): ' + ' '.join(f'{np.nanquantile(r, q):.3f}' for q in (.01, .1, .5, .9, .99, 1.0)))
    # periodicity: change share by phase for several candidate periods
    for per in (30, 60, 120, 300, 600, 900):
        ph = (tt // 1000) % per; nbin = min(per, 30); w = per / nbin
        k = (ph // w).astype(int); c1 = np.bincount(k, minlength=nbin).astype(float); c2 = np.bincount(k, weights=cc, minlength=nbin)
        rr = c2 / np.maximum(c1, 1)
        print(f'    period {per:4d}s: change share by phase min {rr.min():.4f} max {rr.max():.4f} ratio {rr.max() / max(rr.min(), 1e-9):.2f} | ' + ' '.join(f'{x:.3f}' for x in rr[:30]))
    # autocorrelation of the binned change rate (lags in 10 s)
    x = np.where(ok, r, np.nanmean(r)); x = x - x.mean(); ac = [float((x[:-l] * x[l:]).mean() / (x * x).mean()) for l in range(1, 91)]
    top = np.argsort(ac)[::-1][:8]
    print('    autocorrelation of 10 s change rate, top lags (s): ' + ', '.join(f'{(l + 1) * 10}:{ac[l]:.3f}' for l in top) + f' | lag10 {ac[0]:.3f} lag60 {ac[5]:.3f} lag300 {ac[29]:.3f} lag600 {ac[59]:.3f}')
print('done %.0fs' % (time.time() - T0))
