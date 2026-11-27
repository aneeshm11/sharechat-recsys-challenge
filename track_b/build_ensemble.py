"""Build an ensemble submission from finished final models and report the matching hold-out blend.
usage (on the kernel): python s12_ens.py <out_name> <run:holdrun/holdtag:finaltag[,finaltag...]:weight_per_final> ...
  holdrun/holdtag names the hold-out model that stands for these finals in the hold-out blend (may live in another run)."""
import os, sys, glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'; R = '/kaggle/working/run'; OUT = '/kaggle/working/subs'; os.makedirs(OUT, exist_ok=True)
name = sys.argv[1]; specs = [a.split(':') for a in sys.argv[2:]]
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
yt = np.concatenate([np.load(f'{T}/y_val_{k}.npy') for k in range(3)]) > 0.5; yf = np.load(f'{T}/y_val.npy') > 0.5
zt = zf = zs = 0; wt = ws = 0.0; manifest = []; allz = []
for sp in specs:
    run, hold, finals = sp[:3]; w = float(sp[3]) if len(sp) > 3 else 1.0
    P = f'{R}/{run}/out/pred'; hr, ht = hold.split('/') if '/' in hold else (run, hold); HP = f'{R}/{hr}/out/pred'
    fl = [f for f in finals.split(',') if os.path.exists(f'{P}/{f}__test.npy')]
    if not fl:
        print('  (skip, no final yet)', run, finals); continue
    a = lg(np.load(f'{HP}/{ht}__valthin.npy').astype(np.float64)); b = lg(np.load(f'{HP}/{ht}__valfull.npy').astype(np.float64))
    print(f'  {ht:22s} thin {roc_auc_score(yt, a):.5f} full {roc_auc_score(yf, b):.5f} | finals used: {len(fl)} x weight {w} {fl}')
    zt = zt + w * len(fl) * a; zf = zf + w * len(fl) * b; wt += w * len(fl)
    for f in fl:
        z = lg(np.load(f'{P}/{f}__test.npy').astype(np.float64)); zs = zs + w * z; ws += w; allz.append(z)
        manifest.append({'run': run, 'file': f + '.ubj', 'weight': w})
print(f'HOLD-OUT blend (weights = sum of final weights per configuration): thin {roc_auc_score(yt, zt / wt):.5f} | full {roc_auc_score(yf, zf / wt):.5f}')
pred = 1 / (1 + np.exp(-zs / ws))
ids = pq.read_table(glob.glob('/kaggle/input/**/test.parquet', recursive=True)[0], columns=['id']).column('id').to_numpy().astype(np.int64)
sub = pd.DataFrame({'id': ids, 'watched': np.clip(pred, 1e-7, 1 - 1e-7)})
ss = pd.read_csv(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)[0], usecols=['id']); sub = ss.merge(sub, on='id', how='left')
assert len(sub) == len(ss) and sub['watched'].notna().all()
sub.to_csv(f'{OUT}/{name}.csv', index=False, float_format='%.7f'); json.dump(manifest, open(f'{OUT}/{name}.models.json', 'w'), indent=1)
c = np.corrcoef(np.stack(allz))
print(f'wrote {OUT}/{name}.csv rows {len(sub)} models {len(allz)} mean {sub["watched"].mean():.5f} | pairwise logit corr min {c[np.triu_indices(len(allz), 1)].min() if len(allz) > 1 else 1:.4f} max {c[np.triu_indices(len(allz), 1)].max() if len(allz) > 1 else 1:.4f}')
