"""Hold-out blend of the track_b production configurations (one hold-out model per configuration) for a few weightings."""
import numpy as np, os, itertools
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'; R = '/kaggle/working/run'
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
yt = np.concatenate([np.load(f'{T}/y_val_{k}.npy') for k in range(3)]) > 0.5; yf = np.load(f'{T}/y_val.npy') > 0.5
H = {'wm': ('go2', 'ph_wm_lr030_s21'), 'wd': ('pd', 'ph_wd_lr030_s31'), 'wfd': ('pf2', 'ph_wfd_lr030_s51'), 'wfud': ('pu', 'ph_wfud_lr030_s61'),
     'cb': ('cb1', 'cbh_wfu_s71'), 'wfum': ('pm', 'ph_wfum_lr030_s81')}
Z = {}
for k, (run, tag) in H.items():
    p = f'{R}/{run}/out/pred/{tag}__valthin.npy'
    if os.path.exists(p):
        Z[k] = (lg(np.load(p).astype(np.float64)), lg(np.load(p.replace('valthin', 'valfull')).astype(np.float64)))
        print(f'{k:5s} thin {roc_auc_score(yt, Z[k][0]):.5f} full {roc_auc_score(yf, Z[k][1]):.5f}')
ks = list(Z)
print('rank corr (thin):'); import pandas as pd
print(pd.DataFrame({k: pd.Series(Z[k][0]).rank() for k in ks}).corr().round(4).to_string())


def ev(w):
    s = sum(w.values()); zt = sum(w[k] * Z[k][0] for k in w) / s; zf = sum(w[k] * Z[k][1] for k in w) / s
    return roc_auc_score(yt, zt), roc_auc_score(yf, zf)


cands = [{k: 1 for k in ks}, {'wm': 1, 'wd': 3, 'wfd': 3, 'wfud': 3}, {'wm': 1, 'wd': 1, 'wfd': 2, 'wfud': 4}, {'wfd': 1, 'wfud': 1}, {'wfd': 1, 'wfud': 2}, {'wfd': 1, 'wfud': 3},
         {'wd': 1, 'wfd': 2, 'wfud': 6}, {'wm': 1, 'wd': 1, 'wfd': 1, 'wfud': 6}, {'wfud': 1}]
for w in cands:
    w = {k: v for k, v in w.items() if k in Z}
    a, b = ev(w); print(f'weights {w}: thin {a:.5f} | full {b:.5f}')
