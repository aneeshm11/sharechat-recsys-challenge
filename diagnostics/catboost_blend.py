"""Hold-out value of adding the CatBoost member to the final XGBoost ensemble (configuration-level proxy), and the test-side correlation."""
import numpy as np, os
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'; R = '/kaggle/working/run'
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
yt = np.concatenate([np.load(f'{T}/y_val_{k}.npy') for k in range(3)]) > 0.5; yf = np.load(f'{T}/y_val.npy') > 0.5
L = lambda run, tag, k: lg(np.load(f'{R}/{run}/out/pred/{tag}__{k}.npy').astype(np.float64))
xg = {k: (24 * L('pu', 'ph_wfud_lr030_s61', k) + 4.5 * L('pf2', 'ph_wfd_lr030_s51', k)) / 28.5 for k in ('valthin', 'valfull')}
cb = {k: L('cb1', 'cbh_wfu_s71', k) for k in ('valthin', 'valfull')}
for w in (0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
    print(f'XGB ensemble proxy + {w:.2f} x CatBoost: thin {roc_auc_score(yt, (xg["valthin"] + w * cb["valthin"]) / (1 + w)):.5f} | full {roc_auc_score(yf, (xg["valfull"] + w * cb["valfull"]) / (1 + w)):.5f}')
p = f'{R}/cb2/out/pred/cbf_wfu_s71__test.npy'
if os.path.exists(p):
    ct = lg(np.load(p).astype(np.float64)); xt = lg(np.load(f'{R}/pu/out/pred/pf_wfud_lr030_s61__test.npy').astype(np.float64))
    print('CatBoost final on test: mean p %.5f | logit corr with an XGB final %.4f (hold-out models: %.4f)' % (np.load(p).mean(), np.corrcoef(ct, xt)[0, 1], np.corrcoef(cb['valfull'], L('pu', 'ph_wfud_lr030_s61', 'valfull'))[0, 1]))
else:
    print('CatBoost final not finished yet')
