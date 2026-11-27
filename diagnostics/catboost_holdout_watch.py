"""Wait for the CatBoost hold-out job to finish, then kill the CatBoost run before its final fit and report the blend value."""
import json, os, signal, time
import numpy as np
from sklearn.metrics import roc_auc_score
T = '/tmp/sc_full1'; R = '/kaggle/working/run'
res = f'{R}/cb1/out/pred/cbh_wfu_s71__res.json'
t0 = time.time()
while not os.path.exists(res) and time.time() - t0 < 1500:
    time.sleep(5)
try:
    pid = int(open('/kaggle/working/logs/cb1.pid').read().strip()); os.killpg(pid, signal.SIGTERM); print('stopped CatBoost run', pid)
except Exception as e:
    print('kill:', e)
if os.path.exists(res):
    r = json.load(open(res)); print('CatBoost hold-out', r['auc'], 'rounds', r['rounds'])
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    yt = np.concatenate([np.load(f'{T}/y_val_{k}.npy') for k in range(3)]) > 0.5; yf = np.load(f'{T}/y_val.npy') > 0.5
    x = (lg(np.load(f'{R}/pu/out/pred/ph_wfud_lr030_s61__valthin.npy').astype(np.float64)), lg(np.load(f'{R}/pu/out/pred/ph_wfud_lr030_s61__valfull.npy').astype(np.float64)))
    c = (lg(np.load(f'{R}/cb1/out/pred/cbh_wfu_s71__valthin.npy').astype(np.float64)), lg(np.load(f'{R}/cb1/out/pred/cbh_wfu_s71__valfull.npy').astype(np.float64)))
    print('rank corr with the XGB model (thin): %.4f' % np.corrcoef(np.argsort(np.argsort(x[0])), np.argsort(np.argsort(c[0])))[0, 1])
    for w in (0, 0.1, 0.2, 0.3, 0.5):
        print(f'  XGB + {w:.1f} x CatBoost: thin {roc_auc_score(yt, (x[0] + w * c[0]) / (1 + w)):.5f} | full {roc_auc_score(yf, (x[1] + w * c[1]) / (1 + w)):.5f}')
else:
    print('no CatBoost result')
