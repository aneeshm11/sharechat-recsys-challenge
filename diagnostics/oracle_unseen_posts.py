"""Does the oracle (true recent watch rate of the post) help on posts unseen in the fit window?  If not, the seen/unseen gap is intrinsic."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'; P = '/kaggle/working/run/fz2/pred'; T = '/tmp/sc_full1'
meta = pd.read_parquet(f'{B}/meta.parquet', columns=['split', 'ts', 'feature18', 'watched']); NTR = int((meta.split == 0).sum())
ts = meta.ts.values[:NTR]; post = meta.feature18.values[:NTR]; y = meta.watched.values[:NTR] > 0.5; fit = ts < 48 * 3600_000
rv = np.flatnonzero(~fit); yv = y[rv]; seen = np.isin(post[rv], np.unique(post[fit]))
F = np.load(f'{T}/oracle_te.npy', mmap_mode='r')
print('val rows on unseen posts: %.4f | of those, share with other rows of the post within +-3 h: %.3f' % ((~seen).mean(), np.isfinite(np.asarray(F[rv, 2]))[~seen].mean()))
for tag in ('base', 'oracle_1h', 'oracle_3h', 'oracle_12h', 'oracle_all', 'past6h'):
    p = np.load(f'{P}/{tag}__val.npy')
    print(f'{tag:11s} all {roc_auc_score(yv, p):.5f} | seen posts {roc_auc_score(yv[seen], p[seen]):.5f} | unseen posts {roc_auc_score(yv[~seen], p[~seen]):.5f}')
# what distinguishes unseen-post rows: base rate, duration, post age
import json
EL = np.load(f'{T}/feat/EL_tr.npy', mmap_mode='r'); art = json.load(open('/kaggle/input/datasets/cassinicraft/sharechat-nb-darkgray-models/artifacts.json'))
import sys; sys.path.insert(0, '/kaggle/input/datasets/cassinicraft/sharechat-nb-darkgray-models'); import elig_feat as EF
names = EF.feature_names(art)
for c in ('feature54', 'post_age_h', 'feature252', 'feature247'):
    v = np.asarray(EL[rv, names.index(c)], np.float64)
    print(f'{c:12s} median seen {np.nanmedian(v[seen]):.2f} | unseen {np.nanmedian(v[~seen]):.2f}')
print('watch rate seen %.4f unseen %.4f' % (yv[seen].mean(), yv[~seen].mean()))
