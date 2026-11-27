"""Sanity of the same-user block on the TEST rows: where it is present the full model should differ from the counter-only model, elsewhere not."""
import numpy as np
R = '/kaggle/working/run'; T = '/tmp/sc_full1'; NTR = 1537638
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
a = lg(np.mean([np.load(f'{R}/pu/out/pred/pf_wfud_lr030_s{s}__test.npy') for s in (61, 62, 63)], 0).astype(np.float64))
b = lg(np.mean([np.load(f'{R}/pf2/out/pred/pf_wfd_lr030_s{s}__test.npy') for s in (51, 52, 53)], 0).astype(np.float64))
um = np.load(f'{T}/blk/um_full.npy', mmap_mode='r'); has = np.isfinite(np.asarray(um[NTR:, 0])); gap = np.expm1(np.asarray(um[NTR:, 0], np.float64)) / 3600
print('test rows %d | with a later session of the same user %.4f' % (len(a), has.mean()))
for nm, m in (('with later session', has), ('without', ~has)):
    print(f'  {nm:20s} rows {m.sum():7d} | mean p full {1 / (1 + np.exp(-a[m])).mean():.4f} counter-only {1 / (1 + np.exp(-b[m])).mean():.4f} | corr of logits {np.corrcoef(a[m], b[m])[0, 1]:.4f} | sd of logit difference {np.std(a[m] - b[m]):.3f}')
for lo, hi in [(0, 1), (1, 6), (6, 24), (24, 1e9)]:
    m = has & (gap >= lo) & (gap < hi)
    print(f'  later session after {lo}-{hi if hi < 1e8 else "inf"} h: rows {m.sum():6d} | sd of logit difference {np.std(a[m] - b[m]):.3f} | share with |diff| > 2: {(np.abs(a[m] - b[m]) > 2).mean():.3f}')
# same statistics on the dense hold-out for comparison (hold-out models)
yv = np.load(f'{T}/y_val.npy') > 0.5
av = lg(np.load(f'{R}/pu/out/pred/ph_wfud_lr030_s61__valfull.npy').astype(np.float64)); bv = lg(np.load(f'{R}/pf2/out/pred/ph_wfd_lr030_s51__valfull.npy').astype(np.float64))
hv = np.isfinite(np.asarray(um[1010303:NTR, 0]))
from sklearn.metrics import roc_auc_score
print('hold-out: with later session %.4f | AUC there: full %.4f vs counter-only %.4f | elsewhere: full %.4f vs counter-only %.4f | sd of logit diff there %.3f elsewhere %.3f' % (
    hv.mean(), roc_auc_score(yv[hv], av[hv]), roc_auc_score(yv[hv], bv[hv]), roc_auc_score(yv[~hv], av[~hv]), roc_auc_score(yv[~hv], bv[~hv]), np.std(av[hv] - bv[hv]), np.std(av[~hv] - bv[~hv])))
