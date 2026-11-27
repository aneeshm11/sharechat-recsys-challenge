"""E3 only: pseudo-labelling for the cross-row model (target rows at test-like density, soft labels = the model's own predictions)."""
import json, os, subprocess, sys, time
import numpy as np
T = '/tmp/sc_full1'; OUT = '/kaggle/working/run/diag2/pred'; P1 = '/kaggle/working/run/diag1/pred'
rows_fit = np.load(f'{T}/rows_fit.npy'); rows_val = np.load(f'{T}/rows_val.npy'); C_ALL = list(range(0, 481))
P = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 50, 'colsample_bytree': 0.4, 'reg_lambda': 5.0}
S = lambda path, rows, cols=None: {'path': path, 'rows': rows, 'cols': cols}


def srcs(g, l, k=None):
    sfx = 'full' if k is None else f'cb{k}'
    return [S(f'{T}/feat/EL_tr.npy', g), S(f'{T}/blk/{sfx}.npy', l, C_ALL), S(f'{T}/blk/fh_{sfx}.npy', l), S(f'{T}/blk/um_{sfx}.npy', l)]


dense = lambda rp, yp, n: {'srcs': srcs(rp, rp), 'y': yp, 'n': n}
pt = np.load(f'{P1}/L5_trackb__valthin.npy').astype(np.float32); o = 0; parts = []; ev = []
for k in range(3):
    g = f'{T}/g_val_{k}.npy'; l = f'{T}/l_val_{k}.npy'; m = len(np.load(g)); yp = f'{OUT}/e3_pl{k}.npy'; np.save(yp, pt[o:o + m]); o += m
    parts.append({'srcs': srcs(g, l, k), 'y': yp, 'n': m}); ev.append({'srcs': srcs(g, l, k), 'y': f'{T}/y_val_{k}.npy', 'n': m})
j = {'tag': 'e3_pseudo_trackb', 'params': P, 'train': [dense(f'{T}/rows_fit.npy', f'{T}/y_fit.npy', len(rows_fit))] + parts,
     'evals': {'valfull': [dense(f'{T}/rows_val.npy', f'{T}/y_val.npy', len(rows_val))], 'valthin': ev}, 'es_set': 'valthin', 'max_rounds': 4000, 'es': 100, 'out_dir': OUT, 'seed': 1,
     'device': 'cuda', 'gpu': int(os.environ.get('SC_GPU', '1')), 'chunk': 100000, 'log_period': 1000}
jp = f'{OUT}/e3_pseudo_trackb.json'; json.dump(j, open(jp, 'w'))
r = subprocess.run([sys.executable, f'{T}/xgbjob.py', jp], capture_output=True, text=True)
print(r.stdout[-600:]); print(r.stderr[-400:])
