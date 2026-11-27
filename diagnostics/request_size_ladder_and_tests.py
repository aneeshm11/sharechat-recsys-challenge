"""DIAGNOSTICS for the organisers (not used by either track's inference).
Reads request_id of the TRAINING file only to measure what request size is worth and whether it can be obtained otherwise.
A. request size vs label; reconstruction of request size WITHOUT the request_id column (train and test files)
B. information ladder on one hold-out (fit first 48 h, val last 24 h): row / +counters / +trace, each with and without request size
C. can request size be derived from the row's own features?   D. a labels-based test for 'does a score already contain request size'"""
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
T0 = time.time(); T = '/tmp/sc_full1'; D = '/tmp/diag1'; OUT = '/kaggle/working/run/diag1/pred'; os.makedirs(D, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DS = os.path.dirname([p for p in glob.glob('/kaggle/input/**/trace_feat.py', recursive=True) if 'darkgray' in p][0]); sys.path.insert(0, DS)
import elig_feat as EF, trace_feat as TF
art = json.load(open(DS + '/artifacts.json')); names = EF.feature_names(art); fut = json.load(open(DS + '/fut_cols.json'))['fut_cols']
tfn = TF.feature_names() + TF.future_names(fut); RESULT = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *a, flush=True)


TRAIN = glob.glob('/kaggle/input/**/train.parquet', recursive=True)[0]; TEST = glob.glob('/kaggle/input/**/test.parquet', recursive=True)[0]


def keys(path):
    t = pq.read_table(path, columns=['request_id', 'feature5', 'feature1'])
    return pd.factorize(t.column('request_id').to_numpy())[0], pd.factorize(t.column('feature5').to_numpy())[0], t.column('feature1').cast(pa.int64()).to_numpy()


# ------------------------------------------------------------------ A
rq, us, ts = keys(TRAIN); NTR = len(rq)
y = np.concatenate([np.load(f'{T}/y_fit.npy'), np.load(f'{T}/y_val.npy')]).astype(np.float64)
n = pd.Series(rq).map(pd.Series(rq).value_counts()).values.astype(np.int64)
for nm, (r_, u_, t_) in (('train', (rq, us, ts)), ('test', keys(TEST))):
    n_true = pd.Series(r_).map(pd.Series(r_).value_counts()).values
    ut = pd.factorize(pd.Series(u_.astype(np.int64) * 10 ** 14 + (t_ - t_.min())))[0]; n_ut = pd.Series(ut).map(pd.Series(ut).value_counts()).values
    chg = np.r_[True, (u_[1:] != u_[:-1]) | (t_[1:] != t_[:-1])]; run = np.cumsum(chg) - 1; n_adj = pd.Series(run).map(pd.Series(run).value_counts()).values
    RESULT[f'A_recon_{nm}'] = {'rows': int(len(r_)), 'requests': int(r_.max() + 1), 'equal_user_ts': float((n_ut == n_true).mean()), 'equal_adjacent': float((n_adj == n_true).mean()),
                               'requests_contiguous_in_file': bool((np.r_[True, r_[1:] != r_[:-1]].sum()) == r_.max() + 1)}
    log(f'A. {nm}: rows {len(r_)} requests {r_.max() + 1} | request size rebuilt WITHOUT request_id: count of rows with same (user, timestamp) equals the true size on {100 * (n_ut == n_true).mean():.4f} % of rows; '
        f'run length of adjacent rows with same (user, timestamp): {100 * (n_adj == n_true).mean():.4f} %')
tab = pd.DataFrame({'n': np.minimum(n, 10), 'y': y}).groupby('n')['y'].agg(['mean', 'size']); tab['share'] = tab['size'] / NTR
RESULT['A_rate_by_n'] = {int(k): [float(v['mean']), float(v['share'])] for k, v in tab.iterrows()}
log('A. watch rate by request size (10 = 10+): ' + ' | '.join(f'n={k}: {v["mean"]:.4f} ({100 * v["share"]:.1f}% rows)' for k, v in tab.iterrows()))
log('A. AUC of -request_size alone as a score (whole train): %.5f' % roc_auc_score(y > 0.5, -n.astype(float)))
# request blocks (training rows)
np.save(f'{D}/rq_n.npy', np.stack([n, 1.0 / n, np.log(n)], 1).astype(np.float32))
EL = np.load(f'{T}/feat/EL_tr.npy', mmap_mode='r'); FULL = np.load(f'{T}/blk/full.npy', mmap_mode='r')
ROWC = ['feature54', 'plat_u4p6_bu4_bp6', 'plat_u6p3_bu6_bp3', 'feature419', 'feature418', 'feature411', 'feature416', 'feature417', 'feature53', 'feature50', 'dot_u6p3', 'dot_u4p6']
TRC = ['tr_z_c2', 'tr_z_c3', 'tr_z_u4p6', 'tr_z_u6p3', 'tr_c900_c2', 'tr_c600_c2', 'tr_amp_u4p6', 'tr_zbest_u4p6']


def rel(vals):
    out = []; g = pd.Series(rq)
    for v_ in vals:
        v = pd.Series(np.asarray(v_, np.float64)); gb = v.groupby(rq); mean = gb.transform('mean').to_numpy(); mx = gb.transform('max').to_numpy(); rk = gb.rank(method='first', ascending=False).to_numpy()
        second = v.where(rk == 2).groupby(rq).transform('max').to_numpy(); other = np.where(rk == 1, second, mx); vv = v.to_numpy()
        out += [vv - mean, vv - other, np.where(np.isfinite(vv), (rk - 1) / np.maximum(n - 1, 1), np.nan), mx]
    return np.stack(out, 1).astype(np.float32)


np.save(f'{D}/rq_rowrel.npy', rel([EL[:, names.index(c)] for c in ROWC])); np.save(f'{D}/rq_trrel.npy', rel([FULL[:NTR, tfn.index(c)] for c in TRC]))
log('request blocks built')
# ------------------------------------------------------------------ job plumbing
rows_fit = np.load(f'{T}/rows_fit.npy'); rows_val = np.load(f'{T}/rows_val.npy'); NF_ = len(rows_fit)
C_TL = list(range(0, 55)); C_FU = list(range(55, 325)); C_W = list(range(325, 481))
P = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 50, 'colsample_bytree': 0.4, 'reg_lambda': 5.0}


def S(path, rows, cols=None):
    return {'path': path, 'rows': rows, 'cols': cols}


def part(which, B, thin_k=None):
    if thin_k is None:
        rp = f'{T}/rows_{which}.npy'; yp = f'{T}/y_{which}.npy'; nrow = len(np.load(rp)); g = l = rp; blk = f'{T}/blk/full.npy'; fh = f'{T}/blk/fh_full.npy'; um = f'{T}/blk/um_full.npy'
    else:
        g = f'{T}/g_val_{thin_k}.npy'; l = f'{T}/l_val_{thin_k}.npy'; yp = f'{T}/y_val_{thin_k}.npy'; nrow = len(np.load(g)); blk = f'{T}/blk/cb{thin_k}.npy'; fh = f'{T}/blk/fh_cb{thin_k}.npy'; um = f'{T}/blk/um_cb{thin_k}.npy'
    s = [S(f'{T}/feat/EL_tr.npy', g)]; bc = (C_TL if 'tr' in B else []) + (C_FU if 'fu' in B else []) + (C_W if 'w' in B else [])
    if bc:
        s.append(S(blk, l, bc))
    if 'fh' in B:
        s.append(S(fh, l))
    if 'um' in B:
        s.append(S(um, l))
    for nm in ('n', 'rowrel', 'trrel', 'predn'):
        if nm in B:
            s.append(S(f'{D}/' + {'n': 'rq_n', 'rowrel': 'rq_rowrel', 'trrel': 'rq_trrel', 'predn': 'pred_logn'}[nm] + '.npy', g))
    return {'srcs': s, 'y': yp, 'n': nrow}


def job(tag, B, gpu, thin=True, params=None, train=None, evals=None):
    ev = evals or ({'valfull': [part('val', B)], 'valthin': [part('val', B, k) for k in range(3)]} if thin else {'valfull': [part('val', B)]})
    j = {'tag': tag, 'params': params or P, 'train': train or [part('fit', B)], 'evals': ev, 'es_set': 'valthin' if (thin and not evals) else None, 'max_rounds': 4000, 'es': 100, 'out_dir': OUT, 'seed': 1,
         'device': 'cuda', 'gpu': gpu, 'chunk': 100000, 'log_period': 1000}
    jp = f'{OUT}/{tag}.json'; json.dump(j, open(jp, 'w'))
    return subprocess.Popen([sys.executable, f'{T}/xgbjob.py', jp], stdout=open(f'{OUT}/{tag}.log', 'w'), stderr=subprocess.STDOUT)


def run(specs):
    pend = list(specs); live = {}
    while pend or live:
        for g in (0, 1):
            if g not in live and pend:
                tag, B, kw = pend.pop(0); live[g] = (tag, job(tag, B, g, **kw)); log('started', tag, 'gpu', g)
        time.sleep(3)
        for g in list(live):
            tag, pr = live[g]
            if pr.poll() is not None:
                del live[g]; rp = f'{OUT}/{tag}__res.json'
                if os.path.exists(rp):
                    r = json.load(open(rp)); RESULT.setdefault('B_ladder', {})[tag] = {**{k: round(v, 5) for k, v in r['auc'].items()}, 'rounds': r['rounds']}
                    log('  done', tag, {k: round(v, 5) for k, v in r['auc'].items()}, 'rounds', r['rounds'])
                else:
                    log('  FAILED', tag, open(f'{OUT}/{tag}.log').read()[-400:])
        json.dump(RESULT, open('/kaggle/working/run/diag1/diag1.json', 'w'), indent=1)


# ------------------------------------------------------------------ C: request size predicted from the row's own features (out-of-fold)
ln = np.log(n).astype(np.float32); fold = np.random.default_rng(3).integers(0, 2, NF_)
PR = dict(P, objective='reg:squarederror', eval_metric='rmse', max_depth=6)
for f_ in (0, 1):
    a, b = rows_fit[fold == f_], rows_fit[fold != f_]
    np.save(f'{D}/c_tr{f_}.npy', a); np.save(f'{D}/c_te{f_}.npy', b); np.save(f'{D}/c_ytr{f_}.npy', ln[a]); np.save(f'{D}/c_yte{f_}.npy', ln[b])
np.save(f'{D}/c_yval.npy', ln[rows_val])
mk = lambda r, yy, nn: {'srcs': [S(f'{T}/feat/EL_tr.npy', r)], 'y': yy, 'n': nn}
cj = [(f'c_logn_f{f_}', None, dict(params=PR, train=[mk(f'{D}/c_tr{f_}.npy', f'{D}/c_ytr{f_}.npy', int((fold == f_).sum()))],
                                    evals={'val': [mk(f'{T}/rows_val.npy', f'{D}/c_yval.npy', len(rows_val))], 'oof': [mk(f'{D}/c_te{f_}.npy', f'{D}/c_yte{f_}.npy', int((fold != f_).sum()))]})) for f_ in (0, 1)]
run(cj)
pl = np.full(NTR, np.nan, np.float32)
for f_ in (0, 1):
    pl[rows_fit[fold != f_]] = np.load(f'{OUT}/c_logn_f{f_}__oof.npy')
pl[rows_val] = (np.load(f'{OUT}/c_logn_f0__val.npy') + np.load(f'{OUT}/c_logn_f1__val.npy')) / 2
np.save(f'{D}/pred_logn.npy', pl[:, None]); v_ = rows_val
r2 = 1 - np.mean((pl[v_] - ln[v_]) ** 2) / np.var(ln[v_])
RESULT['C_pred_n'] = {'r2_log_n_val': float(r2), 'auc_singleton_val': float(roc_auc_score(n[v_] == 1, -pl[v_])), 'spearman_val': float(pd.Series(pl[v_]).corr(pd.Series(ln[v_]), method='spearman'))}
log('C. request size predicted from the row alone (hold-out): R2 of log n %.4f | Spearman %.4f | AUC for "request has one row" %.4f' % (r2, RESULT['C_pred_n']['spearman_val'], RESULT['C_pred_n']['auc_singleton_val']))
# ------------------------------------------------------------------ B: ladder
run([('L0_row', set(), {}), ('L1_row+n', {'n'}, {}), ('L2_row+n+within_request', {'n', 'rowrel'}, {}), ('L7_row+predicted_n', {'predn'}, {}),
     ('L3_row+counters', {'fu', 'fh'}, {}), ('L4_row+counters+n', {'fu', 'fh', 'n'}, {}),
     ('L6_trackb+n', {'tr', 'fu', 'w', 'fh', 'um', 'n'}, {}), ('L6c_trackb+n+within_request', {'tr', 'fu', 'w', 'fh', 'um', 'n', 'rowrel', 'trrel'}, {'thin': False}),
     ('L5_trackb', {'tr', 'fu', 'w', 'fh', 'um'}, {})])
# ------------------------------------------------------------------ D: does a score already contain request size?  (labels-based test)
yv = np.load(f'{T}/y_val.npy') > 0.5; nv = n[rows_val]; lnv = np.log(nv); rqv = rq[rows_val]; half = (pd.factorize(rqv)[0] % 2 == 0)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
RESULT['D_test'] = {}
for tag in ('L0_row', 'L3_row+counters', 'L5_trackb', 'L7_row+predicted_n', 'L1_row+n', 'L4_row+counters+n', 'L6_trackb+n'):
    p = f'{OUT}/{tag}__valfull.npy'
    if not os.path.exists(p):
        continue
    z = lg(np.load(p).astype(np.float64)); X = np.stack([z, lnv, z * lnv], 1)
    m = LogisticRegression(C=1e6, max_iter=500).fit(X[half], yv[half]); a0 = roc_auc_score(yv[~half], z[~half]); a1 = roc_auc_score(yv[~half], m.decision_function(X[~half]))
    rate = pd.DataFrame({'n': np.minimum(nv, 6), 'p': 1 / (1 + np.exp(-z)), 'y': yv}).groupby('n')[['p', 'y']].mean()
    RESULT['D_test'][tag] = {'auc_score': float(a0), 'auc_score_plus_request_size': float(a1), 'gain': float(a1 - a0), 'coef_log_n': float(m.coef_[0][1]),
                             'mean_pred_by_n': {int(k): float(v) for k, v in rate['p'].items()}, 'rate_by_n': {int(k): float(v) for k, v in rate['y'].items()}}
    log(f'D. {tag:28s} AUC {a0:.5f} -> with request size added to the score {a1:.5f} (gain {a1 - a0:+.5f}) | mean prediction by n=1..6+: ' + ' '.join(f'{v:.3f}' for v in rate['p'].values) + ' | actual: ' + ' '.join(f'{v:.3f}' for v in rate['y'].values))
json.dump(RESULT, open('/kaggle/working/run/diag1/diag1.json', 'w'), indent=1)
log('DIAG DONE')
