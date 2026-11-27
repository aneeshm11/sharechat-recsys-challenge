"""DIAGNOSTICS part 2: two ideas the hosts were asked about and did not rule on, which neither track uses - measured on the hold-out.
E1  pseudo-labelling: add the target-period rows with the model's own soft predictions as labels, retrain, score the target period.
E2  distillation from a REQUEST-SIZE-aware teacher (training-time only) into a student that never sees request size.
Both for the row-only model (track_a) and, E3/E4, for the cross-row model (track_b)."""
import json, os, subprocess, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
T0 = time.time(); T = '/tmp/sc_full1'; D = '/tmp/diag1'; OUT = '/kaggle/working/run/diag2/pred'; P1 = '/kaggle/working/run/diag1/pred'; os.makedirs(OUT, exist_ok=True)
RES = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *a, flush=True)


def wait(p, tmax=3000):
    t = time.time()
    while not os.path.exists(p) and time.time() - t < tmax:
        time.sleep(5)
    return os.path.exists(p)


rows_fit = np.load(f'{T}/rows_fit.npy'); rows_val = np.load(f'{T}/rows_val.npy'); yfit = np.load(f'{T}/y_fit.npy').astype(np.float32); yv = np.load(f'{T}/y_val.npy') > 0.5
C_ALL = list(range(0, 481))
P = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 50, 'colsample_bytree': 0.4, 'reg_lambda': 5.0}
S = lambda path, rows, cols=None: {'path': path, 'rows': rows, 'cols': cols}


def srcs(g, l, tb, n, k=None):
    s = [S(f'{T}/feat/EL_tr.npy', g)]
    if tb:
        sfx = 'full' if k is None else f'cb{k}'
        s += [S(f'{T}/blk/{sfx}.npy', l, C_ALL), S(f'{T}/blk/fh_{sfx}.npy', l), S(f'{T}/blk/um_{sfx}.npy', l)]
    if n:
        s.append(S(f'{D}/rq_n.npy', g))
    return s


def dense(rowsp, yp, nrow, tb=False, n=False):
    return {'srcs': srcs(rowsp, rowsp, tb, n), 'y': yp, 'n': nrow}


def thin(k, yp=None, tb=True):
    g = f'{T}/g_val_{k}.npy'; return {'srcs': srcs(g, f'{T}/l_val_{k}.npy', tb, False, k), 'y': yp or f'{T}/y_val_{k}.npy', 'n': len(np.load(g))}


def sv(name, a):
    p = f'{OUT}/{name}.npy'; np.save(p, a); return p


def run(specs):
    pend = list(specs); live = {}
    while pend or live:
        for g in (0, 1):
            if g not in live and pend:
                tag, train, evals, kw = pend.pop(0)
                j = {'tag': tag, 'params': P, 'train': train, 'evals': evals, 'max_rounds': 4000, 'es': 100, 'out_dir': OUT, 'seed': 1, 'device': 'cuda', 'gpu': g, 'chunk': 100000, 'log_period': 1000, **kw}
                jp = f'{OUT}/{tag}.json'; json.dump(j, open(jp, 'w'))
                live[g] = (tag, subprocess.Popen([sys.executable, f'{T}/xgbjob.py', jp], stdout=open(f'{OUT}/{tag}.log', 'w'), stderr=subprocess.STDOUT)); log('started', tag, 'gpu', g)
        time.sleep(3)
        for g in list(live):
            tag, pr = live[g]
            if pr.poll() is not None:
                del live[g]; rp = f'{OUT}/{tag}__res.json'
                if os.path.exists(rp):
                    r = json.load(open(rp)); RES[tag] = {**{k: round(v, 5) for k, v in r['auc'].items()}, 'rounds': r['rounds']}; log('  done', tag, RES[tag])
                else:
                    log('  FAILED', tag, open(f'{OUT}/{tag}.log').read()[-300:])
        json.dump(RES, open('/kaggle/working/run/diag2/diag2.json', 'w'), indent=1)


VAL_A = {'val': [dense(f'{T}/rows_val.npy', f'{T}/y_val.npy', len(rows_val))]}
FITP = f'{T}/rows_fit.npy'; NF = len(rows_fit)
# ---------------- E2: request-size-aware teacher (out-of-fold on the fit rows) -> row-only student
fold = np.random.default_rng(11).integers(0, 2, NF); tj = []
for f_ in (0, 1):
    a, b = rows_fit[fold == f_], rows_fit[fold != f_]
    tj.append((f'e2_teacher_f{f_}', [dense(sv(f'e2_a{f_}', a), sv(f'e2_ya{f_}', yfit[fold == f_]), len(a), n=True)], {'oof': [dense(sv(f'e2_b{f_}', b), sv(f'e2_yb{f_}', yfit[fold != f_]), len(b), n=True)]}, {}))
run(tj)
q = np.empty(NF, np.float32)
for f_ in (0, 1):
    q[fold != f_] = np.load(f'{OUT}/e2_teacher_f{f_}__oof.npy')
RES['e2_teacher_oof_auc'] = float(roc_auc_score(yfit > 0.5, q)); log('E2 request-size-aware teacher: out-of-fold AUC on fit rows %.5f (mean %.4f vs rate %.4f)' % (RES['e2_teacher_oof_auc'], q.mean(), yfit.mean()))
jobs = [(f'e2_student_lam{int(l * 10)}', [dense(FITP, sv(f'e2_soft{int(l * 10)}', (l * q + (1 - l) * yfit).astype(np.float32)), NF)], VAL_A, {}) for l in (0.5, 1.0)]
# ---------------- E1: pseudo-labelling for the row-only model (soft labels = the hold-out model's own predictions on the target rows)
if wait(f'{P1}/L0_row__valfull.npy'):
    time.sleep(3); pl = np.load(f'{P1}/L0_row__valfull.npy').astype(np.float32)
    jobs.append(('e1_pseudo_row', [dense(FITP, f'{T}/y_fit.npy', NF), dense(f'{T}/rows_val.npy', sv('e1_pl', pl), len(rows_val))], VAL_A, {}))
    jobs.append(('e1_pseudo_row_hard', [dense(FITP, f'{T}/y_fit.npy', NF), dense(f'{T}/rows_val.npy', sv('e1_plh', (pl > np.quantile(pl, 0.95)).astype(np.float32)), len(rows_val))], VAL_A, {}))
run(jobs)
# ---------------- E3: pseudo-labelling for the cross-row model, target rows at test-like density
if wait(f'{P1}/L5_trackb__valthin.npy'):
    time.sleep(3); pt = np.load(f'{P1}/L5_trackb__valthin.npy').astype(np.float32); o = 0; parts = []
    for k in range(3):
        m = len(np.load(f'{T}/g_val_{k}.npy')); parts.append(thin(k, sv(f'e3_pl{k}', pt[o:o + m]))); o += m
    run([('e3_pseudo_trackb', [dense(FITP, f'{T}/y_fit.npy', NF, tb=True)] + parts, {'valfull': [dense(f'{T}/rows_val.npy', f'{T}/y_val.npy', len(rows_val), tb=True)], 'valthin': [thin(k) for k in range(3)]}, {'es_set': 'valthin'})])
# ---------------- E4: request-size-aware teacher (with the cross-row blocks) -> cross-row student without request size
if os.environ.get('SC_E4', '1') == '1':
    tj = []
    for f_ in (0, 1):
        tj.append((f'e4_teacher_f{f_}', [dense(f'{OUT}/e2_a{f_}.npy', f'{OUT}/e2_ya{f_}.npy', int((fold == f_).sum()), tb=True, n=True)],
                   {'oof': [dense(f'{OUT}/e2_b{f_}.npy', f'{OUT}/e2_yb{f_}.npy', int((fold != f_).sum()), tb=True, n=True)]}, {}))
    run(tj)
    q4 = np.empty(NF, np.float32)
    for f_ in (0, 1):
        q4[fold != f_] = np.load(f'{OUT}/e4_teacher_f{f_}__oof.npy')
    RES['e4_teacher_oof_auc'] = float(roc_auc_score(yfit > 0.5, q4)); log('E4 teacher (cross-row + request size): out-of-fold AUC %.5f' % RES['e4_teacher_oof_auc'])
    run([('e4_student_lam5', [dense(FITP, sv('e4_soft5', (0.5 * q4 + 0.5 * yfit).astype(np.float32)), NF, tb=True)],
          {'valfull': [dense(f'{T}/rows_val.npy', f'{T}/y_val.npy', len(rows_val), tb=True)], 'valthin': [thin(k) for k in range(3)]}, {'es_set': 'valthin'})])
json.dump(RES, open('/kaggle/working/run/diag2/diag2.json', 'w'), indent=1)
log('DIAG2 DONE', json.dumps(RES))
