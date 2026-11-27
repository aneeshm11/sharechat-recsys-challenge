"""ShareChat track S - ROW-INDEPENDENT models (inference reads only the row's own columns via elig_feat + frozen training vocabularies).

Experiments (hold-out: fit = first 48 h of train.parquet, val = last 24 h):
  * hard-label XGBoost, two settings (depth 5 / depth 7)                                -> the pure, fully compliant family
  * training-time distillation: a teacher that sees cross-row 'trace' features of the TRAINING file (out-of-fold), whose
    predictions soften the training labels of the row-only student.  The student never sees a cross-row feature and
    scores test rows independently.  A row-only teacher is run as a control (generic label smoothing vs privileged information).
No request_id is read anywhere in this file."""
import os, sys, glob, json, time, subprocess, gc, base64
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


SMOKE = int(os.environ.get('SC_SMOKE', '0'))
ROOT = os.environ.get('SC_INPUT', '/kaggle/input')
OUT = os.environ.get('SC_OUT', '/kaggle/working'); TMP = os.environ.get('SC_TMP', '/tmp/scs')
PRED = f'{OUT}/pred'; JOBS = f'{TMP}/jobs'; FEAT = f'{TMP}/feat'
for d in (OUT, TMP, PRED, JOBS, FEAT):
    os.makedirs(d, exist_ok=True)


def find(name, must=None):
    r = sorted(glob.glob(f'{ROOT}/**/{name}', recursive=True))
    if must:
        r = [x for x in r if must in x]
    assert r, f'not found: {name}'
    return r[0]


TRAIN = find('train.parquet'); TEST = find('test.parquet'); SSUB = find('sample_submission.csv')
DS = os.environ.get('SC_DS') or os.path.dirname(find('trace_feat.py', 'darkgray'))
sys.path.insert(0, DS)
open(f'{TMP}/xgbjob.py', 'wb').write(base64.b64decode(XGBJOB_B64))
import elig_feat as EF, trace_feat as TF
import torch
GPU = torch.cuda.is_available(); NGPU = torch.cuda.device_count() if GPU else 0
_eigh0 = torch.linalg.eigh


def _eigh_cpu(A, *a, **k):
    if A.is_cuda:
        w, V = _eigh0(A.cpu(), *a, **k)
        return w.to(A.device), V.to(A.device)
    return _eigh0(A, *a, **k)


torch.linalg.eigh = _eigh_cpu
log('MODE', MODE, 'SMOKE', SMOKE, 'GPU', GPU, NGPU)
art = json.load(open(DS + '/artifacts.json')); names = EF.feature_names(art); NF = len(names)
fut_cols = json.load(open(DS + '/fut_cols.json'))['fut_cols']; fidx = np.array([names.index(c) for c in fut_cols])
tf_names = TF.feature_names() + TF.future_names(fut_cols)


def build_file(path, extra_cols, emb):
    if SMOKE:
        pf = pq.ParquetFile(path)
        need = sorted(set(art['num_cols']) | {EF.TS_COL, 'feature57', 'feature58'} | set(EF.CAT_COLS) | set(EF.USER_E) | set(EF.POST_E) | set(extra_cols))
        b = next(pf.iter_batches(batch_size=SMOKE, columns=need))
        F, E = EF.batch_features(b, art, True); ex = {}
        for c in extra_cols:
            col = b.column(c)
            ex[c] = np.asarray(col.cast(pa.int64()).to_numpy(zero_copy_only=False)) if pa.types.is_timestamp(col.type) else np.asarray(col.to_numpy(zero_copy_only=False))
        return F, E.astype(np.float32), ex
    if emb:
        return EF.build(path, art, batch=100000, return_emb=True, emb_dtype=np.float32, extra_cols=extra_cols, log=log)
    F, ex = EF.build(path, art, batch=100000, extra_cols=extra_cols, log=log)
    return F, None, ex


SF = None if SMOKE else os.environ.get('SC_FEAT')                       # reuse row features already built by another run on this machine
EL_TR = f'{SF or FEAT}/EL_tr.npy'; TF_TR = f'{FEAT}/TF_tr.npy'; EL_TE = f'{SF or FEAT}/EL_te.npy'
if SF:
    rc = pq.read_table(TRAIN, columns=['feature1', 'feature18', 'watched'])
    ex = {'feature1': rc.column('feature1').cast(pa.int64()).to_numpy(), 'feature18': rc.column('feature18').to_numpy(), 'watched': rc.column('watched').to_numpy()}
    F = np.load(EL_TR, mmap_mode='r'); E = np.load(f'{SF}/E_tr.npy', mmap_mode='r'); del rc
else:
    F, E, ex = build_file(TRAIN, ('feature1', 'feature18', 'watched'), True); np.save(EL_TR, F)
ts = ex['feature1'].astype(np.int64); post = pd.factorize(ex['feature18'])[0].astype(np.int64); y = ex['watched'].astype(np.float32)
N = len(ts); t = ts - ts.min(); HR = 3600_000
T_SPLIT = int(np.quantile(t, 0.67)) if SMOKE else 48 * HR
fit = t < T_SPLIT
assert F.shape == (N, NF)
# teacher-only features: cross-row trace inside the TRAINING file (posts split into partitions that fit a T4)
X = np.lib.format.open_memmap(TF_TR, mode='w+', dtype=np.float32, shape=(N, len(tf_names))); n_tl = len(TF.feature_names())
X[:, n_tl:] = TF.build_future(ts, post, F[:, fidx], fut_cols)
npart = max(1, int(np.ceil(N / 600000))); q = post % npart
for i in range(npart):
    m = np.flatnonzero(q == i); Em = E[m]
    X[m, :n_tl] = TF.build_trace(ts[m], pd.factorize(post[m])[0].astype(np.int64), Em[:, :7], Em[:, 7:])
    del Em
    if GPU:
        torch.cuda.empty_cache()
X.flush(); del X, E, F; gc.collect()
log('train features done', N, 'rate %.5f' % y.mean())
if SF:
    ids = pq.read_table(TEST, columns=['id']).column('id').to_numpy().astype(np.int64); NTE = len(ids)
    assert np.load(EL_TE, mmap_mode='r').shape == (NTE, NF)
else:
    Ft, _, ext = build_file(TEST, ('id',), False); ids = ext['id'].astype(np.int64); NTE = len(ids); np.save(EL_TE, Ft); del Ft; gc.collect()
log('test features done', NTE)


def sidx(name, arr):
    p = f'{TMP}/{name}.npy'; np.save(p, arr); return p


def src(path, rows=None, cols=None):
    return {'path': path, 'rows': rows, 'cols': cols}


rows_fit = np.flatnonzero(fit); rows_val = np.flatnonzero(~fit); rows_all = np.arange(N)
P = {'fit': sidx('rows_fit', rows_fit), 'val': sidx('rows_val', rows_val), 'all': sidx('rows_all', rows_all)}
Y = {'fit': sidx('y_fit', y[rows_fit]), 'val': sidx('y_val', y[rows_val]), 'all': sidx('y_all', y)}
NN = {'fit': len(rows_fit), 'val': len(rows_val), 'all': N}


def part(which, teacher=False, ypath=None, rows=None, name=None):
    if rows is not None:
        rp = sidx('r_' + name, rows); yp = ypath or sidx('y_' + name, y[rows]); n = len(rows)
    else:
        rp, yp, n = P[which], ypath or Y[which], NN[which]
    return {'srcs': [src(EL_TR, rp)] + ([src(TF_TR, rp)] if teacher else []), 'y': yp, 'n': n}


BASE = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8}
LR = float(os.environ.get('SC_LR', '0.05'))
CFG = {'d5': dict(BASE, learning_rate=LR, max_depth=5, min_child_weight=50, colsample_bytree=0.5, reg_lambda=5.0),
       'd7': dict(BASE, learning_rate=LR, max_depth=7, min_child_weight=200, colsample_bytree=0.3, reg_lambda=20.0),
       't6': dict(BASE, learning_rate=0.1, max_depth=6, min_child_weight=50, colsample_bytree=0.4, reg_lambda=5.0)}
RES = {}


def mkjob(tag, cfg, train, evals=None, seed=1, fixed_rounds=None, predict=None, fnames=None):
    return {'tag': tag, 'params': dict(CFG[cfg]), 'train': train, 'evals': evals or {}, 'es_set': None, 'max_rounds': 40 if SMOKE else 10000,
            'es': 20 if SMOKE else 150, 'out_dir': PRED, 'fnames': fnames, 'seed': seed, 'fixed_rounds': fixed_rounds, 'predict': predict,
            'device': 'cuda' if GPU else 'cpu', 'chunk': 100000, 'log_period': 10 if SMOKE else 250}


def run_jobs(jobs):
    slots = ([int(g) for g in os.environ['SC_GPUS'].split(',')] if os.environ.get('SC_GPUS') else list(range(NGPU))) if GPU else [0]
    pending = list(jobs); running = {}
    while pending or running:
        for g in slots:
            if g not in running and pending:
                j = pending.pop(0); j['gpu'] = g
                jp = f"{JOBS}/{j['tag']}.json"; json.dump(j, open(jp, 'w')); lf = open(f"{JOBS}/{j['tag']}.log", 'w')
                running[g] = (j, subprocess.Popen([sys.executable, f'{TMP}/xgbjob.py', jp], stdout=lf, stderr=subprocess.STDOUT), lf)
                log('started', j['tag'], 'on gpu', g)
        time.sleep(3)
        for g in list(running):
            j, pr, lf = running[g]
            if pr.poll() is not None:
                lf.close(); del running[g]
                tail = open(f"{JOBS}/{j['tag']}.log").read().splitlines()
                keep = [l for l in tail if '[' + j['tag'] + ']' in l and ('rounds' in l or 'predicted' in l)][-3:]
                print('\n'.join(keep if pr.returncode == 0 else tail[-40:]), flush=True)
                rp = f"{PRED}/{j['tag']}__res.json"
                RES[j['tag']] = json.load(open(rp)) if (pr.returncode == 0 and os.path.exists(rp)) else {'tag': j['tag'], 'failed': True}
        json.dump(RES, open(f'{OUT}/results_{MODE}.json', 'w'), indent=1)


from sklearn.metrics import roc_auc_score
yv = y[rows_val] > 0.5
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def vauc(tags):
    z = np.mean([lg(np.load(f'{PRED}/{g}__val.npy').astype(np.float64)) for g in tags], 0)
    return float(roc_auc_score(yv, z))


VAL = {'val': [part('val')]}; VALT = {'val': [part('val', teacher=True)]}
# ---------------------------------------------------------------- wave 1: hard-label baselines + teacher round count
run_jobs([mkjob('h_d5', 'd5', [part('fit')], VAL, fnames=names), mkjob('h_d7', 'd7', [part('fit')], VAL, fnames=names),
          mkjob('teach_ho', 't6', [part('fit', teacher=True)], VALT)])
log('hard d5 %.5f | hard d7 %.5f | blend %.5f | cross-row teacher %.5f' % (vauc(['h_d5']), vauc(['h_d7']), vauc(['h_d5', 'h_d7']), vauc(['teach_ho'])))
# ---------------------------------------------------------------- wave 2: out-of-fold teachers on the fit rows
KF = 4; fold = np.random.default_rng(7).integers(0, KF, len(rows_fit))
jobs = []
for j in range(KF):
    a, b = rows_fit[fold != j], rows_fit[fold == j]
    jobs.append(mkjob(f'tp{j}', 't6', [part(None, True, rows=a, name=f'kfa{j}')], {'oof': [part(None, True, rows=b, name=f'kfb{j}')]}, fixed_rounds=int(RES['teach_ho']['rounds'])))
    jobs.append(mkjob(f'tr{j}', 'd5', [part(None, False, rows=a, name=f'kfa{j}')], {'oof': [part(None, False, rows=b, name=f'kfb{j}')]}, fixed_rounds=int(RES['h_d5']['rounds'])))
run_jobs(jobs)
yf = y[rows_fit]; Q = {}
for nm in ('tp', 'tr'):
    qq = np.empty(len(rows_fit), np.float32)
    for j in range(KF):
        qq[fold == j] = np.load(f'{PRED}/{nm}{j}__oof.npy')
    Q[nm] = qq
    log(f'teacher {nm}: out-of-fold AUC %.5f | mean %.5f (rate %.5f) | Brier %.5f (labels vs rate: %.5f)' % (roc_auc_score(yf > 0.5, qq), qq.mean(), yf.mean(), np.mean((qq - yf) ** 2), np.mean((yf.mean() - yf) ** 2)))
# ---------------------------------------------------------------- wave 3: students
soft = lambda nm, lam: sidx(f'ysoft_{nm}_{int(lam * 10)}', (lam * Q[nm] + (1 - lam) * yf).astype(np.float32))
ST = [('s_d5_p10', 'd5', 'tp', 1.0), ('s_d5_p05', 'd5', 'tp', 0.5), ('s_d7_p05', 'd7', 'tp', 0.5), ('s_d5_r05', 'd5', 'tr', 0.5), ('s_d7_p10', 'd7', 'tp', 1.0), ('s_d7_r05', 'd7', 'tr', 0.5)]
run_jobs([mkjob(tag, cfg, [part('fit', ypath=soft(nm, lam))], VAL) for tag, cfg, nm, lam in ST])
log('================ TRACK S hold-out AUC ================')
for g in ['h_d5', 'h_d7'] + [s[0] for s in ST]:
    log(f'  {g:10s} %.5f  rounds {RES[g].get("rounds")}' % vauc([g]))
B = {'hard d5+d7': vauc(['h_d5', 'h_d7']), 'privileged 0.5 d5+d7': vauc(['s_d5_p05', 's_d7_p05']), 'privileged 1.0 d5+d7': vauc(['s_d5_p10', 's_d7_p10']),
     'row-teacher 0.5 d5+d7': vauc(['s_d5_r05', 's_d7_r05']), 'hard + privileged 0.5 (4 models)': vauc(['h_d5', 'h_d7', 's_d5_p05', 's_d7_p05'])}
for k, v in B.items():
    log(f'  blend {k:34s} %.5f' % v)
RES['blends'] = B; json.dump(RES, open(f'{OUT}/results_{MODE}.json', 'w'), indent=1)
# ---------------------------------------------------------------- production: pure hard-label models on all 72 h (always), distilled (if it helps)
TESTP = {'test': [{'srcs': [src(EL_TE, sidx('rows_te', np.arange(NTE)))], 'n': NTE}]}
SEEDS = [31, 32, 33]
rd = lambda g: int(1.15 * RES[g]['rounds']) + 1


def write_sub(tags, name):
    z = np.mean([lg(np.load(f'{PRED}/{g}__test.npy').astype(np.float64)) for g in tags], 0)
    sub = pd.DataFrame({'id': ids, 'watched': np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)})
    if not SMOKE:
        ss = pd.read_csv(SSUB, usecols=['id']); sub = ss.merge(sub, on='id', how='left')
        assert len(sub) == len(ss) and sub['watched'].notna().all()
    sub.to_csv(f'{OUT}/{name}.csv', index=False, float_format='%.7f'); log('wrote', name, len(sub), 'models', len(tags), 'mean %.5f' % sub['watched'].mean())


pure = [mkjob(f'p_{c}_s{s}', c, [part('all')], seed=s, fixed_rounds=rd(f'h_{c}'), predict=TESTP) for s in SEEDS for c in ('d5', 'd7')]
run_jobs(pure); write_sub([j['tag'] for j in pure], 'submission_S_pure')
gain = max(B['privileged 0.5 d5+d7'], B['privileged 1.0 d5+d7']) - B['hard d5+d7']
log('distillation gain on the two-model blend: %+.5f' % gain)
if gain > 0.0003 or os.environ.get('SC_FORCE_DIST'):
    lam = 0.5 if B['privileged 0.5 d5+d7'] >= B['privileged 1.0 d5+d7'] else 1.0; sfx = 'p05' if lam == 0.5 else 'p10'
    fold_a = np.random.default_rng(8).integers(0, KF, N); jobs = []
    for j in range(KF):
        a, b = rows_all[fold_a != j], rows_all[fold_a == j]
        jobs.append(mkjob(f'ta{j}', 't6', [part(None, True, rows=a, name=f'afa{j}')], {'oof': [part(None, True, rows=b, name=f'afb{j}')]}, fixed_rounds=int(1.1 * RES['teach_ho']['rounds'])))
    run_jobs(jobs)
    qa = np.empty(N, np.float32)
    for j in range(KF):
        qa[fold_a == j] = np.load(f'{PRED}/ta{j}__oof.npy')
    log('all-train teacher out-of-fold AUC %.5f' % roc_auc_score(y > 0.5, qa))
    ys = sidx('ysoft_all', (lam * qa + (1 - lam) * y).astype(np.float32))
    dist = [mkjob(f'd_{c}_s{s}', c, [part('all', ypath=ys)], seed=s, fixed_rounds=rd(f's_{c}_{sfx}'), predict=TESTP) for s in SEEDS for c in ('d5', 'd7')]
    run_jobs(dist); write_sub([j['tag'] for j in dist], 'submission_S_dist')
json.dump(RES, open(f'{OUT}/results_{MODE}.json', 'w'), indent=1)
log('ALL DONE')
