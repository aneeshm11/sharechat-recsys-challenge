"""ShareChat round B - cross-row trace model WITHOUT request_id (track C), density-matched training.

DISCLOSURE: NOT row-independent. Uses later/earlier rows of the SAME POST (train and test files, one joint timeline).
No request_id, no request grouping, no grouping by user is read or derived anywhere in this file.

Design
  * features: row-level (elig_feat, 751) + trace matched filter (trace_feat, 55) + later-snapshot counter deltas (270)
              + per-post-whitened increment aggregates / cumulative cosines (feats2: W main heads, V other heads)
  * geometry: the test file has ~0.68x the row density of the training file, one 5 h hole and a file end.  Training rows are
              therefore re-featurised on 3 'test-like' copies of the timeline: 70 % of training rows kept at random, two 5 h
              holes removed, followed by the (label-free) test rows.  Hold-out rows are scored under the same geometry.
  * hold-out: fit = first 48 h of train, val = last 24 h.  Final models: all 72 h, 1.15 x early-stopped rounds."""
import os, sys, glob, json, time, subprocess, gc, base64
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


SMOKE = int(os.environ.get('SC_SMOKE', '0'))
STAGES = os.environ.get('SC_STAGES', 'feat,abl,prod').split(',')
ROOT = os.environ.get('SC_INPUT', '/kaggle/input')
OUT = os.environ.get('SC_OUT', '/kaggle/working'); TMP = os.environ.get('SC_TMP', '/tmp/scw')
PRED = f'{OUT}/pred'; JOBS = f'{TMP}/jobs'; BLK = f'{TMP}/blk'
for d in (OUT, TMP, PRED, JOBS, BLK):
    os.makedirs(d, exist_ok=True)
log(subprocess.run(f'df -h {TMP} {OUT} | cat; free -g | head -2', shell=True, capture_output=True, text=True).stdout)


def find(name, must=None, opt=False):
    r = sorted(glob.glob(f'{ROOT}/**/{name}', recursive=True))
    if must:
        r = [x for x in r if must in x]
    if not r and opt:
        return None
    assert r, f'not found: {name}'
    return r[0]


TRAIN = find('train.parquet'); TEST = find('test.parquet'); SSUB = find('sample_submission.csv')
DS = os.environ.get('SC_DS') or os.path.dirname(find('trace_feat.py', 'darkgray'))
sys.path.insert(0, DS); sys.path.insert(0, TMP)
for _n, _b in (('xgbjob.py', XGBJOB_B64), ('feats2.py', FEATS2_B64), ('catjob.py', CATJOB_B64)):
    _q = f'{TMP}/{_n}.{os.getpid()}.tmp'; open(_q, 'wb').write(base64.b64decode(_b)); os.replace(_q, f'{TMP}/{_n}')
_wp = os.environ.get('SC_WAIT_PID')
while _wp and os.path.exists(f'/proc/{_wp}'):               # chained runs: wait for an earlier run to release its GPU
    time.sleep(5)
import elig_feat as EF, trace_feat as TF, feats2 as F2
import torch
GPU = torch.cuda.is_available(); NGPU = torch.cuda.device_count() if GPU else 0
_eigh0 = torch.linalg.eigh


def _eigh_cpu(A, *a, **k):
    """Batched 32x32 eigendecompositions on the CPU: the GPU solver's workspace (6.9 GB per 10k matrices) does not fit a T4."""
    if A.is_cuda:
        w, V = _eigh0(A.cpu(), *a, **k)
        return w.to(A.device), V.to(A.device)
    return _eigh0(A, *a, **k)


torch.linalg.eigh = _eigh_cpu
for _i in range(NGPU):                               # create the CUDA contexts / BLAS handles from the main thread before any worker thread
    _x = torch.randn(4, 8, 8, device=f'cuda:{_i}', dtype=torch.float64); _ = torch.bmm(_x, _x) @ _x[0]; _ = torch.zeros(3, 8, device=f'cuda:{_i}').index_add_(0, torch.tensor([0, 1, 1], device=f'cuda:{_i}'), torch.ones(3, 8, device=f'cuda:{_i}'))
    del _x
log('MODE', MODE, 'SMOKE', SMOKE, 'STAGES', STAGES, 'GPU', GPU, NGPU)

art = json.load(open(DS + '/artifacts.json')); names = EF.feature_names(art); NF = len(names)
fut_cols = json.load(open(DS + '/fut_cols.json'))['fut_cols']; fidx = np.array([names.index(c) for c in fut_cols])
tf_names = TF.feature_names() + TF.future_names(fut_cols)
w_names = F2.feature_names(F2.PAIRS_W, F2.COMBO_W, 'w'); v_names = F2.feature_names(F2.PAIRS_V, F2.COMBO_V, 'v')
blk_names = tf_names + w_names + v_names
C_TF = list(range(len(tf_names))); C_W = list(range(len(tf_names), len(tf_names) + len(w_names))); C_V = list(range(len(tf_names) + len(w_names), len(blk_names)))
log('features: row', NF, 'trace+future', len(tf_names), 'W', len(w_names), 'V', len(v_names))


def build_file(path, extra_cols):
    if SMOKE:
        pf = pq.ParquetFile(path)
        need = sorted(set(art['num_cols']) | {EF.TS_COL, 'feature57', 'feature58'} | set(EF.CAT_COLS) | set(EF.USER_E) | set(EF.POST_E) | set(extra_cols))
        b = next(pf.iter_batches(batch_size=SMOKE, columns=need))
        F, E = EF.batch_features(b, art, True)
        return F, E.astype(np.float32)
    F, E, _ = EF.build(path, art, batch=100000, return_emb=True, emb_dtype=np.float32, extra_cols=(), log=log)
    return F, E


def read_cols(path, cols):
    pf = pq.ParquetFile(path)
    tb = next(pf.iter_batches(batch_size=SMOKE, columns=cols)) if SMOKE else pq.read_table(path, columns=cols)
    out = {}
    for c in cols:
        col = tb.column(c)
        if pa.types.is_timestamp(col.type):
            col = col.cast(pa.int64())
        out[c] = np.asarray(col.to_numpy(zero_copy_only=False)) if SMOKE else col.to_numpy()
    return out


# ------------------------------------------------------------------ base features (reuse an earlier run's output if attached)
fe = None if SMOKE else find('EL_te.npy', opt=True)
if fe and all(os.path.exists(os.path.join(os.path.dirname(fe), f)) for f in ('EL_tr.npy', 'E_tr.npy', 'E_te.npy')):
    FEAT = os.path.dirname(fe); log('reusing base features from', FEAT)
else:
    FEAT = f'{TMP}/feat'; os.makedirs(FEAT, exist_ok=True)
    for tag, path in (('tr', TRAIN), ('te', TEST)):
        if os.path.exists(f'{FEAT}/EL_{tag}.npy') and os.path.exists(f'{FEAT}/E_{tag}.npy') and 'feat' not in STAGES:
            log('base features present:', tag); continue
        F, E = build_file(path, ()); np.save(f'{FEAT}/EL_{tag}.npy', F); np.save(f'{FEAT}/E_{tag}.npy', E); log('built', tag, F.shape); del F, E; gc.collect()
EL_TR = f'{FEAT}/EL_tr.npy'; EL_TE = f'{FEAT}/EL_te.npy'
rtr = read_cols(TRAIN, ['feature1', 'feature18', 'watched']); rte = read_cols(TEST, ['feature1', 'feature18', 'id'])
NTR, NTE = len(rtr['feature1']), len(rte['feature1']); NALL = NTR + NTE
ts_all = np.concatenate([rtr['feature1'], rte['feature1']]).astype(np.int64)
post_all = pd.factorize(np.concatenate([rtr['feature18'], rte['feature18']]))[0].astype(np.int64)
y = rtr['watched'].astype(np.float32); ids = rte['id'].astype(np.int64)
t = ts_all - ts_all[:NTR].min(); HR = 3600_000
assert np.load(EL_TR, mmap_mode='r').shape == (NTR, NF) and np.load(EL_TE, mmap_mode='r').shape == (NTE, NF)
log('rows train', NTR, 'test', NTE, 'posts', post_all.max() + 1, 'train span h %.2f' % (t[:NTR].max() / HR), 'test span h %.2f..%.2f' % (t[NTR:].min() / HR, t.max() / HR), 'rate %.5f' % y.mean())
T_SPLIT = int(np.quantile(t[:NTR], 0.67)) if SMOKE else 48 * HR
T_END = int(t[:NTR].max()) + 1
fit = t[:NTR] < T_SPLIT

# ------------------------------------------------------------------ test-like copies of the training timeline
rng = np.random.default_rng(20261125); grp = rng.integers(0, 10, NTR)
DROPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
HOLES = [[(20, 25), (56, 61)], [(8, 13), (34, 39)], [(27, 32), (63, 68)]]            # hours since train start, 5 h each
if SMOKE:
    HOLES = [[(a * T_END / HR / 72, b * T_END / HR / 72) for a, b in hs] for hs in HOLES]
th = t[:NTR] / HR
KEEP = []
for dr, hs in zip(DROPS, HOLES):
    k = ~np.isin(grp, dr)
    for a, b in hs:
        k &= ~((th >= a) & (th < b))
    KEEP.append(np.flatnonzero(k))
    log('copy', len(KEEP) - 1, 'keeps', len(KEEP[-1]), 'train rows (%.3f)' % (len(KEEP[-1]) / NTR))
cov = np.zeros(NTR, np.int32)
for k in KEEP:
    cov[k] += 1
log('train rows by number of copies containing them:', np.bincount(cov).tolist())


def make_block(rows, keep_n, path):
    """Cross-row features for the timeline made of `rows` (sorted indices into train+test); the first keep_n rows are stored."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    tsr = ts_all[rows]; pr = post_all[rows]
    X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(keep_n, len(blk_names)))
    n_tl = len(TF.feature_names())
    X[:, n_tl:len(tf_names)] = TF.build_future(tsr, pr, Fsub_all[rows], fut_cols)[:keep_n]
    # every statistic is keyed by post, so posts are split into partitions that fit a T4 (global whitening statistics are
    # re-estimated per partition from several hundred thousand rows, i.e. identical up to sampling noise)
    npart = max(1, int(np.ceil(len(rows) / int(os.environ.get('SC_PART_ROWS', '700000'))))); q = pr % npart
    devs = queue.Queue()
    for d in ([f'cuda:{i}' for i in range(NGPU)] if GPU else ['cpu']):
        devs.put(d)

    def one(i):
        m = np.flatnonzero(q == i); nk = int((m < keep_n).sum())
        if nk == 0:
            return i
        dev = devs.get()
        try:
            Em = E_all[rows[m]]; pm = pd.factorize(pr[m])[0].astype(np.int64); tm = tsr[m]; U, P = Em[:, :7], Em[:, 7:]
            a = TF.build_trace(tm, pm, U, P, device=dev)[:nk]
            b = F2.build_w(tm, pm, U, P, F2.PAIRS_W, F2.COMBO_W, 'w', device=dev)[:nk]
            c = np.full((nk, len(v_names)), np.nan, np.float32) if os.environ.get('SC_NO_V') else F2.build_w(tm, pm, U, P, F2.PAIRS_V, F2.COMBO_V, 'v', device=dev)[:nk]
            return i, m[:nk], np.hstack([a, b, c])
        finally:
            if GPU:
                torch.cuda.empty_cache()
            devs.put(dev)
    with ThreadPoolExecutor(max_workers=max(1, NGPU)) as ex:
        for r in ex.map(one, range(npart)):
            if isinstance(r, tuple):
                _, mk, blk = r
                X[mk, :n_tl] = blk[:, :n_tl]; X[mk, C_W[0]:] = blk[:, n_tl:]
    X.flush(); del X
    gc.collect()


FULL = f'{BLK}/full.npy'; CB = [f'{BLK}/cb{k}.npy' for k in range(3)]
NO_COPIES = bool(os.environ.get('SC_NO_COPIES'))          # inference only needs the features on the full timeline
if 'feat' in STAGES:
    E_all = np.concatenate([np.load(f'{FEAT}/E_tr.npy', mmap_mode='r'), np.load(f'{FEAT}/E_te.npy', mmap_mode='r')])
    Fsub_all = np.concatenate([np.load(EL_TR, mmap_mode='r')[:, fidx], np.load(EL_TE, mmap_mode='r')[:, fidx]])
    log('embeddings', E_all.shape, 'counter columns', Fsub_all.shape)
    make_block(np.arange(NALL), NALL, FULL); log('block full done')
    for k in ([] if NO_COPIES else range(3)):
        make_block(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), CB[k]); log('block copy', k, 'done')
    del E_all, Fsub_all; gc.collect()


# ------------------------------------------------------------------ job plumbing
def sidx(name, arr):
    p = f'{TMP}/{name}.npy'; q = f'{TMP}/{name}.{os.getpid()}.tmp.npy'
    np.save(q, arr); os.replace(q, p)                      # atomic: several runs may share this directory
    return p


def src(path, rows=None, cols=None):
    return {'path': path, 'rows': rows, 'cols': None if cols is None else [int(c) for c in cols]}


rows_fit = np.flatnonzero(fit); rows_val = np.flatnonzero(~fit); rows_tr = np.arange(NTR)
P_FIT, P_VAL, P_TR = sidx('rows_fit', rows_fit), sidx('rows_val', rows_val), sidx('rows_tr', rows_tr)
P_TE_EL = sidx('rows_te_el', np.arange(NTE)); P_TE_FULL = sidx('rows_te_full', np.arange(NTR, NALL))
Y_FIT, Y_VAL, Y_TR = sidx('y_fit', y[rows_fit]), sidx('y_val', y[rows_val]), sidx('y_tr', y)
THIN = {'fit': [], 'val': [], 'all': []}; VAL_ROWS = []
for k in range(3):
    idx = KEEP[k]; loc = np.arange(len(idx)); fk = fit[idx]
    for nm, m in (('fit', fk), ('val', ~fk), ('all', np.ones(len(idx), bool))):
        THIN[nm].append({'g': sidx(f'g_{nm}_{k}', idx[m]), 'l': sidx(f'l_{nm}_{k}', loc[m]), 'y': sidx(f'y_{nm}_{k}', y[idx[m]]), 'n': int(m.sum()), 'blk': CB[k]})
    VAL_ROWS.append(idx[~fk])


# extra cross-row blocks stored next to the main ones: name -> (file for the full timeline, files for the 3 test-like copies, column names)
HZ = [300, 600, 1200, 2400]
fh_names = [f'fh{h}_{c}' for h in HZ for c in fut_cols] + [f'fh{h}_gap' for h in HZ]
FW2 = [120, 180, 450, 900, 1800, 3600]; BW = [300, 600, 1200]
KEYC = [c for c in ['feature247', 'feature252', 'feature248', 'feature53', 'feature50', 'feature48', 'feature429', 'feature418', 'feature419', 'feature417',
                    'feature430', 'feature416', 'feature428', 'feature431', 'feature224', 'feature410'] if c in fut_cols]
RC = ['feature247', 'feature248', 'feature252']; TRP = ['u4p6', 'u6p3']
fh2_names = ([f'fh{h}_{c}' for h in FW2 for c in KEYC] + [f'bh{h}_{c}' for h in BW for c in KEYC] + [f'bh{h}_gap' for h in BW]
             + [f'fr{h}_{c}' for h in BW for c in RC] + [f'trn{h}_{p}_{c}' for h in (600, 1200) for p in TRP for c in RC])
UMC = ['feature141', 'feature133', 'feature113', 'feature352', 'feature145', 'feature125', 'feature121', 'feature346', 'feature137', 'feature140', 'feature340',
       'feature132', 'feature337', 'feature112', 'feature358', 'feature149', 'feature345', 'feature339', 'feature351', 'feature493', 'feature144', 'feature331',
       'feature120', 'feature124', 'feature579', 'feature577', 'feature334', 'feature574', 'feature349', 'feature136', 'feature153', 'feature361', 'feature330',
       'feature336', 'feature490', 'feature484', 'feature348', 'feature580', 'feature148', 'feature343']
UMC = [c for c in UMC if c in names]
um_names = (['um_gap_next', 'um_gap_prev'] + [f'um_dn_{c}' for c in UMC] + [f'um_dp_{c}' for c in UMC]
            + [f'um_cos_u{a}p{b}' for a, b in EF.SIG.items()] + [f'um_dot_u{a}p{b}' for a, b in EF.SIG.items()] + ['um_dnorm_u4', 'um_dnorm_u6']
            + [f'um_pcos_u{a}p{b}' for a, b in EF.SIG.items()])
AHC = [c for c in [f'feature{i}' for i in list(range(253, 287)) + [440, 473, 474] + list(range(611, 617))] if c in names]      # author-level real-time columns
AHF = [300, 600, 1200, 2400]; AHB = [600]
ah_names = ([f'ah{h}_{c}' for h in AHF for c in AHC] + [f'ab{h}_{c}' for h in AHB for c in AHC] + [f'ah{h}_gap' for h in AHF] + [f'ab{h}_gap' for h in AHB]
            + ['ah_cnt600', 'ah_cntb600'])
ub_names = [f'ub_dn_{c}' for c in EF.USER_B] + [f'ub_dp_{c}' for c in EF.USER_B] + ['ub_dn_sum', 'ub_dn_abs']
EXTRA = {'ub': (f'{BLK}/ub_full.npy', [f'{BLK}/ub_cb{k}.npy' for k in range(3)], ub_names),
         'ah': (f'{BLK}/ah_full.npy', [f'{BLK}/ah_cb{k}.npy' for k in range(3)], ah_names),
         'fh': (f'{BLK}/fh_full.npy', [f'{BLK}/fh_cb{k}.npy' for k in range(3)], fh_names),
         'fh2': (f'{BLK}/fh2_full.npy', [f'{BLK}/fh2_cb{k}.npy' for k in range(3)], fh2_names),
         'um': (f'{BLK}/um_full.npy', [f'{BLK}/um_cb{k}.npy' for k in range(3)], um_names)}


def parts_thin(which, cols, extra=()):
    return [{'srcs': [src(EL_TR, p['g']), src(p['blk'], p['l'], cols)] + [src(EXTRA[e][1][k], p['l']) for e in extra], 'y': p['y'], 'n': p['n']} for k, p in enumerate(THIN[which])]


def part_full(which, cols, extra=()):
    rp, yp, n = {'fit': (P_FIT, Y_FIT, len(rows_fit)), 'val': (P_VAL, Y_VAL, len(rows_val)), 'all': (P_TR, Y_TR, NTR)}[which]
    return {'srcs': [src(EL_TR, rp), src(FULL, rp, cols)] + [src(EXTRA[e][0], rp) for e in extra], 'y': yp, 'n': n}


def part_test(cols, extra=()):
    return {'srcs': [src(EL_TE, P_TE_EL), src(FULL, P_TE_FULL, cols)] + [src(EXTRA[e][0], P_TE_FULL) for e in extra], 'n': NTE}


PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'learning_rate': 0.1,
          'max_depth': 6, 'min_child_weight': 50, 'colsample_bytree': 0.4, 'reg_lambda': 5.0}
RES = {}


def mkjob(tag, train, evals, es_set=None, fnames=None, seed=1, fixed_rounds=None, lr=0.1, predict=None, save_model=False, es=None, max_rounds=None, pov=None):
    return {'tag': tag, 'params': dict(PARAMS, learning_rate=lr, **(pov or {})), 'train': train, 'evals': evals, 'es_set': es_set,
            'max_rounds': max_rounds or (40 if SMOKE else 8000), 'es': es or (20 if SMOKE else 100), 'out_dir': PRED, 'fnames': fnames, 'seed': seed,
            'fixed_rounds': fixed_rounds, 'predict': predict, 'save_model': save_model, 'device': 'cuda' if GPU else 'cpu', 'chunk': 100000,
            'log_period': 10 if SMOKE else 250}


def run_jobs(jobs):
    slots = ([int(g) for g in os.environ['SC_GPUS'].split(',')] if os.environ.get('SC_GPUS') else list(range(NGPU))) if GPU else [0]
    pending = list(jobs); running = {}
    while pending or running:
        for g in slots:
            if g not in running and pending:
                j = pending.pop(0); j['gpu'] = g
                jp = f"{JOBS}/{j['tag']}.json"; json.dump(j, open(jp, 'w')); lf = open(f"{JOBS}/{j['tag']}.log", 'w')
                running[g] = (j, subprocess.Popen([sys.executable, f"{TMP}/{j.get('runner', 'xgbjob.py')}", jp], stdout=lf, stderr=subprocess.STDOUT), lf)
                log('started', j['tag'], 'on gpu', g)
        time.sleep(3)
        for g in list(running):
            j, pr, lf = running[g]
            if pr.poll() is not None:
                lf.close(); del running[g]
                tail = open(f"{JOBS}/{j['tag']}.log").read().splitlines()
                keep = [l for l in tail if '[' + j['tag'] + ']' in l or 'Traceback' in l][-10:]
                print('\n'.join(keep if pr.returncode == 0 else tail[-40:]), flush=True)
                rp = f"{PRED}/{j['tag']}__res.json"
                if pr.returncode == 0 and os.path.exists(rp):
                    RES[j['tag']] = json.load(open(rp))
                else:
                    log('JOB FAILED', j['tag'], 'rc', pr.returncode); RES[j['tag']] = {'tag': j['tag'], 'failed': True}
        json.dump(RES, open(f'{OUT}/results_{MODE}.json', 'w'), indent=1)


from sklearn.metrics import roc_auc_score
vt_rows = np.concatenate(VAL_ROWS); vt_y = y[vt_rows] > 0.5; vt_t = th[vt_rows]
hole_end = np.zeros(len(vt_rows), bool); o = 0                                   # val rows within 1 h before a hole of their copy
for k in range(3):
    n = len(VAL_ROWS[k])
    for a, b in HOLES[k]:
        hole_end[o:o + n] |= (th[VAL_ROWS[k]] >= a - 1) & (th[VAL_ROWS[k]] < a)
    o += n


def report(tag):
    r = {}
    p = f'{PRED}/{tag}__valthin.npy'
    if os.path.exists(p):
        pv = np.load(p); r['thin'] = roc_auc_score(vt_y, pv)
        if hole_end.sum() > 50 and vt_y[hole_end].any():
            r['thin_prehole'] = roc_auc_score(vt_y[hole_end], pv[hole_end]); r['thin_other'] = roc_auc_score(vt_y[~hole_end], pv[~hole_end])
    p = f'{PRED}/{tag}__valfull.npy'
    if os.path.exists(p):
        r['full'] = roc_auc_score(y[rows_val] > 0.5, np.load(p))
    log(f'  {tag:10s} ' + ' | '.join(f'{k} {v:.5f}' for k, v in r.items()) + f" | rounds {RES.get(tag, {}).get('rounds')}")
    RES.setdefault(tag, {})['report'] = r
    return r


FN = {'c2': names + tf_names, 'w': names + tf_names + w_names, 'wv': names + blk_names}
CO = {'c2': C_TF, 'w': C_TF + C_W, 'wv': C_TF + C_W + C_V}
XT = {'c2': (), 'w': (), 'wv': ()}                       # extra blocks per configuration
for _c, _e in (('wf', ('fh',)), ('wg', ('fh', 'fh2')), ('wfu', ('fh', 'um')), ('wgu', ('fh', 'fh2', 'um')), ('wfa', ('fh', 'ah')), ('wfau', ('fh', 'ah', 'um')), ('wfub', ('fh', 'um', 'ub'))):
    CO[_c] = CO['w']; XT[_c] = _e; FN[_c] = FN['w'] + [n_ for e in _e for n_ in EXTRA[e][2]]


def fut_fixed(tsr, pr, M, keep_n, path):
    """Deltas of the post's counters at the first row of the same post >= t+h (h in HZ)."""
    N = len(tsr); ts_ = tsr - tsr.min(); o = np.lexsort((ts_, pr)); key = pr[o] * 10 ** 10 + ts_[o]; po = pr[o]; tso = ts_[o]; Ms = M[o]; k = M.shape[1]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]
    X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(keep_n, len(fh_names)))
    for i, h in enumerate(HZ):
        j = np.searchsorted(key, key + h * 1000, side='left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po
        X[:, i * k:(i + 1) * k] = np.where(ok[:, None], Ms[jj] - Ms, np.nan)[sel]
        X[:, len(HZ) * k + i] = np.where(ok, np.log1p(np.maximum(tso[jj] - tso, 0) / 1000.0), np.nan)[sel]
    X.flush(); del X


def fut_fixed2(tsr, pr, M, keep_n, path, blk_path):
    """More horizons for the key counters, backward (baseline traffic) windows, forward/backward rate ratios, and trace amplitudes
    normalised by the number of counted events in the same window."""
    N = len(tsr); ts_ = tsr - tsr.min(); o = np.lexsort((ts_, pr)); key = pr[o] * 10 ** 10 + ts_[o]; po = pr[o]; tso = ts_[o]
    kc = [fut_cols.index(c) for c in KEYC]; Ms = M[o][:, kc].astype(np.float64); nk = len(kc)
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]
    X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(keep_n, len(fh2_names))); c0 = 0

    def fwd(h):
        j = np.searchsorted(key, key + h * 1000, side='left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po
        return np.where(ok[:, None], Ms[jj] - Ms, np.nan), np.where(ok, (tso[jj] - tso) / 1000.0, np.nan)

    def bwd(h):
        j = np.searchsorted(key, key - h * 1000, side='right') - 1; ok = j >= 0; jj = np.maximum(j, 0); ok &= po[jj] == po
        return np.where(ok[:, None], Ms - Ms[jj], np.nan), np.where(ok, (tso - tso[jj]) / 1000.0, np.nan)
    for h in FW2:
        d, _ = fwd(h); X[:, c0:c0 + nk] = d[sel]; c0 += nk
    BD = {}
    for h in BW:
        d, g = bwd(h); BD[h] = (d, g); X[:, c0:c0 + nk] = d[sel]; c0 += nk
    for h in BW:
        X[:, c0] = np.log1p(np.maximum(BD[h][1], 0))[sel]; c0 += 1
    FD = {h: fwd(h) for h in BW}
    for h in BW:
        for c in RC:
            q = KEYC.index(c)
            with np.errstate(all='ignore'):
                rf = np.maximum(FD[h][0][:, q], 0) / np.maximum(FD[h][1], 1.0) * 600.0; rb = np.maximum(BD[h][0][:, q], 0) / np.maximum(BD[h][1], 1.0) * 600.0
            X[:, c0] = (np.log1p(rf) - np.log1p(rb))[sel]; c0 += 1
    blk = np.load(blk_path, mmap_mode='r')
    for h in (600, 1200):
        for p in TRP:
            amp = np.asarray(blk[:keep_n, blk_names.index(f'wc_n{h}_mf_{p}')], np.float64)
            for c in RC:
                X[:, c0] = amp / np.sqrt(np.maximum(FD[h][0][:, KEYC.index(c)], 0)[sel] + 1.0); c0 += 1
    assert c0 == len(fh2_names), (c0, len(fh2_names))
    X.flush(); del X


def user_mirror(rows, keep_n, path):
    """Other SESSIONS of the same user (rows of the same user at a different timestamp; rows sharing the row's own timestamp are
    skipped, nothing is grouped or counted per request): change of the user's counters and of the user's embedding heads at the
    user's next / previous session, the latter projected on this row's post heads."""
    global _UM
    if '_UM' not in globals():
        u5 = pd.factorize(np.concatenate([read_cols(TRAIN, ['feature5'])['feature5'], read_cols(TEST, ['feature5'])['feature5']]))[0].astype(np.int64)
        ci = [names.index(c) for c in UMC]
        Mu = np.concatenate([np.load(EL_TR, mmap_mode='r')[:, ci], np.load(EL_TE, mmap_mode='r')[:, ci]]).astype(np.float32)
        _UM = (u5, Mu, np.load(f'{FEAT}/E_tr.npy', mmap_mode='r'), np.load(f'{FEAT}/E_te.npy', mmap_mode='r'))
    u5, Mu, Etr, Ete = _UM

    def getE(g):                                             # embeddings [len(g), 14, 32] for global row ids (any order)
        out = np.zeros((len(g), 14, 32), np.float32); a = g < NTR
        if a.any():
            s = np.argsort(g[a]); out[np.flatnonzero(a)[s]] = Etr[g[a][s]]
        if (~a).any():
            s = np.argsort(g[~a]); out[np.flatnonzero(~a)[s]] = Ete[g[~a][s] - NTR]
        return out
    N = len(rows); us = u5[rows]; tsr = ts_all[rows] - ts_all[rows].min(); o = np.lexsort((tsr, us)); key = us[o] * 10 ** 10 + tsr[o]; uo = us[o]; tso = tsr[o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]                       # positions (in sorted order) of the rows to store
    jn = np.searchsorted(key, key, side='right'); okn = jn < N; jn = np.minimum(jn, N - 1); okn &= uo[jn] == uo
    jp = np.searchsorted(key, key, side='left') - 1; okp = jp >= 0; jp = np.maximum(jp, 0); okp &= uo[jp] == uo
    X = np.full((keep_n, len(um_names)), np.nan, np.float32); K = len(UMC)
    X[:, 0] = np.where(okn, np.log1p((tso[jn] - tso) / 1000.0), np.nan)[sel]; X[:, 1] = np.where(okp, np.log1p((tso - tso[jp]) / 1000.0), np.nan)[sel]
    Ms = Mu[rows][o]
    X[:, 2:2 + K] = np.where(okn[:, None], Ms[jn] - Ms, np.nan)[sel]; X[:, 2 + K:2 + 2 * K] = np.where(okp[:, None], Ms - Ms[jp], np.nan)[sel]
    c0 = 2 + 2 * K; g_sorted = np.asarray(rows)[o]
    for which, jj, ok, col in (('n', jn, okn, c0), ('p', jp, okp, c0 + 16)):
        m = np.flatnonzero(ok[sel])                                                             # stored rows that have the neighbour session
        if len(m) == 0:
            continue
        own = getE(g_sorted[sel[m]]); nb = getE(g_sorted[jj[sel[m]]])
        for q, (a, b) in enumerate(EF.SIG.items()):
            d = (nb[:, a] - own[:, a]) if which == 'n' else (own[:, a] - nb[:, a]); pv = own[:, 7 + b]
            dn = np.linalg.norm(d, axis=1); pn = np.linalg.norm(pv, axis=1); dot = (d * pv).sum(1); v = (dn > 0) & (pn > 0)
            cosv = np.where(v, dot / (dn * pn + 1e-12), np.nan)
            if which == 'n':
                X[m, col + q] = cosv; X[m, col + 7 + q] = np.where(v, dot, np.nan)
                if a in (4, 6):
                    X[m, col + 14 + (0 if a == 4 else 1)] = dn
            else:
                X[m, col + q] = cosv
    np.save(path, X)
    return float(np.isfinite(X[:, 0]).mean())


um3_names = ([f'um3_cosp_u{a}p{b}' for a, b in EF.SIG.items()] + [f'um3_dotp_u{a}p{b}' for a, b in EF.SIG.items()] + [f'um3_along_u{a}' for a in range(7)]
             + [f'um3_perpn_u{a}' for a in range(7)])


def user_mirror3(rows, keep_n, path):
    """Sharper user-side mirror: the change of the user's head a at the user's next session, with the component along the user's own
    (current) head removed - posts shown to one user all lie close to that direction - projected on this row's post head b treated
    the same way.  Also the size of the change along / across the user's own direction."""
    u5 = pd.factorize(np.concatenate([read_cols(TRAIN, ['feature5'])['feature5'], read_cols(TEST, ['feature5'])['feature5']]))[0].astype(np.int64) if '_U5' not in globals() else _U5
    globals()['_U5'] = u5
    Etr = np.load(f'{FEAT}/E_tr.npy', mmap_mode='r'); Ete = np.load(f'{FEAT}/E_te.npy', mmap_mode='r')

    def getE(g):
        out = np.zeros((len(g), 14, 32), np.float32); a = g < NTR
        if a.any():
            s = np.argsort(g[a]); out[np.flatnonzero(a)[s]] = Etr[g[a][s]]
        if (~a).any():
            s = np.argsort(g[~a]); out[np.flatnonzero(~a)[s]] = Ete[g[~a][s] - NTR]
        return out
    N = len(rows); us = u5[rows]; tsr = ts_all[rows] - ts_all[rows].min(); o = np.lexsort((tsr, us)); key = us[o] * 10 ** 10 + tsr[o]; uo = us[o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]; g_sorted = np.asarray(rows)[o]
    jn = np.searchsorted(key, key, side='right'); okn = jn < N; jn = np.minimum(jn, N - 1); okn &= uo[jn] == uo
    X = np.full((keep_n, len(um3_names)), np.nan, np.float32); m = np.flatnonzero(okn[sel])
    if len(m):
        own = getE(g_sorted[sel[m]]).astype(np.float64); nb = getE(g_sorted[jn[sel[m]]]).astype(np.float64)
        for q, (a, b) in enumerate(EF.SIG.items()):
            u = own[:, a]; un = np.linalg.norm(u, axis=1, keepdims=True); uh = u / (un + 1e-12)
            d = nb[:, a] - u; pv = own[:, 7 + b]
            al = (d * uh).sum(1); dp = d - al[:, None] * uh; pp = pv - (pv * uh).sum(1, keepdims=True) * uh
            dn = np.linalg.norm(dp, axis=1); pn = np.linalg.norm(pp, axis=1); dot = (dp * pp).sum(1); v = (dn > 0) & (pn > 0) & (un[:, 0] > 0)
            X[m, q] = np.where(v, dot / (dn * pn + 1e-12), np.nan); X[m, 7 + q] = np.where(v, dot, np.nan)
            X[m, 14 + a] = np.where(un[:, 0] > 0, al, np.nan); X[m, 21 + a] = np.where(un[:, 0] > 0, dn, np.nan)
    np.save(path, X)


EXTRA['um3'] = (f'{BLK}/um3_full.npy', [f'{BLK}/um3_cb{k}.npy' for k in range(3)], um3_names)
CO['wfu3'] = CO['w']; XT['wfu3'] = ('fh', 'um', 'um3'); FN['wfu3'] = FN['w'] + fh_names + um_names + um3_names


def user_bias(rows, keep_n, path):
    """Change of the user's 7 embedding bias scalars at the user's next / previous session (same neighbour rule as user_mirror)."""
    global _UB
    if '_UB' not in globals():
        u5 = pd.factorize(np.concatenate([read_cols(TRAIN, ['feature5'])['feature5'], read_cols(TEST, ['feature5'])['feature5']]))[0].astype(np.int64)
        ci = [names.index(c) for c in EF.USER_B]
        _UB = (u5, np.concatenate([np.load(EL_TR, mmap_mode='r')[:, ci], np.load(EL_TE, mmap_mode='r')[:, ci]]).astype(np.float64))
    u5, Mb = _UB
    N = len(rows); us = u5[rows]; tsr = ts_all[rows] - ts_all[rows].min(); o = np.lexsort((tsr, us)); key = us[o] * 10 ** 10 + tsr[o]; uo = us[o]; Ms = Mb[rows][o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]
    jn = np.searchsorted(key, key, side='right'); okn = jn < N; jn = np.minimum(jn, N - 1); okn &= uo[jn] == uo
    jp = np.searchsorted(key, key, side='left') - 1; okp = jp >= 0; jp = np.maximum(jp, 0); okp &= uo[jp] == uo
    dn = np.where(okn[:, None], Ms[jn] - Ms, np.nan); dp = np.where(okp[:, None], Ms - Ms[jp], np.nan)
    X = np.hstack([dn, dp, np.nansum(dn, 1, keepdims=True) + np.where(okn, 0, np.nan)[:, None], np.nansum(np.abs(dn), 1, keepdims=True) + np.where(okn, 0, np.nan)[:, None]]).astype(np.float32)[sel]
    assert X.shape[1] == len(ub_names)
    np.save(path, X)


def author_fixed(rows, keep_n, path):
    """Author-level real-time counters read at later / earlier rows of the SAME AUTHOR (any of the author's posts): deltas at the
    first such row >= t+h, a baseline window before t, and how many sampled rows of the author fall in those windows.
    Rows sharing the row's own timestamp are never used; nothing is grouped per request."""
    global _AH
    if '_AH' not in globals():
        a5 = pd.factorize(np.concatenate([read_cols(TRAIN, ['feature23'])['feature23'], read_cols(TEST, ['feature23'])['feature23']]))[0].astype(np.int64)
        ci = [names.index(c) for c in AHC]
        _AH = (a5, np.concatenate([np.load(EL_TR, mmap_mode='r')[:, ci], np.load(EL_TE, mmap_mode='r')[:, ci]]).astype(np.float32))
    a5, Ma = _AH
    N = len(rows); au = a5[rows]; tsr = ts_all[rows] - ts_all[rows].min(); o = np.lexsort((tsr, au)); key = au[o] * 10 ** 10 + tsr[o]; ao = au[o]; tso = tsr[o]; Ms = Ma[rows][o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]; K = len(AHC)
    X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(keep_n, len(ah_names))); c0 = 0; gaps = []
    for h in AHF:
        j = np.searchsorted(key, key + h * 1000, side='left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= ao[jj] == ao
        X[:, c0:c0 + K] = np.where(ok[:, None], Ms[jj] - Ms, np.nan)[sel]; c0 += K; gaps.append(np.where(ok, np.log1p(np.maximum(tso[jj] - tso, 0) / 1000.0), np.nan))
    for h in AHB:
        j = np.searchsorted(key, key - h * 1000, side='right') - 1; ok = j >= 0; jj = np.maximum(j, 0); ok &= ao[jj] == ao
        X[:, c0:c0 + K] = np.where(ok[:, None], Ms - Ms[jj], np.nan)[sel]; c0 += K; gaps.append(np.where(ok, np.log1p(np.maximum(tso - tso[jj], 0) / 1000.0), np.nan))
    for g in gaps:
        X[:, c0] = g[sel]; c0 += 1
    after = np.searchsorted(key, key, side='right'); before = np.searchsorted(key, key, side='left')
    X[:, c0] = np.log1p(np.searchsorted(key, key + 600_000, side='right') - after)[sel]; X[:, c0 + 1] = np.log1p(before - np.searchsorted(key, key - 600_000, side='left'))[sel]; c0 += 2
    assert c0 == len(ah_names)
    X.flush(); del X


def ensure_extra(which):
    """Build the extra blocks (full timeline + 3 test-like copies) if they are not on disk yet."""
    full_p, cb_p, _ = EXTRA[which]
    if NO_COPIES:
        cb_p = []
    if all(os.path.exists(p) for p in [full_p] + cb_p):
        return
    if which == 'um3':
        for r, kn, p in [(np.arange(NALL), NALL, full_p)] + [(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), cb_p[k]) for k in range(len(cb_p))]:
            if not os.path.exists(p):
                q = p + f'.{os.getpid()}.tmp.npy'; user_mirror3(r, kn, q); os.replace(q, p); log('extra block um3', os.path.basename(p), 'done')
        return
    if which == 'ub':
        for r, kn, p in [(np.arange(NALL), NALL, full_p)] + [(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), cb_p[k]) for k in range(len(cb_p))]:
            if not os.path.exists(p):
                q = p + f'.{os.getpid()}.tmp.npy'; user_bias(r, kn, q); os.replace(q, p); log('extra block ub', os.path.basename(p), 'done')
        return
    if which == 'ah':
        for r, kn, p in [(np.arange(NALL), NALL, full_p)] + [(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), cb_p[k]) for k in range(3)]:
            if not os.path.exists(p):
                q = p + f'.{os.getpid()}.tmp.npy'; author_fixed(r, kn, q); os.replace(q, p); log('extra block ah', os.path.basename(p), 'done')
        ah = np.load(full_p, mmap_mode='r'); gi = ah_names.index('ah600_gap')
        log('ah coverage (author seen again >= 10 min later): train %.4f | test %.4f ; post-level fh600 for comparison: see fh block' % (np.isfinite(ah[:NTR:7, gi]).mean(), np.isfinite(ah[NTR::7, gi]).mean()))
        return
    if which == 'um':
        for r, kn, p in [(np.arange(NALL), NALL, full_p)] + [(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), cb_p[k]) for k in range(len(cb_p))]:
            if not os.path.exists(p):
                q = p + f'.{os.getpid()}.tmp.npy'; cov = user_mirror(r, kn, q); os.replace(q, p); log('extra block um', os.path.basename(p), 'done; rows with a later session of the same user: %.4f' % cov)
        um = np.load(full_p, mmap_mode='r')
        log('um coverage (later session exists): fit %.4f | val %.4f | test %.4f' % (np.isfinite(um[:NTR, 0][fit]).mean(), np.isfinite(um[:NTR, 0][~fit]).mean(), np.isfinite(um[NTR:, 0]).mean()))
        return
    Fs = np.concatenate([np.load(EL_TR, mmap_mode='r')[:, fidx], np.load(EL_TE, mmap_mode='r')[:, fidx]])
    todo = [(np.arange(NALL), NALL, full_p, FULL)] + [(np.concatenate([KEEP[k], np.arange(NTR, NALL)]), len(KEEP[k]), cb_p[k], CB[k]) for k in range(len(cb_p))]
    for r, kn, p, bp in todo:
        if os.path.exists(p):
            continue
        q = p + f'.{os.getpid()}.tmp.npy'
        if which == 'fh':
            fut_fixed(ts_all[r], post_all[r], Fs[r], kn, q)
        else:
            fut_fixed2(ts_all[r], post_all[r], Fs[r], kn, q, bp)
        os.replace(q, p); log('extra block', which, os.path.basename(p), 'done')
    del Fs; gc.collect()


if 'predict' in STAGES:
    # inference from saved boosters: rebuild the test rows' features and average the models listed in the manifest (logit space)
    import xgboost as xgb
    MD = os.environ.get('SC_MODELS') or os.path.dirname(find('track_b_manifest.json'))
    man = json.load(open(f'{MD}/track_b_manifest.json'))
    _mm = {}

    def chunks(part, ch=100000):
        srcs = []
        for s in part['srcs']:
            rows = np.load(s['rows']); cols = None if s.get('cols') is None else np.asarray(s['cols'], np.int64)
            if s['path'] not in _mm:
                _mm[s['path']] = np.load(s['path'], mmap_mode='r')
            srcs.append((_mm[s['path']], int(rows[0]), cols))
        for a in range(0, part['n'], ch):
            b = min(part['n'], a + ch)
            yield np.hstack([np.asarray(M[st + a:st + b], np.float32) if c is None else np.asarray(M[st + a:st + b], np.float32)[:, c] for M, st, c in srcs])
    z = np.zeros(NTE, np.float64); wsum = 0.0
    for m in man['models']:
        cfg = m['cfg']
        for _e in XT[cfg]:
            ensure_extra(_e)
        part = part_test(CO[cfg], XT[cfg])
        if m['file'].endswith('.cbm'):
            from catboost import CatBoostClassifier
            cbm = CatBoostClassifier(); cbm.load_model(f"{MD}/{m['file']}")
            p = np.concatenate([cbm.predict_proba(X)[:, 1] for X in chunks(part)]).astype(np.float64)
        else:
            bst = xgb.Booster(); bst.load_model(f"{MD}/{m['file']}"); bst.set_param({'device': 'cuda' if GPU else 'cpu', 'nthread': os.cpu_count()})
            assert bst.num_features() == NF + len(CO[cfg]) + sum(len(EXTRA[e][2]) for e in XT[cfg]), (m['file'], bst.num_features())
            p = np.concatenate([np.asarray(bst.inplace_predict(X)) for X in chunks(part)]).astype(np.float64)
        p = np.clip(p, 1e-7, 1 - 1e-7); z += m['weight'] * np.log(p / (1 - p)); wsum += m['weight']
        log('predicted', m['file'], cfg, 'weight', m['weight'], 'mean %.5f' % p.mean())
    pred = 1 / (1 + np.exp(-z / wsum))
    sub = pd.DataFrame({'id': ids, 'watched': np.clip(pred, 1e-7, 1 - 1e-7)})
    if not SMOKE:
        ss = pd.read_csv(SSUB, usecols=['id']); sub = ss.merge(sub, on='id', how='left')
        assert len(sub) == len(ss) and sub['watched'].notna().all()
    sub.to_csv(f'{OUT}/submission.csv', index=False, float_format='%.7f')
    log('wrote submission.csv', len(sub), 'mean %.5f' % sub['watched'].mean(), 'models', len(man['models']))

if 'cb' in STAGES:
    # a CatBoost member for model-family diversity: dense training, hold-out for the round count, then all 72 h -> test predictions
    cfg = os.environ.get('SC_CFG', 'wfu'); sd = int(os.environ.get('SC_SEED', '71')); CP = {'learning_rate': float(os.environ.get('SC_LR', '0.08')), 'depth': int(os.environ.get('SC_DEPTH', '8'))}
    for _e in XT[cfg]:
        ensure_extra(_e)
    cj = lambda tag, train, evals, **k: dict(mkjob(tag, train, evals, seed=sd, **k), params=CP, runner='catjob.py', metric_period=100, es=300, max_rounds=60 if SMOKE else 6000, log_period=20 if SMOKE else 200)
    ht = f'cbh_{cfg}_s{sd}'
    if os.environ.get('SC_ROUNDS'):
        rounds = int(os.environ['SC_ROUNDS'])
    else:
        run_jobs([cj(ht, [part_full('fit', CO[cfg], XT[cfg])], {'valthin': parts_thin('val', CO[cfg], XT[cfg]), 'valfull': [part_full('val', CO[cfg], XT[cfg])]}, es_set='valfull')])
        report(ht); rounds = int(1.15 * RES[ht]['rounds']) + 1
    if not os.environ.get('SC_HOLD_ONLY'):
        trp_ = part_full('all', CO[cfg], XT[cfg])
        if os.environ.get('SC_CB_ROWS'):                       # CatBoost holds the whole matrix in host memory: fit the final model on a random row subset
            rs = np.sort(np.random.default_rng(sd).choice(NTR, int(os.environ['SC_CB_ROWS']), replace=False)); rp = sidx(f'rows_cbsub_{sd}', rs)
            for s_ in trp_['srcs']:
                s_['rows'] = rp
            trp_['y'] = sidx(f'y_cbsub_{sd}', y[rs]); trp_['n'] = len(rs)
        run_jobs([cj(f'cbf_{cfg}_s{sd}', [trp_], {}, fixed_rounds=rounds, predict={'test': [part_test(CO[cfg], XT[cfg])]}, save_model=True)])

if 'xc' in STAGES:
    # generic feature-set experiment: dense training, lr 0.1, seeds 1 and 2, early stopping on the test-like hold-out
    for c in os.environ['SC_XCFG'].split(','):
        for _e in XT[c]:
            ensure_extra(_e)
        xe = {'valfull': [part_full('val', CO[c], XT[c])], 'valthin': parts_thin('val', CO[c], XT[c])}
        sd = [int(s) for s in os.environ.get('SC_XSEEDS', '1,2').split(',')]
        run_jobs([mkjob(f'xc_{c}_s{s}', [part_full('fit', CO[c], XT[c])], xe, es_set='valthin', fnames=FN[c], seed=s) for s in sd])
        log(f'================ XC {c} (dense training, lr 0.1) ================')
        for s in sd:
            report(f'xc_{c}_s{s}')
        pv = np.mean([np.load(f'{PRED}/xc_{c}_s{s}__valthin.npy') for s in sd], 0); pf = np.mean([np.load(f'{PRED}/xc_{c}_s{s}__valfull.npy') for s in sd], 0)
        log(f'  xc_{c} {len(sd)}-seed average: thin %.5f | full %.5f' % (roc_auc_score(vt_y, pv), roc_auc_score(y[rows_val] > 0.5, pf)))
        if 'um' in XT[c]:
            um = np.load(EXTRA['um'][0], mmap_mode='r'); has = np.isfinite(np.asarray(um[rows_val, 0])); yv_ = y[rows_val] > 0.5
            log('  dense val rows with a later session of the same user: %.4f | AUC there %.5f | AUC elsewhere %.5f' % (has.mean(), roc_auc_score(yv_[has], pf[has]), roc_auc_score(yv_[~has], pf[~has])))

if 'x3' in STAGES:
    # experiment: tree-shape / sampling hyper-parameters on the current best feature set (dense training, lr 0.1, one seed each)
    c = os.environ.get('SC_CFG', 'wf')
    for _e in XT[c]:
        ensure_extra(_e)
    GRID = {'base': {}, 'd8': dict(max_depth=8, min_child_weight=100), 'd10': dict(max_depth=10, min_child_weight=200, colsample_bytree=0.3),
            'cs25': dict(colsample_bytree=0.25), 'cs60': dict(colsample_bytree=0.6), 'mcw20': dict(min_child_weight=20),
            'mcw200l20': dict(min_child_weight=200, reg_lambda=20.0), 'sub6': dict(subsample=0.6), 'd8cs25': dict(max_depth=8, min_child_weight=100, colsample_bytree=0.25)}
    only = os.environ.get('SC_GRID'); keys = only.split(',') if only else list(GRID)
    xe = {'valfull': [part_full('val', CO[c], XT[c])], 'valthin': parts_thin('val', CO[c], XT[c])}
    run_jobs([mkjob(f'x3_{c}_{k}', [part_full('fit', CO[c], XT[c])], xe, es_set='valthin', seed=1, pov=GRID[k]) for k in keys])
    log(f'================ X3: hyper-parameters on {c} (dense training, lr 0.1, seed 1) ================')
    for k in keys:
        report(f'x3_{c}_{k}')

if 'x2' in STAGES:
    # experiment: does the second counter block (more horizons, baseline windows, ratios) add over W + fixed-horizon deltas?
    ensure_extra('fh'); ensure_extra('fh2')
    xe = lambda c: {'valfull': [part_full('val', CO[c], XT[c])], 'valthin': parts_thin('val', CO[c], XT[c])}
    run_jobs([mkjob(f'x2_{c}_s{s}', [part_full('fit', CO[c], XT[c])], xe(c), es_set='valthin', fnames=FN[c], seed=s) for s in (1, 2) for c in ('wg',)])
    log('================ X2: second counter block (dense training, lr 0.1) ================')
    for g in ('x2_wg_s1', 'x2_wg_s2'):
        report(g)
    pv = np.mean([np.load(f'{PRED}/x2_wg_s{s}__valthin.npy') for s in (1, 2)], 0); pf = np.mean([np.load(f'{PRED}/x2_wg_s{s}__valfull.npy') for s in (1, 2)], 0)
    log('  x2_wg 2-seed average: thin %.5f | full %.5f' % (roc_auc_score(vt_y, pv), roc_auc_score(y[rows_val] > 0.5, pf)))

def write_sub(tags, name):
    ps = [np.clip(np.load(f'{PRED}/{g}__test.npy').astype(np.float64), 1e-7, 1 - 1e-7) for g in tags]
    z = np.mean([np.log(p / (1 - p)) for p in ps], 0); pred = 1 / (1 + np.exp(-z))
    sub = pd.DataFrame({'id': ids, 'watched': np.clip(pred, 1e-7, 1 - 1e-7)})
    if not SMOKE:
        ss = pd.read_csv(SSUB, usecols=['id']); sub = ss.merge(sub, on='id', how='left')
        assert len(sub) == len(ss) and sub['watched'].notna().all()
    sub.to_csv(f'{OUT}/{name}.csv', index=False, float_format='%.7f')
    log('wrote', name, len(sub), 'mean %.5f' % sub['watched'].mean(), 'models', len(ps), 'min pairwise corr %.5f' % (min(np.corrcoef(ps[i], ps[j])[0, 1] for i in range(len(ps)) for j in range(i)) if len(ps) > 1 else 1.0))


if 'go' in STAGES:
    LR = float(os.environ.get('SC_LR', '0.1')); ESR = 20 if SMOKE else int(os.environ.get('SC_ES', '100'))
    ev = lambda c: {'valfull': [part_full('val', CO[c])], 'valthin': parts_thin('val', CO[c])}
    J = lambda tag, train, c, es_set='valthin': mkjob(tag, train, ev(c), es_set=es_set, fnames=FN[c], lr=LR, es=ESR)
    # wave 1: the comparisons that decide the production configuration
    run_jobs([J('th_w', parts_thin('fit', CO['w']), 'w'), J('th_mix', [part_full('fit', CO['w'])] + parts_thin('fit', CO['w']), 'w'),
              J('th_wv', parts_thin('fit', CO['wv']), 'wv'), J('un_c2', [part_full('fit', CO['c2'])], 'c2', 'valfull')])
    log('================ WAVE 1 (hold-out AUC; thin = test-like geometry, full = dense training-file geometry) ================')
    R = {g: report(g) for g in ('un_c2', 'th_w', 'th_mix', 'th_wv')}
    g_v = R['th_wv'].get('thin', 0) - R['th_w'].get('thin', 0); g_mix = R['th_mix'].get('thin', 0) - R['th_w'].get('thin', 0)
    cfg = os.environ.get('SC_CFG') or ('wv' if g_v > 0.0003 else 'w'); mix = int(os.environ.get('SC_MIX', '1' if g_mix > 0.0003 else '0'))
    ref = 'th_mix' if (mix and cfg == 'w') else f'th_{cfg}'
    rounds = int(1.15 * RES[ref]['rounds']) + 1
    log(f'DECISION: V gain {g_v:+.5f} -> cfg {cfg}; dense-copy gain {g_mix:+.5f} -> mix {mix}; rounds {rounds} (from {ref})')
    tr_parts = lambda: ([part_full('all', CO[cfg])] if mix else []) + parts_thin('all', CO[cfg])
    fin = lambda s: mkjob(f'pf_{cfg}{"m" if mix else ""}_s{s}', tr_parts(), {}, fnames=FN[cfg], seed=s, lr=LR, fixed_rounds=rounds, predict={'test': [part_test(CO[cfg])]}, save_model=True)
    S1 = [11, 12]; S2 = [13, 14]
    run_jobs([fin(s) for s in S1])
    write_sub([f'pf_{cfg}{"m" if mix else ""}_s{s}' for s in S1], 'submission_C1')
    json.dump({'cfg': cfg, 'mix': mix, 'rounds': rounds, 'lr': LR, 'fnames': FN[cfg]}, open(f'{OUT}/prod_config.json', 'w'))
    # wave 2: remaining ablations (what the new feature block and the density matching are each worth) + two more seeds
    run_jobs([J('th_c2', parts_thin('fit', CO['c2']), 'c2'), J('un_w', [part_full('fit', CO['w'])], 'w', 'valfull')] + [fin(s) for s in S2])
    log('================ WAVE 2 ================')
    for g in ('un_c2', 'th_c2', 'un_w', 'th_w', 'th_mix', 'th_wv'):
        report(g)
    write_sub([f'pf_{cfg}{"m" if mix else ""}_s{s}' for s in S1 + S2], 'submission_C2')

if 'x1' in STAGES:
    # experiment: later-snapshot deltas of the post's public counters at fixed horizons (first row of the same post >= t+h)
    HZ = [300, 600, 1200, 2400]
    fh_names = [f'fh{h}_{c}' for h in HZ for c in fut_cols] + [f'fh{h}_gap' for h in HZ]

    def fut_h(tsr, pr, M, keep_n, path):
        N = len(tsr); ts = tsr - tsr.min(); o = np.lexsort((ts, pr)); key = pr[o] * 10 ** 10 + ts[o]; po = pr[o]; tso = ts[o]; Ms = M[o]; k = M.shape[1]
        inv = np.empty(N, np.int64); inv[o] = np.arange(N); sel = inv[:keep_n]
        X = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(keep_n, len(fh_names)))
        for i, h in enumerate(HZ):
            j = np.searchsorted(key, key + h * 1000, side='left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po
            X[:, i * k:(i + 1) * k] = np.where(ok[:, None], Ms[jj] - Ms, np.nan)[sel]
            X[:, len(HZ) * k + i] = np.where(ok, np.log1p((tso[jj] - tso) / 1000.0), np.nan)[sel]
        X.flush(); del X
    FHF = f'{BLK}/fh_full.npy'; FHC = [f'{BLK}/fh_cb{k}.npy' for k in range(3)]
    if not all(os.path.exists(p) for p in [FHF] + FHC):
        Fsub_all = np.concatenate([np.load(EL_TR, mmap_mode='r')[:, fidx], np.load(EL_TE, mmap_mode='r')[:, fidx]])
        fut_h(ts_all, post_all, Fsub_all, NALL, FHF + '.tmp.npy'); os.replace(FHF + '.tmp.npy', FHF); log('fh full done')
        for k in range(3):
            r = np.concatenate([KEEP[k], np.arange(NTR, NALL)])
            fut_h(ts_all[r], post_all[r], Fsub_all[r], len(KEEP[k]), FHC[k] + '.tmp.npy'); os.replace(FHC[k] + '.tmp.npy', FHC[k]); log('fh copy', k, 'done')
        del Fsub_all; gc.collect()

    def xfull(which, cols, fh):
        p = part_full(which, cols)
        if fh:
            p['srcs'].append(src(FHF, p['srcs'][0]['rows']))
        return p

    def xthin(which, cols, fh):
        ps = parts_thin(which, cols)
        if fh:
            for k, p in enumerate(ps):
                p['srcs'].append(src(FHC[k], THIN[which][k]['l']))
        return ps
    xe = lambda fh: {'valfull': [xfull('val', CO['w'], fh)], 'valthin': xthin('val', CO['w'], fh)}
    jobs = []
    for s in (1, 2):
        jobs.append(mkjob(f'x_fh_s{s}', [xfull('fit', CO['w'], True)], xe(True), es_set='valthin', fnames=FN['w'] + fh_names, seed=s))
        jobs.append(mkjob(f'x_base_s{s}', [xfull('fit', CO['w'], False)], xe(False), es_set='valthin', fnames=FN['w'], seed=s))
    run_jobs(jobs)
    log('================ X1: fixed-horizon counter deltas (dense training, lr 0.1) ================')
    for g in ('x_base_s1', 'x_base_s2', 'x_fh_s1', 'x_fh_s2'):
        report(g)
    for nm in ('x_base', 'x_fh'):
        pv = np.mean([np.load(f'{PRED}/{nm}_s{s}__valthin.npy') for s in (1, 2)], 0); pf = np.mean([np.load(f'{PRED}/{nm}_s{s}__valfull.npy') for s in (1, 2)], 0)
        log(f'  {nm} 2-seed average: thin %.5f | full %.5f' % (roc_auc_score(vt_y, pv), roc_auc_score(y[rows_val] > 0.5, pf)))

if 'go2' in STAGES:
    # production at a lower learning rate: hold-out (early stopping, test-like geometry) for each seed -> refit on all 72 h -> test predictions
    cfg = os.environ.get('SC_CFG', 'w'); mix = int(os.environ.get('SC_MIX', '1')); LR = float(os.environ.get('SC_LR', '0.03'))
    HS = [int(s) for s in os.environ.get('SC_HSEEDS', '21,22').split(',')]; FS = [int(s) for s in os.environ.get('SC_FSEEDS', '21,22').split(',')]
    trainset = os.environ.get('SC_TRAINSET') or ('mix' if mix else 'thin')          # mix = dense copy + 3 test-like copies | thin | dense
    tg = f'{cfg}{dict(mix="m", thin="t", dense="d")[trainset]}_lr{int(round(LR * 1000)):03d}'
    for _e in XT[cfg]:
        ensure_extra(_e)
    ev = {'valfull': [part_full('val', CO[cfg], XT[cfg])], 'valthin': parts_thin('val', CO[cfg], XT[cfg])}
    trp = lambda w: ([part_full(w, CO[cfg], XT[cfg])] if trainset in ('mix', 'dense') else []) + (parts_thin(w, CO[cfg], XT[cfg]) if trainset in ('mix', 'thin') else [])
    if os.environ.get('SC_ROUNDS'):
        rounds = int(os.environ['SC_ROUNDS'])
    else:
        run_jobs([mkjob(f'ph_{tg}_s{s}', trp('fit'), ev, es_set='valthin', fnames=FN[cfg], seed=s, lr=LR, es=20 if SMOKE else 200) for s in HS])
        log(f'================ HOLD-OUT {tg} ================')
        for s in HS:
            report(f'ph_{tg}_s{s}')
        pv = np.mean([np.load(f'{PRED}/ph_{tg}_s{s}__valthin.npy') for s in HS], 0); pf = np.mean([np.load(f'{PRED}/ph_{tg}_s{s}__valfull.npy') for s in HS], 0)
        log('  seed-average: thin %.5f | full %.5f' % (roc_auc_score(vt_y, pv), roc_auc_score(y[rows_val] > 0.5, pf)))
        rounds = int(1.15 * np.mean([RES[f'ph_{tg}_s{s}']['rounds'] for s in HS])) + 1
    log('final rounds', rounds)
    run_jobs([mkjob(f'pf_{tg}_s{s}', trp('all'), {}, fnames=FN[cfg], seed=s, lr=LR, fixed_rounds=rounds, predict={'test': [part_test(CO[cfg], XT[cfg])]}, save_model=True) for s in FS])
    write_sub([f'pf_{tg}_s{s}' for s in FS], f'submission_C_{tg}')
    json.dump({'cfg': cfg, 'mix': mix, 'rounds': rounds, 'lr': LR, 'fnames': FN[cfg]}, open(f'{OUT}/prod_config_{tg}.json', 'w'))

if 'abl' in STAGES:
    ev = lambda c: {'valfull': [part_full('val', CO[c])], 'valthin': parts_thin('val', CO[c])}
    jobs = [mkjob('th_w', parts_thin('fit', CO['w']), ev('w'), es_set='valthin', fnames=FN['w']),
            mkjob('un_w', [part_full('fit', CO['w'])], ev('w'), es_set='valfull', fnames=FN['w']),
            mkjob('th_wv', parts_thin('fit', CO['wv']), ev('wv'), es_set='valthin', fnames=FN['wv']),
            mkjob('th_c2', parts_thin('fit', CO['c2']), ev('c2'), es_set='valthin', fnames=FN['c2']),
            mkjob('un_c2', [part_full('fit', CO['c2'])], ev('c2'), es_set='valfull', fnames=FN['c2'])]
    run_jobs(jobs)
    log('================ ABLATION (hold-out AUC; thin = test-like geometry, full = dense training-file geometry) ================')
    for tag in ('un_c2', 'th_c2', 'un_w', 'th_w', 'th_wv'):
        report(tag)

if 'prod' in STAGES:
    try:
        gain_v = RES['th_wv']['report']['thin'] - RES['th_w']['report']['thin']
    except Exception:
        gain_v = 0.0
    cfg = os.environ.get('SC_CFG') or ('wv' if gain_v > 0.0003 else 'w')
    LR = float(os.environ.get('SC_LR', '0.05')); SEEDS = [int(s) for s in os.environ.get('SC_SEEDS', '11,12').split(',')]
    log(f'PRODUCTION config {cfg} (V gain {gain_v:+.5f}) lr {LR} seeds {SEEDS}')
    ev = {'valfull': [part_full('val', CO[cfg])], 'valthin': parts_thin('val', CO[cfg])}
    run_jobs([mkjob(f'ph_{cfg}_s{s}', parts_thin('fit', CO[cfg]), ev, es_set='valthin', fnames=FN[cfg], seed=s, lr=LR, es=150) for s in SEEDS])
    for s in SEEDS:
        report(f'ph_{cfg}_s{s}')
    pv = np.mean([np.load(f'{PRED}/ph_{cfg}_s{s}__valthin.npy') for s in SEEDS], 0)
    log('  hold-out seed-average (test-like geometry): %.5f' % roc_auc_score(vt_y, pv))
    rounds = int(1.15 * np.mean([RES[f'ph_{cfg}_s{s}']['rounds'] for s in SEEDS])) + 1
    log('final rounds', rounds)
    run_jobs([mkjob(f'pf_{cfg}_s{s}', parts_thin('all', CO[cfg]), {}, fnames=FN[cfg], seed=s, lr=LR, fixed_rounds=rounds,
                    predict={'test': [part_test(CO[cfg])]}, save_model=True) for s in SEEDS])
    ps = [np.clip(np.load(f'{PRED}/pf_{cfg}_s{s}__test.npy').astype(np.float64), 1e-7, 1 - 1e-7) for s in SEEDS]
    z = np.mean([np.log(p / (1 - p)) for p in ps], 0); pred = 1 / (1 + np.exp(-z))
    sub = pd.DataFrame({'id': ids, 'watched': np.clip(pred, 1e-7, 1 - 1e-7)})
    if not SMOKE:
        ss = pd.read_csv(SSUB, usecols=['id']); sub = ss.merge(sub, on='id', how='left')
        assert len(sub) == len(ss) and sub['watched'].notna().all()
    sub.to_csv(f'{OUT}/submission_C_{cfg}.csv', index=False, float_format='%.7f')
    log('wrote submission', len(sub), 'mean %.5f' % sub['watched'].mean(), 'seed corr %.5f' % (np.corrcoef(ps[0], ps[-1])[0, 1] if len(ps) > 1 else 1.0))

json.dump(RES, open(f'{OUT}/results_{MODE}.json', 'w'), indent=1)
log('ALL DONE')
