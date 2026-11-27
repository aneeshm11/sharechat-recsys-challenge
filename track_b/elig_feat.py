"""Row-independent feature builder (ShareChat RecSys Challenge) — used identically for local training and Kaggle inference.

Every output value is a function of ONE input row plus artifacts frozen from the TRAINING file (category vocabularies,
training-set entity counts).  Nothing here groups rows, reads request_id, or looks at any other row of the file being scored.

Feature blocks
  num   : every numeric masked column of the row, as float32 (schema order, frozen in the config)
  time  : minute of day (IST) of the row's timestamp + sin/cos
  age   : post age and account age at request time (from the row's own timestamps)
  cat   : low/medium-cardinality hashed categories -> integer code by training frequency rank (unseen = -1, null = -2)
  geo   : the row's 7 user x 7 post embedding heads: 49 dots, 49 cosines, 14 norms, 14 validity flags
  plat  : matched-head scores  <u_a, p_s(a)>  (+ the row's bias scalars) = the serving model's own per-head logits
"""
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

USER_E = [f'feature{i}' for i in range(26, 33)]
POST_E = [f'feature{i}' for i in range(40, 47)]
USER_B = [f'feature{i}' for i in range(33, 40)]          # 7 user-level scalars
POST_B = [f'feature{i}' for i in range(47, 54)]          # 7 post-level scalars
SIG = {0: 1, 1: 0, 2: 5, 3: 2, 4: 6, 5: 4, 6: 3}          # user head -> matched post head
CAT_COLS = ['feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
            'feature16', 'feature17', 'feature19', 'feature20', 'feature21', 'feature22', 'feature25']
TS_COL = 'feature1'
SKIP = {'request_id', 'id', 'watched', TS_COL}


def numeric_columns(path):
    sch = pq.read_schema(path)
    return [f.name for f in sch if f.name not in SKIP and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))]


def fit_artifacts(train_path, batch=200000):
    """Category vocabularies from the TRAINING file only."""
    from collections import Counter
    cnt = {c: Counter() for c in CAT_COLS}
    pf = pq.ParquetFile(train_path)
    for b in pf.iter_batches(batch_size=batch, columns=CAT_COLS):
        for c in CAT_COLS:
            vc = b.column(c).value_counts()
            for v, n in zip(vc.field(0).to_pylist(), vc.field(1).to_pylist()):
                if v is not None:
                    cnt[c][v] += n
    art = {'num_cols': numeric_columns(train_path),
           'vocab': {c: {v: i for i, (v, _) in enumerate(cnt[c].most_common())} for c in CAT_COLS}}
    return art


def _lists(col, n):
    a = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    valid = ~np.asarray(a.is_null().to_numpy(zero_copy_only=False))
    offs = np.asarray(a.offsets); lens = np.diff(offs)
    vals = np.asarray(a.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    ok = valid & (lens == 32)
    m = np.zeros((n, 32), np.float32)
    if ok.any():
        m[ok] = vals[offs[:-1][ok][:, None] + np.arange(32)[None, :]]
    return m, ok


def feature_names(art):
    names = list(art['num_cols']) + ['t_min', 't_sin', 't_cos', 'post_age_h', 'post_age_log', 'user_age_h', 'user_age_log']
    names += ['cat_' + c for c in CAT_COLS]
    names += [f'dot_u{a}p{b}' for a in range(7) for b in range(7)] + [f'cos_u{a}p{b}' for a in range(7) for b in range(7)]
    names += [f'nu{a}' for a in range(7)] + [f'np{b}' for b in range(7)] + [f'vu{a}' for a in range(7)] + [f'vp{b}' for b in range(7)]
    names += [f'plat_u{a}p{b}_bu{a}_bp{b}' for a, b in SIG.items()] + [f'plat_u{a}p{b}_sumb' for a, b in SIG.items()]
    names += ['plat_max', 'plat_mean', 'bias_u_sum', 'bias_p_sum']
    return names


def batch_features(b, art, return_emb=False):
    """b: pyarrow RecordBatch/Table with the raw columns of some rows. Returns float32 [rows, F] (and raw embeddings if asked)."""
    n = b.num_rows
    cols = {name: b.column(name) for name in b.schema.names}
    num = np.empty((n, len(art['num_cols'])), np.float32)
    for j, c in enumerate(art['num_cols']):
        num[:, j] = cols[c].to_numpy(zero_copy_only=False).astype(np.float32)
    ts = cols[TS_COL]
    ts_ms = np.asarray(ts.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)
    tmin = ((ts_ms // 60000 + 330) % 1440).astype(np.float32)
    ang = tmin * (2 * np.pi / 1440)
    ix = {c: j for j, c in enumerate(art['num_cols'])}
    ts_s = ts_ms / 1000.0
    f57 = cols['feature57'].to_numpy(zero_copy_only=False).astype(np.float64)
    f58 = cols['feature58'].to_numpy(zero_copy_only=False).astype(np.float64)
    pa_h = (ts_s - f58) / 3600.0; ua_h = (ts_s - f57) / 3600.0
    with np.errstate(invalid='ignore', divide='ignore'):
        tblock = np.stack([tmin, np.sin(ang), np.cos(ang), pa_h, np.log1p(np.maximum(pa_h, 0)), ua_h, np.log1p(np.maximum(ua_h, 0))], 1).astype(np.float32)
    cat = np.empty((n, len(CAT_COLS)), np.float32)
    for j, c in enumerate(CAT_COLS):
        v = cols[c].to_pylist(); d = art['vocab'][c]
        cat[:, j] = [(-2 if x is None else d.get(x, -1)) for x in v]
    U = np.empty((n, 7, 32), np.float32); P = np.empty((n, 7, 32), np.float32); vu = np.empty((n, 7), bool); vp = np.empty((n, 7), bool)
    for k in range(7):
        U[:, k], vu[:, k] = _lists(cols[USER_E[k]], n)
        P[:, k], vp[:, k] = _lists(cols[POST_E[k]], n)
    dots = np.einsum('nad,nbd->nab', U, P)
    nu = np.linalg.norm(U, axis=2); npn = np.linalg.norm(P, axis=2)
    cos = dots / (nu[:, :, None] * npn[:, None, :] + 1e-8)
    both = vu[:, :, None] & vp[:, None, :]
    dots = np.where(both, dots, np.nan).astype(np.float32); cos = np.where(both, cos, np.nan).astype(np.float32)
    bu = np.stack([num[:, ix[c]] for c in USER_B], 1); bp = np.stack([num[:, ix[c]] for c in POST_B], 1)
    md = np.stack([dots[:, a, b] for a, b in SIG.items()], 1)
    plat1 = np.stack([dots[:, a, b] + bu[:, a] + bp[:, b] for a, b in SIG.items()], 1)
    sb = (np.nansum(bu, 1) + np.nansum(bp, 1))[:, None]
    plat2 = md + sb
    with np.errstate(all='ignore'):
        extra = np.stack([np.nanmax(md, 1), np.nanmean(md, 1), np.nansum(bu, 1), np.nansum(bp, 1)], 1)
    F = np.hstack([num, tblock, cat, dots.reshape(n, 49), cos.reshape(n, 49),
                   np.where(vu, nu, np.nan), np.where(vp, npn, np.nan), vu.astype(np.float32), vp.astype(np.float32),
                   plat1, plat2, extra]).astype(np.float32)
    if return_emb:
        return F, np.concatenate([U, P], 1)
    return F


def build(path, art, batch=100000, return_emb=False, extra_cols=(), log=None, emb_dtype=np.float16):
    """Features for every row of a parquet file, in file order. extra_cols: raw columns returned alongside (e.g. id, watched, feature1)."""
    pf = pq.ParquetFile(path)
    need = sorted(set(art['num_cols']) | {TS_COL, 'feature57', 'feature58'} | set(CAT_COLS) | set(USER_E) | set(POST_E) | set(extra_cols))
    Fs, Es, ex = [], [], {c: [] for c in extra_cols}
    done = 0
    for b in pf.iter_batches(batch_size=batch, columns=need):
        r = batch_features(b, art, return_emb)
        if return_emb:
            Fs.append(r[0]); Es.append(r[1].astype(emb_dtype))
        else:
            Fs.append(r)
        for c in extra_cols:
            col = b.column(c)
            ex[c].append(np.asarray(col.cast(pa.int64()).to_numpy(zero_copy_only=False)) if pa.types.is_timestamp(col.type) else np.asarray(col.to_numpy(zero_copy_only=False)))
        done += b.num_rows
        if log is not None:
            log(f'  {path.split("/")[-1]}: {done} rows')
    out = [np.vstack(Fs)]
    if return_emb:
        out.append(np.concatenate(Es))
    out.append({c: np.concatenate(v) for c, v in ex.items()})
    return tuple(out)
