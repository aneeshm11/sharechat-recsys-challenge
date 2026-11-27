"""feats2: richer cross-row 'trace' statistics in the per-post noise-whitened space.

DISCLOSURE: NOT row-independent.  For a row (user U, post P, time t) these read the embedding of the SAME POST at later rows.
No request_id, no request grouping and no grouping by user is used anywhere in this file.

Preparation (whitening, projection, per-post increment covariance) is the same as trace_feat.build_trace.  On top of it:
  wa_*  aggregates (max / sum / signed root-sum-square) of per-increment whitened cosines (zc) and amplitudes (mf) between
        consecutive snapshots of the post on a time grid, over all increments and over latency-restricted windows
  wc_*  cumulative whitened cosine of (post embedding at first row >= t+h) - (post embedding at this row), for 15 horizons
"""
import numpy as np
import torch

H14 = [0, 30, 60, 120, 180, 300, 450, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200]
H_MF = (600, 900, 1200, 1800)
PAIRS_W = [(4, 6), (6, 3), (5, 4)]
COMBO_W = [('c2', [(0, 1.0), (1, 1.0)]), ('c3', [(0, 1.0), (1, 1.0), (2, -1.0)])]
PAIRS_V = [(0, 1), (1, 0), (2, 5), (3, 2)]
COMBO_V = [('c2', [(0, 1.0), (1, 1.0)]), ('c4', [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)])]
_SUBS = [('all', ('max', 'sum')), ('lat', ('max', 'sum', 'rss')), ('lat2', ('max', 'sum'))]


def feature_names(pairs=PAIRS_W, combos=COMBO_W, prefix='w'):
    n = [f'{prefix}_ninc_u{a}p{b}' for a, b in pairs] + [f'{prefix}_nval_u{a}p{b}' for a, b in pairs]
    tags = [f'zc_u{a}p{b}' for a, b in pairs] + [f'mf_u{a}p{b}' for a, b in pairs] + [f'zc_{c}' for c, _ in combos] + [f'mf_{combos[0][0]}']
    for t in tags:
        for sub, sts in _SUBS:
            n += [f'{prefix}a_{t}_{sub}_{s}' for s in sts]
    for h in H14:
        n += [f'{prefix}c_n{h}_zc_u{a}p{b}' for a, b in pairs] + [f'{prefix}c_n{h}_zc_{c}' for c, _ in combos]
        if h in H_MF:
            n += [f'{prefix}c_n{h}_mf_u{a}p{b}' for a, b in pairs]
    return n


@torch.no_grad()
def build_w(ts_ms, post, U, P, pairs=PAIRS_W, combos=COMBO_W, prefix='w', device=None, kappa=64.0, log=None):
    """ts_ms int64 [N]; post int codes [N]; U, P float32 [N,7,32].  Returns float32 [N, F] in input row order."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    N = len(ts_ms); ts = (np.asarray(ts_ms, np.int64) - int(np.min(ts_ms))); post = np.asarray(post, np.int64)
    o = np.lexsort((ts, post)); key = post[o] * 10 ** 10 + ts[o]; po = post[o]; tso = ts[o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); inv_t = torch.from_numpy(inv).to(dev)
    pc = torch.from_numpy(po).to(dev); NP = int(post.max()) + 1; ot = torch.from_numpy(o).to(dev)
    same_next = torch.from_numpy(np.r_[po[1:] == po[:-1], False]).to(dev)
    eye = torch.eye(32, device=dev, dtype=torch.float64)
    Utn, Pt, uok, NINC = [], [], [], []
    for a, b in pairs:                                                                   # ---- identical to trace_feat.build_trace
        Ua = torch.from_numpy(np.ascontiguousarray(U[:, a])).to(dev)[ot]; Pb = torch.from_numpy(np.ascontiguousarray(P[:, b])).to(dev)[ot]
        ok_u = Ua.norm(dim=1) > 0
        Xr = Ua[ok_u].double(); mu = Xr.mean(0); Xc = Xr - mu
        C = (Xc.T @ Xc) / len(Xc); C += 1e-4 * torch.trace(C) / 32 * torch.eye(32, device=dev, dtype=torch.float64)
        ev, evec = torch.linalg.eigh(C); Wm = evec @ torch.diag(ev.clamp_min(1e-12) ** -0.5) @ evec.T
        ut = (Ua.double() - mu) @ Wm; pt = Pb.double() @ Wm
        e1 = mu @ Wm; e1 = e1 / e1.norm()
        utv = torch.where(ok_u[:, None], ut, torch.zeros_like(ut))
        Ssum = torch.zeros(NP, 32, device=dev, dtype=torch.float64).index_add_(0, pc, utv); cnt = torch.zeros(NP, device=dev, dtype=torch.float64).index_add_(0, pc, ok_u.double())
        m = Ssum / cnt.clamp_min(1)[:, None]; m = m - (m @ e1)[:, None] * e1[None]
        e2p = torch.where((cnt >= 20)[:, None], m / (m.norm(dim=1, keepdim=True) + 1e-12), torch.zeros_like(m))
        e2r = e2p[pc]

        def perp(x):
            x = x - (x @ e1)[:, None] * e1[None]
            return x - (x * e2r).sum(1, keepdim=True) * e2r
        pt = perp(pt); ut = perp(ut)
        Cp = torch.zeros(NP, 1024, device=dev, dtype=torch.float64); ninc = torch.zeros(NP, device=dev, dtype=torch.float64)
        for s in range(0, N - 1, 100000):
            e = min(N - 1, s + 100000); d = pt[s + 1:e + 1] - pt[s:e]; dn = d.norm(dim=1); v = same_next[s:e] & (dn > 0)
            dh = d[v] / dn[v][:, None]; pp = pc[s:e][v]
            Cp.index_add_(0, pp, (dh[:, :, None] * dh[:, None, :]).reshape(-1, 1024)); ninc.index_add_(0, pp, torch.ones(len(pp), device=dev, dtype=torch.float64))
        Cs = (32.0 * Cp.reshape(NP, 32, 32) + kappa * eye[None]) / (ninc + kappa)[:, None, None]
        Cs = Cs + e1[:, None] * e1[None, :] + e2p[:, :, None] * e2p[:, None, :]
        A = torch.empty(NP, 32, 32, device=dev, dtype=torch.float64)
        for s in range(0, NP, 10000):
            w, Vv = torch.linalg.eigh(Cs[s:s + 10000]); A[s:s + 10000] = Vv @ torch.diag_embed(w.clamp_min(1e-6) ** -0.5) @ Vv.transpose(1, 2)
        del Cp, Cs
        for s in range(0, N, 100000):
            Ar = A[pc[s:s + 100000]]
            pt[s:s + 100000] = torch.bmm(pt[s:s + 100000, None, :], Ar).squeeze(1); ut[s:s + 100000] = torch.bmm(ut[s:s + 100000, None, :], Ar).squeeze(1)
        un = ut.norm(dim=1)
        uok.append(ok_u); Utn.append((ut / (un[:, None] + 1e-12)).float()); Pt.append(pt.float()); NINC.append(ninc[pc].float())
        del A, Ua, Pb, ut, pt, utv, Ssum, m, e2p, e2r
        if log:
            log(f'  feats2[{prefix}]: pair u{a}p{b} prepared')

    def look(h):
        j = np.searchsorted(key, key + h * 1000, side='right' if h == 0 else 'left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po
        return torch.from_numpy(jj).to(dev), torch.from_numpy(ok).to(dev), torch.from_numpy(np.where(ok, (tso[jj] - tso) / 1000.0, 0).astype(np.float32)).to(dev)
    idx = [torch.arange(N, device=dev)]; okg = [torch.ones(N, dtype=torch.bool, device=dev)]; off = [torch.zeros(N, device=dev)]
    for h in H14:
        j, ok, of = look(h); idx.append(j); okg.append(ok); off.append(of)
    K = len(H14); start = torch.stack(off[:-1]); end = torch.stack(off[1:])              # [K, N]
    nan = torch.tensor(float('nan'), device=dev); NEG = -1e9
    Z, M, V = [], [], []
    for p in range(len(pairs)):
        z = torch.zeros(K, N, device=dev); mm = torch.zeros(K, N, device=dev); vv = torch.zeros(K, N, dtype=torch.bool, device=dev)
        for k in range(K):
            j1, j2 = idx[k], idx[k + 1]; valid = okg[k] & okg[k + 1] & (j2 != j1)
            d = Pt[p][j2] - Pt[p][j1]; dn = d.norm(dim=1); dt = (Utn[p] * d).sum(1); v = valid & (dn > 0) & uok[p]
            z[k] = torch.where(v, dt / (dn + 1e-12), torch.zeros_like(dt)); mm[k] = torch.where(v, dt, torch.zeros_like(dt)); vv[k] = v
        Z.append(z); M.append(mm); V.append(vv)
    cols = [n for n in NINC] + [v.float().sum(0) for v in V]

    def agg(S, Vm):
        for sub, mask in (('all', Vm), ('lat', Vm & (start < 900)), ('lat2', Vm & (start < 1800) & (end > 30))):
            any_ = mask.any(0); s0 = torch.where(mask, S, torch.zeros_like(S))
            cols.append(torch.where(any_, torch.where(mask, S, torch.full_like(S, NEG)).max(0).values, nan))
            cols.append(torch.where(any_, s0.sum(0), nan))
            if sub == 'lat':
                cols.append(torch.where(any_, torch.sign(s0.sum(0)) * torch.sqrt((s0 ** 2).sum(0)), nan))

    def combo(T, spec):
        s = sum(w * T[i] for i, w in spec) / (len(spec) ** 0.5); vm = V[spec[0][0]]
        for i, _ in spec[1:]:
            vm = vm & V[i]
        return s, vm
    for p in range(len(pairs)):
        agg(Z[p], V[p])
    for p in range(len(pairs)):
        agg(M[p], V[p])
    for _, spec in combos:
        agg(*combo(Z, spec))
    s, vm = combo(M, combos[0][1]); agg(s * (len(combos[0][1]) ** 0.5), vm)
    del Z, M
    for hi, h in enumerate(H14):
        j2 = idx[hi + 1]; zs = []; ms = []
        for p in range(len(pairs)):
            d = Pt[p][j2] - Pt[p]; dn = d.norm(dim=1); v = okg[hi + 1] & (dn > 0) & uok[p]; dt = (Utn[p] * d).sum(1)
            zs.append(torch.where(v, dt / (dn + 1e-12), nan)); ms.append(torch.where(v, dt, nan))
        cols += zs + [sum(w * zs[i] for i, w in spec) / (len(spec) ** 0.5) for _, spec in combos]
        if h in H_MF:
            cols += ms
    out = torch.stack([c.float()[inv_t] for c in cols], 1).cpu().numpy().astype(np.float32)
    assert out.shape[1] == len(feature_names(pairs, combos, prefix)), (out.shape, len(feature_names(pairs, combos, prefix)))
    return out
