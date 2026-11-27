"""Cross-row 'trace' features computed INSIDE ONE FILE (gray-area variant C2).

DISCLOSURE: these features are NOT row-independent. For a row (user U, post P, time t) they read the embedding of the same post
at LATER rows of the same file. They use no request_id and no grouping by request, but they do use other rows.

Mechanism: the serving model is trained online; a view moves the post's embedding head b along the viewing user's paired head a.
Statistic: latency-weighted matched filter of the post's later embedding increments against the row's user vector, in a
per-post noise-whitened space (global user covariance -> project out global/audience mean directions -> per-post increment covariance)."""
import numpy as np
import torch

PAIRS = [(4, 6), (6, 3), (5, 4)]
H_GRID = [0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600, 720, 840, 960, 1200, 1500, 1800, 2700, 3600]
H_CUM = [300, 450, 600, 900, 1200, 1800, 3600]
_TF = [0, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 720, 840, 960, 1200, 1500, 1800, 2700, 3600, 7200.]
_VF = [0, .015, .03, .065, .11, .17, .25, .35, .46, .56, .64, .71, .81, .87, .91, .95, .97, .98, .99, .995, 1.0]


def feature_names():
    n = []
    for a, b in PAIRS:
        n += [f'tr_z_u{a}p{b}', f'tr_amp_u{a}p{b}', f'tr_info_u{a}p{b}', f'tr_zbest_u{a}p{b}', f'tr_un_u{a}p{b}']
    n += ['tr_z_c2', 'tr_z_c3']
    for h in H_CUM:
        n += [f'tr_c{h}_u{a}p{b}' for a, b in PAIRS] + [f'tr_c{h}_c2', f'tr_c{h}_c3']
    n += ['tr_gap_next', 'tr_gap_900', 'tr_ninc']
    return n


@torch.no_grad()
def build_trace(ts_ms, post, U, P, device=None, kappa=64.0, log=None):
    """ts_ms int64 [N]; post int codes [N]; U, P float32 [N,7,32] (user heads, post heads). Returns float32 [N, F] in input row order."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    N = len(ts_ms); ts = (np.asarray(ts_ms, np.int64) - int(np.min(ts_ms))); post = np.asarray(post, np.int64)
    o = np.lexsort((ts, post)); key = post[o] * 10 ** 10 + ts[o]; po = post[o]; tso = ts[o]
    inv = np.empty(N, np.int64); inv[o] = np.arange(N); inv_t = torch.from_numpy(inv).to(dev)
    pc = torch.from_numpy(po).to(dev); NP = int(post.max()) + 1; ot = torch.from_numpy(o).to(dev)
    same_next = torch.from_numpy(np.r_[po[1:] == po[:-1], False]).to(dev)
    eye = torch.eye(32, device=dev, dtype=torch.float64)
    Utn, Pt, uok, UN, ninc0 = [], [], [], [], None
    for a, b in PAIRS:
        Ua = torch.from_numpy(np.ascontiguousarray(U[:, a])).to(dev)[ot]; Pb = torch.from_numpy(np.ascontiguousarray(P[:, b])).to(dev)[ot]
        ok_u = Ua.norm(dim=1) > 0
        Xr = Ua[ok_u].double(); mu = Xr.mean(0); Xc = Xr - mu
        C = (Xc.T @ Xc) / len(Xc); C += 1e-4 * torch.trace(C) / 32 * torch.eye(32, device=dev, dtype=torch.float64)
        ev, evec = torch.linalg.eigh(C); Wm = evec @ torch.diag(ev.clamp_min(1e-12) ** -0.5) @ evec.T          # float64
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
            w, V = torch.linalg.eigh(Cs[s:s + 10000]); A[s:s + 10000] = V @ torch.diag_embed(w.clamp_min(1e-6) ** -0.5) @ V.transpose(1, 2)
        del Cp, Cs
        for s in range(0, N, 100000):
            Ar = A[pc[s:s + 100000]]
            pt[s:s + 100000] = torch.bmm(pt[s:s + 100000, None, :], Ar).squeeze(1); ut[s:s + 100000] = torch.bmm(ut[s:s + 100000, None, :], Ar).squeeze(1)
        un = ut.norm(dim=1)
        uok.append(ok_u); UN.append(un); Utn.append(ut / (un[:, None] + 1e-12)); Pt.append(pt)
        if ninc0 is None: ninc0 = ninc[pc]
        del A, Ua, Pb, ut, utv, Ssum, m, e2p, e2r
        if log: log(f'  trace: pair u{a}p{b} prepared')
    tF = torch.tensor(_TF, device=dev); vF = torch.tensor(_VF, device=dev)
    def Fcdf(t):
        t = t.clamp(0, 7200); i = torch.searchsorted(tF, t.contiguous(), right=True).clamp(1, len(tF) - 1)
        return vF[i - 1] + (vF[i] - vF[i - 1]) * (t - tF[i - 1]) / (tF[i] - tF[i - 1])
    def look(h):
        j = np.searchsorted(key, key + h * 1000, side='right' if h == 0 else 'left'); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po
        return torch.from_numpy(jj).to(dev), torch.from_numpy(ok).to(dev), torch.from_numpy(np.where(ok, (tso[jj] - tso) / 1000.0, 0).astype(np.float32)).to(dev)
    idx = [torch.arange(N, device=dev)]; okg = [torch.ones(N, dtype=torch.bool, device=dev)]; off = [torch.zeros(N, device=dev)]
    for h in H_GRID:
        j, ok, of = look(h); idx.append(j); okg.append(ok); off.append(of)
    nan = torch.tensor(float('nan'), device=dev, dtype=torch.float64); NEG = -1e9
    cols = []; zs = []
    for p, (a, b) in enumerate(PAIRS):
        num = torch.zeros(N, device=dev, dtype=torch.float64); den2 = torch.zeros(N, device=dev, dtype=torch.float64); best = torch.full((N,), NEG, device=dev, dtype=torch.float64)
        for k in range(len(H_GRID)):
            j1, j2 = idx[k], idx[k + 1]; valid = okg[k] & okg[k + 1] & (j2 != j1)
            d = Pt[p][j2] - Pt[p][j1]; dn = d.norm(dim=1); x = (Utn[p] * d).sum(1); v = valid & (dn > 0) & uok[p]
            w = (Fcdf(off[k + 1]) - Fcdf(off[k])).clamp_min(0).double(); s2 = (dn ** 2 / 32).clamp_min(1e-30); z0 = torch.zeros_like(x)
            num += torch.where(v, w * x / s2, z0); den2 += torch.where(v, w * w / s2, z0)
            best = torch.maximum(best, torch.where(v & (w > 0.05), x / torch.sqrt(s2), torch.full_like(x, NEG)))
        ok = den2 > 0; z = torch.where(ok, num / torch.sqrt(den2.clamp_min(1e-30)), nan); zs.append(z)
        cols += [z, torch.where(ok, num / den2.clamp_min(1e-30), nan), torch.where(ok, torch.sqrt(den2), nan), torch.where(best > -1e8, best, nan), torch.where(uok[p], UN[p], nan)]
    cols += [(zs[0] + zs[1]) / 2 ** 0.5, (zs[0] + zs[1] - zs[2]) / 3 ** 0.5]
    for h in H_CUM:
        j2, ok2, _ = look(h); c = []
        for p in range(len(PAIRS)):
            d = Pt[p][j2] - Pt[p]; dn = d.norm(dim=1); v = ok2 & (dn > 0) & uok[p]
            c.append(torch.where(v, (Utn[p] * d).sum(1) / (dn + 1e-12), nan))
        cols += c + [(c[0] + c[1]) / 2 ** 0.5, (c[0] + c[1] - c[2]) / 3 ** 0.5]
    j0, ok0, of0 = look(0); j9, ok9, of9 = look(900)
    cols += [torch.where(ok0, of0.double(), nan), torch.where(ok9, of9.double(), nan), ninc0]
    out = torch.stack([c.float()[inv_t] for c in cols], 1).cpu().numpy().astype(np.float32)
    assert out.shape[1] == len(feature_names())
    return out


def future_names(colnames):
    return [f'fn_{c}' for c in colnames] + [f'f1_{c}' for c in colnames] + [f'fp_{c}' for c in colnames] + ['fu_gap_next', 'fu_gap_1h', 'fu_gap_prev']


def build_future(ts_ms, post, M, colnames):
    """Later/earlier snapshots of the same post's counters inside the file (cross-row, no request_id).
    M: float32 [N, k] the row's own values of the post-level columns `colnames`.
    fn_ = value at the post's next row minus own; f1_ = value at the first row >= 1 h later minus own; fp_ = own minus value at the previous row."""
    N = len(ts_ms); ts = np.asarray(ts_ms, np.int64) - int(np.min(ts_ms)); post = np.asarray(post, np.int64)
    o = np.lexsort((ts, post)); key = post[o] * 10 ** 10 + ts[o]; po = post[o]; tso = ts[o]
    Ms = np.asarray(M, np.float32)[o]
    def nxt(h, side):
        j = np.searchsorted(key, key + h * 1000, side=side); ok = j < N; jj = np.minimum(j, N - 1); ok &= po[jj] == po; return jj, ok
    j1, ok1 = nxt(0, 'right'); jh, okh = nxt(3600, 'left')
    jp = np.searchsorted(key, key, side='left') - 1; okp = jp >= 0; jp = np.maximum(jp, 0); okp &= po[jp] == po
    out = np.empty((N, 3 * Ms.shape[1] + 3), np.float32); k = Ms.shape[1]
    out[:, :k] = np.where(ok1[:, None], Ms[j1] - Ms, np.nan)
    out[:, k:2 * k] = np.where(okh[:, None], Ms[jh] - Ms, np.nan)
    out[:, 2 * k:3 * k] = np.where(okp[:, None], Ms - Ms[jp], np.nan)
    out[:, 3 * k] = np.where(ok1, np.log1p((tso[j1] - tso) / 1000.0), np.nan)
    out[:, 3 * k + 1] = np.where(okh, np.log1p((tso[jh] - tso) / 1000.0), np.nan)
    out[:, 3 * k + 2] = np.where(okp, np.log1p((tso - tso[jp]) / 1000.0), np.nan)
    res = np.empty_like(out); res[o] = out
    return res
