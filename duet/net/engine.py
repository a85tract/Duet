"""A fast inference engine for the two-block circuit network (duet.net.model.Net, head "lines"): bit-packed detector
bits (B, ceil(m / 8)) uint8 on the GPU and each shot's pattern index -> logits c (B, n_obs), confidence min_P |c_P|
and decisions, in Triton kernels over the detector sites only.

Adapted from CoDecoder's deployed engine (surface_v2/timing/engine.py and deploy_tables in surface_v2/convlines/net.py,
License 2.0, see third_party/CoDecoder). Kept: tiles of sites of one type, the stem's in-program bit gather, the
zero-padded dense grid for the residual blocks' 3 x 3 x 3 gathers, several taps per product, the folded read-out
(query-key product in one matrix, the constant bias table, per-tile pooled partial sums, one read-out program per line
with the last one forming c through an atomic counter). Changed: two blocks with shared weights (tiles never mix
blocks); bit-packed input decoded in the stem; the constant channels and the role flags folded into a per-pattern site
table; the gate coupling (new kernel); one head per product over its blocks' sites; g = D A mod 2 from the packed bits
in the read-out; each shot's pattern selects its tables.

Kernels per call (L residual blocks, L + 1 couplings), each over (shot, tile of BM sites):
  stem x = P[pattern, site] + sum over the 3 x 5 x 5 taps of bit * W[type, tap]; epilogue: coupling 0's
               u = LN(x) Wd (site-indexed)
  couple k x += Up(sum over the gates beside the site's layer of the partner block's u at the gate's two layers,
               times the map of (side, role, type)); the partner's u on layers t - 1, t, t + 1 is gathered one layer
               at a time and multiplied by the Cb-row slices of the maps (slots) it enters, the role of each row
               masking the slots it does not have; the partner's u at a layer without a detector there is
               down(LN(0)), outside the layers 0. Epilogue: block k's u = SiLU(LN(x) Wd + bd) on the padded grid, or
               (last coupling) each product's line-attention partial sums over the tile (the head LayerNorm's affine
               folded into the score and value weights)
  block k x += Up(SiLU(type-specific 3 x 3 x 3 convolution of u + bc)); epilogue: coupling k + 1's u
  read-out one program per (shot, product P, line j): the partial sums added / N0, out-projection, residual with
               the query, LayerNorm, MLP -> z; t = w (-1)^g z with g from the packed bits; the shot's last program sums
Arithmetic: by default every matrix product runs as three fp16 tensor-core products of operands split into hi + lo
halves (hi.hi + lo.hi + hi.lo, about 2^-21 relative error, fp32 accumulation); activations that other sites gather are
stored as hi and lo; the residual stream, LayerNorm, gates and the read-out are fp32 (mode "precise"). Mode "onepass"
runs one fp16 product per matrix product.
Widths that are not powers of two (the width C, the bottleneck Cb, the head width) run zero-padded to the next power of
two (Cb and the head width to at least 16): every padded weight and bias is 0, so the padded channels stay 0, and
LayerNorm takes its statistics over the C true channels. Power-of-two networks compile the same kernels as before.

The call the analysis uses:
    eng = engine(net, specs, mode="precise" | "onepass", chunk=None, latency=False)
    c, conf, dec = predict(eng, det, pid) # a whole shot set from host memory -> numpy
    out, dec = eng(pk, pid, out, dec) # one batch on the GPU (fixed buffers, CUDA-graph capturable)"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from duet.net import geometry, model as M


def _p2(n):
    return 1 << max(0, int(n) - 1).bit_length()


# ---------------------------------------------------------------------------------------------------------- tables

@torch.no_grad()
def deploy(net, specs):
    """Inference tables (CPU, fp32 / int32 numpy-backed tensors) of a Net for one or more CNOT patterns. specs: the
    geometry.analyse output of each pattern's circuit (one geometry); index k of specs is pattern index k at run time.
    Engine site order: (block, type, t, i, j); groups: the row ranges of the four (block, type) groups (the engine cuts
    them into tiles)."""
    assert net.head == "lines", "the engine implements the line head"
    net = copy.deepcopy(net).float().cpu().eval()
    t = trunk_tables(net, specs, net)
    s0 = specs[0]
    d, m, C, blk = s0["d"], s0["m"], net.C, s0["block"]
    inv = np.empty(m, np.int64)
    inv[t["order"]] = np.arange(m)

    # heads: per product, the folded score matrix, the bias table (engine rows), values, read-out
    DP = max(16, _p2(d))
    heads = []
    for k, (hd, Pr) in enumerate(zip(net.heads, s0["products"])):
        Hh, dh = hd.h, hd.dh
        q = hd.wq(hd.queries()).view(d, Hh, dh) * dh ** -0.5
        Wsc = torch.zeros(C, Hh, DP)
        for h in range(Hh):
            Wsc[:, h, :d] = hd.k.weight[h * dh:(h + 1) * dh].t() @ q[:, h].t()
        sel = np.flatnonzero(np.isin(blk, Pr["blocks"]))
        bias = torch.zeros(m, Hh, DP)
        bias[torch.as_tensor(inv[sel]), :, :d] = hd.bias().permute(0, 2, 1)
        heads.append(dict(ln_w=hd.ln.weight, ln_b=hd.ln.bias, eps=hd.ln.eps, Wsc=Wsc.reshape(C, Hh * DP),
                          bias=bias.reshape(m, Hh * DP), Wv=hd.v.weight.t(), bv=hd.v.bias, Uq=hd.queries(),
                          Wo=hd.o.weight.t(), bo=hd.o.bias, m_ln_w=hd.mlp[0].weight, m_ln_b=hd.mlp[0].bias,
                          m_eps=hd.mlp[0].eps, W1=hd.mlp[1].weight.t(), b1=hd.mlp[1].bias,
                          W2=hd.mlp[3].weight.reshape(-1), b2=hd.mlp[3].bias.reshape(())))
    t.update(heads=heads, DP=DP, Hh=net.heads[0].h, dh=net.heads[0].dh, n0=M.N0)
    return _detached(t)


def trunk_tables(net, specs, geo):
    """deploy's tables of the trunk (stem, couplings, residual blocks), the site order and each pattern's gate roles
    and A, for a network already copied to the CPU in fp32. geo holds the grid masks and constant channels (mx, mz,
    const): the network itself (Net), or one depth's tables of a variable-depth network (model_ro, engine_ro)."""
    assert getattr(net, "couple_mode", "cross") == "cross", "the engine implements the cross-block coupling"
    s0 = specs[0]
    for s in specs[1:]:
        assert geometry.same_geometry(s0, s), "the patterns' geometries differ"
    d, m, nobs, T, H, W = (s0[k] for k in ("d", "m", "n_obs", "T", "H", "W"))
    C = net.C
    blk, tij, typ = s0["block"], s0["tij"], s0["type"]
    order = np.lexsort((tij[:, 2], tij[:, 1], tij[:, 0], typ, blk))       # engine row r = detector order[r]
    inv = np.empty(m, np.int64)
    inv[order] = np.arange(m)
    eb, et, ety = blk[order], tij[order], typ[order]
    groups = []
    for b in (0, 1):
        for ty in (0, 1):
            rr = np.flatnonzero((eb == b) & (ety == ty))
            if len(rr):
                groups.append((int(rr[0]), int(rr[-1]) + 1, b, ty))
    det_at = {(int(b), int(r[0]), int(r[1]), int(r[2])): k for k, (b, r) in enumerate(zip(blk, tij))}

    # stem: tap detectors (circuit order; m = a zero bit), tap weights per type, per-pattern site table P
    st = net.stem
    kt, ki, kj = net.cfg["stem_k"]
    taps = [(a - kt // 2, c - ki // 2, e - kj // 2) for a in range(kt) for c in range(ki) for e in range(kj)]
    KS = len(taps)
    K1 = KS if KS & (KS - 1) == 0 else 1 << (KS.bit_length() - 1)               # taps in two products: K1 + K2
    K2 = max(16, _p2(KS - K1)) if KS > K1 else 0
    KSP = K1 + K2
    nbd = np.full((m, KSP), m, np.int32)
    for r in range(m):
        b, (t, i, j) = int(eb[r]), et[r]
        for q, (dt, di, dj) in enumerate(taps):
            k = det_at.get((b, t + dt, i + di, j + dj))
            if k is not None:
                nbd[r, q] = k
    w0 = st.weight[:, 0].reshape(2 * C, KS)
    Wb = torch.zeros(2, KSP, C)
    Wb[0, :KS], Wb[1, :KS] = w0[:C].t(), w0[C:].t()
    yc = F.conv3d(geo.const, st.weight[:, 1:], st.bias, padding=st.padding).permute(0, 2, 3, 4, 1)   # (2, T, H, W, 2C)
    P, role, Apk = [], [], []
    NBY = (m + 7) // 8
    NBYP = _p2(NBY + 1)
    for s in specs:
        gt = torch.as_tensor(geometry.gate_table(s))                                                # (2, T, 2, 2)
        y = yc + net.role(gt.flatten(2))[:, :, None, None] * (geo.mx + geo.mz)
        x = y[..., :C] * geo.mx + y[..., C:] * geo.mz                                                # (2, T, H, W, C)
        P.append(x[torch.as_tensor(eb), torch.as_tensor(et[:, 0]), torch.as_tensor(et[:, 1]),
                   torch.as_tensor(et[:, 2])])
        g = gt.numpy()[eb, et[:, 0]]                                                                 # (m, side, role)
        role.append(np.where(g.sum(-1) > 0, 1 + g.argmax(-1), 0).astype(np.int32))    # 0 none, 1 ctl, 2 tgt
        A = np.asarray(s["A"], np.uint8)                                                             # (m, nobs d)
        pk = np.zeros((A.shape[1], NBYP), np.uint8)
        pk[:, :NBY] = np.packbits(A.T, axis=1, bitorder="little")
        Apk.append(pk)

    # coupling: partner rows at t - 1, t, t + 1 (-1 outside the layers, -2 no detector there)
    pc = np.zeros((m, 3), np.int32)
    for r in range(m):
        b, (t, i, j) = int(eb[r]), et[r]
        for q, k in enumerate((-1, 0, 1)):
            if not 0 <= t + k < T:
                pc[r, q] = -1
            else:
                k2 = det_at.get((1 - b, t + k, i, j))
                pc[r, q] = inv[k2] if k2 is not None else -2
    couples = []
    for cp in net.couple:
        Cb = cp.Cb
        mp = [cp.maps[side, ro] for side in (0, 1) for ro in (0, 1)]                 # slot 2 (2 side + role) + layer
        couples.append(dict(ln_w=cp.ln.weight, ln_b=cp.ln.bias, eps=cp.ln.eps, Wd=cp.down.weight.t(),
                            c0=cp.down(cp.ln(torch.zeros(1, C)))[0],
                            M=torch.stack([torch.cat([w[:, ty * Cb:(ty + 1) * Cb] for w in mp], 0) for ty in (0, 1)]),
                            Wu=cp.up.weight.t(), bu=cp.up.bias))

    # residual blocks on the zero-padded grid (2, T + 2 pt, H + 2 pi, W + 2 pj)
    bk = tuple(net.cfg["block_k"])
    pt, pi, pj = (v // 2 for v in bk)
    GT, GH, GW = T + 2 * pt, H + 2 * pi, W + 2 * pj
    gpos = ((eb * GT + et[:, 0] + pt) * GH + et[:, 1] + pi) * GW + et[:, 2] + pj
    blocks = []
    for bl in net.blocks:
        Cb = bl.Cb
        wc = bl.conv.weight                                                        # (2 Cb, Cb, kt, ki, kj)
        blocks.append(dict(ln_w=bl.ln.weight, ln_b=bl.ln.bias, eps=bl.ln.eps, Wd=bl.down.weight.t(), bd=bl.down.bias,
                           Wc=torch.stack([wc[ty * Cb:(ty + 1) * Cb].permute(2, 3, 4, 1, 0).reshape(-1, Cb)
                                           for ty in (0, 1)]),
                           bc=torch.stack([bl.conv.bias[:Cb], bl.conv.bias[Cb:]]), Wu=bl.up.weight.t(), bu=bl.up.bias))

    return dict(d=d, m=m, nobs=nobs, T=T, H=H, W=W, C=C, Cb=net.couple[0].Cb, L=len(net.blocks), order=order,
                groups=groups, KSP=KSP, K1=K1, K2=K2, nbd=nbd, Wb=Wb, P=torch.stack(P), role=np.stack(role), NBY=NBY,
                NBYP=NBYP, Apk=np.stack(Apk), pc=pc, couples=couples, grid=(GT, GH, GW), block_k=bk,
                gpos=gpos.astype(np.int32), blocks=blocks, prod_blocks=[list(Pr["blocks"]) for Pr in s0["products"]],
                w=net.w.reshape(-1), b=net.b, cfg=dict(net.cfg))


def _detached(t):
    if torch.is_tensor(t):
        return t.detach().float().clone().contiguous()
    if isinstance(t, dict):
        return {k: _detached(v) for k, v in t.items()}
    if isinstance(t, list):
        return [_detached(v) for v in t]
    return t


# --------------------------------------------------------------------------------------------------------- kernels
# Every matrix is stored as fp16 hi (K, N) followed by fp16 lo = W - hi (K, N). X3 = False: one product of fp16
# inputs (activations rounded to fp16). X3 = True: three products hi.hi + lo.hi + hi.lo of the split operands, about
# fp32 accuracy (relative error near 2^-21 instead of 2^-11), at three times the tensor-core work. Stored activations
# (the couplings' and blocks' u, gathered by other sites) hold fp16 hi, then lo if X3, in rows of UW = CB or 2 CB.

@triton.jit
def _mm(a, w_ptr, woff, KN: tl.constexpr, X3: tl.constexpr):
    """a (M, K) fp32 times W (K, N) (fp16 hi at w_ptr + woff, lo at w_ptr + KN + woff) -> (M, N) fp32."""
    ah = a.to(tl.float16)
    wh = tl.load(w_ptr + woff)
    acc = tl.dot(ah, wh)
    if X3:
        acc += tl.dot((a - ah.to(tl.float32)).to(tl.float16), wh) + tl.dot(ah, tl.load(w_ptr + KN + woff))
    return acc


@triton.jit
def _mm2(ah, al, w_ptr, woff, KN: tl.constexpr, X3: tl.constexpr):
    """a given as fp16 hi (and lo if X3) times W, as _mm."""
    wh = tl.load(w_ptr + woff)
    acc = tl.dot(ah, wh)
    if X3:
        acc += tl.dot(al, wh) + tl.dot(ah, tl.load(w_ptr + KN + woff))
    return acc


@triton.jit
def _put(ptr, roff, u, rm, CB: tl.constexpr, UW: tl.constexpr, X3: tl.constexpr):
    """u (BM, CB) fp32 -> row roff of a stored activation: fp16 hi, then lo = u - hi if the rows hold it."""
    cb = tl.arange(0, CB)
    hi = u.to(tl.float16)
    tl.store(ptr + roff[:, None] * UW + cb[None, :], hi, mask=rm[:, None])
    if UW == 2 * CB:
        tl.store(ptr + roff[:, None] * UW + CB + cb[None, :], (u - hi.to(tl.float32)).to(tl.float16), mask=rm[:, None])


@triton.jit
def _norm(x, cols, C: tl.constexpr, CN: tl.constexpr, eps):
    """LayerNorm statistics over the CN true channels of the C (padded) columns."""
    mean = tl.sum(x, 1) / CN
    xc = x - mean[:, None]
    if CN < C:
        xc = tl.where((cols < CN)[None, :], xc, 0.0)
    return xc * tl.rsqrt(tl.sum(xc * xc, 1) / CN + eps)[:, None]


@triton.jit
def _ln(x, w_ptr, b_ptr, cols, C: tl.constexpr, CN: tl.constexpr, eps):
    return _norm(x, cols, C, CN, eps) * tl.load(w_ptr + cols)[None, :] + tl.load(b_ptr + cols)[None, :]


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _tile(tiles_ptr, tile, BM: tl.constexpr):
    r0 = tl.load(tiles_ptr + tile * 4)
    nr = tl.load(tiles_ptr + tile * 4 + 1)
    typ = tl.load(tiles_ptr + tile * 4 + 3)
    i = tl.arange(0, BM)
    return r0 + i, i < nr, typ


@triton.jit
def _uc(x, rows, rm, b, m, lnw, lnb, eps, wd_ptr, uc_ptr, C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr,
        UW: tl.constexpr, X3: tl.constexpr):
    """A coupling's u = LN(x) Wd (no bias) of the tile rows -> uc (B, m, UW)."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    u = _mm(_ln(x, lnw, lnb, cols, C, CN, eps), wd_ptr, cols[:, None] * CB + cb[None, :], C * CB, X3)
    _put(uc_ptr, b * m + rows, u, rm, CB, UW, X3)


@triton.jit
def _ub(x, gp, rm, b, NG, lnw, lnb, eps, wd_ptr, bd_ptr, ub_ptr, C: tl.constexpr, CN: tl.constexpr,
        CB: tl.constexpr, UW: tl.constexpr, X3: tl.constexpr):
    """A residual block's u = SiLU(LN(x) Wd + bd) of the tile rows -> ub (B, NG, UW) at grid positions gp."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    u = _silu(_mm(_ln(x, lnw, lnb, cols, C, CN, eps), wd_ptr, cols[:, None] * CB + cb[None, :], C * CB, X3) +
              tl.load(bd_ptr + cb)[None, :])
    _put(ub_ptr, b * NG + gp, u, rm, CB, UW, X3)


@triton.jit
def _stem_taps(allb, nbd_ptr, rows, rm, m, Wb_ptr, typ, t0, KP: tl.constexpr, KSP: tl.constexpr, C: tl.constexpr,
               BM: tl.constexpr, X3: tl.constexpr):
    """Taps t0.. t0 + KP of the stem: the (BM, KP) bit tile gathered from the shot's packed bytes, times W[type]."""
    cols = tl.arange(0, C)
    taps = t0 + tl.arange(0, KP)
    nd = tl.reshape(tl.load(nbd_ptr + rows[:, None] * KSP + taps[None, :], mask=rm[:, None], other=m), (BM * KP,))
    bit = tl.reshape(((tl.gather(allb, nd >> 3, 0) >> (nd & 7)) & 1).to(tl.float16), (BM, KP))
    woff = typ * 2 * KSP * C + taps[:, None] * C + cols[None, :]                               # Wb (type, hi / lo)
    acc = tl.dot(bit, tl.load(Wb_ptr + woff))
    if X3:
        acc += tl.dot(bit, tl.load(Wb_ptr + KSP * C + woff))                                    # the bits are exact
    return acc


@triton.jit
def _k_stem(pk_ptr, pid_ptr, tiles_ptr, nbd_ptr, P_ptr, Wb_ptr, x_ptr, uc_ptr, lnw, lnb, eps, wd_ptr, m, NBY,
            C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, KSP: tl.constexpr, K1: tl.constexpr,
            K2: tl.constexpr, NBYP: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr, PDL: tl.constexpr):
    """The shot's packed bytes are read once; the bit tiles of the taps are gathered from them in the program."""
    if PDL:
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ = _tile(tiles_ptr, tl.program_id(1), BM)
    pat = tl.load(pid_ptr + b)
    cols = tl.arange(0, C)
    by = tl.arange(0, NBYP)
    allb = tl.load(pk_ptr + b * NBY + by, mask=by < NBY, other=0).to(tl.int32)                   # (NBYP,)
    acc = _stem_taps(allb, nbd_ptr, rows, rm, m, Wb_ptr, typ, 0, K1, KSP, C, BM, X3)
    if K2 > 0:
        acc += _stem_taps(allb, nbd_ptr, rows, rm, m, Wb_ptr, typ, K1, K2, KSP, C, BM, X3)
    x = tl.load(P_ptr + (pat * m + rows)[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + acc
    tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    _uc(x, rows, rm, b, m, lnw, lnb, eps, wd_ptr, uc_ptr, C, CN, CB, UW, X3)


@triton.jit
def _couple(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, pat, m,
            C: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr):
    """x[rows] + Up(V) (BM, C) fp32, V = sum over the gates beside the row's layer t (side 0 before, 1 after) of the
    partner's u at the gate's two layers times the map of (side, the row's role there, the tile's type). The partner's
    u at layer t - 1 + k is gathered once per k; the map is applied in Cb-row slices (slot 2 (2 side + role) + half)."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    rbase = (pat * m + rows) * 2
    r0 = tl.load(role_ptr + rbase, mask=rm, other=0)               # role code on each side: 0 none, 1 ctl, 2 tgt
    r1 = tl.load(role_ptr + rbase + 1, mask=rm, other=0)
    mb = M_ptr + typ * 2 * 8 * CB * CB
    v = tl.zeros((BM, CB), dtype=tl.float32)
    for k in tl.static_range(3):                                    # partner layer t - 1 + k
        p = tl.load(pc_ptr + rows * 3 + k, mask=rm, other=-1)
        src = uc_ptr + (b * m + p)[:, None] * UW + cb[None, :]
        lm = (p >= 0)[:, None]
        miss = (p == -2)[:, None]                                   # no detector there: u = down(LN(0))
        uh = tl.where(miss, tl.load(c0_ptr + cb)[None, :], tl.load(src, mask=lm, other=0.0))
        ul = uh
        if X3:
            ul = tl.where(miss, tl.load(c0_ptr + CB + cb)[None, :], tl.load(src + CB, mask=lm, other=0.0))
        zero = tl.zeros_like(uh)
        for side in tl.static_range(2):
            if k >= side:
                if k <= side + 1:
                    if side == 0:
                        rr = r0
                    else:
                        rr = r1
                    for ro in tl.static_range(2):
                        sel = (rr == ro + 1)[:, None]
                        slot = (side * 2 + ro) * 2 + k - side
                        v += _mm2(tl.where(sel, uh, zero), tl.where(sel, ul, zero), mb,
                                  (slot * CB + cb[:, None]) * CB + cb[None, :], 8 * CB * CB, X3)
    up = _mm(v, Wu_ptr, cb[:, None] * C + cols[None, :], CB * C, X3) + tl.load(bu_ptr + cols)[None, :]
    return tl.load(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + up


@triton.jit
def _k_couple_block(x_ptr, uc_ptr, pid_ptr, tiles_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, gpos_ptr,
                    ub_ptr, lnw, lnb, eps, wd_ptr, bd_ptr, m, NG, C: tl.constexpr, CN: tl.constexpr,
                    CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ = _tile(tiles_ptr, tl.program_id(1), BM)
    pat = tl.load(pid_ptr + b)
    x = _couple(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, pat, m, C, CB, UW,
                BM, X3)
    cols = tl.arange(0, C)
    tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
    _ub(x, gp, rm, b, NG, lnw, lnb, eps, wd_ptr, bd_ptr, ub_ptr, C, CN, CB, UW, X3)


@triton.jit
def _k_couple_head(x_ptr, uc_ptr, pid_ptr, tiles_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, pair_ptr,
                   heps, wsc_ptr, bias_ptr, wv_ptr, bv_ptr, pa_ptr, m, NPAIR, C: tl.constexpr, CN: tl.constexpr,
                   CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, HD: tl.constexpr, DH: tl.constexpr,
                   DN: tl.constexpr, DP: tl.constexpr, D: tl.constexpr, NOBS: tl.constexpr, HW: tl.constexpr,
                   STORE_X: tl.constexpr, X3: tl.constexpr, XH: tl.constexpr, PDL: tl.constexpr):
    """The last coupling (x stored only if STORE_X, for checks), then for each product P containing the tile and each
    attention head h: with y = LN(x) (its affine folded into Wsc and Wv), gates sigmoid(y Wsc + bias) per line, values
    y Wv + bv of the head's channels, and the tile's partial sums sum_s gate v_s -> pa (B, NPAIR, D, C). HW: one head
    at a time (faster at head width 16; also every padded head width: head h's channels are h DN.. h DN + DN - 1 of
    DH >= DN computed); else every head's scores in one product (faster at head width 32)."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    rows, rm, typ = _tile(tiles_ptr, tile, BM)
    pat = tl.load(pid_ptr + b)
    x = _couple(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, pat, m, C, CB, UW,
                BM, X3)
    cols = tl.arange(0, C)
    if STORE_X:
        tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    y = _norm(x, cols, C, CN, heps)
    yh = y.to(tl.float16)
    yl = (y - yh.to(tl.float32)).to(tl.float16)
    jd = tl.arange(0, DP)                                            # line j of a head (padded to DP)
    ch = tl.arange(0, DH)
    for P in tl.range(0, NOBS):
        pr = tl.load(pair_ptr + tile * NOBS + P)
        if pr >= 0:
            if HW:              # one head at a time (loop not unrolled, so one head's weights are live)
                for h in tl.range(0, HD):
                    hj = h * DP + jd                                 # score columns h DP + j
                    s = _mm2(yh, yl, wsc_ptr + P * 2 * C * HD * DP, cols[:, None] * (HD * DP) + hj[None, :],
                             C * HD * DP, XH) + tl.load(bias_ptr + (P * m + rows)[:, None] * (HD * DP) + hj[None, :],
                                                        mask=rm[:, None], other=0.0)
                    gt = tl.trans(tl.where(rm[:, None], 1.0 / (1.0 + tl.exp(-s)), 0.0))   # (DP, BM) gates
                    gh = gt.to(tl.float16)
                    gl = (gt - gh.to(tl.float32)).to(tl.float16)
                    cc = h * DN + ch                                 # the head's value columns (DN < DH: padded)
                    v = _mm2(yh, yl, wv_ptr + P * 2 * C * C, cols[:, None] * C + cc[None, :], C * C, XH) + \
                        tl.load(bv_ptr + P * C + cc)[None, :]
                    vh = v.to(tl.float16)
                    pp = tl.dot(gh, vh)
                    if XH:
                        pp += tl.dot(gl, vh) + tl.dot(gh, (v - vh.to(tl.float32)).to(tl.float16))
                    keep = (jd < D)[:, None]
                    if DN < DH:
                        keep = keep & (ch < DN)[None, :]
                    tl.store(pa_ptr + ((b * NPAIR + pr) * D + jd[:, None]) * C + cc[None, :], pp, mask=keep)
            else:               # every head's scores in one product, then the values one head (DH columns) at a time
                hj = tl.arange(0, HD * DP)
                s = _mm2(yh, yl, wsc_ptr + P * 2 * C * HD * DP, cols[:, None] * (HD * DP) + hj[None, :],
                         C * HD * DP, XH) + tl.load(bias_ptr + (P * m + rows)[:, None] * (HD * DP) + hj[None, :],
                                                    mask=rm[:, None], other=0.0)
                gt = tl.trans(tl.where(rm[:, None], 1.0 / (1.0 + tl.exp(-s)), 0.0))       # (HD DP, BM) gates
                gh = gt.to(tl.float16)
                gl = (gt - gh.to(tl.float32)).to(tl.float16)
                for c0 in tl.static_range(0, C, DH):
                    cc = c0 + ch
                    v = _mm2(yh, yl, wv_ptr + P * 2 * C * C, cols[:, None] * C + cc[None, :], C * C, XH) + \
                        tl.load(bv_ptr + P * C + cc)[None, :]
                    vh = v.to(tl.float16)
                    pp = tl.dot(gh, vh)                              # every head's lines; keep the head's own
                    if XH:
                        pp += tl.dot(gl, vh) + tl.dot(gh, (v - vh.to(tl.float32)).to(tl.float16))
                    kp = ((hj // DP)[:, None] == c0 // DH) & ((hj % DP) < D)[:, None]
                    tl.store(pa_ptr + ((b * NPAIR + pr) * D + (hj % DP)[:, None]) * C + cc[None, :], pp, mask=kp)


@triton.jit
def _conv_up(x_ptr, ub_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, m, NG, GH: tl.constexpr,
             GW: tl.constexpr, KT: tl.constexpr, KI: tl.constexpr, KJ: tl.constexpr, C: tl.constexpr,
             CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, STAGES: tl.constexpr, TPI: tl.constexpr,
             X3: tl.constexpr):
    """x[rows] + Up(SiLU(sum over taps of u[neighbour] Wc[type, tap] + bc[type])) (BM, C) fp32; the neighbour of a tap
    is at a constant offset from the site's grid position; TPI taps per product (weights zero-padded to whole
    iterations)."""
    NTAP: tl.constexpr = KT * KI * KJ
    NIT: tl.constexpr = (NTAP + TPI - 1) // TPI
    KN: tl.constexpr = NIT * TPI * CB * CB
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    qq = tl.arange(0, TPI * CB)
    acc = tl.zeros((BM, CB), dtype=tl.float32)
    base = (b * NG + gp)[:, None] * UW
    wbase = Wc_ptr + typ * 2 * KN
    for it in tl.range(0, NIT, num_stages=STAGES):
        o = it * TPI + qq // CB
        delta = ((o // (KI * KJ) - KT // 2) * GH + (o // KJ % KI - KI // 2)) * GW + (o % KJ - KJ // 2)
        src = ub_ptr + base + (delta * UW + qq % CB)[None, :]
        msk = rm[:, None] & (o < NTAP)[None, :]
        ah = tl.load(src, mask=msk, other=0.0)
        al = ah
        if X3:
            al = tl.load(src + CB, mask=msk, other=0.0)
        acc += _mm2(ah, al, wbase, (it * TPI * CB + qq[:, None]) * CB + cb[None, :], KN, X3)
    v = _silu(acc + tl.load(bc_ptr + typ * CB + cb)[None, :])
    up = _mm(v, Wu_ptr, cb[:, None] * C + cols[None, :], CB * C, X3) + tl.load(bu_ptr + cols)[None, :]
    return tl.load(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + up


@triton.jit
def _k_block(x_ptr, ub_ptr, tiles_ptr, gpos_ptr, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, uc_ptr, lnw, lnb, eps, wd_ptr, m, NG,
             GH: tl.constexpr, GW: tl.constexpr, KT: tl.constexpr, KI: tl.constexpr, KJ: tl.constexpr,
             C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr,
             STAGES: tl.constexpr, TPI: tl.constexpr, X3: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ = _tile(tiles_ptr, tl.program_id(1), BM)
    gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
    x = _conv_up(x_ptr, ub_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, m, NG, GH, GW, KT, KI, KJ, C, CB,
                 UW, BM, STAGES, TPI, X3)
    cols = tl.arange(0, C)
    tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    _uc(x, rows, rm, b, m, lnw, lnb, eps, wd_ptr, uc_ptr, C, CN, CB, UW, X3)


@triton.jit
def _k_readout(pa_ptr, pst_ptr, FIRST_ptr, Wo_ptr, bo_ptr, mw_ptr, mb_ptr, meps, W1_ptr, b1_ptr, W2_ptr, b2_ptr, w_ptr,
               bP_ptr, inv_n0, pk_ptr, pid_ptr, A_ptr, t_ptr, ctr_ptr, out_ptr, dec_ptr, NPAIR, NBY,
               C: tl.constexpr, CN: tl.constexpr, D: tl.constexpr, NOBS: tl.constexpr, NPP: tl.constexpr,
               FF: tl.constexpr, FC: tl.constexpr, NBYP: tl.constexpr, KDP: tl.constexpr, PDL: tl.constexpr):
    """Program (b, P D + j): line j of product P: z from the partial sums (fp32 weights), t = w (-1)^g z with
    g = (D A)_{P D + j} mod 2 from the packed bits; the shot's last program (atomic counter) forms c_P = sum_j t + b_P,
    the confidence min_P |c_P| and the decisions, and resets the counter."""
    if PDL:
        gdc_wait()
    b = tl.program_id(0).to(tl.int64)
    pj = tl.program_id(1)
    P = pj // D
    j = pj % D
    cols = tl.arange(0, C)
    tt = tl.arange(0, NPP)
    p0 = tl.load(pst_ptr + P)
    npr = tl.load(pst_ptr + P + 1) - p0
    u = tl.load(FIRST_ptr + (P * D + j) * C + cols) + tl.load(bo_ptr + P * C + cols)
    for r0 in tl.static_range(0, C, FC):                           # u += a Wo, a = the pooled sums / N0, by rows
        rr = r0 + tl.arange(0, FC)
        msk = (tt < npr)[:, None]
        if CN < C:                                                 # the padded columns of pa are never written
            msk = msk & (rr < CN)[None, :]
        a = tl.sum(tl.load(pa_ptr + ((b * NPAIR + p0 + tt[:, None]) * D + j) * C + rr[None, :],
                           mask=msk, other=0.0), 0) * inv_n0
        u += tl.sum(a[:, None] * tl.load(Wo_ptr + P * C * C + rr[:, None] * C + cols[None, :]), 0)
    mean = tl.sum(u, 0) / CN
    uc = u - mean
    if CN < C:
        uc = tl.where(cols < CN, uc, 0.0)
    y = uc * tl.rsqrt(tl.sum(uc * uc, 0) / CN + meps) * tl.load(mw_ptr + P * C + cols) + tl.load(mb_ptr + P * C + cols)
    z = tl.load(b2_ptr + P)
    for f0 in tl.static_range(0, FF, FC):
        ff = f0 + tl.arange(0, FC)
        h = tl.sum(y[:, None] * tl.load(W1_ptr + P * C * FF + cols[:, None] * FF + ff[None, :]), 0) + \
            tl.load(b1_ptr + P * FF + ff)
        z += tl.sum(_silu(h) * tl.load(W2_ptr + P * FF + ff), 0)
    pat = tl.load(pid_ptr + b)
    by = tl.arange(0, NBYP)
    v = tl.load(pk_ptr + b * NBY + by, mask=by < NBY, other=0).to(tl.int32) & \
        tl.load(A_ptr + (pat * KDP + pj) * NBYP + by).to(tl.int32)
    v = v - ((v >> 1) & 0x55)
    v = (v & 0x33) + ((v >> 2) & 0x33)
    v = (v + (v >> 4)) & 0x0F
    sgn = 1.0 - 2.0 * (tl.sum(v, 0) & 1).to(tl.float32)
    tl.store(t_ptr + b * KDP + pj, tl.load(w_ptr + pj) * sgn * z)
    tl.debug_barrier()                                             # the store precedes the release below
    done = tl.atomic_add(ctr_ptr + b, 1, sem="acq_rel", scope="gpu")
    if done == NOBS * D - 1:
        tl.debug_barrier()
        kd = tl.arange(0, KDP)
        tv = tl.load(t_ptr + b * KDP + kd, mask=kd < NOBS * D, other=0.0)
        conf = 3.0e38
        dec = 0
        for Q in tl.static_range(NOBS):
            cq = tl.sum(tl.where((kd >= Q * D) & (kd < Q * D + D), tv, 0.0), 0) + tl.load(bP_ptr + Q)
            tl.store(out_ptr + b * (NOBS + 1) + Q, cq)
            conf = tl.minimum(conf, tl.abs(cq))
            dec += (cq > 0).to(tl.int32) << Q
        tl.store(out_ptr + b * (NOBS + 1) + NOBS, conf)
        tl.store(dec_ptr + b, dec.to(tl.uint8))
        tl.atomic_xchg(ctr_ptr + b, 0)


# ---------------------------------------------------------------------------------------------------------- engine

def _hilo(w):
    """fp32 (..., K, N) -> fp16 (..., 2, K, N): hi, lo = w - hi."""
    w = torch.as_tensor(w).float()
    hi = w.half()
    return torch.stack([hi, (w - hi.float()).half()], -3)


def _pad(w, *shape):
    """w (fp32) with its last len(shape) dimensions zero-padded at the end to `shape`."""
    w = torch.as_tensor(w).float()
    return F.pad(w, [q for n, k in zip(reversed(shape), reversed(w.shape[w.dim() - len(shape):])) for q in (0, n - k)])


MODES = {"precise": True, "onepass": False}
FC = 32                                                                          # read-out rows per step


class Engine:
    """Engine of deploy tables t. __call__(pk (B, NBY) uint8 on the GPU, pid (B,) int32 pattern indices or None for
    pattern 0) -> (out (B, n_obs + 1) fp32: c_P then the confidence, dec (B,) uint8: bit P = c_P > 0). mode "precise"
    runs every matrix product as three fp16 products of split operands (about fp32 accuracy), "onepass" one fp16
    product; precise=True / False is the older spelling of the two, and a string of kernel letters (s stem,
    c couplings, b blocks, h head) runs those kernels in three passes and the rest in one. BM sites per tile.
    Buffers are kept per (batch size, slot); a call passes the batch through the kernels in chunks of `chunk` shots
    (all if None)."""

    def __init__(self, t, device="cuda", mode="precise", BM=64, warps=4, warps_ro=4, stages=3, tpi=2, chunk=None,
                 pdl=False, precise=None):
        dev = torch.device(device)
        self.dev, self.t = dev, t
        f32, i32 = torch.float32, torch.int32
        h = lambda v: _hilo(v).to(dev).contiguous()
        f = lambda v: torch.as_tensor(v).to(dev, f32).contiguous()
        n = lambda v: torch.as_tensor(np.asarray(v)).to(dev, i32).contiguous()
        if precise is None:
            precise = MODES[mode]
        mode = {True: "precise", False: "onepass"}.get(precise, precise)
        self.mode = mode
        self.cfg = dict(BM=BM, mode=mode, warps=warps, warps_ro=warps_ro, stages=stages, tpi=tpi, chunk=chunk,
                        pdl=pdl)
        prec = {True: "scbh", False: ""}.get(precise, precise)                  # three-pass kernels: stem,
        self.X = {k: k in prec for k in "scbh"}                                  # couplings, blocks, head
        self.X3 = self.X["c"] or self.X["b"]
        self.d, self.m, self.nobs, self.L = (t[k] for k in ("d", "m", "nobs", "L"))
        self.CN, CBN = t["C"], t["Cb"]                         # true widths; the kernels run padded ones (C, CB)
        self.C, self.CB = C, CB = _p2(self.CN), max(16, _p2(CBN))
        assert C >= FC, "the read-out takes FC rows per step"
        self.UW = 2 * self.CB if self.X3 else self.CB                             # stored u row: hi (and lo)
        self.NBY, self.NBYP, self.KSP, self.DP = t["NBY"], t["NBYP"], t["KSP"], t["DP"]
        tiles = [(r, min(BM, r1 - r), b, ty) for r0, r1, b, ty in t["groups"] for r in range(r0, r1, BM)]
        pair_of, pst = np.full((len(tiles), self.nobs), -1, np.int32), [0]          # product k: pairs pst[k]..
        for k, blocks in enumerate(t["prod_blocks"]):
            for ti, tile in enumerate(tiles):
                if tile[2] in blocks:
                    pair_of[ti, k] = pst[-1] + (pair_of[:, k] >= 0).sum()
            pst.append(pst[-1] + int((pair_of[:, k] >= 0).sum()))
        self.NT = len(tiles)
        self.tiles, self.pair, self.pst = n(tiles), n(pair_of), n(pst)
        self.NPAIR = pst[-1]
        self.NPP = _p2(int(np.diff(pst).max()))
        self.nbd, self.Wb, self.P = n(t["nbd"]), h(_pad(t["Wb"], self.KSP, C)), f(_pad(t["P"], C))
        self.role, self.pc, self.gpos = n(t["role"]), n(t["pc"]), n(t["gpos"])
        self.npat = len(t["role"])
        KD = self.nobs * self.d
        self.KDP = _p2(KD)
        A = np.zeros((self.npat, self.KDP, self.NBYP), np.uint8)
        A[:, :KD] = t["Apk"]
        self.A = torch.as_tensor(A).to(dev).contiguous()
        self.cp = [dict(ln_w=f(_pad(c["ln_w"], C)), ln_b=f(_pad(c["ln_b"], C)), eps=float(c["eps"]),
                        Wd=h(_pad(c["Wd"], C, CB)), c0=h(_pad(c["c0"], CB)[None])[..., 0, :].contiguous(),
                        M=h(_pad(c["M"].reshape(2, 8, CBN, CBN), CB, CB).reshape(2, 8 * CB, CB)),
                        Wu=h(_pad(c["Wu"], CB, C)), bu=f(_pad(c["bu"], C))) for c in t["couples"]]
        self.KT, self.KI, self.KJ = t["block_k"]
        GT, self.GH, self.GW = t["grid"]
        self.NG = 2 * GT * self.GH * self.GW
        ntap = self.KT * self.KI * self.KJ
        pad = (-ntap) % tpi
        self.blk = []
        for p in t["blocks"]:
            Wc = _pad(p["Wc"].reshape(2, ntap, CBN, CBN), CB, CB)
            Wc = torch.cat([Wc, Wc.new_zeros(2, pad, CB, CB)], 1).reshape(2, -1, CB)
            self.blk.append(dict(ln_w=f(_pad(p["ln_w"], C)), ln_b=f(_pad(p["ln_b"], C)), eps=float(p["eps"]),
                                 Wd=h(_pad(p["Wd"], C, CB)), bd=f(_pad(p["bd"], CB)), Wc=h(Wc),
                                 bc=f(_pad(p["bc"], CB)), Wu=h(_pad(p["Wu"], CB, C)), bu=f(_pad(p["bu"], C))))
        w = torch.zeros(self.KDP)
        w[:KD] = t["w"]
        self.w, self.bP = f(w), f(t["b"])
        self._init_head(t, f, h)
        self._bufs = {}
        self._pid0 = {}
        self.store_x = False                                       # True: the last coupling stores x (checks)

    def _init_head(self, t, f, h):
        """The line head's and read-out's tables (engine_ro.EngineRO has its own)."""
        C = self.C
        hs = t["heads"]
        st = lambda k: torch.stack([torch.as_tensor(x[k]).float() for x in hs])
        assert len({float(x["eps"]) for x in hs}) == 1 and len({float(x["m_eps"]) for x in hs}) == 1
        self.HD, self.DN = t["Hh"], t["dh"]                                       # heads, true head width
        self.DH = max(16, _p2(self.DN))
        assert (self.HD - 1) * self.DN + self.DH <= C, "a padded head's value columns run past the padded width"
        FF = st("W1").shape[-1]
        self.FF = FFP = -(-FF // FC) * FC
        g = lambda k: torch.stack([torch.as_tensor(x[k]).double() for x in hs])
        lw, lb, Wsc, Wv = g("ln_w")[:, :, None], g("ln_b")[:, None], g("Wsc"), g("Wv")      # LN affine folded:
        self.hd = dict(eps=float(hs[0]["eps"]), Wsc=h(_pad((lw * Wsc).float(), C, Wsc.shape[-1])),
                       bias=f(g("bias") + lb @ Wsc), Wv=h(_pad((lw * Wv).float(), C, C)),
                       bv=f(_pad((g("bv") + (lb @ Wv)[:, 0]).float(), C)), Uq=f(_pad(st("Uq"), C)),
                       Wo=f(_pad(st("Wo"), C, C)), bo=f(_pad(st("bo"), C)), mw=f(_pad(st("m_ln_w"), C)),
                       mb=f(_pad(st("m_ln_b"), C)), meps=float(hs[0]["m_eps"]), W1=f(_pad(st("W1"), C, FFP)),
                       b1=f(_pad(st("b1"), FFP)), W2=f(_pad(st("W2"), FFP)), b2=f(st("b2")))
        self.inv_n0 = 1.0 / float(t["n0"])

    def buffers(self, B, slot=0):
        if (B, slot) not in self._bufs:
            dev, m, f16 = self.dev, self.m, torch.float16
            self._bufs[B, slot] = dict(
                x=torch.empty(B, m, self.C, device=dev), uc=torch.empty(B, m, self.UW, device=dev, dtype=f16),
                ub=torch.zeros(B, self.NG, self.UW, device=dev, dtype=f16), **self._head_buffers(B))
        return self._bufs[B, slot]

    def _head_buffers(self, B):
        dev = self.dev
        return dict(pa=torch.empty(B, self.NPAIR, self.d, self.C, device=dev), tz=torch.empty(B, self.KDP, device=dev),
                    ctr=torch.zeros(B, device=dev, dtype=torch.int32))

    def launches(self, pk, pid, out, dec, slot=0):
        """[(name, launch)] of one pass of the kernels over the shots of pk (one chunk)."""
        B = pk.shape[0]
        bf = self.buffers(B, slot)
        c = self.cfg
        BM, nw, X = c["BM"], c["warps"], self.X
        grid = (B, self.NT)
        cp0 = self.cp[0]
        pdl = c["pdl"]
        sz = dict(C=self.C, CN=self.CN, CB=self.CB, UW=self.UW, BM=BM, num_warps=nw, PDL=pdl,
                  **({"launch_pdl": True} if pdl else {}))
        steps = [("stem", lambda: _k_stem[grid](
            pk, pid, self.tiles, self.nbd, self.P, self.Wb, bf["x"], bf["uc"], cp0["ln_w"], cp0["ln_b"], cp0["eps"],
            cp0["Wd"], self.m, self.NBY, KSP=self.KSP, K1=self.t["K1"], K2=self.t["K2"], NBYP=self.NBYP, X3=X["s"],
            **(dict(sz, launch_pdl=False) if pdl else sz)))]
        common = lambda k: (bf["x"], bf["uc"], pid, self.tiles, self.pc, self.role, self.cp[k]["c0"], self.cp[k]["M"],
                            self.cp[k]["Wu"], self.cp[k]["bu"])
        for k in range(self.L):
            p = self.blk[k]
            steps.append((f"couple{k}", lambda k=k, p=p: _k_couple_block[grid](
                *common(k), self.gpos, bf["ub"], p["ln_w"], p["ln_b"], p["eps"], p["Wd"], p["bd"], self.m,
                self.NG, X3=X["c"], **sz)))
            q = self.cp[k + 1]
            steps.append((f"block{k}", lambda p=p, q=q: _k_block[grid](
                bf["x"], bf["ub"], self.tiles, self.gpos, p["Wc"], p["bc"], p["Wu"], p["bu"], bf["uc"], q["ln_w"],
                q["ln_b"], q["eps"], q["Wd"], self.m, self.NG, GH=self.GH, GW=self.GW, KT=self.KT, KI=self.KI,
                KJ=self.KJ, STAGES=c["stages"], TPI=c["tpi"], X3=X["b"], **sz)))
        return steps + self._head_steps(pk, pid, out, dec, bf, grid, sz, common(self.L))

    def _head_steps(self, pk, pid, out, dec, bf, grid, sz, common):
        """The last coupling with the line head, then the read-out (engine_ro.EngineRO has its own); common: the last
        coupling's arguments."""
        B, c, X, hd, pdl = pk.shape[0], self.cfg, self.X, self.hd, self.cfg["pdl"]
        return [(f"couple{self.L}_head", lambda: _k_couple_head[grid](
                    *common, self.pair, hd["eps"], hd["Wsc"], hd["bias"], hd["Wv"], hd["bv"],
                    bf["pa"], self.m, self.NPAIR, HD=self.HD, DH=self.DH, DN=self.DN, DP=self.DP, D=self.d,
                    NOBS=self.nobs, HW=self.DH < 32 or self.DN < self.DH, STORE_X=self.store_x, X3=X["c"], XH=X["h"],
                    **sz)),
                ("readout", lambda: _k_readout[(B, self.nobs * self.d)](
                    bf["pa"], self.pst, hd["Uq"], hd["Wo"], hd["bo"], hd["mw"], hd["mb"], hd["meps"], hd["W1"],
                    hd["b1"], hd["W2"], hd["b2"], self.w, self.bP, self.inv_n0, pk, pid, self.A, bf["tz"], bf["ctr"],
                    out, dec, self.NPAIR, self.NBY, C=self.C, CN=self.CN, D=self.d, NOBS=self.nobs, NPP=self.NPP,
                    FF=self.FF, FC=FC, NBYP=self.NBYP, KDP=self.KDP, num_warps=c["warps_ro"], PDL=pdl,
                    **({"launch_pdl": True} if pdl else {})))]

    def pid0(self, B):
        if B not in self._pid0:
            self._pid0[B] = torch.zeros(B, device=self.dev, dtype=torch.int32)
        return self._pid0[B]

    def plan(self, pk, pid, out, dec, slot=0):
        """All launches of a call: the batch in chunks of cfg['chunk'] shots (all of it if None)."""
        B = pk.shape[0]
        ch = self.cfg["chunk"] or B
        steps = []
        for s in range(0, B, ch):
            e = min(B, s + ch)
            steps += self.launches(pk[s:e], pid[s:e], out[s:e], dec[s:e], slot)
        return steps

    def __call__(self, pk, pid=None, out=None, dec=None, slot=0):
        B = pk.shape[0]
        pid = self.pid0(B) if pid is None else pid
        out = torch.empty(B, self.nobs + 1, device=self.dev) if out is None else out
        dec = torch.empty(B, device=self.dev, dtype=torch.uint8) if dec is None else dec
        for _, fn in self.plan(pk, pid, out, dec, slot):
            fn()
        return out, dec


def engine(net, specs, mode="precise", chunk=None, latency=False, device="cuda"):
    """The engine of a trained Net for the CNOT patterns of `specs` (geometry.analyse of each pattern's circuit; a
    shot's pattern index is its position in specs). mode "precise" or "onepass";
    chunk: shots per pass through the kernels (None or 0: the whole batch); latency=True: the batch-one settings
    (tiles of 16 sites, programmatic dependent launch)."""
    tabs = deploy(net, specs)
    if latency:
        return Engine(tabs, device, mode, BM=16, tpi=2, pdl=True, stages=3)
    return Engine(tabs, device, mode, chunk=chunk or None)


@torch.no_grad()
def predict(eng, det, pid=None, batch=1024):
    """Every shot of det (N, ceil(m / 8)) uint8, stim's bit packing, with pattern indices pid (N,) (None: pattern 0),
    both numpy or tensors on any device, in batches of `batch` -> numpy."""
    N = len(det)
    pk = torch.as_tensor(det).to(eng.dev).contiguous()
    pid = eng.pid0(N) if pid is None else torch.as_tensor(pid).to(eng.dev, torch.int32).contiguous()
    out = torch.empty(N, eng.nobs + 1, device=eng.dev)
    dec = torch.empty(N, dtype=torch.uint8, device=eng.dev)
    for i in range(0, N, batch):
        eng(pk[i:i + batch], pid[i:i + batch], out[i:i + batch], dec[i:i + batch])
    out, dec = out.cpu().numpy(), dec.cpu().numpy()
    return out[:, :eng.nobs], out[:, eng.nobs], ((dec[:, None] >> np.arange(eng.nobs)) & 1).astype(bool)
