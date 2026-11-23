"""The inference engine (duet.net.engine) with the per-layer read-out of the earlier study (duet.net.model_ro.NetRO, ro_b's
read-out): the same trunk kernels (stem, couplings, residual blocks over the detector sites, every matrix product in
three fp16 passes in mode "precise"), then three kernels in place of the line head and its read-out:

  couple L + head the last coupling; then per product P and attention head h, over the tile's sites (both blocks
                   feed every product): gates sigmoid(y Wsc + bias) per line j, with the bias of the site's
                   (type, offset bin to line j) if its block is in r_P(t) at the site's layer t, else the
                   outside-the-support bias (model_ro.LayerLines; r_P(t) is the pattern's labels_ro.supports);
                   values y Wv + bv; the gates of each layer of the tile in its own rows of the gate matrix, so one
                   product gives the tile's partial sums per (layer, line) -> pa (B, n_obs, pairs, d, C)
  read-out program (shot, P, block of RB rows (line j, layer t)): a = the partial sums of layer t (its tile
                   pairs, a fixed order) / N0; u = a Wo + bo + query_j + emb[r_P(t)]; l_{P,j,t} = MLP(LN(u)); the
                   three matrix products on tensor cores (three passes in mode "precise")
  soft XOR program per shot: z_{P,j} = the soft XOR over t of l_{P,j,t} in fp32 and in log space, the same
                   formula as model_ro.soft_xor (clamps included, libdevice's exp, expm1, log, log1p);
                   g = (D A)_{P,j} mod 2 from the packed bits; c_P = sum_j w (-1)^g z + b_P, the confidence

One engine per depth L (the trunk's grid and the head's sites depend on it); one batch may mix the depth's CNOT
patterns (per-pattern tables: site table, gate roles, A, r_P(t)). Widths must be powers of two here (64 and 128).

    eng = engine(net, specs, L, mode="precise" | "onepass", chunk=None, latency=False)
    c, conf, dec = engine.predict(eng, det, pid) # duet.net.engine.predict: a set in host memory -> numpy
    out, dec = eng(pk, pid, out, dec) # one batch on the GPU (fixed buffers, CUDA-graph capturable)
    c, l = reference(net, specs, L, det, pid) # eager fp32 (TF32 off): logits and per-layer logits"""
import copy

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from duet.net import engine as E, labels_ro as LR, model_ro as MR
from duet.net.engine import _couple, _ln, _mm, _mm2, _norm, _p2, _silu, _tile

FCR = 64            # read-out MLP columns per step


@torch.no_grad()
def deploy(net, specs, L):
    """Inference tables of a NetRO at depth L for the CNOT patterns of specs (geometry.analyse of each pattern's
    circuit of depth L; one geometry; index k of specs is pattern index k at run time): engine.trunk_tables on the
    depth's grid, then the per-layer head's. The head's sites, types and offset bins are the network's own (its
    geometry of depth L), as in the eager forward."""
    net = copy.deepcopy(net).float().cpu().eval()
    geo = net.geo[str(L)]
    t = E._detached(E.trunk_tables(net, specs, geo))
    s0 = specs[0]
    d, m, nobs, T, H, W, C = (t[k] for k in ("d", "m", "nobs", "T", "H", "W", "C"))
    assert T == geo.T == L + 1 and m == geo.m, "the specs are not of the network's depth L"
    blk, tij = s0["block"], s0["tij"]
    flat = ((blk * T + tij[:, 0]) * H + tij[:, 1]) * W + tij[:, 2]
    assert np.array_equal(geo.gidx.numpy()[flat], np.arange(m)), "the network's grid of depth L differs from the specs'"
    order = t["order"]
    eb, et, ety = blk[order], tij[order], s0["type"][order]
    cell = {int(v): s for s, v in enumerate(geo.hw.tolist())}
    cs = np.array([cell[int(i) * W + int(j)] for i, j in et[:, 1:]])                  # each engine row's site cell
    ix = tuple(torch.as_tensor(v) for v in (eb, et[:, 0], cs))
    assert geo.on[ix].bool().all() and int(geo.on.sum()) == m, "the head's sites are not the detectors"
    assert np.array_equal(geo.ty[ix].numpy(), ety)
    ety_t = torch.as_tensor(ety)
    DP = max(16, _p2(d))
    heads = []
    for k, hd in enumerate(net.heads):
        Hh, dh, R = hd.h, hd.dh, hd.R
        off = getattr(geo, f"off{k}")[torch.as_tensor(cs)]                               # (m, d) offset bins
        b_in = hd.b_line[:, ety_t[:, None], off]                                         # (H, m, d), fp32
        b_out = hd.b_line[:, ety_t, 2 * R + 1][..., None]                                # (H, m, 1)
        sel = lambda ins: (b_out + ins * (b_in - b_out)).permute(1, 0, 2)               # eager's expression, fp32
        bias = torch.zeros(2, m, Hh, DP)
        bias[0, :, :, :d], bias[1, :, :, :d] = sel(0.0), sel(1.0)                        # outside / inside r_P(t)
        h64 = copy.deepcopy(hd).double()
        q = h64.wq(h64.queries()).view(d, Hh, dh) * dh ** -0.5
        Wsc = torch.zeros(C, Hh, DP, dtype=torch.float64)
        for h in range(Hh):
            Wsc[:, h, :d] = h64.k.weight[h * dh:(h + 1) * dh].t() @ q[:, h].t()
        heads.append(_kept(dict(ln_w=h64.ln.weight, ln_b=h64.ln.bias, eps=hd.ln.eps, Wsc=Wsc.reshape(C, Hh * DP),
                          bias=bias.reshape(2, m, Hh * DP), Wv=h64.v.weight.t(), bv=h64.v.bias,
                          Uq=h64.queries() + h64.o.bias, emb=hd.emb, Wo=h64.o.weight.t(), m_ln_w=hd.mlp[0].weight,
                          m_ln_b=hd.mlp[0].bias, m_eps=hd.mlp[0].eps, W1=hd.mlp[1].weight.t(), b1=hd.mlp[1].bias,
                          W2=hd.mlp[3].weight.reshape(-1), b2=hd.mlp[3].bias.reshape(()))))
    sup = np.stack([LR.supports(s) for s in specs]).astype(np.int32)                    # (patterns, n_obs, T, 2)
    t.update(heads=heads, DP=DP, Hh=net.heads[0].h, dh=net.heads[0].dh, n0=MR.N0, tr=et[:, 0].astype(np.int32),
             sup=sup, depth=L)
    return t


def _kept(x):
    """The head's tables detached, in their own precision (the folded ones float64 until EngineRO folds them)."""
    return {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in x.items()}


# --------------------------------------------------------------------------------------------------------- kernels

@triton.jit
def _k_couple_head_ro(x_ptr, uc_ptr, pid_ptr, tiles_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, tr_ptr,
                      tq_ptr, sup_ptr, heps, wsc_ptr, bias_ptr, wv_ptr, bv_ptr, pa_ptr, m, T, NQ, C: tl.constexpr,
                      CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, HD: tl.constexpr,
                      DH: tl.constexpr, DN: tl.constexpr, DP: tl.constexpr, D: tl.constexpr, NOBS: tl.constexpr,
                      NLP: tl.constexpr, STORE_X: tl.constexpr, X3: tl.constexpr, XH: tl.constexpr,
                      PDL: tl.constexpr):
    """The last coupling (x stored only if STORE_X), then per product P and head h: y = LN(x) (affine folded into
    Wsc, bias, Wv, bv), gates sigmoid(y Wsc + bias) with bias[inside r_P(t)] per row, values y Wv + bv of the head's
    channels; gate rows (slot l, line j) keep the tile's rows of layer t0 + l, so G (NLP DP, BM) v gives the tile's
    partial sums per layer -> pa (B, NOBS, NQ, D, C) at pair tq[tile, l] (none: -1)."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    rows, rm, typ = _tile(tiles_ptr, tile, BM)
    blk = tl.load(tiles_ptr + tile * 4 + 2)
    pat = tl.load(pid_ptr + b)
    x = _couple(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, pat, m, C, CB, UW,
                BM, X3)
    cols = tl.arange(0, C)
    if STORE_X:
        tl.store(x_ptr + (b * m + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
    y = _norm(x, cols, C, CN, heps)
    yh = y.to(tl.float16)
    yl = (y - yh.to(tl.float32)).to(tl.float16)
    tr = tl.load(tr_ptr + rows, mask=rm, other=0)                      # each row's layer
    t0 = tl.load(tr_ptr + tl.load(tiles_ptr + tile * 4))
    slot = tl.where(rm, tr - t0, -1)
    sel = slot[None, :] == tl.arange(0, NLP)[:, None]                 # (NLP, BM): row in layer t0 + l
    jd = tl.arange(0, DP)
    ch = tl.arange(0, DH)
    rr = tl.arange(0, NLP * DP)
    jj = rr % DP
    q = tl.load(tq_ptr + tile * NLP + rr // DP)                        # pair of each gate row's layer
    keep = ((q >= 0) & (jj < D))[:, None]
    if DN < DH:
        keep = keep & (ch < DN)[None, :]
    for P in tl.range(0, NOBS, num_stages=1):                         # unpipelined: shared memory at width 128
        ins = tl.load(sup_ptr + ((pat * NOBS + P) * T + tr) * 2 + blk, mask=rm, other=0)
        for h in tl.range(0, HD, num_stages=1):
            hj = h * DP + jd
            boff = (P * 2 * m + rows)[:, None] * (HD * DP) + hj[None, :]
            bias = tl.where((ins != 0)[:, None], tl.load(bias_ptr + boff + m * HD * DP, mask=rm[:, None], other=0.0),
                            tl.load(bias_ptr + boff, mask=rm[:, None], other=0.0))
            s = _mm2(yh, yl, wsc_ptr + P * 2 * C * HD * DP, cols[:, None] * (HD * DP) + hj[None, :], C * HD * DP,
                     XH) + bias
            gt = tl.trans(tl.where(rm[:, None], 1.0 / (1.0 + tl.exp(-s)), 0.0))                    # (DP, BM)
            g = tl.reshape(tl.where(sel[:, None, :], gt[None, :, :], 0.0), (NLP * DP, BM))
            gh = g.to(tl.float16)
            gl = (g - gh.to(tl.float32)).to(tl.float16)
            cc = h * DN + ch
            v = _mm2(yh, yl, wv_ptr + P * 2 * C * C, cols[:, None] * C + cc[None, :], C * C, XH) + \
                tl.load(bv_ptr + P * C + cc)[None, :]
            vh = v.to(tl.float16)
            pp = tl.dot(gh, vh)
            if XH:
                pp += tl.dot(gl, vh) + tl.dot(gh, (v - vh.to(tl.float32)).to(tl.float16))
            tl.store(pa_ptr + (((b * NOBS + P) * NQ + q) * D + jj)[:, None] * C + cc[None, :], pp, mask=keep)


@triton.jit
def _k_readout_ro(pa_ptr, lq_ptr, pid_ptr, sup_ptr, FIRST_ptr, emb_ptr, Wo_ptr, mw_ptr, mb_ptr, meps, W1_ptr, b1_ptr,
                  W2_ptr, b2_ptr, inv_n0, l_ptr, T, NQ, NQT, C: tl.constexpr, D: tl.constexpr, NOBS: tl.constexpr,
                  RB: tl.constexpr, FF: tl.constexpr, FC: tl.constexpr, X3: tl.constexpr, PDL: tl.constexpr):
    """Program (b, P, rows r = j T + t of block RB): a = sum of layer t's pairs lq[t:] of pa / N0 (in pair order);
    l (B, NOBS, D, T)."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    b = tl.program_id(0).to(tl.int64)
    P = tl.program_id(1)
    r = tl.program_id(2) * RB + tl.arange(0, RB)
    rm = r < D * T
    j = tl.where(rm, r // T, 0)
    t = tl.where(rm, r % T, 0)
    cols = tl.arange(0, C)
    a = tl.zeros((RB, C), dtype=tl.float32)
    for k in tl.range(0, NQT):
        q = tl.load(lq_ptr + t * NQT + k, mask=rm, other=-1)
        a += tl.load(pa_ptr + (((b * NOBS + P) * NQ + q) * D + j)[:, None] * C + cols[None, :],
                     mask=(q >= 0)[:, None], other=0.0)
    a = a * inv_n0
    pat = tl.load(pid_ptr + b)
    sb = sup_ptr + ((pat * NOBS + P) * T + t) * 2
    code = tl.load(sb, mask=rm, other=0) + 2 * tl.load(sb + 1, mask=rm, other=0)
    u = _mm(a, Wo_ptr + P * 2 * C * C, cols[:, None] * C + cols[None, :], C * C, X3) + \
        tl.load(FIRST_ptr + (P * D + j)[:, None] * C + cols[None, :]) + \
        tl.load(emb_ptr + (P * 4 + code)[:, None] * C + cols[None, :])
    y = _ln(u, mw_ptr + P * C, mb_ptr + P * C, cols, C, C, meps)
    yh = y.to(tl.float16)
    yl = (y - yh.to(tl.float32)).to(tl.float16)
    z = tl.zeros((RB,), dtype=tl.float32) + tl.load(b2_ptr + P)
    for f0 in tl.static_range(0, FF, FC):
        ff = f0 + tl.arange(0, FC)
        hc = _mm2(yh, yl, W1_ptr + P * 2 * C * FF, cols[:, None] * FF + ff[None, :], C * FF, X3) + \
            tl.load(b1_ptr + P * FF + ff)[None, :]
        z += tl.sum(_silu(hc) * tl.load(W2_ptr + P * FF + ff)[None, :], 1)
    tl.store(l_ptr + ((b * NOBS + P) * D + j) * T + t, z, mask=rm)


@triton.jit
def _log1mexp(a):
    """log(1 - exp(-a)), a > 0: model_ro._log1mexp."""
    lo = tl.minimum(a, 0.6931)
    hi = tl.maximum(a, 0.6931)
    return tl.where(a < 0.6931, libdevice.log(-libdevice.expm1(-lo)), libdevice.log1p(-libdevice.exp(-hi)))


@triton.jit
def _k_xor_ro(l_ptr, pk_ptr, pid_ptr, A_ptr, w_ptr, bP_ptr, out_ptr, dec_ptr, T, NBY, D: tl.constexpr,
              NOBS: tl.constexpr, KDP: tl.constexpr, TP: tl.constexpr, NBYP: tl.constexpr, PDL: tl.constexpr):
    """Program b: z = model_ro.soft_xor over t of l (B, NOBS D, T) per row P D + j; g = (D A)_{P D + j} mod 2 from the
    packed bits; c_P = sum_j w (-1)^g z + b_P -> out (B, NOBS + 1) (c, then the confidence min_P |c_P|), dec (bit P =
    c_P > 0)."""
    if PDL:
        gdc_wait()
    b = tl.program_id(0).to(tl.int64)
    kd = tl.arange(0, KDP)
    tt = tl.arange(0, TP)
    km = kd < NOBS * D
    msk = km[:, None] & (tt < T)[None, :]
    lv = tl.load(l_ptr + (b * NOBS * D + kd)[:, None] * T + tt[None, :], mask=msk, other=0.0)
    a = tl.minimum(tl.maximum(tl.abs(lv), 1e-6), 80.0)
    S = tl.sum(tl.where(msk, _log1mexp(a) - libdevice.log1p(libdevice.exp(-a)), 0.0), 1)
    S = tl.minimum(S, -1e-37)
    mag = libdevice.log1p(libdevice.exp(S)) - _log1mexp(-S)
    odd = (tl.sum(tl.where(msk & (lv > 0), 1, 0), 1) % 2) == 1
    z = tl.where(odd, mag, -mag)
    pat = tl.load(pid_ptr + b)
    by = tl.arange(0, NBYP)
    v = tl.load(pk_ptr + b * NBY + by, mask=by < NBY, other=0).to(tl.int32)[None, :] & \
        tl.load(A_ptr + (pat * KDP + kd)[:, None] * NBYP + by[None, :]).to(tl.int32)
    v = v - ((v >> 1) & 0x55)
    v = (v & 0x33) + ((v >> 2) & 0x33)
    v = (v + (v >> 4)) & 0x0F
    sgn = 1.0 - 2.0 * (tl.sum(v, 1) & 1).to(tl.float32)
    tv = tl.where(km, tl.load(w_ptr + kd) * sgn * z, 0.0)
    conf = 3.0e38
    dec = 0
    for Q in tl.static_range(NOBS):
        cq = tl.sum(tl.where((kd >= Q * D) & (kd < Q * D + D), tv, 0.0), 0) + tl.load(bP_ptr + Q)
        tl.store(out_ptr + b * (NOBS + 1) + Q, cq)
        conf = tl.minimum(conf, tl.abs(cq))
        dec += (cq > 0).to(tl.int32) << Q
    tl.store(out_ptr + b * (NOBS + 1) + NOBS, conf)
    tl.store(dec_ptr + b, dec.to(tl.uint8))


# ---------------------------------------------------------------------------------------------------------- engine

class EngineRO(E.Engine):
    """engine.Engine with the per-layer head (deploy tables of this module). RB: read-out rows (line, layer) per
    program. After a call, buffers(chunk)["l"] holds the last chunk's per-layer logits (B, n_obs, d, T)."""

    def __init__(self, t, device="cuda", mode="precise", BM=64, warps=4, warps_ro=4, stages=3, tpi=2, chunk=None,
                 pdl=False, RB=32):
        self.RB = RB
        super().__init__(t, device, mode, BM, warps, warps_ro, stages, tpi, chunk, pdl)

    def _init_head(self, t, f, h):
        dev, C, m, d, nobs = self.dev, self.C, self.m, self.d, self.nobs
        assert C == self.CN, "the per-layer head runs power-of-two widths only"
        self.T = T = t["T"]
        self.depth = t["depth"]
        BM = self.cfg["BM"]
        tr = np.asarray(t["tr"])
        tiles = [(r, min(BM, r1 - r)) for r0, r1, _, _ in t["groups"] for r in range(r0, r1, BM)]
        lay = [sorted(set(tr[r:r + n].tolist())) for r, n in tiles]                     # layers of each tile
        self.NLP = NLP = _p2(max(ls[-1] - ls[0] + 1 for ls in lay))
        tq, pairs = np.full((len(tiles), NLP), -1, np.int32), []
        for i, ls in enumerate(lay):
            for tt in ls:
                tq[i, tt - ls[0]] = len(pairs)
                pairs.append(tt)
        self.NQ = len(pairs)
        per = [[q for q, tt in enumerate(pairs) if tt == u] for u in range(T)]
        self.NQT = max(len(v) for v in per)
        lq = np.full((T, self.NQT), -1, np.int32)
        for u, v in enumerate(per):
            lq[u, :len(v)] = v
        i32 = lambda v: torch.as_tensor(np.asarray(v)).to(dev, torch.int32).contiguous()
        self.tr, self.tq, self.lq, self.sup = i32(tr), i32(tq), i32(lq), i32(t["sup"])
        assert self.sup.shape == (self.npat, nobs, T, 2)
        hs = t["heads"]
        self.HD, self.DN, self.DP = t["Hh"], t["dh"], t["DP"]
        self.DH = max(16, _p2(self.DN))
        assert self.HD * self.DN == C and self.DH == self.DN, "heads of power-of-two width"
        g = lambda k: torch.stack([torch.as_tensor(x[k]).double() for x in hs])
        st = lambda k: torch.stack([torch.as_tensor(x[k]).float() for x in hs])
        assert len({float(x["eps"]) for x in hs}) == 1 and len({float(x["m_eps"]) for x in hs}) == 1
        self.FF = st("W1").shape[-1]
        assert self.FF % FCR == 0
        lw, lb, Wsc, Wv = g("ln_w")[:, :, None], g("ln_b")[:, None], g("Wsc"), g("Wv")      # LN affine folded:
        self.hd = dict(eps=float(hs[0]["eps"]), Wsc=h((lw * Wsc).float()),
                       bias=f(g("bias") + (lb @ Wsc)[:, None]), Wv=h((lw * Wv).float()),
                       bv=f((g("bv") + (lb @ Wv)[:, 0]).float()), Uq=f(g("Uq").float()), emb=f(st("emb")),
                       Wo=h(g("Wo").float()), mw=f(st("m_ln_w")), mb=f(st("m_ln_b")), meps=float(hs[0]["m_eps"]),
                       W1=h(st("W1")), b1=f(st("b1")), W2=f(st("W2")), b2=f(st("b2")))
        self.inv_n0 = 1.0 / float(t["n0"])
        self.NRB = -(-d * T // self.RB)
        self.TP = _p2(T)

    def _head_buffers(self, B):
        dev = self.dev
        return dict(pa=torch.empty(B, self.nobs, self.NQ, self.d, self.C, device=dev),
                    l=torch.empty(B, self.nobs, self.d, self.T, device=dev))

    def _head_steps(self, pk, pid, out, dec, bf, grid, sz, common):
        B, c, X, hd, pdl = pk.shape[0], self.cfg, self.X, self.hd, self.cfg["pdl"]
        lp = {"launch_pdl": True} if pdl else {}
        return [(f"couple{self.L}_head", lambda: _k_couple_head_ro[grid](
                    *common, self.tr, self.tq, self.sup, hd["eps"], hd["Wsc"], hd["bias"], hd["Wv"], hd["bv"],
                    bf["pa"], self.m, self.T, self.NQ, HD=self.HD, DH=self.DH, DN=self.DN, DP=self.DP, D=self.d,
                    NOBS=self.nobs, NLP=self.NLP, STORE_X=self.store_x, X3=X["c"], XH=X["h"], **sz)),
                ("readout", lambda: _k_readout_ro[(B, self.nobs, self.NRB)](
                    bf["pa"], self.lq, pid, self.sup, hd["Uq"], hd["emb"], hd["Wo"], hd["mw"], hd["mb"], hd["meps"],
                    hd["W1"], hd["b1"], hd["W2"], hd["b2"], self.inv_n0, bf["l"], self.T, self.NQ, self.NQT,
                    C=self.C, D=self.d, NOBS=self.nobs, RB=self.RB, FF=self.FF, FC=FCR, X3=X["h"],
                    num_warps=c["warps_ro"], PDL=pdl, **lp)),
                ("xor", lambda: _k_xor_ro[(B,)](
                    bf["l"], pk, pid, self.A, self.w, self.bP, out, dec, self.T, self.NBY, D=self.d, NOBS=self.nobs,
                    KDP=self.KDP, TP=self.TP, NBYP=self.NBYP, num_warps=4, PDL=pdl, **lp))]


NLP_MAX = 8     # layer slots of the head kernel (a tile's layer span, rounded up to a power of two): at 16 (d = 3,
                # L >= 8, tiles of 64 rows) it needs 106,496 bytes of shared memory, above an RTX PRO 6000 block's
                # 101,376; 8 slots compile and match eager (d = 3 at L <= 7, d = 5)


def tile_rows(t, BM=64):
    """The engine's tile (rows per program): BM, halved while a tile would span more than NLP_MAX layer slots. At d = 3
    a 64-row tile spans the whole depth (L + 1 layers), so the tile is halved from L = 8 on; at d >= 5 it stays 64
    (tests/test_d35.py checks every P2-A depth at d = 3... 11)."""
    tr = np.asarray(t["tr"])
    while BM > 16 and _p2(max(int(tr[r:min(r + BM, r1)].max() - tr[r:min(r + BM, r1)].min()) + 1
                              for r0, r1, _, _ in t["groups"] for r in range(r0, r1, BM))) > NLP_MAX:
        BM //= 2
    return BM


def engine(net, specs, L, mode="precise", chunk=None, latency=False, device="cuda"):
    """The engine of a NetRO at depth L for the CNOT patterns of specs (a shot's pattern index is its position in
    specs); mode, chunk and latency as engine.engine; the tile size from tile_rows (64 at d >= 5)."""
    tabs = deploy(net, specs, L)
    if latency:
        return EngineRO(tabs, device, mode, BM=16, tpi=2, pdl=True, stages=3)
    return EngineRO(tabs, device, mode, BM=tile_rows(tabs), chunk=chunk or None)


@torch.no_grad()
def reference(net, specs, L, det, pid=None, batch=256, layers=False):
    """The eager fp32 forward (TF32 off) of a NetRO at depth L, batches grouped by pattern:
    logits c (N, n_obs) float32, and with layers=True the per-layer logits l (N, n_obs, d, T)."""
    from duet.net import geometry, model as M0
    if batch <= 0:
        raise ValueError("eager batch must be positive")
    tf = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dev = next(net.parameters()).device
    pid = np.zeros(len(det), np.int64) if pid is None else np.asarray(pid)
    s0 = specs[0]
    m, nobs, d = s0["m"], s0["n_obs"], s0["d"]
    c_out = np.zeros((len(det), nobs), np.float32)
    l_out = np.zeros((len(det), nobs, d, L + 1), np.float32) if layers else None
    try:
        for k, s in enumerate(specs):
            A = torch.as_tensor(s["A"], dtype=torch.float32, device=dev)
            gate = torch.as_tensor(geometry.gate_table(s), device=dev)[None]
            sup = torch.as_tensor(LR.supports(s), device=dev)[None]
            idx = np.flatnonzero(pid == k)
            for i in range(0, len(idx), batch):
                j = idx[i:i + batch]
                bits = M0.unpack(torch.as_tensor(det[j], device=dev), m)
                g = M0.gf2(bits, A).view(-1, nobs, d)
                c, _, l = net(bits, gate.expand(len(j), -1, -1, -1, -1), g, L, sup.expand(len(j), -1, -1, -1))
                c_out[j] = c.float().cpu().numpy()
                if layers:
                    l_out[j] = l.float().cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
    return c_out, l_out
