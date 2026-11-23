"""The Triton engine of the Clifford network NetCL: the
function of duet.net.model_cl.NetCL.forward on one circuit's shots, from stim's bit-packed detector bits to the logits
and decisions, in a few Triton kernels over the detector sites only. The kernels and their arithmetic are those of the
engines for Net and NetRO, which are imported, not changed:
tiles of BM sites of one (block, type) group, the stem's bit gather from the packed bytes, the zero-padded grid for the
residual blocks' 3 x 3 x 3 gathers, the head's per-layer partial sums in a fixed order, the soft XOR in fp32 log space.
What NetCL changes, all of it in the per-circuit tables (circuit, into static device buffers that every circuit of
one shape (blocks, layers, product slots) reuses, so the CUDA graph of a batch size is captured once):
  stem the constant channels (preparation and readout flags per block, `bound`), the bias and the role term of
            the gates (8 flags per block and layer) folded into a site table P; the bit taps per site as detector
            indices (absent: a bit that is always 0)
  couple CoupleCL: per site, side (the gate before / after its layer) and slot (the gate's two layers) the partner's
            row (the other block at the same cell, at R90 / R270 of it, or the same block at the S partner cell; the
            partner block from `pblk`), -2 where that site holds no detector (u = down(LN(0))), -1 where there is none
            (0); per site and side the role (control, target, S) that selects the Cb x Cb slices of maps[side, role]
            for the site's type; roles absent from a tile are skipped
  head LayerLinesCL, one head shared by the product slots: the key projection folded with the fixed queries and the
            LayerNorm's affine (scores and values computed once per site), per slot the line-bias table
            b_line[type, offset bin] of each site, line and head (plus the folded LayerNorm bias); per slot and layer
            the queries' part-count term
  read-out per (shot, slot, line, layer): u = a Wo + bo + query_j + part term, l = MLP(LN(u)); soft XOR over the layers,
            g = (D A) mod 2 from the packed bits, c_k = sum_j w_j (-1)^g z_kj + b, the confidence min_k |c_k| over the
            circuit's products and the decision bits
Kernels per call (L residual blocks): stem, L x (coupling + block u, block + next coupling's u), last coupling + head,
read-out, soft XOR: 2 L + 4, each over (shot, tile); the call runs in chunks of `chunk` shots (buffers of one chunk stay
in the L2 cache) and is captured in one CUDA graph per batch size; the caller copies the shots in and the decision bits
out. Mode "precise": every matrix product as three fp16 tensor-core products of split operands (about fp32 accuracy),
"onepass": one fp16 product. Widths: C and the head width powers of two (NetCL's 128 and 32).

    eng = EngineCL(net, mode="precise") # net: NetCL (any device); the engine runs on `device`
    circ = eng.circuit(layers, prep, meas, noise) # or spec=a spec_cl.build dict
    c, conf, dec = eng.decode(circ, det) # numpy, all shots of that circuit
    call = eng.caller(circ, hin_pinned, B) # call(s0): shots s0.. s0 + B, synchronised (timing)"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from duet.net import model_cl as M
from duet.net import spec_cl
from duet.net.engine import _conv_up, _hilo, _ln, _mm, _mm2, _norm, _p2, _pad, _silu, _stem_taps, _ub, _uc
from duet.net.engine_ro import FCR, NLP_MAX, _log1mexp
from duet.net.model import N0

NSLOT = tl.constexpr(12)   # coupling map slices per type: (side, role control / target / S, slot of the gate's two layers)


# --------------------------------------------------------------------------------------------------------- kernels

@triton.jit
def _tile4(tiles_ptr, tile, BM: tl.constexpr):
    r0 = tl.load(tiles_ptr + tile * 4)
    nr = tl.load(tiles_ptr + tile * 4 + 1)
    i = tl.arange(0, BM)
    return r0 + i, i < nr, tl.load(tiles_ptr + tile * 4 + 3), nr


@triton.jit
def _couple_cl(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, MR,
               C: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr):
    """x[rows] + Up(V) (BM, C) fp32: V = sum over the sides with a gate of the partners' u at the gate's two layers times
    the Cb x Cb slices of maps[side, role] of the tile's type, the role of each row selecting its slices."""
    cols = tl.arange(0, C)
    cb = tl.arange(0, CB)
    mb = M_ptr + typ * 2 * NSLOT * CB * CB
    v = tl.zeros((BM, CB), dtype=tl.float32)
    for side in tl.static_range(2):
        rr = tl.load(role_ptr + rows * 2 + side, mask=rm, other=0)        # 0 none, 1 control, 2 target, 3 S
        if tl.max(rr, 0) > 0:
            for o in tl.static_range(2):                                   # the gate's layers: t - 1 + side + o
                p = tl.load(pc_ptr + rows * 4 + side * 2 + o, mask=rm, other=-1)
                src = uc_ptr + (b * MR + p)[:, None] * UW + cb[None, :]
                lm = (p >= 0)[:, None]
                miss = (p == -2)[:, None]                                  # no detector there: u = down(LN(0))
                uh = tl.where(miss, tl.load(c0_ptr + cb)[None, :], tl.load(src, mask=lm, other=0.0))
                ul = uh
                if X3:
                    ul = tl.where(miss, tl.load(c0_ptr + CB + cb)[None, :], tl.load(src + CB, mask=lm, other=0.0))
                zero = tl.zeros_like(uh)
                for ro in tl.static_range(3):
                    sel = rr == ro + 1
                    if tl.max(sel.to(tl.int32), 0) > 0:
                        slot = (side * 3 + ro) * 2 + o
                        v += _mm2(tl.where(sel[:, None], uh, zero), tl.where(sel[:, None], ul, zero), mb,
                                  (slot * CB + cb[:, None]) * CB + cb[None, :], NSLOT * CB * CB, X3)
    up = _mm(v, Wu_ptr, cb[:, None] * C + cols[None, :], CB * C, X3) + tl.load(bu_ptr + cols)[None, :]
    return tl.load(x_ptr + (b * MR + rows)[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + up


@triton.jit
def _k_stem_cl(pk_ptr, s0, cm_ptr, tiles_ptr, nbd_ptr, P_ptr, Wb_ptr, x_ptr, uc_ptr, lnw, lnb, eps, wd_ptr, MR,
               C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, KSP: tl.constexpr,
               K1: tl.constexpr, K2: tl.constexpr, NBYP: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr):
    """x = P[site] + sum over the stem's taps of bit * W[type, tap] (the shot's packed bytes read once); coupling 0's u."""
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ, nr = _tile4(tiles_ptr, tl.program_id(1), BM)
    if nr > 0:
        nby = tl.load(cm_ptr + 1)
        cols = tl.arange(0, C)
        by = tl.arange(0, NBYP)
        allb = tl.load(pk_ptr + (s0 + b) * nby + by, mask=by < nby, other=0).to(tl.int32)
        acc = _stem_taps(allb, nbd_ptr, rows, rm, 0, Wb_ptr, typ, 0, K1, KSP, C, BM, X3)
        if K2 > 0:
            acc += _stem_taps(allb, nbd_ptr, rows, rm, 0, Wb_ptr, typ, K1, K2, KSP, C, BM, X3)
        x = tl.load(P_ptr + rows[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0) + acc
        tl.store(x_ptr + (b * MR + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
        _uc(x, rows, rm, b, MR, lnw, lnb, eps, wd_ptr, uc_ptr, C, CN, CB, UW, X3)


@triton.jit
def _k_couple_block_cl(x_ptr, uc_ptr, tiles_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, gpos_ptr, ub_ptr,
                       lnw, lnb, eps, wd_ptr, bd_ptr, MR, NG, C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr,
                       UW: tl.constexpr, BM: tl.constexpr, X3: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ, nr = _tile4(tiles_ptr, tl.program_id(1), BM)
    if nr > 0:
        x = _couple_cl(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, MR, C, CB,
                       UW, BM, X3)
        cols = tl.arange(0, C)
        tl.store(x_ptr + (b * MR + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
        gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
        _ub(x, gp, rm, b, NG, lnw, lnb, eps, wd_ptr, bd_ptr, ub_ptr, C, CN, CB, UW, X3)


@triton.jit
def _k_block_cl(x_ptr, ub_ptr, tiles_ptr, gpos_ptr, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, uc_ptr, lnw, lnb, eps, wd_ptr, MR,
                NG, GH: tl.constexpr, GW: tl.constexpr, KT: tl.constexpr, KI: tl.constexpr, KJ: tl.constexpr,
                C: tl.constexpr, CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr,
                STAGES: tl.constexpr, TPI: tl.constexpr, X3: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    rows, rm, typ, nr = _tile4(tiles_ptr, tl.program_id(1), BM)
    if nr > 0:
        gp = tl.load(gpos_ptr + rows, mask=rm, other=0)
        x = _conv_up(x_ptr, ub_ptr, gp, Wc_ptr, bc_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, MR, NG, GH, GW, KT, KI, KJ,
                     C, CB, UW, BM, STAGES, TPI, X3)
        cols = tl.arange(0, C)
        tl.store(x_ptr + (b * MR + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
        _uc(x, rows, rm, b, MR, lnw, lnb, eps, wd_ptr, uc_ptr, C, CN, CB, UW, X3)


@triton.jit
def _k_couple_head_cl(x_ptr, uc_ptr, tiles_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, cm_ptr, tr_ptr,
                      tq_ptr, heps, wsc_ptr, bias_ptr, wv_ptr, bv_ptr, pa_ptr, MR, NQ, C: tl.constexpr,
                      CN: tl.constexpr, CB: tl.constexpr, UW: tl.constexpr, BM: tl.constexpr, HD: tl.constexpr,
                      DH: tl.constexpr, DP: tl.constexpr, D: tl.constexpr, S: tl.constexpr, NLP: tl.constexpr,
                      STORE_X: tl.constexpr, X3: tl.constexpr, XH: tl.constexpr):
    """The last coupling (x stored only if STORE_X), then per attention head h: y = LN(x) (affine folded), scores
    y Wsc and values y Wv + bv once; per product slot P < n_obs the gates sigmoid(score + bias[P, row]) in the rows of
    their layer slot, so G (NLP DP, BM) v gives the tile's partial sums per (layer, line) -> pa (B, S, NQ, D, C) at
    pair tq[tile, layer slot] (none: -1)."""
    b = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    rows, rm, typ, nr = _tile4(tiles_ptr, tile, BM)
    if nr > 0:
        x = _couple_cl(x_ptr, uc_ptr, pc_ptr, role_ptr, c0_ptr, M_ptr, Wu_ptr, bu_ptr, rows, rm, typ, b, MR, C, CB,
                       UW, BM, X3)
        cols = tl.arange(0, C)
        if STORE_X:
            tl.store(x_ptr + (b * MR + rows)[:, None] * C + cols[None, :], x, mask=rm[:, None])
        y = _norm(x, cols, C, CN, heps)
        yh = y.to(tl.float16)
        yl = (y - yh.to(tl.float32)).to(tl.float16)
        tr = tl.load(tr_ptr + rows, mask=rm, other=0)                      # each row's layer
        t0 = tl.load(tr_ptr + tl.load(tiles_ptr + tile * 4))
        sel = tl.where(rm, tr - t0, -1)[None, :] == tl.arange(0, NLP)[:, None]     # (NLP, BM): row in layer t0 + l
        jd = tl.arange(0, DP)
        ch = tl.arange(0, DH)
        rr = tl.arange(0, NLP * DP)
        jj = rr % DP
        q = tl.load(tq_ptr + tile * NLP + rr // DP)                        # pair of each gate row's layer
        keep = ((q >= 0) & (jj < D))[:, None]
        nobs = tl.load(cm_ptr + 2)
        for h in tl.range(0, HD, num_stages=1):
            hj = h * DP + jd
            sc = _mm2(yh, yl, wsc_ptr, cols[:, None] * (HD * DP) + hj[None, :], C * HD * DP, XH)
            cc = h * DH + ch
            v = _mm2(yh, yl, wv_ptr, cols[:, None] * C + cc[None, :], C * C, XH) + tl.load(bv_ptr + cc)[None, :]
            vh = v.to(tl.float16)
            vl = (v - vh.to(tl.float32)).to(tl.float16)
            for P in tl.range(0, nobs, num_stages=1):
                s = sc + tl.load(bias_ptr + (P * MR + rows)[:, None] * (HD * DP) + hj[None, :], mask=rm[:, None],
                                 other=0.0)
                gt = tl.trans(tl.where(rm[:, None], 1.0 / (1.0 + tl.exp(-s)), 0.0))                  # (DP, BM)
                g = tl.reshape(tl.where(sel[:, None, :], gt[None, :, :], 0.0), (NLP * DP, BM))
                gh = g.to(tl.float16)
                pp = tl.dot(gh, vh)
                if XH:
                    pp += tl.dot((g - gh.to(tl.float32)).to(tl.float16), vh) + tl.dot(gh, vl)
                tl.store(pa_ptr + (((b * S + P) * NQ + q) * D + jj)[:, None] * C + cc[None, :], pp, mask=keep)


@triton.jit
def _k_readout_cl(pa_ptr, lq_ptr, cm_ptr, Uq_ptr, Uc_ptr, Wo_ptr, mw_ptr, mb_ptr, meps, W1_ptr, b1_ptr, W2_ptr,
                  b2_ptr, inv_n0, l_ptr, NQ, T, NQT: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
                  S: tl.constexpr, RB: tl.constexpr, FF: tl.constexpr, FC: tl.constexpr, X3: tl.constexpr):
    """Program (b, P, rows r = j T + t of block RB), P < n_obs: a = the partial sums of layer t's pairs lq[t:] (fixed
    l (B, S, D, T)."""
    b = tl.program_id(0).to(tl.int64)
    P = tl.program_id(1)
    if P < tl.load(cm_ptr + 2):
        r = tl.program_id(2) * RB + tl.arange(0, RB)
        rm = r < D * T
        j = tl.where(rm, r // T, 0)
        t = tl.where(rm, r % T, 0)
        cols = tl.arange(0, C)
        a = tl.zeros((RB, C), dtype=tl.float32)
        for k in tl.range(0, NQT):
            q = tl.load(lq_ptr + t * NQT + k, mask=rm, other=-1)
            a += tl.load(pa_ptr + (((b * S + P) * NQ + q) * D + j)[:, None] * C + cols[None, :],
                         mask=(q >= 0)[:, None], other=0.0)
        a = a * inv_n0
        u = _mm(a, Wo_ptr, cols[:, None] * C + cols[None, :], C * C, X3) + \
            tl.load(Uq_ptr + j[:, None] * C + cols[None, :]) + tl.load(Uc_ptr + (P * T + t)[:, None] * C + cols[None, :])
        y = _ln(u, mw_ptr, mb_ptr, cols, C, C, meps)
        yh = y.to(tl.float16)
        yl = (y - yh.to(tl.float32)).to(tl.float16)
        z = tl.zeros((RB,), dtype=tl.float32) + tl.load(b2_ptr)
        for f0 in tl.static_range(0, FF, FC):
            ff = f0 + tl.arange(0, FC)
            hc = _mm2(yh, yl, W1_ptr, cols[:, None] * FF + ff[None, :], C * FF, X3) + tl.load(b1_ptr + ff)[None, :]
            z += tl.sum(_silu(hc) * tl.load(W2_ptr + ff)[None, :], 1)
        tl.store(l_ptr + ((b * S + P) * D + j) * T + t, z, mask=rm)


@triton.jit
def _k_xor_cl(l_ptr, pk_ptr, s0, cm_ptr, A_ptr, w_ptr, b0_ptr, out_ptr, dec_ptr, T, D: tl.constexpr, S: tl.constexpr,
              KDP: tl.constexpr, TP: tl.constexpr, NBYP: tl.constexpr):
    """Program b: z = model_ro.soft_xor over t of l (B, S D, T) per row P D + j (engine_ro's formula); g = (D A) mod 2
    from the packed bits; c_P = sum_j w_j (-1)^g z + b -> out (B, S + 1) (c, then min over P < n_obs of |c_P|), dec
    (bit P = c_P > 0, P < n_obs)."""
    b = tl.program_id(0).to(tl.int64)
    nby = tl.load(cm_ptr + 1)
    nobs = tl.load(cm_ptr + 2)
    kd = tl.arange(0, KDP)
    tt = tl.arange(0, TP)
    km = kd < nobs * D
    msk = km[:, None] & (tt < T)[None, :]
    lv = tl.load(l_ptr + (b * S * D + kd)[:, None] * T + tt[None, :], mask=msk, other=0.0)
    a = tl.minimum(tl.maximum(tl.abs(lv), 1e-6), 80.0)
    Ssum = tl.sum(tl.where(msk, _log1mexp(a) - libdevice.log1p(libdevice.exp(-a)), 0.0), 1)
    Ssum = tl.minimum(Ssum, -1e-37)
    mag = libdevice.log1p(libdevice.exp(Ssum)) - _log1mexp(-Ssum)
    odd = (tl.sum(tl.where(msk & (lv > 0), 1, 0), 1) % 2) == 1
    z = tl.where(odd, mag, -mag)
    by = tl.arange(0, NBYP)
    v = tl.load(pk_ptr + (s0 + b) * nby + by, mask=by < nby, other=0).to(tl.int32)[None, :] & \
        tl.load(A_ptr + kd[:, None] * NBYP + by[None, :]).to(tl.int32)
    v = v - ((v >> 1) & 0x55)
    v = (v & 0x33) + ((v >> 2) & 0x33)
    v = (v + (v >> 4)) & 0x0F
    sgn = 1.0 - 2.0 * (tl.sum(v, 1) & 1).to(tl.float32)
    tv = tl.where(km, tl.load(w_ptr + kd) * sgn * z, 0.0)
    conf = 3.0e38
    dec = 0
    ob = (s0 + b) * (S + 1)
    for Q in tl.static_range(S):
        cq = tl.sum(tl.where((kd >= Q * D) & (kd < Q * D + D), tv, 0.0), 0) + tl.load(b0_ptr)
        tl.store(out_ptr + ob + Q, cq)
        conf = tl.where(Q < nobs, tl.minimum(conf, tl.abs(cq)), conf)
        dec += ((cq > 0) & (Q < nobs)).to(tl.int32) << Q
    tl.store(out_ptr + ob + S, conf)
    tl.store(dec_ptr + s0 + b, dec.to(tl.uint8))


# ---------------------------------------------------------------------------------------------------------- tables

def cell_maps(d):
    """(6, H W) the partner cell per pcode (model_cl.NetCL.cellmap): none, same cell, R90, R270, mirror in frame 0 / 1."""
    g90, g270 = spec_cl.grid_maps(d)
    mir = spec_cl.mirror_maps(d)
    return np.stack([np.full_like(g90, -1), np.arange(len(g90)), g90, g270, mir[0], mir[1]])


def tiles_of(t_row, groups, BM):
    """[(r0, nr, block, type)] tiles of at most BM rows within each (block, type) group, and the largest layer span."""
    tiles = [(r, min(BM, r1 - r), b, ty) for r0, r1, b, ty in groups for r in range(r0, r1, BM)]
    span = max(int(t_row[r:r + n].max() - t_row[r:r + n].min()) + 1 for r, n, _, _ in tiles)
    return tiles, span


class EngineCL:
    """The engine of a NetCL (module docstring). mode "precise" / "onepass"; BM sites per tile (None: 64, halved while a
    tile would span more than NLP_MAX layers); chunk: shots per pass through the kernels (None: the whole batch);
    tpi: the residual blocks' taps per product, tpi_small at batches of at most `small` shots (fewer, larger steps
    shorten each program's chain where the batch leaves the GPU idle); nlp: the head's layer slots per tile (None: from
    BM and d); store_x: the last coupling stores x (checks)."""

    def __init__(self, net, mode="precise", BM=None, chunk=64, warps=4, warps_ro=4, stages=3, tpi=2, tpi_small=4,
                 small=8, RB=32, nlp=None, device="cuda"):
        self.dev = dev = torch.device(device)
        self.mode = mode
        X3 = {"precise": True, "onepass": False}[mode]
        self.X3 = X3
        net = copy.deepcopy(net).float().cpu().eval()
        self.net = net
        d, C, Cb = net.d, net.C, net.couple[0].Cb
        self.d, self.C, self.CB, self.H, self.L = d, C, Cb, net.H, len(net.blocks)
        assert C == _p2(C) and Cb == _p2(Cb) and Cb >= 16, "the engine runs power-of-two widths"
        self.UW = 2 * Cb if X3 else Cb
        n = (d * d - 1) // 2                                                # checks of one type per layer
        if BM is None:
            BM = 64
            while BM > 16 and _p2(-(-(BM - 1) // n) + 1) > NLP_MAX:
                BM //= 2
        self.BM = BM
        self.NLP = nlp or _p2(-(-(BM - 1) // n) + 1)
        self.cfg = dict(mode=mode, BM=BM, chunk=chunk, warps=warps, warps_ro=warps_ro, stages=stages, tpi=tpi,
                        tpi_small=tpi_small, small=small, RB=RB, nlp=self.NLP)
        self.store_x = False
        h = lambda v: _hilo(v).to(dev).contiguous()                            # noqa: E731
        f = lambda v: torch.as_tensor(v).to(dev, torch.float32).contiguous()   # noqa: E731
        with torch.no_grad():
            # stem: the bit taps' weights per type, as engine.trunk_tables
            st = net.stem
            self.kst = tuple(st.weight.shape[2:])
            KS = int(np.prod(self.kst))
            K1 = KS if KS & (KS - 1) == 0 else 1 << (KS.bit_length() - 1)
            K2 = max(16, _p2(KS - K1)) if KS > K1 else 0
            self.KS, self.K1, self.K2, self.KSP = KS, K1, K2, K1 + K2
            w0 = st.weight[:, 0].reshape(2 * C, KS)
            Wb = torch.zeros(2, self.KSP, C)
            Wb[0, :KS], Wb[1, :KS] = w0[:C].t(), w0[C:].t()
            self.Wb = h(Wb)
            # couplings: per type the 12 Cb x Cb slices (side, role, slot) of the maps' type columns; c0 = down(LN(0))
            self.cp = []
            for cp in net.couple:
                Mt = torch.stack([torch.cat([cp.maps[s, r][o * Cb:(o + 1) * Cb, ty * Cb:(ty + 1) * Cb]
                                             for s in (0, 1) for r in range(3) for o in (0, 1)]) for ty in (0, 1)])
                self.cp.append(dict(ln_w=f(cp.ln.weight), ln_b=f(cp.ln.bias), eps=float(cp.ln.eps),
                                    Wd=h(cp.down.weight.t()), c0=h(cp.down(cp.ln(torch.zeros(1, C))))[:, 0].contiguous(),
                                    M=h(Mt), Wu=h(cp.up.weight.t()), bu=f(cp.up.bias)))
            # residual blocks, as engine.trunk_tables and engine.Engine
            self.bk = tuple(net.cfg["block_k"])
            ntap = int(np.prod(self.bk))
            pad = -ntap % 4                         # taps padded to a multiple of 4: tpi 2 or 4 alike
            self.blk = []
            for bl in net.blocks:
                wc = bl.conv.weight
                Wc = torch.stack([wc[ty * Cb:(ty + 1) * Cb].permute(2, 3, 4, 1, 0).reshape(ntap, Cb, Cb)
                                  for ty in (0, 1)])
                Wc = torch.cat([Wc, Wc.new_zeros(2, pad, Cb, Cb)], 1).reshape(2, -1, Cb)
                self.blk.append(dict(ln_w=f(bl.ln.weight), ln_b=f(bl.ln.bias), eps=float(bl.ln.eps),
                                     Wd=h(bl.down.weight.t()), bd=f(bl.down.bias), Wc=h(Wc),
                                     bc=f(torch.stack([bl.conv.bias[:Cb], bl.conv.bias[Cb:]])), Wu=h(bl.up.weight.t()),
                                     bu=f(bl.up.bias)))
            # head: keys folded with the queries and the LayerNorm's affine (float64), values, read-out
            hd = copy.deepcopy(net.head).double()
            Hh, dh = hd.h, hd.dh
            assert Hh * dh == C and dh == _p2(dh) and dh >= 16, "heads of power-of-two width"
            self.HD, self.DH, self.DP = Hh, dh, max(16, _p2(d))
            q = hd.wq(hd.queries()).view(d, Hh, dh)
            ws = torch.einsum("hec,jhe->chj", hd.k.weight.view(Hh, dh, C), q) * dh ** -0.5     # (C, Hh, d)
            lw, lb = hd.ln.weight, hd.ln.bias
            Wsc = torch.zeros(C, Hh, self.DP, dtype=torch.float64)
            Wsc[..., :d] = lw[:, None, None] * ws
            self.bfold = torch.einsum("c,chj->hj", lb, ws).float()                               # (Hh, d)
            Wv = hd.v.weight.t()
            self.R = hd.R
            self.b_line = net.head.b_line.detach().float().reshape(Hh, -1)                       # (Hh, 2 (2R + 2))
            self.part = net.head.part.detach().float()
            FF = hd.mlp[1].weight.shape[0]
            assert FF % FCR == 0
            self.hd = dict(eps=float(net.head.ln.eps), Wsc=h(Wsc.reshape(C, -1).float()),
                           Wv=h((lw[:, None] * Wv).float()), bv=f(hd.v.bias + lb @ Wv),
                           Uq=f(hd.queries() + hd.o.bias), Wo=h(hd.o.weight.t().float()), mw=f(hd.mlp[0].weight),
                           mb=f(hd.mlp[0].bias), meps=float(hd.mlp[0].eps), W1=h(hd.mlp[1].weight.t().float()),
                           b1=f(hd.mlp[1].bias), W2=f(hd.mlp[3].weight.reshape(-1)), b2=f(hd.mlp[3].bias.reshape(1)),
                           FF=FF)
            self.w_d, self.b0 = net.w.detach().float(), f(net.b.detach().reshape(1))
        self.shape, self.t, self.graphs, self.bufs = None, None, {}, {}

    # ------------------------------------------------------------------------------------------------ per circuit
    def _shape(self, K, T, S):
        """Static sizes of one circuit shape (blocks, layers, product slots)."""
        d, BM = self.d, self.BM
        n = (d * d - 1) // 2
        self.shape = (K, T, S)
        self.K, self.T, self.S = K, T, S
        self.MR = K * T * (d * d - 1)                                        # the most detectors
        self.NT = 2 * K * -(-T * n // BM)                                    # the most tiles
        self.NQ = self.NT * self.NLP                                         # the most (tile, layer) pairs
        self.NBYM = (self.MR + 7) // 8
        self.NBYP = _p2(self.NBYM + 1)                                       # one zero byte past every shot's
        self.KDP = _p2(S * d)
        pt, pi, pj = (v // 2 for v in self.bk)
        self.GT, self.GH, self.GW = T + 2 * pt, self.H + 2 * pi, self.H + 2 * pj
        self.NG = K * self.GT * self.GH * self.GW
        w = torch.zeros(self.KDP)
        w[:S * d] = self.w_d.repeat(S)
        self.w = w.to(self.dev)

    def tables(self, s):
        """The per-circuit tables (numpy, and P on the engine's device) of a spec_cl.build dict s."""
        d, H, C = self.d, self.H, self.C
        HW = H * H
        K, T, S, m = s["K"], s["T"], s["slots"], s["m"]
        if self.shape is None:
            self._shape(K, T, S)
        assert (K, T, S) == self.shape, ("another circuit shape", self.shape, (K, T, S))
        MR = self.MR
        flat = np.asarray(s["flat"], np.int64)
        mxf, mzf = s["mx"].reshape(-1), s["mz"].reshape(-1)
        assert ((mxf[flat] + mzf[flat]) == 1).all() and int(mxf.sum() + mzf.sum()) == m
        ty = mzf[flat].astype(np.int64)
        blk, tt, cell = flat // (T * HW), flat // HW % T, flat % HW
        order = np.lexsort((cell % H, cell // H, tt, ty, blk))                 # engine row r = detector order[r]
        eb, et, ec, ety = blk[order], tt[order], cell[order], ty[order]
        groups = []
        for b in range(K):
            for y in (0, 1):
                rr = np.flatnonzero((eb == b) & (ety == y))
                if len(rr):
                    groups.append((int(rr[0]), int(rr[-1]) + 1, b, y))
        tiles, span = tiles_of(et, groups, self.BM)
        assert len(tiles) <= self.NT and span <= self.NLP, ("tiles", len(tiles), self.NT, "span", span, self.NLP)
        tl_ = np.zeros((self.NT, 4), np.int32)
        tl_[:len(tiles)] = tiles
        row_at = np.full(K * T * HW, -1, np.int64)                             # grid site -> engine row
        row_at[flat[order]] = np.arange(m)
        det_at = np.full(K * T * HW, -1, np.int64)                             # grid site -> detector (bit) index
        det_at[flat] = np.arange(m)
        nby = (m + 7) // 8
        # stem taps (cross-correlation, zero padding k // 2): the tap's detector, or bit 8 nby (a zero byte)
        kt, ki, kj = self.kst
        a, c_, e = (v.reshape(1, -1) for v in np.meshgrid(np.arange(kt), np.arange(ki), np.arange(kj), indexing="ij"))
        t2, i2, j2 = et[:, None] + a - kt // 2, (ec // H)[:, None] + c_ - ki // 2, (ec % H)[:, None] + e - kj // 2
        ok = (t2 >= 0) & (t2 < T) & (i2 >= 0) & (i2 < H) & (j2 >= 0) & (j2 < H)
        dt = det_at[np.where(ok, ((eb[:, None] * T + t2) * H + i2) * H + j2, 0)]
        nbd = np.full((MR, self.KSP), 8 * nby, np.int32)
        nbd[:m, :self.KS] = np.where(ok & (dt >= 0), dt, 8 * nby)
        # gate roles per side: 0 none, 1 control, 2 target, 3 S
        g3 = s["gate"][..., :3]
        rc = np.where(g3.sum(-1) > 0, 1 + g3.argmax(-1), 0)                    # (K, T, side)
        role = np.zeros((MR, 2), np.int32)
        role[:m] = rc[eb, et]
        # partners per (side, slot): the partner block's row at layer t - 1 + side + slot, -2 no detector, -1 none
        cmap = cell_maps(d)
        pc = np.full((MR, 4), -1, np.int32)
        for sd in (0, 1):
            c2 = cmap[s["pcode"][eb, et, sd].astype(np.int64), ec]
            b2 = s["pblk"][eb, et, sd].astype(np.int64)
            for o in (0, 1):
                lay = et - 1 + sd + o
                ok = (c2 >= 0) & (lay >= 0) & (lay < T)
                r2 = row_at[np.where(ok, (b2 * T + lay.clip(0, T - 1)) * HW + c2.clip(0), 0)]
                pc[:m, 2 * sd + o] = np.where(ok, np.where(r2 >= 0, r2, -2), -1)
        # residual blocks' padded grid
        pt, pi, pj = (v // 2 for v in self.bk)
        gpos = np.zeros(MR, np.int32)
        gpos[:m] = ((eb * self.GT + et + pt) * self.GH + ec // H + pi) * self.GW + ec % H + pj
        tr = np.zeros(MR, np.int32)
        tr[:m] = et
        # head: per (tile, layer slot) a pair; per layer its pairs in a fixed order
        tq = np.full((self.NT, self.NLP), -1, np.int32)
        pairs = []
        for i, (r, nr, _, _) in enumerate(tiles):
            for u in np.unique(et[r:r + nr]):
                tq[i, u - et[r]] = len(pairs)
                pairs.append(int(u))
        lq = np.full((T, self.NT), -1, np.int32)
        for u in range(T):
            p_ = [k for k, v in enumerate(pairs) if v == u]
            lq[u, :len(p_)] = p_
        # line-bias table per slot, row, head, line (+ the LayerNorm bias folded with the keys)
        offb = M.offset_bins(d, s["lines"], s["orient"], s["mx"], s["mz"])        # (S, T, K, H W, d)
        idx = ety[None, :, None] * (2 * self.R + 2) + offb[:, et, eb, ec].astype(np.int64)   # (S, m, d)
        bias = np.zeros((S, MR, self.HD, self.DP), np.float32)
        bias[:, :m, :, :d] = (self.b_line.numpy()[:, idx].transpose(1, 2, 0, 3) + self.bfold.numpy()[None, None])
        cnt = M.part_counts(s["parts"])                                          # (S, T, 3)
        Uc = (torch.as_tensor(cnt, dtype=torch.float32)[..., None] * self.part).sum(-2)   # (S, T, C), as the forward
        A = np.zeros((self.KDP, self.NBYP), np.uint8)
        A[:S * d, :nby] = np.packbits(np.asarray(s["A"], np.uint8).T, axis=1, bitorder="little")
        cm = np.array([m, nby, s["n_obs"], 0], np.int32)
        return dict(tiles=tl_, nbd=nbd, role=role, pc=pc, gpos=gpos, tr=tr, tq=tq, lq=lq, bias=bias.reshape(S, MR, -1),
                    Uc=Uc.numpy(), A=A, cm=cm, P=self._site_table(s, eb, et, ec, ety)), \
            dict(m=m, nb=nby, n_obs=s["n_obs"], S=S, tiles=len(tiles), pairs=len(pairs), span=span)

    @torch.no_grad()
    def _site_table(self, s, eb, et, ec, ety):
        """P (MR, C) on the device: the stem's constant channels (with `bound`), its bias and the gates' role term, of
        each site's type (fp32, TF32 off, as FastCL's x0)."""
        net, dev, C, H = self.net, self.dev, self.C, self.H
        K, T = self.K, self.T
        f32 = dict(dtype=torch.float32, device=dev)
        mx = torch.as_tensor(s["mx"], **f32)
        mz = torch.as_tensor(s["mz"], **f32)
        on = mx + mz
        bd = torch.as_tensor(s["bound"], **f32)
        const = torch.stack([mx, mz, on * bd[..., 0, None, None], on * bd[..., 1, None, None]], 1)   # (K, 4, T, H, W)
        st = net.stem
        tf = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            y = F.conv3d(const, st.weight[:, 1:].to(dev), st.bias.to(dev), padding=st.padding).permute(0, 2, 3, 4, 1)
            y = y + F.linear(torch.as_tensor(s["gate"], **f32).flatten(2), net.role.weight.to(dev))[:, :, None, None] \
                * on[..., None]
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf
        x0 = y[..., :C] * mx[..., None] + y[..., C:] * mz[..., None]                  # (K, T, H, W, C)
        P = torch.zeros(self.MR, C, **f32)
        ix = [torch.as_tensor(v, device=dev) for v in (eb, et, ec // H, ec % H)]
        P[:len(eb)] = x0[ix[0], ix[1], ix[2], ix[3]]
        return P

    @torch.no_grad()
    def circuit(self, layers, prep, meas, noise="uniform", p=0.003, spec=None, build=None):
        """Load one circuit's tables into the static buffers (spec: a spec_cl.build dict; build: the function that makes
        it, spec_cl.build by default); returns its sizes."""
        s = spec if spec is not None else (build or spec_cl.build)(self.d, layers, prep, meas, p, noise)
        t, info = self.tables(s)
        t = {k: torch.as_tensor(v) for k, v in t.items()}
        if self.t is None:
            self.t = {k: v.to(self.dev).contiguous() for k, v in t.items()}
        else:
            for k, v in t.items():
                self.t[k].copy_(v, non_blocking=True)
            for bf in self.bufs.values():
                bf["ub"].zero_()                     # grid sites without a detector in this circuit hold u = 0
        return info

    # ------------------------------------------------------------------------------------------------ the call
    def buffers(self, B):
        if B not in self.bufs:
            dev, f16 = self.dev, torch.float16
            self.bufs[B] = dict(x=torch.empty(B, self.MR, self.C, device=dev),
                                uc=torch.empty(B, self.MR, self.UW, device=dev, dtype=f16),
                                ub=torch.zeros(B, self.NG, self.UW, device=dev, dtype=f16),
                                pa=torch.empty(B, self.S, self.NQ, self.d, self.C, device=dev),
                                l=torch.empty(B, self.S, self.d, self.T, device=dev))
        return self.bufs[B]

    def launches(self, xin, out, dec, s0, B, tpi=None):
        """[(name, launch)]: the kernels over shots s0.. s0 + B of the packed input xin (one chunk)."""
        t, c, bf = self.t, self.cfg, self.buffers(B)
        X3 = self.X3
        grid = (B, self.NT)
        sz = dict(C=self.C, CN=self.C, CB=self.CB, UW=self.UW, BM=self.BM, num_warps=c["warps"])
        cp0 = self.cp[0]
        steps = [("stem", lambda: _k_stem_cl[grid](
            xin, s0, t["cm"], t["tiles"], t["nbd"], t["P"], self.Wb, bf["x"], bf["uc"], cp0["ln_w"], cp0["ln_b"],
            cp0["eps"], cp0["Wd"], self.MR, KSP=self.KSP, K1=self.K1, K2=self.K2, NBYP=self.NBYP, X3=X3, **sz))]
        cpl = lambda k: (bf["x"], bf["uc"], t["tiles"], t["pc"], t["role"], self.cp[k]["c0"], self.cp[k]["M"],  # noqa
                         self.cp[k]["Wu"], self.cp[k]["bu"])
        for k in range(self.L):
            p, q = self.blk[k], self.cp[k + 1]
            steps.append((f"couple{k}", lambda k=k, p=p: _k_couple_block_cl[grid](
                *cpl(k), t["gpos"], bf["ub"], p["ln_w"], p["ln_b"], p["eps"], p["Wd"], p["bd"], self.MR, self.NG,
                X3=X3, **sz)))
            steps.append((f"block{k}", lambda p=p, q=q: _k_block_cl[grid](
                bf["x"], bf["ub"], t["tiles"], t["gpos"], p["Wc"], p["bc"], p["Wu"], p["bu"], bf["uc"], q["ln_w"],
                q["ln_b"], q["eps"], q["Wd"], self.MR, self.NG, GH=self.GH, GW=self.GW, KT=self.bk[0], KI=self.bk[1],
                KJ=self.bk[2], STAGES=c["stages"], TPI=tpi or c["tpi"], X3=X3, **sz)))
        hd = self.hd
        steps.append((f"couple{self.L}_head", lambda: _k_couple_head_cl[grid](
            *cpl(self.L), t["cm"], t["tr"], t["tq"], hd["eps"], hd["Wsc"], t["bias"], hd["Wv"], hd["bv"], bf["pa"],
            self.MR, self.NQ, HD=self.HD, DH=self.DH, DP=self.DP, D=self.d, S=self.S, NLP=self.NLP,
            STORE_X=self.store_x, X3=X3, XH=X3, **sz)))
        steps.append(("readout", lambda: _k_readout_cl[(B, self.S, -(-self.d * self.T // c["RB"]))](
            bf["pa"], t["lq"], t["cm"], hd["Uq"], t["Uc"], hd["Wo"], hd["mw"], hd["mb"], hd["meps"], hd["W1"], hd["b1"],
            hd["W2"], hd["b2"], 1.0 / N0, bf["l"], self.NQ, self.T, NQT=self.NT, C=self.C, D=self.d, S=self.S,
            RB=c["RB"], FF=hd["FF"], FC=FCR, X3=X3, num_warps=c["warps_ro"])))
        steps.append(("xor", lambda: _k_xor_cl[(B,)](
            bf["l"], xin, s0, t["cm"], t["A"], self.w, self.b0, out, dec, self.T, D=self.d, S=self.S, KDP=self.KDP,
            TP=_p2(self.T), NBYP=self.NBYP, num_warps=4)))
        return steps

    def plan(self, xin, out, dec, B):
        c = self.cfg
        ch, tpi = c["chunk"] or B, c["tpi_small"] if B <= c["small"] else c["tpi"]
        return [st for s0 in range(0, B, ch) for st in self.launches(xin, out, dec, s0, min(ch, B - s0), tpi)]

    @torch.no_grad()
    def graph(self, B):
        """The CUDA graph of one batch size: (graph, static input (B NBYM bytes), out (B, S + 1), dec (B,))."""
        if B not in self.graphs:
            xin = torch.zeros(B * self.NBYM + self.NBYP, dtype=torch.uint8, device=self.dev)
            out = torch.zeros(B, self.S + 1, device=self.dev)
            dec = torch.zeros(B, dtype=torch.uint8, device=self.dev)
            steps = self.plan(xin, out, dec, B)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    for _, fn in steps:
                        fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _, fn in steps:
                    fn()
            self.graphs[B] = (g, xin, out, dec)
        return self.graphs[B]

    def caller(self, circ, hin, B):
        """call(s0): shots s0.. s0 + B of hin (pinned (n, bytes per shot) uint8) -> decision bits in pinned memory."""
        g, xin, _, dec = self.graph(B)
        x2 = xin[:B * circ["nb"]].view(B, circ["nb"])
        hout = torch.zeros(B, dtype=torch.uint8).pin_memory()
        self.hout = hout                                # the last caller's decision bits (host)

        def call(s0):
            x2.copy_(hin[s0:s0 + B], non_blocking=True)
            g.replay()
            hout.copy_(dec, non_blocking=True)
            torch.cuda.synchronize()
        return call

    @torch.no_grad()
    def decode(self, circ, det, B=1024):
        """All shots of det (numpy (n, bytes per shot) uint8) of the loaded circuit -> c (n, n_obs), the confidence
        min_P |c_P| (n,), decisions (n, n_obs) bool."""
        n, k = len(det), circ["n_obs"]
        g, xin, out, dec = self.graph(B)
        cs, confs, ds = [], [], []
        for s0 in range(0, n, B):
            b = min(B, n - s0)
            xb = np.zeros((B, circ["nb"]), np.uint8)
            xb[:b] = det[s0:s0 + b]
            xin[:B * circ["nb"]].view(B, circ["nb"]).copy_(torch.from_numpy(xb))
            g.replay()
            o, dd = out[:b].cpu().numpy(), dec[:b].cpu().numpy()
            cs.append(o[:, :k])
            confs.append(o[:, self.S])
            ds.append((dd[:, None] >> np.arange(k)) & 1)
        return np.concatenate(cs), np.concatenate(confs), np.concatenate(ds).astype(bool)
