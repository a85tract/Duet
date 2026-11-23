"""The fast decode path of NetCL: the function of
duet.net.model_cl.NetCL.forward on one two-block circuit's shots, from stim's bit-packed detector bits to the decisions,
rearranged for speed. model_cl is not changed; FastCL reads a NetCL's weights.

Per circuit (circuit, into static device buffers that every circuit of the same depth reuses):
  stem the constant channels' convolution, its bias and the role term of the gates, selected per site type: x0;
            the bit channel's 3 x 5 x 5 taps as a gather table per site (the site type folded in: 2 x 75 columns)
  exchange per coupling, (block, layer) and side the gate-weighted sum of the role maps (one 4 Cb x 2 Cb matrix per
            block and layer) and the partners' flat indices of both sides and slots in one gather table
  read-out the line-bias table looked up (S, d, heads, K, T, H W), the site mask / N0, the queries plus the part
            counts' term per slot and layer
  parity A's rows in grid order, so g = grid bits A mod 2
Per weights (once): the key projection folded with the queries (scores straight from the head LayerNorm, one matrix
with the value projection), the stem's bit weights as a (C, 150) matrix; bf16 copies of the linear and convolution
weights for the bf16 path (autocast as in training: LayerNorm, the sigmoid, soft XOR and c stay fp32).
Per call (one CUDA graph per batch size, captured once and replayed for every circuit of that depth): packed bits ->
grid bits (one gather) -> stem (one product) -> couplings (LayerNorm, down, one gather, one batched product over
(block, layer), up) and the residual blocks (model.Block's arithmetic on the same sites) -> head (both product slots at once) -> soft XOR -> c, decisions c > 0.
The caller copies the shots in (one copy into the static input) and the decisions out (one copy).
compile: 0 the graph holds eager kernels, 1 torch.compile (default), 2 torch.compile max-autotune-no-cudagraphs.
conv (the residual blocks' 3 x 3 x 3 convolution): 0 cuDNN as torch.compile lowers it (on the grid), 2 a gather of the
27 taps of each site and one matrix product (in batch chunks of at most 2 GB of gathered taps); -1 (default): 2 in
fp32 (cuDNN's fp32 3-D kernels without TF32 are slow at every batch size) and in bf16 up to 4,096 sites per batch
(shots x blocks x layers x sites), 0 in bf16 above.
compact (default 1): every tensor of C or Cb channels holds per block and layer only the d^2 - 1 cells of the measure
qubits (the same cells in every frame) and one cell that never holds a check, instead of the (d + 1)^2 grid cells; the
other cells' x is 0 in NetCL, so their partner value down(LN(0)) is the spare cell's and their convolution input 0.

    fast = FastCL(net, bf16=True) # net: NetCL on the GPU, eval mode
    circ = fast.circuit(layers, prep, meas, noise)
    c, dec = fast.decode(det_packed) # numpy, all shots of that circuit
    call = fast.caller(circ, hin_pinned, B) # call(s0): shots s0.. s0 + B, synchronised (timing)
The single-layer residual runs on two-slice circuits (FastCL builds
whole circuits only): its blocks and couplings as the trunk's, its routing (RouteCL) as one gather of the neighbours'
sites with the per-relation means folded into a (sites, cap, relations) weight table, and its product head folded
like the read-out. The single-layer path of (model_cl NetCL.l1path) is a NetCL: FastCL(net.l1path) decodes
single-layer circuits with it, its trunk routing (troute) as the residual's and its idle-tick channels (flags) folded
into the stem's constant term; FastCL(net) refuses them and decodes every deeper circuit with the network."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from duet.net import labels_ro as LR
from duet.net import model_cl as M
from duet.net import spec_cl
from duet.net.model import N0
from duet.net.model_ro import soft_xor


def stem_taps(K, T, H, kt, kh, kw):
    """(N, kt kh kw) int64: grid index of each site's stem tap in the weight's order (cross-correlation, zero padding
    k // 2), N = K T H H where the tap falls outside the block's grid."""
    k, t, i, j = (v.reshape(-1, 1) for v in np.meshgrid(np.arange(K), np.arange(T), np.arange(H), np.arange(H),
                                                          indexing="ij"))
    a, b, e = (v.reshape(1, -1) for v in np.meshgrid(np.arange(kt), np.arange(kh), np.arange(kw), indexing="ij"))
    tt, ii, jj = t + a - kt // 2, i + b - kh // 2, j + e - kw // 2
    ok = (tt >= 0) & (tt < T) & (ii >= 0) & (ii < H) & (jj >= 0) & (jj < H)
    return np.where(ok, ((k * T + tt) * H + ii) * H + jj, K * T * H * H)


class FastCL:
    def __init__(self, net, bf16=True, compile=1, bench=1, conv=-1, compact=1):
        self.net, self.bf16, self.compile, self.bench = net, bool(bf16), int(compile), int(bench)
        self.conv, self.compact = int(conv), int(compact)
        self.dev = next(net.parameters()).device
        self.dt = torch.bfloat16 if self.bf16 else torch.float32
        self.d, self.C, self.H, self.Cb = net.d, net.C, net.H, net.couple[0].Cb
        hd = net.head
        self.heads, self.dh = hd.h, hd.dh
        fw = copy.deepcopy(net)                 # the forward's weights: linear and convolution weights in bf16 for
        if self.bf16:                           # the bf16 path (what autocast casts them to at every call)
            for mod in fw.modules():
                if isinstance(mod, (nn.Linear, nn.Conv3d)):
                    mod.to(torch.bfloat16)
        self.res = getattr(fw, "res", None)
        self.troute = getattr(fw, "troute", None)
        for blk in list(fw.blocks) + (list(self.res.blocks) if self.res is not None else []):
            blk.w27 = blk.conv.weight.detach().permute(0, 2, 3, 4, 1).reshape(2 * self.Cb, -1).contiguous()
        self.fw = fw
        with torch.no_grad():
            st = net.stem
            self.kst = tuple(st.weight.shape[2:])
            wb = st.weight[:, 0].reshape(2 * self.C, -1).double()                       # (2C, 75)
            self.w_stem = torch.cat([wb[:self.C], wb[self.C:]], 1).to(self.dt)          # (C, 150): X taps, Z taps
            d, Hh, dh, C = self.d, self.heads, self.dh, self.C
            q = hd.wq(hd.queries()).double().view(d, Hh, dh)
            wk = hd.k.weight.double().view(Hh, dh, C)
            ws = torch.einsum("hec,jhe->jhc", wk, q) * dh ** -0.5                       # (d, Hh, C): scores
            self.w_sv = torch.cat([ws.reshape(d * Hh, C), hd.v.weight.double()]).to(self.dt)
            self.b_sv = torch.cat([torch.zeros(d * Hh, dtype=torch.float64, device=self.dev),
                                   hd.v.bias.double()]).to(self.dt)
            self.w_fin, self.b_fin = net.w.detach().float().clone(), net.b.detach().float().clone()
            if self.res is not None and hasattr(self.res, "route"):
                self.rmaps = [rt.maps.detach().to(self.dt) for rt in net.res.route]
            if self.troute is not None:
                self.tmaps = [rt.maps.detach().to(self.dt) for rt in net.troute]
            if self.res is not None and hasattr(self.res, "prod"):
                pd = net.res.prod
                q = pd.wq(pd.queries()).double().view(d, Hh, dh)
                ws = torch.einsum("hec,jhe->jhc", pd.k.weight.double().view(Hh, dh, C), q) * dh ** -0.5
                self.w_sv2 = torch.cat([ws.reshape(d * Hh, C), pd.v.weight.double()]).to(self.dt)
                self.b_sv2 = torch.cat([torch.zeros(d * Hh, dtype=torch.float64, device=self.dev),
                                        pd.v.bias.double()]).to(self.dt)
        self.t, self.shape, self.graphs, self.taps = None, None, {}, None
        fn = self._fwd
        if self.compile:
            fn = torch.compile(fn, dynamic=False, mode="max-autotune-no-cudagraphs" if self.compile == 2 else None)
        self.fn = fn

    # ------------------------------------------------------------------------------------------------ per circuit
    @torch.no_grad()
    def circuit(self, layers, prep, meas, noise="uniform", p=0.003):
        """Build one circuit's tables into the static buffers; returns its sizes (m, bytes per shot, n_obs)."""
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # folds in fp32 always
        try:
            return self._circuit(layers, prep, meas, noise, p)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32

    def _circuit(self, layers, prep, meas, noise, p):
        net, hd, dev = self.net, self.net.head, self.dev
        d, H, C, Cb = self.d, self.H, self.C, self.Cb
        s = spec_cl.build(d, layers, prep, meas, p, noise)
        offb = M.offset_bins(d, s["lines"], s["orient"], s["mx"], s["mz"])           # (S, T, K, H W, d)
        cnt = M.part_counts(s["parts"])                                               # (S, T, 3)
        K, T, m, S = s["K"], s["T"], s["m"], s["slots"]
        HW = H * H
        N = K * T * HW
        assert K == 2, "FastCL reads two-block circuits (no pblk / bound of K blocks)"
        assert T != 2 or not hasattr(net, "l1path"), "single-layer circuits: FastCL(net.l1path)"
        assert T == 2 or self.troute is None, "the trunk's routing runs on single-layer circuits only"
        shape = (K, T, S)
        assert self.shape in (None, shape), ("another circuit shape", self.shape, shape)
        f32 = dict(dtype=torch.float32, device=dev)
        inv = np.full(N + 1, -1, np.int64)
        inv[s["flat"]] = np.arange(m)
        det = np.maximum(inv, 0)
        if self.taps is None:
            self.taps = torch.as_tensor(stem_taps(K, T, H, *self.kst), device=dev)    # (N, 75)
            # sites: every grid cell, or (compact) per block and layer the d^2 - 1 cells of the measure qubits (the
            # same cells in every frame) and one cell that never holds a check (its x stays 0, as on the grid); the
            # cells from an idle reference with both check types, since a first circuit may hold one type only
            #
            ref = spec_cl.build(d, ["II"], ("Z", "X"), ("Z", "X"), p, noise)
            occ = (ref["mx"] | ref["mz"]).reshape(-1, HW).any(0) | (s["mx"] | s["mz"]).reshape(K * T, HW).any(0)
            self.cells = (np.concatenate([np.flatnonzero(occ), np.flatnonzero(~occ)[:1]]) if self.compact
                          else np.arange(HW))
            assert not self.compact or len(self.cells) == d * d, ("check cells", int(occ.sum()))
            P_ = len(self.cells)
            sel = (np.arange(K * T)[:, None] * HW + self.cells[None]).reshape(-1)      # site -> grid index
            d2s = np.full(N + 1, P_ - 1)              # grid -> site: a cell without a check -> the first spare cell
            d2s[sel] = np.arange(len(sel))
            d2s[N] = len(sel)                          # outside / none -> the zero row
            self.sel, self.d2s = sel, d2s
            t27 = stem_taps(K, T, H, *net.blocks[0].conv.weight.shape[2:])[sel]
            dense = np.where(np.isin(np.arange(N), sel), d2s[:N], len(sel))
            self.taps27 = torch.as_tensor(d2s[t27], device=dev)                          # (sites, 27) -> site
            self.sel_t = torch.as_tensor(sel, device=dev)
            self.dense_of = torch.as_tensor(dense, device=dev)                           # grid -> site or zero row
        assert (((s["mx"] | s["mz"]).reshape(K * T, HW).any(0) & ~np.isin(np.arange(HW), self.cells)) == 0).all()
        sel, d2s, P_ = self.sel, self.d2s, len(self.cells)
        Ns = len(sel)
        i150 = torch.cat([self.taps, self.taps], 1)
        mx = torch.as_tensor(s["mx"], **f32)                                          # (K, T, H, W)
        mz = torch.as_tensor(s["mz"], **f32)
        on = mx + mz
        typ = torch.cat([mx.reshape(N, 1).expand(N, i150.shape[1] // 2), mz.reshape(N, 1).expand(N, i150.shape[1] // 2)], 1)
        A = np.zeros((N, s["A"].shape[1]), np.float32)
        A[s["flat"]] = s["A"]
        # stem: the constant channels (whole circuit: preparation at t = 0, readout at T - 1), the bias, the role term
        st = net.stem
        fl = torch.zeros(T, **f32)
        ll = torch.zeros(T, **f32)
        fl[0], ll[-1] = 1, 1
        const = torch.stack([mx, mz, on * fl[None, :, None, None], on * ll[None, :, None, None]], 1)   # (K, 4, T, H, W)
        y0 = F.conv3d(const, st.weight[:, 1:5].float(), st.bias.float(), padding=st.padding)   # p channel (pin): 0 at the read p
        gate = torch.as_tensor(s["gate"], **f32)                                      # (K, T, 2, 4)
        if hasattr(net, "flag"):                     # the idle-tick channels (model_cl.sched_flags)
            fg = M.sched_flags(gate[None])[0].float()                                  # (K, 2)
            y0 = y0 + F.conv3d(on[:, None] * fg[:, :, None, None, None], net.flag.weight.float(), None,
                               padding=st.padding)
        y0 = y0.permute(0, 2, 3, 4, 1)
        y0 = y0 + net.role(gate.flatten(2))[:, :, None, None] * on[..., None]
        x0 = y0[..., :C] * mx[..., None] + y0[..., C:] * mz[..., None]               # (K, T, H, W, C)
        # exchange: partners of both sides and slots; per coupling the gate-weighted role maps (0 / 1 weights: exact)
        P = net.partners(torch.as_tensor(s["pcode"], device=dev)[None], T)[0]         # (K, T, side, slot, H W)
        src = P.permute(0, 1, 4, 2, 3).reshape(N, 4).cpu().numpy()[sel]               # (sites, side slot), grid
        g3 = gate[..., :3]
        meff = torch.stack([torch.cat([(g3[:, :, sd, :, None, None] * cp.maps[sd].float()).sum(2) for sd in (0, 1)], 2)
                            .reshape(K * T, 4 * Cb, 2 * Cb) for cp in net.couple]).to(self.dt)
        # read-out: line-bias table per slot, site mask / N0, queries + part counts
        R = hd.R
        ty = torch.as_tensor(s["mz"], device=dev).reshape(K, T, HW).long()
        idx = ty.transpose(0, 1)[None, ..., None] * (2 * R + 2) + torch.as_tensor(offb, device=dev).long()
        bias = hd.b_line.float().reshape(self.heads, -1).t()[idx].permute(0, 4, 5, 2, 1, 3)   # (S, d, Hh, K, T, HW)
        cl = torch.as_tensor(self.cells, device=dev)
        part = (torch.as_tensor(cnt, **f32)[..., None] * hd.part.float()).sum(-2)     # (S, T, C)
        qp = hd.queries().float()[None, :, None] + part[:, None]                      # (S, d, T, C)
        hasd = torch.as_tensor(inv >= 0, **f32)
        site = lambda v: v.reshape(N, -1)[self.sel_t].reshape(K, T, P_, -1)          # noqa: E731 grid -> sites
        t = dict(boff=torch.as_tensor(det // 8, device=dev), sh=torch.as_tensor(det % 8, dtype=torch.uint8, device=dev),
                 has=hasd.to(self.dt), nb=torch.tensor((m + 7) // 8, device=dev), i150=i150[self.sel_t],
                 m150=typ[self.sel_t].to(self.dt), A=torch.as_tensor(A, device=dev), x0=site(x0), mx6=site(mx),
                 mz6=site(mz), on6=site(on), src=torch.as_tensor(d2s[src].reshape(-1), device=dev), meff=meff,
                 bias=bias[..., cl], onh=on.reshape(K, T, HW)[..., cl] / N0, qp=qp)
        if self.res is not None and T == 2:          # the single-layer residual's tables
            res = net.res
            t["meff_r"] = torch.stack([torch.cat([(g3[:, :, sd, :, None, None] * cp.maps[sd].float()).sum(2)
                                                  for sd in (0, 1)], 2).reshape(K * T, 4 * Cb, 2 * Cb)
                                       for cp in res.couple]).to(self.dt)
            if hasattr(res, "route") or self.troute is not None:
                nbr, rel, _ = spec_cl.routes(s, LR.extended_dem(s)[0])
                at = d2s[s["flat"]]                                                 # detector -> site
                ok = nbr < m
                rn = np.full((Ns, nbr.shape[1]), Ns, np.int64)
                rn[at] = np.where(ok, at[np.minimum(nbr, m - 1)], Ns)
                oh = np.zeros((Ns, nbr.shape[1], spec_cl.N_REL), np.float32)
                i_, k_ = np.nonzero(ok)
                oh[at[i_], k_, rel[i_, k_]] = 1
                t["rnbr"] = torch.as_tensor(rn, device=dev)
                t["rw"] = torch.as_tensor(oh / np.maximum(oh.sum(1, keepdims=True), 1), device=dev).to(self.dt)
            if hasattr(res, "prod"):
                pd = res.prod
                b2 = pd.b_line.float().reshape(self.heads, -1).t()[idx].permute(0, 4, 5, 2, 1, 3)
                t["bias2"] = b2[..., cl]
                pt = (torch.as_tensor(cnt, **f32)[..., None] * pd.part.float()).sum((-2, -3))     # (S, C)
                t["qp2"] = pd.queries().float()[None] + pt[:, None]                            # (S, d, C)
        if self.t is None:
            self.t = {k: v.contiguous().clone() for k, v in t.items()}
            self.shape = shape
            self.nbmax = (N + 7) // 8
        else:
            for k, v in t.items():
                self.t[k].copy_(v)
        return dict(m=m, nb=(m + 7) // 8, n_obs=s["n_obs"], S=S)

    # ------------------------------------------------------------------------------------------------ the decode
    def _couple(self, cp, meff, x, t):
        B, K, T, P, C = x.shape
        Cb = self.Cb
        u = cp.down(cp.ln(x)).reshape(B, K * T * P, Cb)
        u = torch.cat([u, u.new_zeros(B, 1, Cb)], 1)
        pair = u[:, t["src"]].view(B, K * T, P, 4 * Cb)
        out = torch.einsum("bgsk,gkn->bgsn", pair, meff).reshape(B, K, T, P, 2 * Cb)
        v = out[..., :Cb] * t["mx6"] + out[..., Cb:] * t["mz6"]
        return x + cp.up(v) * t["on6"]

    def _route(self, rt, maps, x, t):
        B, K, T, P, C = x.shape
        Cb = self.Cb
        u = rt.down(rt.ln(x)).reshape(B, K * T * P, Cb)
        u = torch.cat([u, u.new_zeros(B, 1, Cb)], 1)
        agg = torch.einsum("bskc,skr->bsrc", u[:, t["rnbr"]], t["rw"])
        out = torch.einsum("bsrc,rcn->bsn", agg, maps).reshape(B, K, T, P, 2 * Cb)
        return x + rt.up(out[...,:Cb] * t["mx6"] + out[..., Cb:] * t["mz6"]) * t["on6"]

    def _block(self, blk, x, t):
        B, K, T, P, C = x.shape
        Cb, H = self.Cb, self.H
        u = F.silu(blk.down(blk.ln(x))) * t["on6"]
        u = torch.cat([u.reshape(B, K * T * P, Cb), u.new_zeros(B, 1, Cb)], 1)
        n = B * K * T * P                              # sites of the batch
        conv = self.conv if self.conv >= 0 else 0 if self.bf16 and n > 4096 else 2
        if conv == 2:                                  # the 27 taps of each site gathered, one product per chunk of
            ch = max(1, -(-n * self.taps27.shape[1] * Cb * u.element_size() // 2 ** 31))     # at most 2 GB
            st = -(-B // ch)
            vs = [F.linear(u[b0:b1][:, self.taps27].reshape(b1 - b0, K * T * P, -1), blk.w27, blk.conv.bias)
                  for b0, b1 in zip(range(0, B, st), [*range(st, B, st), B])]
            v = vs[0] if len(vs) == 1 else torch.cat(vs)
        else:                                          # cuDNN on the grid (as torch.compile lowers it)
            ud = u[:, self.dense_of] if self.compact else u[:, :-1]
            v = blk.conv(ud.reshape(B * K, T, H, H, Cb).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
            v = v.reshape(B, K * T * H * H, 2 * Cb)
            v = v[:, self.sel_t] if self.compact else v
        v = v.reshape(B, K, T, P, 2 * Cb)
        v = F.silu(v[..., :Cb] * t["mx6"] + v[..., Cb:] * t["mz6"])
        return x + blk.up(v) * t["on6"]

    def _fwd(self, xin, t):
        K, T, S = self.shape
        d, H, C, Hh, dh = self.d, self.H, self.C, self.heads, self.dh
        P = len(self.cells)
        N = K * T * H * H
        B = xin.shape[0] // self.nbmax
        fw = self.fw
        rows = torch.arange(B, device=xin.device)[:, None] * t["nb"] + t["boff"]
        grid = ((xin[rows] >> t["sh"]) & 1).to(self.dt) * t["has"]                    # (B, N + 1), the last 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            x = F.linear(grid[:, t["i150"]] * t["m150"], self.w_stem).view(B, K, T, P, C) + t["x0"]
            x = self._couple(fw.couple[0], t["meff"][0], x, t)
            for i in range(len(fw.blocks)):
                if self.troute is not None:
                    x = self._route(fw.troute[i], self.tmaps[i], x, t)
                x = self._couple(fw.couple[i + 1], t["meff"][i + 1], self._block(fw.blocks[i], x, t), t)
            res = getattr(fw, "res", None) if T == 2 else None
            if res is not None:                       # the single-layer residual (whole circuits only here)
                for i in range(len(res.blocks)):
                    if hasattr(res, "route"):
                        x = self._route(res.route[i], self.rmaps[i], x, t)
                    x = self._couple(res.couple[i], t["meff_r"][i], self._block(res.blocks[i], x, t), t)
            hd = fw.head
            sv = F.linear(hd.ln(x), self.w_sv, self.b_sv)                             # (B, K, T, P, d Hh + C)
            sc = sv[..., :d * Hh].reshape(B, K, T, P, d, Hh).permute(0, 4, 5, 1, 2, 3)
            v = sv[..., d * Hh:].reshape(B, K, T, P, Hh, dh)
            p = torch.sigmoid(sc.float()[:, None] + t["bias"]) * t["onh"]             # (B, S, d, Hh, K, T, P)
            a = torch.einsum("bajhnts,bntshe->bajthe", p.to(v.dtype), v).reshape(B, S, d, T, C)
            l = hd.mlp(hd.o(a) + t["qp"]).squeeze(-1).float()                         # (B, S, d, T)
            dz = 0
            if res is not None and hasattr(res, "prod"):
                pd = res.prod
                sv = F.linear(pd.ln(x), self.w_sv2, self.b_sv2)
                sc = sv[..., :d * Hh].reshape(B, K, T, P, d, Hh).permute(0, 4, 5, 1, 2, 3)
                v = sv[..., d * Hh:].reshape(B, K, T, P, Hh, dh)
                p = torch.sigmoid(sc.float()[:, None] + t["bias2"]) * t["onh"]
                a = torch.einsum("bajhnts,bntshe->bajhe", p.to(v.dtype), v).reshape(B, S, d, C)
                dz = pd.mlp(pd.o(a) + t["qp2"]).squeeze(-1).float()                     # (B, S, d)
        with torch.autocast("cuda", enabled=False):
            g = torch.remainder(grid[:, :N].float() @ t["A"], 2).view(B, S, d)
            c = ((soft_xor(l) + dz) * (1.0 - 2.0 * g) * self.w_fin).sum(-1) + self.b_fin
        return c, c > 0

    # ------------------------------------------------------------------------------------------------ graphs
    @torch.no_grad()
    def graph(self, B):
        """The CUDA graph of one batch size."""
        if B not in self.graphs:
            xin = torch.zeros(B * self.nbmax, dtype=torch.uint8, device=self.dev)
            prev = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = bool(self.bench)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.fn(xin, self.t)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                c, dec = self.fn(xin, self.t)
            torch.backends.cudnn.benchmark = prev
            self.graphs[B] = (g, xin, c, dec)
        return self.graphs[B]

    def caller(self, circ, hin, B):
        """call(s0): shots s0.. s0 + B of hin (pinned (n, bytes per shot) uint8) -> decisions in pinned host memory."""
        g, xin, _, dec = self.graph(B)
        x2 = xin[:B * circ["nb"]].view(B, circ["nb"])
        hout = torch.zeros(dec.shape, dtype=torch.bool).pin_memory()
        self.hout = hout

        def call(s0):
            x2.copy_(hin[s0:s0 + B], non_blocking=True)
            g.replay()
            hout.copy_(dec, non_blocking=True)
            torch.cuda.synchronize()
        return call

    def decode(self, circ, det, B=1024):
        """All shots of det (numpy (n, bytes per shot) uint8) of the loaded circuit -> c (n, n_obs), decisions."""
        n = len(det)
        cs = []
        for s0 in range(0, n, B):
            b = min(B, n - s0)
            g, xin, c, _ = self.graph(B)
            xb = np.zeros((B, circ["nb"]), np.uint8)
            xb[:b] = det[s0:s0 + b]
            xin[:B * circ["nb"]].view(B, circ["nb"]).copy_(torch.from_numpy(xb))
            g.replay()
            cs.append(c[:b, :circ["n_obs"]].cpu().numpy())
        c = np.concatenate(cs)
        return c, c > 0
