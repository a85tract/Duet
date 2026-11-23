"""The network of first design step: the trunk (the pilot's stem,
residual blocks and gate coupling on a per-depth grid of T = L + 1 layers) with a read-out whose only dependence on the
depth is the number of terms of a soft XOR. Reuses model.Block, model.Couple, N0, loss_fn, gf2 and unpack.

Read-out, per product P, line j and layer t = 0.. L: one logit l_{P,t,j} for u_{P,t,j}, the flip of P's representative
j at the end caused by the errors assigned to layer t (duet.net.labels_ro). The head is the pilot's line queries over the
sites of both blocks on layer t only (sigmoid-weighted sum per layer with the normaliser N0; a layer has the same sites
at every depth): score bias b_line[type, offset to line j] for sites of the blocks in r_P(t), the support of P's
representative back-propagated to layer t (an input per shot), and one learned bias (initially OUT0, about off) for
sites of the other block, as model4's support bias; an embedding of r_P(t) is added to the query before the MLP, whose
last bias starts at FLIP0 (no flip). No time bias. Combination: z_{P,j} = soft_xor over t of l_{P,t,j} (fp32, log
space); c_P = sum_j w_{P,j} (-1)^{g_{P,j}} z_{P,j} + b_P as the pilot.

Input as model_vd.NetVD plus sup (B, n_obs, T, 2) 0/1, r_P(t) per shot (labels_ro.supports of its pattern): bits (B, m_L),
gate (B, 2, T, 2, 2), g (B, n_obs, d), L. Output c (B, n_obs), z (B, n_obs, d), l (B, n_obs, d, T)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from duet.net.model import N0, Block, Couple, gf2, loss_fn, n_params, unpack   # noqa: F401 (re-exported)

FLIP0 = -4.0         # the per-layer logits' initial value (no flip): model4's xor initialisation
OUT0 = -6.0          # the initial score bias of a site outside the support (about off): model4's support bias
AMAX = 80.0          # |l| is clamped here in the parity (exp(-80) is a normal fp32 number)


def _log1mexp(a):
    """log(1 - exp(-a)) for a > 0, accurate at both ends (Maechler)."""
    lo, hi = a.clamp(max=0.6931), a.clamp(min=0.6931)     # both branches finite, so where has finite gradients
    return torch.where(a < 0.6931, torch.log(-torch.expm1(-lo)), torch.log1p(-torch.exp(-hi)))


def soft_xor(l):
    """The logit of the parity of independent bits with flip logits l (..., T) -> (...): tanh(-z/2) = prod_t tanh(-l_t/2),
    in fp32 and in log space: |z| = 2 atanh(exp(S)), S = sum_t log tanh(|l_t|/2), the sign from the parity of the layers
    with l_t > 0. Exact (to fp32) up to |z| of about AMAX, with a gradient there (a clamp of the product at 1 - 1e-6
    would stop it at |z| = 14.5)."""
    with torch.autocast(l.device.type, enabled=False):
        l = l.float()
        a = l.abs().clamp(1e-6, AMAX)
        S = (_log1mexp(a) - torch.log1p(torch.exp(-a))).sum(-1).clamp(max=-1e-37)
        mag = torch.log1p(torch.exp(S)) - _log1mexp(-S)
        odd = torch.remainder((l > 0).sum(-1), 2) == 1
        return torch.where(odd, mag, -mag)


def soft_xor_prod(l):
    """model4's product form (reference for tests): -2 atanh(clamp(prod tanh(-l/2)))."""
    s = torch.tanh(-l.double() / 2).prod(-1)
    return -2 * torch.atanh(s.clamp(-1 + 1e-15, 1 - 1e-15))


class _GeoRO(nn.Module):
    """One depth's tables (non-persistent buffers): detector -> grid map, masks and constant channels (as model_vd);
    the S site cells (i, j) shared by both blocks and all layers, with per (block, layer, cell) the detector's presence
    and type; per product the offset bin of each cell to each line (the same for both blocks: identical layouts)."""

    def __init__(self, spec, R=None):
        super().__init__()
        blk, tij, typ = spec["block"], spec["tij"], spec["type"]
        T, H, W, m, d = (spec[k] for k in ("T", "H", "W", "m", "d"))
        self.T, self.m = T, m
        flat = ((blk * T + tij[:, 0]) * H + tij[:, 1]) * W + tij[:, 2]
        n = 2 * T * H * W
        gidx = np.full(n, m, np.int64)
        gidx[flat] = np.arange(m)
        mx, mz, fl, ll = (np.zeros(n, np.float32) for _ in range(4))
        mx[flat[typ == 0]] = 1
        mz[flat[typ == 1]] = 1
        fl[flat[tij[:, 0] == 0]] = 1
        ll[flat[tij[:, 0] == T - 1]] = 1
        reg = lambda k, v: self.register_buffer(k, torch.as_tensor(v), persistent=False)   # noqa: E731
        reg("gidx", gidx)
        reg("mx", mx.reshape(2, T, H, W, 1))
        reg("mz", mz.reshape(2, T, H, W, 1))
        reg("const", np.ascontiguousarray(np.stack([mx, mz, fl, ll]).reshape(4, 2, T, H, W).transpose(1, 0, 2, 3, 4)))
        cells = sorted({(int(i), int(j)) for i, j in tij[:, 1:]})
        cid = {c: s for s, c in enumerate(cells)}
        S = len(cells)
        on, ty = np.zeros((2, T, S), np.float32), np.zeros((2, T, S), np.int64)
        cs = np.array([cid[(int(i), int(j))] for i, j in tij[:, 1:]])
        on[blk, tij[:, 0], cs] = 1
        ty[blk, tij[:, 0], cs] = typ
        reg("hw", np.array([i * W + j for i, j in cells], np.int64))
        reg("on", on)
        reg("ty", ty)
        self.R = []
        for k, P in enumerate(spec["products"]):
            row = {}
            for det in np.flatnonzero(np.isin(blk, P["blocks"])):
                o = tuple(P["off"][det].tolist())
                assert row.setdefault(cs[det], o) == o, "a cell's line offsets differ between its detectors"
            assert len(row) == S, "a product's blocks do not cover every site cell"
            off = np.array([row[s] for s in range(S)], np.int64)                  # (S, d)
            r = int(np.abs(off).max())
            if R is not None:            # the network's bins: a depth whose cells reach fewer offsets
                assert r <= R[k], f"product {k}: line offsets up to {r}, the network's bins up to {R[k]}"
                r = R[k]
            self.R.append(r)
            reg(f"off{k}", off + self.R[-1])


class LayerLines(nn.Module):
    """The pilot's line queries with one logit per line and layer (module docstring)."""

    def __init__(self, C, heads, n, R, mlp=2):
        super().__init__()
        self.n, self.h, self.dh, self.R = n, heads, C // heads, R
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(n, C))
        bins = torch.arange(-R, R + 1, dtype=torch.float32)
        b0 = torch.cat([-1.0 - 0.5 * bins.abs(), torch.tensor([OUT0])])            # last bin: outside the support
        self.b_line = nn.Parameter(b0.repeat(heads, 2, 1))                          # (H, type, 2R + 2)
        self.emb = nn.Parameter(torch.zeros(4, C))                                  # r_P(t) as A + 2 B (0: unused)
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        nn.init.constant_(self.mlp[-1].bias, FLIP0)

    def queries(self):
        return self.q0 + self.e

    def forward(self, f, ty, off, on, sup):
        """f (B, 2, T, S, C) the site cells' features; ty (2, T, S); off (S, n) offset bins 0.. 2R; on (2, T, S);
        sup (B, T, 2) 0/1 = r_P(t) -> l (B, n, T)."""
        B, nb, T, S, C = f.shape
        n, H, dh = self.n, self.h, self.dh
        y = self.ln(f)
        q = self.wq(self.queries()).view(n, H, dh)
        k = self.k(y).view(B, nb, T, S, H, dh)
        v = self.v(y).view(B, nb, T, S, H, dh)
        sc = torch.einsum("bntshe,jhe->bjhnts", k, q) * dh ** -0.5                  # (B, n, H, nb, T, S)
        b_in = self.b_line[:, ty[..., None], off[None, None]].permute(4, 0, 1, 2, 3)  # (n, H, nb, T, S)
        b_out = self.b_line[:, ty, 2 * self.R + 1]                                    # (H, nb, T, S)
        ins = sup.transpose(1, 2).to(b_out.dtype)[:, None, None, :, :, None]          # (B, 1, 1, nb, T, 1)
        bias = b_out + ins * (b_in - b_out)
        p = torch.sigmoid(sc.float() + bias) / N0 * on
        a = torch.einsum("bjhnts,bntshe->bjthe", p.to(v.dtype), v).reshape(B, n, T, C)
        code = (sup[..., 0] + 2 * sup[..., 1]).long()                                 # (B, T)
        u = self.queries()[None, :, None] + self.o(a) + self.emb[code][:, None]
        return self.mlp(u).squeeze(-1).float()


class NetRO(nn.Module):
    def __init__(self, specs, width=64, bott=16, blocks=1, heads=4, stem_k=(3, 5, 5), block_k=(3, 3, 3), mlp=2,
                 head="lines", R=None):
        """specs: {L: geometry.analyse of any pattern of depth L} for every depth the network will see. R: per product
        the line-offset range of the heads' bias bins (default: the depths' own, which must agree); a trained network
        passes its own, since at L = 1 the cells reach offsets up to 2d - 3 only, against 2d - 1 at every L >= 2."""
        super().__init__()
        assert head == "lines", "the per-layer read-out has line queries only"
        s0 = next(iter(specs.values()))
        d, H, W, n_obs = (s0[k] for k in ("d", "H", "W", "n_obs"))
        assert all((s["d"], s["H"], s["W"], s["n_obs"]) == (d, H, W, n_obs) for s in specs.values())
        self.d, self.C, self.n_obs, self.HW, self.head = d, width, n_obs, (H, W), head
        self.cfg = dict(width=width, bott=bott, blocks=blocks, heads=heads, stem_k=list(stem_k),
                        block_k=list(block_k), mlp=mlp, head=head)
        self.geo = nn.ModuleDict({str(L): _GeoRO(s, R) for L, s in specs.items()})
        R = {tuple(g.R) for g in self.geo.values()}
        assert len(R) == 1, f"the line offsets differ between depths: {R}"
        self.stem = nn.Conv3d(5, 2 * width, stem_k, padding=tuple(v // 2 for v in stem_k))
        self.role = nn.Linear(4, 2 * width, bias=False)
        self.blocks = nn.ModuleList(Block(width, bott, tuple(block_k)) for _ in range(blocks))
        self.couple = nn.ModuleList(Couple(width, bott) for _ in range(blocks + 1))
        self.heads = nn.ModuleList(LayerLines(width, heads, d, r, mlp) for r in R.pop())
        self.w = nn.Parameter(torch.full((n_obs, d), 1.0 / d))
        self.b = nn.Parameter(torch.zeros(n_obs))

    def depths(self):
        return sorted(int(L) for L in self.geo)

    def features(self, bits, gate, geo):                 # (B, m_L), (B, 2, T, 2, 2) -> (B, 2, T, H, W, C)
        B = bits.shape[0]
        T, (H, W), C = geo.T, self.HW, self.C
        pad = torch.cat([bits.to(geo.mx.dtype), bits.new_zeros(B, 1, dtype=geo.mx.dtype)], 1)
        grid = pad[:, geo.gidx].view(B * 2, 1, T, H, W)
        st = self.stem
        y = F.conv3d(grid.to(st.weight.dtype), st.weight[:, :1], None, padding=st.padding).view(B, 2, 2 * C, T, H, W) \
            + F.conv3d(geo.const, st.weight[:, 1:], st.bias, padding=st.padding)[None]
        y = y.permute(0, 1, 3, 4, 5, 2)
        y = y + self.role(gate.flatten(3))[:, :, :, None, None] * (geo.mx + geo.mz)
        x = y[..., :C] * geo.mx + y[..., C:] * geo.mz
        x = self.couple[0](x, gate, geo.mx, geo.mz)
        for blk, cp in zip(self.blocks, self.couple[1:]):
            x = cp(blk(x, geo.mx, geo.mz), gate, geo.mx, geo.mz)
        return x

    def layer_logits(self, bits, gate, L, sup):
        geo = self.geo[str(L)]
        x = self.features(bits, gate, geo)
        B, nb, T, H, W, C = x.shape
        f = x.reshape(B, nb, T, H * W, C)[:, :, :, geo.hw]                          # (B, 2, T, S, C)
        return torch.stack([hd(f, geo.ty, getattr(geo, f"off{k}"), geo.on, sup[:, k])
                            for k, hd in enumerate(self.heads)], 1)                   # (B, n_obs, d, T)

    def forward(self, bits, gate, g, L, sup):
        l = self.layer_logits(bits, gate, L, sup)
        z = soft_xor(l)
        c = (z * (1.0 - 2.0 * g.float()) * self.w).sum(-1) + self.b
        return c, z, l


def loss_ro(c, z, l, y, yl, u, lam):
    """(loss, final part, per-layer part): the pilot's loss (products and lines) plus lam * sum_P mean_{j,t}
    BCE(l_{P,t,j}, u_{P,t,j}); with lam = 0 the per-layer part is computed without a gradient (logged only)."""
    fin = loss_fn(c, z, y, yl, "lines")
    lay = F.binary_cross_entropy_with_logits((l if lam else l.detach()).float(), u.float(),
                                             reduction="none").mean((0, 2, 3)).sum()
    return (fin + lam * lay if lam else fin), fin, lay
