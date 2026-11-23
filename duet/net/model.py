"""The two-block network: CoDecoder's ConvLines per block with shared weights, a gate coupling between the blocks and
one joint read-out head per logical measurement product.

Guang, Yuliang Xiong, Xun Gao and Yueqi Chen; Apache License 2.0, see third_party/CoDecoder). Block and LineQueries
keep CoDecoder's computation; changed: the two blocks are stacked on a block axis with per-block site masks, the
geometry comes from duet.net.geometry instead of a grid file, gate-role flags enter the stem, the gate coupling is new,
LineQueries takes its line offsets, layers and types as arrays, the heads are per product over the sites of the blocks
a product involves, the GF(2) signs are an input, a pooled fallback head is added and the deploy tables are dropped.

Input per shot: detector bits (B, m) in the circuit's order; the gate table of its pattern (B, 2, T, 2, 2)
(geometry.gate_table: [block, layer, gate before / after the layer, control / target]); g (B, n_obs, d) =
D A mod 2 (from the labels in training, from A at inference).
  stem CoDecoder's type-specific 3 x 5 x 5 convolution of the detector bit and the constant channels (X type, Z
            type, first layer, last layer), plus the block's four gate-role flags at the site's layer through a
            1 x 1 x 1 tap: the flags are constant over a layer, so a full kernel would add compute, not information.
  blocks CoDecoder's pre-norm bottleneck residual blocks, the same weights for both blocks.
  coupling after the stem and after every block. For each gate between layers r and r + 1, a site on layer r or r + 1
            receives linear maps of its partner block's features at the same (i, j) on layers r and r + 1. The map
            depends on the receiver's side of the gate (layer before or after), its role (control or target) and its
            site type, so Z sites and X sites both exchange in both directions between control and target. Form:
            x += up(sum over the gates beside the layer of [u(r), u(r + 1)] M[side, role], type-selected), u =
            down(LN(partner)), `up` zero-initialised: the untrained network is two independent CoDecoders.
            blocks up to the read-out) or "late".
  read-out per product P, CoDecoder's LineQueries over the sites of the blocks P involves: query j = representative
            j; score bias by the site's signed distance to line j of its own block, its layer and type; z_{P,j};
            c_P = sum_j w_{P,j} (-1)^{g_{P,j}} z_{P,j} + b_P. head = "pool": one query without line bias, c_P = z_P
            (the plan's fallback: a joint pooled head). couple = "late": the read-out of every product covers the
            sites of both blocks; a site of a block P does not involve takes the line offset of P's site at the same
            (i, j) (a transversal CNOT pairs identical layouts), and a learned score bias per block and head
            (zero-initialised) is added.
Output c (B, n_obs), z (B, n_obs, d or 1); decision c > 0, confidence min over P of |c_P|."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N0 = 64.0                                                # the read-out's constant sum normalisation (CoDecoder)


class Block(nn.Module):
    def __init__(self, C, Cb, k):
        super().__init__()
        self.Cb = Cb
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb)
        self.conv = nn.Conv3d(Cb, 2 * Cb, k, padding=tuple(v // 2 for v in k))     # out: X-site weights, Z-site weights
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, mx, mz):                       # x (B, 2, T, H, W, C); mx / mz (2, T, H, W, 1)
        on = mx + mz
        u = F.silu(self.down(self.ln(x))) * on
        B, nb, T, H, W, Cb = u.shape
        v = self.conv(u.reshape(B * nb, T, H, W, Cb).permute(0, 4, 1, 2, 3))
        v = v.permute(0, 2, 3, 4, 1).reshape(B, nb, T, H, W, 2 * Cb)
        v = F.silu(v[..., :Cb] * mx + v[..., Cb:] * mz)
        return x + self.up(v) * on


class Couple(nn.Module):
    """The gate coupling (see the module docstring)."""

    def __init__(self, C, Cb, cross=True):
        super().__init__()
        self.Cb, self.cross = Cb, cross
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb, bias=False)
        self.maps = nn.Parameter(torch.randn(2, 2, 2 * Cb, 2 * Cb) * (2 * Cb) ** -0.5)   # [side, role]
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, gate, mx, mz):                 # gate (B, 2, T, 2, 2)
        u = self.down(self.ln(x))                       # (B, 2, T, H, W, Cb)
        if self.cross:
            u = u.flip(1)                               # the partner block's features
        zero = torch.zeros_like(u[:, :, :1])
        prev = torch.cat([zero, u[:, :, :-1]], 2)
        nxt = torch.cat([u[:, :, 1:], zero], 2)
        out = 0
        for side, s in enumerate((torch.cat([prev, u], -1), torch.cat([u, nxt], -1))):   # gate before / after t
            for role in (0, 1):
                out = out + gate[:, :, :, side, role, None, None, None] * (s @ self.maps[side, role])
        v = out[..., :self.Cb] * mx + out[..., self.Cb:] * mz
        return x + self.up(v) * (mx + mz)


class LineQueries(nn.Module):
    """CoDecoder's line read-out over site features (B, s, C) -> logits (B, n): off (s, n) the signed distance of each
    site to line j in half-line units, t (s,) its layer, ty (s,) its type (0 X, 1 Z), T the number of layers."""

    def __init__(self, C, heads, off, t, ty, T, mlp=2, blk=None):
        super().__init__()
        n = off.shape[1]
        self.n, self.h, self.dh = n, heads, C // heads
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(n, C))
        R = int(np.abs(off).max())
        bins = torch.arange(-R, R + 1, dtype=torch.float32)
        self.b_line = nn.Parameter((-1.0 - 0.5 * bins.abs()).repeat(heads, 2, 1))  # (H, type, 2R + 1): near the line
        self.b_t = nn.Parameter(torch.zeros(heads, 2, T))                          # (H, type, T)
        self.register_buffer("off", torch.as_tensor(off + R, dtype=torch.long), persistent=False)       # (s, n)
        self.register_buffer("tt", torch.as_tensor(t, dtype=torch.long), persistent=False)
        self.register_buffer("ty", torch.as_tensor(ty, dtype=torch.long), persistent=False)
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        self.b_blk = None                                 # blk (s,) the site's block: a score bias per (head, block)
        if blk is not None:
            self.b_blk = nn.Parameter(torch.zeros(heads, 2))
            self.register_buffer("bk", torch.as_tensor(blk, dtype=torch.long), persistent=False)

    def bias(self):                                       # (s, n, H)
        bl = self.b_line[:, self.ty[:, None], self.off]   # (H, s, n)
        bt = self.b_t[:, self.ty, self.tt]                # (H, s)
        if self.b_blk is not None:
            bt = bt + self.b_blk[:, self.bk]
        return (bl + bt[:, :, None]).permute(1, 2, 0)

    def queries(self):                                    # (n, C)
        return self.q0 + self.e

    def forward(self, f):
        B, s, C = f.shape
        n, H, dh = self.n, self.h, self.dh
        y = self.ln(f)
        q = self.wq(self.queries()).view(n, H, dh)
        k = self.k(y).view(B, s, H, dh)
        v = self.v(y).view(B, s, H, dh)
        sc = torch.einsum("bmhe,jhe->bjhm", k, q) * dh ** -0.5 + self.bias().permute(1, 2, 0)[None].to(k.dtype)
        p = (torch.sigmoid(sc.float()) / N0).to(v.dtype)
        a = torch.einsum("bjhm,bmhe->bjhe", p, v).reshape(B, n, C)
        u = self.queries() + self.o(a)
        return self.mlp(u).squeeze(-1)


class Net(nn.Module):
    def __init__(self, spec, width=64, bott=16, blocks=1, heads=4, stem_k=(3, 5, 5), block_k=(3, 3, 3), mlp=2,
                 head="lines", couple="cross"):
        super().__init__()
        assert head in ("lines", "pool") and couple in ("cross", "self", "none", "late")
        d, m, T, H, W = (spec[k] for k in ("d", "m", "T", "H", "W"))
        self.d, self.m, self.C, self.n_obs, self.shape, self.head = d, m, width, spec["n_obs"], (T, H, W), head
        self.couple_mode = couple
        self.cfg = dict(width=width, bott=bott, blocks=blocks, heads=heads, stem_k=list(stem_k),
                        block_k=list(block_k), mlp=mlp, head=head)
        blk, tij, typ = spec["block"], spec["tij"], spec["type"]
        flat = ((blk * T + tij[:, 0]) * H + tij[:, 1]) * W + tij[:, 2]          # detector -> (block, t, i, j) grid
        n = 2 * T * H * W
        gidx = np.full(n, m, np.int64)
        gidx[flat] = np.arange(m)
        mx, mz, fl, ll = (np.zeros(n, np.float32) for _ in range(4))
        mx[flat[typ == 0]] = 1
        mz[flat[typ == 1]] = 1
        fl[flat[tij[:, 0] == 0]] = 1
        ll[flat[tij[:, 0] == T - 1]] = 1
        self.register_buffer("gidx", torch.as_tensor(gidx), persistent=False)
        self.register_buffer("mx", torch.as_tensor(mx).view(2, T, H, W, 1), persistent=False)
        self.register_buffer("mz", torch.as_tensor(mz).view(2, T, H, W, 1), persistent=False)
        const = np.stack([mx, mz, fl, ll]).reshape(4, 2, T, H, W).transpose(1, 0, 2, 3, 4)
        self.register_buffer("const", torch.as_tensor(np.ascontiguousarray(const)), persistent=False)  # (2, 4, T, H, W)
        self.stem = nn.Conv3d(5, 2 * width, stem_k, padding=tuple(v // 2 for v in stem_k))
        self.role = nn.Linear(4, 2 * width, bias=False)
        self.blocks = nn.ModuleList(Block(width, bott, tuple(block_k)) for _ in range(blocks))
        n_cp = 0 if couple == "none" else blocks + 1
        self.couple = nn.ModuleList(Couple(width, bott, couple == "cross") for _ in range(n_cp))
        self.heads = nn.ModuleList()
        late = couple == "late"
        for k, P in enumerate(spec["products"]):
            sel = np.arange(m) if late else np.flatnonzero(np.isin(blk, P["blocks"]))   # the detectors it reads
            off = both_off(P, blk, tij) if late else P["off"]
            off = off[sel] if head == "lines" else np.zeros((len(sel), 1), np.int64)
            self.heads.append(LineQueries(width, heads, off, tij[sel, 0], typ[sel], T, mlp, blk[sel] if late else None))
            self.register_buffer(f"sel{k}", torch.as_tensor(flat[sel]), persistent=False)
        nl = d if head == "lines" else 1
        self.w = nn.Parameter(torch.full((self.n_obs, nl), 1.0 / nl))
        self.b = nn.Parameter(torch.zeros(self.n_obs))

    def features(self, bits, gate):                      # (B, m), (B, 2, T, 2, 2) -> (B, 2 T H W, C)
        B = bits.shape[0]
        T, H, W = self.shape
        C = self.C
        pad = torch.cat([bits.to(self.mx.dtype), bits.new_zeros(B, 1, dtype=self.mx.dtype)], 1)
        grid = pad[:, self.gidx].view(B * 2, 1, T, H, W)
        st = self.stem
        y = F.conv3d(grid.to(st.weight.dtype), st.weight[:, :1], None, padding=st.padding).view(B, 2, 2 * C, T, H, W) \
            + F.conv3d(self.const, st.weight[:, 1:], st.bias, padding=st.padding)[None]
        y = y.permute(0, 1, 3, 4, 5, 2)                  # (B, 2, T, H, W, 2C)
        y = y + self.role(gate.flatten(3))[:, :, :, None, None] * (self.mx + self.mz)
        x = y[..., :C] * self.mx + y[..., C:] * self.mz
        if self.couple:
            x = self.couple[0](x, gate, self.mx, self.mz)
        for k, blk in enumerate(self.blocks):
            x = blk(x, self.mx, self.mz)
            if self.couple:
                x = self.couple[k + 1](x, gate, self.mx, self.mz)
        return x.reshape(B, 2 * T * H * W, C)

    def forward(self, bits, gate, g):                    # g (B, n_obs, d) 0/1 -> c (B, n_obs), z (B, n_obs, d | 1)
        f = self.features(bits, gate)
        zs = [hd(f[:, getattr(self, f"sel{k}")]) for k, hd in enumerate(self.heads)]
        z = torch.stack(zs, 1).float()
        if self.head == "lines":
            c = (z * (1.0 - 2.0 * g.float()) * self.w).sum(-1) + self.b
        else:
            c = z[..., 0]
        return c, z


def both_off(P, blk, tij):
    """Product P's line offsets (m, d) for the detectors of both blocks: a detector of a block P does not involve takes
    the offset of P's detectors at the same (i, j) (late fusion; a transversal CNOT pairs identical layouts)."""
    off = P["off"].copy()
    own = np.isin(blk, P["blocks"])
    key = tij[:, 1] * (tij[:, 2].max() + 1) + tij[:, 2]
    tab = np.full((key.max() + 1, off.shape[1]), np.iinfo(np.int64).min)
    tab[key[own]] = off[own]
    assert np.array_equal(tab[key[own]], off[own]), "an offset depends on more than the site's (i, j)"
    off[~own] = tab[key[~own]]
    assert (off[~own] != np.iinfo(np.int64).min).all(), "a site of the other block has no counterpart"
    return off


def loss_fn(c, z, y, yl, head="lines"):
    """sum over P of [BCE(c_P, y_P) + mean_j BCE(z_{P,j}, y_{P,j})], each averaged over the batch."""
    loss = F.binary_cross_entropy_with_logits(c.float(), y.float(), reduction="none").mean(0).sum()
    if head == "lines":
        loss = loss + F.binary_cross_entropy_with_logits(z.float(), yl.float(), reduction="none").mean((0, 2)).sum()
    return loss


def gf2(bits, A):
    """(D A) mod 2 as float: bits (B, m) 0/1, A (m, K) float32 on the same device (exact: integer sums in fp32)."""
    with torch.autocast(bits.device.type, enabled=False):
        return torch.remainder(bits.float() @ A, 2.0)


def unpack(packed, n):
    """Stim's little-endian bit-packed rows (B, ceil(n / 8)) uint8 (a tensor on any device) -> (B, n) uint8."""
    sh = torch.arange(8, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> sh) & 1).reshape(packed.shape[0], -1)[:, :n]


def n_params(model):
    return sum(p.numel() for p in model.parameters())
