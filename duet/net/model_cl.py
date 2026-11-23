"""The Clifford network NetCL: NetRO's trunk, exchange and per-layer
read-out generalised to the circuits of duet.clifford_circuits. NetRO (duet/net/model_ro.py) is not touched: the switch is
the class (duet.net.train_cl builds NetCL; every earlier trainer and reader builds NetRO). Changes, all per shot (the
tables of duet.net.spec_cl, gathered by pattern):
  input each block in its own frame on the (2, T, H, W) grid of check cells, H = W = d + 1; the detector -> site
            map and the X / Z site masks come per shot, so a cell's type follows the H gates.
  stem NetRO's: the bit channel and the constant channels (X, Z, first layer, last layer) through the 3 x 5 x 5
            convolution; the role flags per (block, layer) are side x {control, target, S, H} (8 instead of 4).
  trunk model.Block unchanged (its masks broadcast over the batch).
  exchange CoupleCL: model.Couple with the partner chosen per (block, layer, side): the other block at the same cell
            (a CNOT in equal frames), at R90 / R270 of the cell (a CNOT across frames: the rotated pairing), or the same
            block at the S partner cell (the fold's mirror exchange); maps per side and role (control, target, S).
  read-out one LayerLines head shared by the product slots (--slots, 2 now): per layer t and representative j, the
            score bias of a site is b_line[type, offset bin] with the offset to the nearest line of its block's part
            of the site's type (the other part when the block holds one only; the outside bin when none: spec_cl
            offsets); the query gets n_X v_X + n_Z v_Z + n_Y v_Y, n_P the number of blocks whose part of the product
            at layer t is P (one learned vector per part, added once per block, so no weight depends on the number of
            blocks; format 2: the 16-row table of the support code X_A + 2 Z_A + 4 X_B + 8 Z_B was format 1); soft XOR
            over t; c_k = sum_j w_j (-1)^g_{k,j} z_{k,j} + b with w, b shared by the slots. Slots without a product are
            masked.
Input: bits (B, m), gidx (B, m) grid index of each detector, mx / mz (B, K, T, H, W), gate (B, K, T, 2, 4),
pcode (B, K, T, 2), offb (B, S, T, K, H W, d) offset bins, cnt (B, S, T, 3) part counts, g (B, S, d), S product slots
(at least two), and for K blocks pblk (B, K, T, 2) partner blocks and bound (B, K, T, 4) the physical
preparation and readout flags per block and layer (spec_cl); without pblk and bound the network reads two blocks
whole circuits, as before. Output c (B, S), z (B, S, d), l (B, S, d, T).
res > 0: a residual path (ResL1) that runs on whole single-layer circuits only (T = 2;
whole_l1) between the trunk and the read-out: res blocks of (RouteCL, the routing over the circuit's error model, from
nbr / rel (B, m, cap) of spec_cl.routes, with route,) model.Block and CoupleCL, and with prod the product head
(ProdHead); it starts with zero output, and at every other depth it is not run.
l1path: a second NetCL, the single-layer path, that decodes whole single-layer
circuits (whole_l1) in place of the network, which keeps every other input (deeper circuits, cut windows): a copy of
the stem, role projection, trunk, exchange, read-out, w and b with its own residual (res, route, prod), and optionally
troute (a RouteCL before every trunk block, zero output at the start) and flags (two input channels per block from the
layer's schedule, sched_flags, through stem weights that start at zero: a separate convolution, so the copy's own stem
sums stay as they were). load_grown fills both paths from a checkpoint without the path."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from duet.net import spec_cl
from duet.net.model import N0, Block
from duet.net.model_ro import AMAX, FLIP0, OUT0, soft_xor

N_ROLE = 4            # control, target, S, H (H has no exchange)
N_SLOT = 2
P_REF = 0.003         # pin = 1: the p channel carries log(p / P_REF), 0 at the read p


def offset_bins(d, lines, orient, mx, mz):
    """(S, T, K, H W, d) int8 offset bins per product slot and block (numpy; spec_cl tables of one circuit). R = 2d - 1;
    bin R + off, 2R + 1 = outside (no part on the block)."""
    H = d + 1
    R = 2 * d - 1
    T = lines.shape[1]
    cx = np.arange(H * H) // H * 2            # cell x (even coordinates)
    cy = np.arange(H * H) % H * 2
    S, K = lines.shape[0], lines.shape[2]                     # product slots, blocks
    out = np.full((S, T, K, H * H, d), 2 * R + 1, np.int8)
    ty = np.where(mz.reshape(K, T, H * H) > 0, 1, 0)          # site type (X where no detector: unused)
    pos = 2 * np.arange(d) + 1
    for k in range(S):
        for t in range(T):
            for b in range(K):
                have = [orient[k, t, b, p] >= 0 for p in (0, 1)]
                if not any(have):
                    continue
                for part_site in (0, 1):
                    p = part_site if have[part_site] else 1 - part_site
                    o = orient[k, t, b, p]
                    c = cx if o == 1 else cy                     # columns: offset along x; rows: along y
                    m = lines[k, t, b, p]                        # (d,) masks per representative
                    sel = ty[b, t] == part_site
                    for j in range(d):
                        ls = pos[[(int(m[j]) >> kk) & 1 == 1 for kk in range(d)]]
                        if len(ls) == 0:
                            continue
                        diff = c[sel][:, None] - ls[None, :]
                        off = diff[np.arange(len(diff)), np.abs(diff).argmin(1)]
                        out[k, t, b, sel, j] = (off + R).astype(np.int8)
    return out


def part_counts(parts):
    """parts (slots, T, blocks, [X, Z]) -> (slots, T, 3) uint8: how many blocks hold an X, a Z or a Y (both) part."""
    x, z = parts[..., 0].astype(bool), parts[..., 1].astype(bool)
    return np.stack([(x & ~z).sum(-1), (~x & z).sum(-1), (x & z).sum(-1)], -1).astype(np.uint8)


class CoupleCL(nn.Module):
    def __init__(self, C, Cb):
        super().__init__()
        self.Cb = Cb
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb, bias=False)
        self.maps = nn.Parameter(torch.randn(2, 3, 2 * Cb, 2 * Cb) * (2 * Cb) ** -0.5)   # [side, role]
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, gate, src, mx, mz):
        """src (B, K, T, side, slot, H W): flat partner index into the block-layer-cell grid, K T H W = none."""
        B, nb, T, H, W, C = x.shape
        u = self.down(self.ln(x)).reshape(B, nb * T * H * W, self.Cb)
        u = torch.cat([u, u.new_zeros(B, 1, self.Cb)], 1)
        out = 0
        for side in (0, 1):
            pair = torch.cat([torch.gather(u, 1, src[:, :, :, side, o].reshape(B, -1, 1).expand(-1, -1, self.Cb))
                              for o in (0, 1)], -1).view(B, nb, T, H * W, 2 * self.Cb)
            for role in range(3):
                out = out + gate[:, :, :, side, role, None, None].to(u.dtype) * (pair @ self.maps[side, role])
        out = out.reshape(B, nb, T, H, W, 2 * self.Cb)
        v = out[..., :self.Cb] * mx + out[..., self.Cb:] * mz
        return x + self.up(v) * (mx + mz)


class LayerLinesCL(nn.Module):
    lookup = "onehot"     # the line-bias table: "onehot" (a one-hot product, the default) or "gather" (an index gather:
    # the same values in the forward pass and far less memory; its backward sums in another order)
    per_block = False     # (couple = "none"): the read-out per block, joined by soft XOR

    def __init__(self, C, heads, d, mlp=2):
        super().__init__()
        self.n, self.h, self.dh, self.R = d, heads, C // heads, 2 * d - 1
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.e = nn.Parameter(torch.zeros(d, C))
        bins = torch.arange(-self.R, self.R + 1, dtype=torch.float32)
        b0 = torch.cat([-1.0 - 0.5 * bins.abs(), torch.tensor([OUT0])])
        self.b_line = nn.Parameter(b0.repeat(heads, 2, 1))                          # (H, type, 2R + 2)
        self.part = nn.Parameter(torch.zeros(3, C))                                 # v_X, v_Z, v_Y
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        nn.init.constant_(self.mlp[-1].bias, FLIP0)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        if prefix + "emb" in state:
            raise RuntimeError("NetCL checkpoint of format /1 (16-row support-code table): the per-block "
                               "sum of format 2 has no exact conversion from it; retrain, or read it with the code of "
                               "commit f6cce86 or earlier")
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def queries(self):
        return self.q0 + self.e

    def kv(self, f):
        """f (B, 2, T, S, C) -> keys and values, shared by the product slots."""
        B, nb, T, S, C = f.shape
        y = self.ln(f)
        return self.k(y).view(B, nb, T, S, self.h, self.dh), self.v(y).view(B, nb, T, S, self.h, self.dh)

    def forward(self, kv, ty, on, offb, cnt, feats=False):
        """kv from kv(f); ty, on (B, 2, T, S); offb (B, T, 2, S, d) bins; cnt (B, T, 3) -> l (B, d, T); feats: also the
        features u (B, d, T, C) the flip logits come from."""
        k, v = kv
        B, nb, T, S, H, dh = k.shape
        n, C = self.n, H * dh
        q = self.wq(self.queries()).view(n, H, dh)
        sc = torch.einsum("bntshe,jhe->bjhnts", k, q) * dh ** -0.5                  # (B, n, H, nb, T, S)
        idx = ty.transpose(1, 2)[..., None] * (2 * self.R + 2) + offb.long()         # (B, T, nb, S, d)
        # table look-up as a one-hot product: indexing's backward (index_put) was a fifth of the step
        nbin = 2 * (2 * self.R + 2)
        if self.lookup == "gather":       # the same values without the one-hot tensor
            with torch.autocast(idx.device.type, enabled=False):
                bias = F.embedding(idx, self.b_line.reshape(H, nbin).t())           # (B, T, nb, S, d, H)
        else:
            oh = torch.zeros(idx.shape + (nbin,), device=idx.device).scatter_(-1, idx[..., None], 1.0)
            with torch.autocast(idx.device.type, enabled=False):
                bias = oh @ self.b_line.reshape(H, nbin).t()                         # (B, T, nb, S, d, H)
        bias = bias.permute(0, 4, 5, 2, 1, 3)                                        # (B, d, H, nb, T, S)
        p = torch.sigmoid(sc.float() + bias) / N0 * on[:, None, None]
        if self.per_block:                #, couple = "none": each block read alone, the blocks' flip
            # logits joined by the parity only (soft XOR); a block without a part of the
            # product at the layer (all its offset bins outside) is no flip
            assert not feats
            a = torch.einsum("bjhnts,bntshe->bjnthe", p.to(v.dtype), v).reshape(B, n, nb, T, C)
            u = self.queries()[None, :, None, None] + self.o(a) + \
                (cnt.float()[..., None] * self.part).sum(-2)[:, None, None]
            lb = self.mlp(u).squeeze(-1).float()                                         # (B, d, nb, T)
            inv = (offb != 2 * self.R + 1).any(3).permute(0, 3, 2, 1)                    # (B, d, nb, T)
            lb = torch.where(inv, lb, torch.full_like(lb, -AMAX))
            return soft_xor(lb.transpose(2, 3))                                          # (B, d, T)
        a = torch.einsum("bjhnts,bntshe->bjthe", p.to(v.dtype), v).reshape(B, n, T, C)
        u = self.queries()[None, :, None] + self.o(a) + (cnt.float()[..., None] * self.part).sum(-2)[:, None]
        l = self.mlp(u).squeeze(-1).float()
        return (l, u) if feats else l


class PoolCL(nn.Module):
    """pooled read-out (head = "pool"; one
    query without line bias, c_P = z_P) for NetCL: per product slot one query q0 (no representatives, no line offsets,
    no per-layer logits), one attention pooling over every site of every layer and block, the score bias per head,
    site type and support (whether the site's block holds a part of the product at that layer: its offset bins are not
    the outside bin), the query plus the product's part counts averaged over the layers (v_X, v_Z, v_Y as
    LayerLinesCL); the product's flip logit c = mlp(u) directly. Trained on BCE(c, y) alone (loss_cl with z = None)."""
    lookup = "onehot"     # unused (train_cl sets it on every head)

    def __init__(self, C, heads, d, mlp=2):
        super().__init__()
        self.h, self.dh, self.out_bin = heads, C // heads, 2 * (2 * d - 1) + 1
        self.ln = nn.LayerNorm(C)
        self.k = nn.Linear(C, C, bias=False)
        self.wq = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        self.q0 = nn.Parameter(torch.randn(C) * 0.02)
        self.b_sup = nn.Parameter(torch.tensor([-1.0, OUT0]).repeat(heads, 2, 1))       # (H, type, [in, out])
        self.part = nn.Parameter(torch.zeros(3, C))
        self.mlp = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, mlp * C), nn.SiLU(), nn.Linear(mlp * C, 1))
        nn.init.constant_(self.mlp[-1].bias, FLIP0)

    def kv(self, f):
        B, nb, T, S, C = f.shape
        y = self.ln(f)
        return self.k(y).view(B, nb, T, S, self.h, self.dh), self.v(y).view(B, nb, T, S, self.h, self.dh)

    def forward(self, kv, ty, on, offb, cnt):
        """kv from kv(f); ty, on (B, K, T, S); offb (B, T, K, S, d) bins; cnt (B, T, 3) -> c (B,)."""
        k, v = kv
        B, nb, T, S, H, dh = k.shape
        q = self.wq(self.q0).view(H, dh)
        sc = torch.einsum("bntshe,he->bhnts", k, q) * dh ** -0.5                       # (B, H, K, T, S)
        idx = ty.transpose(1, 2).long() * 2 + (offb[..., 0] == self.out_bin).long()    # (B, T, K, S)
        with torch.autocast(idx.device.type, enabled=False):
            bias = F.embedding(idx, self.b_sup.reshape(H, 4).t())                      # (B, T, K, S, H)
        p = torch.sigmoid(sc.float() + bias.permute(0, 4, 2, 1, 3)) / N0 * on[:, None]
        a = torch.einsum("bhnts,bntshe->bhe", p.to(v.dtype), v).reshape(B, H * dh)
        u = self.q0 + self.o(a) + (cnt.float().mean(1)[..., None] * self.part).sum(-2)
        return self.mlp(u).squeeze(-1).float()


class RouteCL(nn.Module):
    """Message passing over the circuit's error model: every detector gathers the
    detectors that share an error mechanism with it (spec_cl.routes), takes the mean of their bottleneck features per
    relation (same or other block, slice and check type), maps each relation's mean into the receiver's X- or Z-site
    half and adds the sum through up, which starts at zero (as CoupleCL)."""

    def __init__(self, C, Cb, n_rel=spec_cl.N_REL):
        super().__init__()
        self.Cb, self.n_rel = Cb, n_rel
        self.ln = nn.LayerNorm(C)
        self.down = nn.Linear(C, Cb, bias=False)
        self.maps = nn.Parameter(torch.randn(n_rel, Cb, 2 * Cb) * Cb ** -0.5)
        self.up = nn.Linear(Cb, C)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, gidx, nbr, rel, mz):
        """x (B, K, T, H, W, C); gidx (B, m) the site of each detector; nbr (B, m, cap) neighbour detectors (m: none);
        rel (B, m, cap) their relations (n_rel: none); mz (B or 1, K, T, H, W, 1) the Z-site mask."""
        B, C, Cb = x.shape[0], x.shape[-1], self.Cb
        m, cap = nbr.shape[1:]
        xf = x.reshape(B, -1, C)
        u = self.down(self.ln(torch.gather(xf, 1, gidx[..., None].expand(-1, -1, C))))        # (B, m, Cb)
        u = torch.cat([u, u.new_zeros(B, 1, Cb)], 1)
        g = torch.gather(u, 1, nbr.reshape(B, -1, 1).expand(-1, -1, Cb)).view(B, m, cap, Cb)
        oh = F.one_hot(rel.long(), self.n_rel + 1)[..., :self.n_rel].to(u.dtype)             # (B, m, cap, R)
        agg = torch.einsum("bmkr,bmkc->bmrc", oh, g) / oh.sum(2).clamp(min=1)[..., None]
        out = torch.einsum("bmrc,rcn->bmn", agg, self.maps.to(agg.dtype))                       # (B, m, 2 Cb)
        z = torch.gather(mz.expand(B, *mz.shape[1:]).reshape(B, -1), 1, gidx)[..., None].to(out.dtype)
        y = self.up(out[..., :Cb] * (1 - z) + out[..., Cb:] * z)
        return xf.scatter_add(1, gidx[..., None].expand(-1, -1, C), y.to(xf.dtype)).view(x.shape)


class ProdHead(LayerLinesCL):
    """The direct product head of the single-layer residual: per product slot and
    representative j a correction dz_j of the flip logit z_j from one attention pooling over the sites of both time
    slices of the routed features (LayerLinesCL's, with each site's own line-offset bias, summed over the slices
    instead of read per slice) and the part counts of both slices; z_j + dz_j enters c as z_j does. Loaded as a copy of
    the read-out (load_grown) with its output layer at zero."""

    def __init__(self, C, heads, d, mlp=2):
        super().__init__(C, heads, d, mlp)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, kv, ty, on, offb, cnt, feats=False):
        """kv from kv(f); ty, on (B, 2, T, S); offb (B, T, 2, S, d); cnt (B, T, 3) -> dz (B, d)."""
        k, v = kv
        B, nb, T, S, H, dh = k.shape
        n, C, nbin = self.n, H * dh, 2 * (2 * self.R + 2)
        q = self.wq(self.queries()).view(n, H, dh)
        sc = torch.einsum("bntshe,jhe->bjhnts", k, q) * dh ** -0.5
        idx = ty.transpose(1, 2)[..., None] * (2 * self.R + 2) + offb.long()
        with torch.autocast(idx.device.type, enabled=False):
            bias = F.embedding(idx, self.b_line.reshape(H, nbin).t()).permute(0, 4, 5, 2, 1, 3)
        p = torch.sigmoid(sc.float() + bias) / N0 * on[:, None, None]
        a = torch.einsum("bjhnts,bntshe->bjhe", p.to(v.dtype), v).reshape(B, n, C)
        u = self.queries()[None] + self.o(a) + (cnt.float()[..., None] * self.part).sum((-2, -3))[:, None]
        return self.mlp(u).squeeze(-1).float()


class ResL1(nn.Module):
    """The residual path of whole single-layer circuits: blocks of (RouteCL with
    route,) model.Block and CoupleCL with the trunk's routing, reading the trunk's features, and with prod the product
    head; every output projection starts at zero, so the network starts as the one it is loaded from."""

    def __init__(self, C, Cb, d, heads, mlp, blocks, block_k, route, prod):
        super().__init__()
        self.blocks = nn.ModuleList(Block(C, Cb, tuple(block_k)) for _ in range(blocks))
        self.couple = nn.ModuleList(CoupleCL(C, Cb) for _ in range(blocks))
        if route:
            self.route = nn.ModuleList(RouteCL(C, Cb) for _ in range(blocks))
        if prod:
            self.prod = ProdHead(C, heads, d, mlp)

    def forward(self, x, args, gidx, nbr, rel):
        for i, (blk, cp) in enumerate(zip(self.blocks, self.couple)):
            if hasattr(self, "route"):
                x = self.route[i](x, gidx, nbr, rel, args[3])
            x = cp(blk(x, args[2], args[3]), *args)
        return x


def sched_flags(gate):
    """(B, K, 2) bool per block of a single-layer circuit (gate (B, K, T, 2, 4), its layer at slice 0, side 0; roles
    control, target, S, H), from duet.clifford_circuits's schedule: [0] the block idles in the layer's gate tick (the
    tick before the round that the CNOTs and H gates share, here on other blocks), [1] it idles in a fold tick (the
    extra tick inside the round of an S on another block)."""
    g = gate[:, :, 0, 0].bool()
    tick, fold = g[..., [0, 1, 3]].any(-1), g[..., 2]
    return torch.stack([tick.any(1, keepdim=True) & ~tick, fold.any(1, keepdim=True) & ~fold], -1)


def whole_l1(bound, B, T, device):
    """(B,) bool: the shot is a whole single-layer circuit (T = 2; per block physical preparation at slice 0 and
    physical readout at slice 1, no cut); bound None: a whole circuit."""
    if T != 2:
        return torch.zeros(B, dtype=torch.bool, device=device)
    if bound is None:
        return torch.ones(B, dtype=torch.bool, device=device)
    return (bound[:, :, 0, 0] == 1).all(1) & (bound[:, :, 1, 1] == 1).all(1) & (bound[..., 2:] == 0).flatten(1).all(1)


class NetCL(nn.Module):
    def __init__(self, d, width=64, bott=16, blocks=1, heads=4, stem_k=(3, 5, 5), block_k=(3, 3, 3), mlp=2,
                 slots=N_SLOT, joint=0, pin=0, res=0, route=0, prod=0, troute=0, flags=0, l1path=None,
                 couple="cross", head="lines"):
        super().__init__()
        self.d, self.C, self.H, self.slots, self.pin = d, width, d + 1, slots, pin
        # ablations (data/training/TRAIN.md): couple = "cross" (the network), "late" (no cross-block exchange
        # in the trunk: a CNOT's exchange reads the site's own cell in its own block, the same maps and parameters;
        # the joint read-out over all blocks is the only place blocks meet) or "none" (the trunk as "late" and the
        # read-out per block, the blocks' flip logits joined by soft XOR: no exchange anywhere); head = "lines" (the
        assert couple in ("cross", "late", "none") and head in ("lines", "pool"), (couple, head)
        assert (couple, head) == ("cross", "lines") or not (joint or res or troute or flags or l1path or pin), \
            "the ablation options alone"
        assert not (couple == "none" and head == "pool"), "the pooled read-out is joint"
        self.couple_mode, self.head_mode = couple, head
        self.cfg = dict(d=d, width=width, bott=bott, blocks=blocks, heads=heads, stem_k=list(stem_k),
                        block_k=list(block_k), mlp=mlp, slots=slots, **({"joint": joint} if joint else {}),
                        **({"pin": pin} if pin else {}), **(dict(res=res, route=route, prod=prod) if res else {}),
                        **({"troute": troute} if troute else {}), **({"flags": flags} if flags else {}),
                        **({"l1path": dict(l1path)} if l1path else {}),
                        **({"couple": couple} if couple != "cross" else {}), **({"head": head} if head != "lines" else {}))
        assert res or not (route or prod), "route and prod are parts of the residual (res blocks)"
        assert not (l1path and (res or troute or flags or joint or pin)), "the single-layer path holds these"
        g90, g270 = spec_cl.grid_maps(d)
        mir = spec_cl.mirror_maps(d)
        # partner cell per pcode (0 none, 1 same cell, 2 R90, 3 R270, 4 / 5 mirror in frame 0 / 1; -1 none) and
        # whether the partner is the other block
        cells = np.stack([np.full_like(g90, -1), np.arange(len(g90)), g90, g270, mir[0], mir[1]])
        self.register_buffer("cellmap", torch.as_tensor(cells, dtype=torch.long), persistent=False)
        self.register_buffer("bflip", torch.tensor([0, 1, 1, 1, 0, 0]), persistent=False)
        self.register_buffer("toff", torch.tensor([[-1, 0], [0, 1]]), persistent=False)   # [side, slot]
        # pin = 1: a sixth input channel log(p / P_REF) at every detector site
        self.stem = nn.Conv3d(5 + pin, 2 * width, stem_k, padding=tuple(v // 2 for v in stem_k))
        self.role = nn.Linear(2 * N_ROLE, 2 * width, bias=False)
        self.blocks = nn.ModuleList(Block(width, bott, tuple(block_k)) for _ in range(blocks))
        self.couple = nn.ModuleList(CoupleCL(width, bott) for _ in range(blocks + 1))
        self.head = PoolCL(width, heads, d, mlp) if head == "pool" else LayerLinesCL(width, heads, d, mlp)
        if couple == "none":
            self.head.per_block = True
        self.w = nn.Parameter(torch.full((d,), 1.0 / d))
        self.b = nn.Parameter(torch.zeros(()))
        if joint:     #: a joint read-out of the two time slices of a single-layer circuit,
            # z = soft XOR of the slices' flip logits + an MLP of both slices' features; zero output at start
            self.joint = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, mlp * width), nn.SiLU(),
                                       nn.Linear(mlp * width, 1))
            nn.init.zeros_(self.joint[-1].weight)
            nn.init.zeros_(self.joint[-1].bias)
        if res:       #: the residual path of whole single-layer circuits
            self.res = ResL1(width, bott, d, heads, mlp, res, block_k, route, prod)
        if troute:    #: routing over the error model before every trunk block
            self.troute = nn.ModuleList(RouteCL(width, bott) for _ in range(blocks))
        if flags:     # C: the idle-tick channels' stem weights (sched_flags), zero at the start
            self.flag = nn.Conv3d(2, 2 * width, stem_k, padding=tuple(v // 2 for v in stem_k), bias=False)
            nn.init.zeros_(self.flag.weight)
        if l1path:    #: the single-layer path, a NetCL of the same shape: l1path's
            # blocks, when given, its own trunk depth
            lp = dict(l1path)
            self.l1path = NetCL(d, width, bott, lp.pop("blocks", blocks), heads, stem_k, block_k, mlp, slots, **lp)

    def partners(self, pcode, T, pblk=None):
        """pcode (B, 2, T, side) -> (B, 2, T, side, slot, H W) flat index of each site's partner in the slot's layer
        (side 0: layers t - 1, t; side 1: t, t + 1); 2 T H W where there is none."""
        HW = self.H * self.H
        pc = pcode.long()
        cnot = (pc >= 1) & (pc <= 3)
        if self.couple_mode != "cross":   #: a CNOT's partner is the site's own cell in its own block
            cell = self.cellmap[torch.where(cnot, torch.ones_like(pc), pc)][:, :, :, :, None]
        else:
            cell = self.cellmap[pc][:, :, :, :, None]                                      # (B, 2, T, 2, 1, HW)
        if pblk is None:                  # two blocks: the partner of codes 1-3 is the other block
            blk = torch.arange(2, device=pc.device)[None, :, None, None] ^ self.bflip[pc]     # (B, 2, T, 2)
        else:
            blk = pblk.long()                                                               # (B, K, T, 2)
        if self.couple_mode != "cross":
            blk = torch.where(cnot, torch.arange(pc.shape[1], device=pc.device)[None, :, None, None], blk)
        t = torch.arange(T, device=pc.device)[:, None, None] + self.toff                  # (T, 2, 2)
        idx = ((blk[..., None] * T + t.clamp(0, T - 1)) * HW)[..., None] + cell
        bad = (cell < 0) | ((t < 0) | (t >= T))[..., None]
        return torch.where(bad, pc.shape[1] * T * HW, idx)

    def features(self, bits, gidx, mx, mz, gate, pcode, pblk=None, bound=None, lp=None, res=False, nbr=None, rel=None):
        """lp (B,) log(p / P_REF) per shot, read when pin = 1; None: p = P_REF (the channel is zero). res: also the
        couplings' arguments (gate, partners, X- and Z-site masks). nbr, rel: the routing tables (troute)."""
        B = bits.shape[0]
        T, H, C = mx.shape[2], self.H, self.C
        st = self.stem
        K = mx.shape[1]
        grid = torch.zeros(B, K * T * H * H, dtype=st.weight.dtype, device=bits.device)
        grid.scatter_(1, gidx, bits.to(st.weight.dtype))
        if bound is None:                 # a whole circuit: preparation at layer 0, readout at the last layer
            fl = torch.zeros(T, device=bits.device, dtype=st.weight.dtype)
            ll = torch.zeros_like(fl)
            fl[0], ll[-1] = 1, 1
            fl, ll = fl[None, None, :, None, None], ll[None, None, :, None, None]
        else:                             # per block: physical preparation and readout (cuts leave them 0)
            bd = bound.to(st.weight.dtype)
            fl, ll = bd[..., 0, None, None], bd[..., 1, None, None]
        mxf, mzf = mx.to(st.weight.dtype), mz.to(st.weight.dtype)
        on = mxf + mzf
        ch = [mxf, mzf, on * fl, on * ll]
        if self.pin and lp is not None:
            ch.append(on * lp.to(st.weight.dtype)[:, None, None, None, None])
        const = torch.stack(ch, 2)
        y = F.conv3d(grid.view(B * K, 1, T, H, H), st.weight[:, :1], None, padding=st.padding) \
            + F.conv3d(const.reshape(B * K, len(ch), T, H, H), st.weight[:, 1:1 + len(ch)], st.bias, padding=st.padding)
        if hasattr(self, "flag"):         # the block's idle ticks, broadcast to its detector sites (every slice)
            fch = on[:, :, None] * sched_flags(gate).to(on.dtype)[..., None, None, None]          # (B, K, 2, T, H, W)
            y = y + F.conv3d(fch.reshape(B * K, 2, T, H, H), self.flag.weight, None, padding=st.padding)
        y = y.view(B, K, 2 * C, T, H, H).permute(0, 1, 3, 4, 5, 2)
        mx6, mz6 = mxf[..., None], mzf[..., None]
        y = y + self.role(gate.flatten(3).to(y.dtype))[:, :, :, None, None] * (mx6 + mz6)
        x = y[..., :C] * mx6 + y[..., C:] * mz6
        args = (gate, self.partners(pcode, T, pblk), mx6, mz6)
        x = self.couple[0](x, *args)
        assert nbr is not None or not hasattr(self, "troute"), "the trunk's routing needs nbr and rel"
        for i, (blk, cp) in enumerate(zip(self.blocks, self.couple[1:])):
            if hasattr(self, "troute"):
                x = self.troute[i](x, gidx, nbr, rel, mz6)
            x = cp(blk(x, mx6, mz6), *args)
        return (x, args) if res else x

    def forward(self, bits, gidx, mx, mz, gate, pcode, offb, cnt, g, pblk=None, bound=None, lp=None, nbr=None,
                rel=None):
        """nbr, rel (B, m, cap): spec_cl.routes of single-layer circuits, read by the residual's routing (res, route).
        The residual runs at T = 2 only (the Python branch: every other depth computes exactly what the network without
        it computes); per shot it replaces the outputs of whole single-layer circuits (whole_l1). The single-layer path
        (l1path) likewise: run at T = 2 only, its outputs replace the network's on whole single-layer circuits, and the
        network itself runs only when a shot of the batch is not one (one host sync on bound)."""
        if hasattr(self, "l1path") and mx.shape[2] == 2:
            out = self.l1path(bits, gidx, mx, mz, gate, pcode, offb, cnt, g, pblk, bound, lp, nbr, rel)
            if bound is None:
                return out
            keep = whole_l1(bound, bits.shape[0], 2, bits.device)
            if bool(keep.all()):
                return out
            own = self._forward(bits, gidx, mx, mz, gate, pcode, offb, cnt, g, pblk, bound, lp, nbr, rel)
            return tuple(torch.where(keep.view(-1, *[1] * (u.dim() - 1)), u, v) for u, v in zip(out, own))
        return self._forward(bits, gidx, mx, mz, gate, pcode, offb, cnt, g, pblk, bound, lp, nbr, rel)

    def _forward(self, bits, gidx, mx, mz, gate, pcode, offb, cnt, g, pblk, bound, lp, nbr, rel):
        S = offb.shape[1]                 # product slots: from the inputs (at least two; K blocks up to K)
        assert cnt.shape[1] == g.shape[1] == S >= self.slots, "inputs for another number of product slots"
        l1 = hasattr(self, "res") and mx.shape[2] == 2
        x = self.features(bits, gidx, mx, mz, gate, pcode, pblk, bound, lp, res=l1, nbr=nbr, rel=rel)
        if l1:
            x, args = x
            keep = whole_l1(bound, x.shape[0], 2, x.device)
            assert nbr is not None or not hasattr(self.res, "route"), "the residual's routing needs nbr and rel"
            x = torch.where(keep[:, None, None, None, None, None], self.res(x, args, gidx, nbr, rel), x)
        B, nb, T, H, W, C = x.shape
        f = x.reshape(B, nb, T, H * W, C)
        ty = mz.reshape(B, nb, T, H * W).long()
        on = (mx + mz).reshape(B, nb, T, H * W).to(f.dtype)
        kv = self.head.kv(f)
        if self.head_mode == "pool":      #: the product's flip logit directly, no z or per-layer logits
            return torch.stack([self.head(kv, ty, on, offb[:, k], cnt[:, k]) for k in range(S)], 1), None, None
        if T == 2 and hasattr(self, "joint"):
            lu = [self.head(kv, ty, on, offb[:, k], cnt[:, k], feats=True) for k in range(S)]
            l = torch.stack([v[0] for v in lu], 1)                                          # (B, S, d, T)
            u = torch.stack([v[1] for v in lu], 1)                                          # (B, S, d, T, C)
            z = soft_xor(l) + self.joint(u.flatten(-2)).squeeze(-1).float()
        else:
            l = torch.stack([self.head(kv, ty, on, offb[:, k], cnt[:, k]) for k in range(S)], 1)  # (B, S, d, T)
            z = soft_xor(l)
        if l1 and hasattr(self.res, "prod"):
            pd = self.res.prod
            kvp = pd.kv(f)
            dz = torch.stack([pd(kvp, ty, on, offb[:, k], cnt[:, k]) for k in range(S)], 1)       # (B, S, d)
            z = torch.where(keep[:, None, None], z + dz, z)
        c = (z * (1.0 - 2.0 * g.float()) * self.w).sum(-1) + self.b
        return c, z, l


def load_grown(net, ck):
    """Load NetCL checkpoint ck (dict with cfg and model) into net, which may have more trunk blocks, a wider trunk
    bottleneck (bott) or the p channel (pin) where ck has none, the rest of the configuration equal, so that net computes the checkpoint's function. The checkpoint's blocks and couplings take
    the first places; added ones keep their initialisation, whose output projections are zero (model.Block, CoupleCL).
    A wider bottleneck Cb0 -> Cb1: the old channels keep their weights (the X- and Z-site halves of Block.conv's outputs
    and CoupleCL.maps' rows (slots) and columns (X, Z) at [0, Cb0) and [Cb1, Cb1 + Cb0)); the new channels' input side
    keeps its initialisation (down's rows, the conv's and maps' new outputs) and their output side starts at zero (up's
    new columns, the conv's taps and the maps' rows from new channels to old ones). The p channel's stem weights start
    at zero. A single-layer residual added to a checkpoint without one keeps its
    initialisation (zero output), except the product head, which starts as a copy of the read-out with its output layer
    at zero. Trunk routing (troute) and the idle-tick channels (flags) added keep their initialisation (zero output). A
    single-layer path added to a checkpoint without one: the network takes the checkpoint's
    weights outside its residual and the path all of them, so that both compute the checkpoint's function where they
    run. A path with options (the p channel) that the checkpoint's path lacks, or with more trunk
    blocks of its own: the network takes the checkpoint's weights and the path loads
    its own as above, the added blocks, couplings and routing at their zero-output start. Returns the number of added
    blocks (the path's, when it grows)."""
    if "l1path" in net.cfg and "l1path" in ck["cfg"] and net.cfg["l1path"] != ck["cfg"]["l1path"]:
        base = {k: v for k, v in ck["cfg"].items() if k != "l1path"}
        assert base == {k: v for k, v in net.cfg.items() if k != "l1path"}, "checkpoint of another network"
        sub = dict(cfg=dict(base, **ck["cfg"]["l1path"]),
                   model={k[7:]: v for k, v in ck["model"].items() if k.startswith("l1path.")})
        added = load_grown(net.l1path, sub)
        assert added == 0 or "blocks" in net.cfg["l1path"], "the single-layer path copies the trunk: equal blocks"
        sd = net.state_dict()
        assert all(k in sd for k in ck["model"]), "keys the network does not have"
        sd.update({k: v for k, v in ck["model"].items() if not k.startswith("l1path.")})
        net.load_state_dict(sd)
        return added
    if "l1path" in net.cfg and "l1path" not in ck["cfg"]:
        base = {k: v for k, v in ck["cfg"].items() if k not in ("res", "route", "prod")}
        assert base == {k: v for k, v in net.cfg.items() if k != "l1path"}, "checkpoint of another network"
        assert load_grown(net.l1path, ck) == 0, "the single-layer path copies the trunk: equal blocks"
        sd = net.state_dict()
        sd.update({k: v for k, v in ck["model"].items() if not k.startswith("res.")})
        net.load_state_dict(sd)
        return 0
    cfg, mine = dict(ck["cfg"]), dict(net.cfg)
    t0, t1 = cfg.pop("troute", 0), mine.pop("troute", 0)
    f0, f1 = cfg.pop("flags", 0), mine.pop("flags", 0)
    assert t0 in (0, t1) and f0 in (0, f1), "checkpoint of another network"
    b0, b1 = cfg.pop("blocks"), mine.pop("blocks")
    c0, c1 = cfg.pop("bott"), mine.pop("bott")
    j0, j1 = cfg.pop("joint", 0), mine.pop("joint", 0)          # a joint read-out added to a checkpoint without one
    p0, p1 = cfg.pop("pin", 0), mine.pop("pin", 0)
    r0, r1 = (tuple(c.pop(k, 0) for k in ("res", "route", "prod")) for c in (cfg, mine))
    assert cfg == mine and b1 >= b0 and c1 >= c0 and j0 in (0, j1) and p0 in (0, p1) and r0 in ((0, 0, 0), r1), \
        "checkpoint of another network"
    sd = net.state_dict()
    assert set(ck["model"]) <= set(sd), "keys the network does not have"
    o = torch.cat([torch.arange(c0), c1 + torch.arange(c0)])     # old (first half, second half) channels in the new
    with torch.no_grad():
        for k, v in ck["model"].items():
            t = sd[k].clone()
            v, o = v.to(t.device), o.to(t.device)
            if t.shape == v.shape:
                t = v
            elif k == "stem.weight":                             # the p channel: zero
                t.zero_()
                t[:, :v.shape[1]] = v
            elif k.endswith(("down.weight", "down.bias")):        # new rows: initialisation
                t[:c0] = v
            elif k.endswith("up.weight"):                         # new columns: zero
                t.zero_()
                t[:, :c0] = v
            elif k.endswith("conv.weight"):                       # old outputs: old taps, zero from new inputs
                t[o] = 0
                t[o, :c0] = v
            elif k.endswith("conv.bias"):
                t[o] = v
            elif k.endswith("maps"):                              # old columns: old rows, zero from new rows
                t[:, :, :, o] = 0
                t[:, :, o[:, None], o[None, :]] = v
            else:
                raise AssertionError(f"{k}: {tuple(v.shape)} into {tuple(t.shape)}")
            sd[k] = t
    net.load_state_dict(sd)
    if r1[2] and not r0[2]:
        with torch.no_grad():
            net.res.prod.load_state_dict(net.head.state_dict())
            nn.init.zeros_(net.res.prod.mlp[-1].weight)
            nn.init.zeros_(net.res.prod.mlp[-1].bias)
    return b1 - b0


def loss_cl(c, z, l, y, yl, u, slot, lam):
    """Masked over the product slots: BCE(c, y) + mean_j BCE(z, yl) + lam mean_{j,t} BCE(l, u), each summed over slots
    and averaged over the batch."""
    s = slot.float()
    fin = (F.binary_cross_entropy_with_logits(c.float(), y.float(), reduction="none") * s).sum(1).mean()
    if z is None:                         # pooled read-out (PoolCL): the final BCE alone
        return fin, fin, torch.zeros((), device=fin.device)
    fin = fin + (F.binary_cross_entropy_with_logits(z.float(), yl.float(), reduction="none").mean(2) * s).sum(1).mean()
    lay = (F.binary_cross_entropy_with_logits((l if lam else l.detach()).float(), u.float(),
                                              reduction="none").mean((2, 3)) * s).sum(1).mean()
    return (fin + lam * lay if lam else fin), fin, lay


def teacher_bce(c, t, tm):
    """The teacher's loss: BCE(c, t) over the slots tm marks, summed over slots and averaged over
    the batch (loss_cl's reduction; rows without a teacher count as zero)."""
    return (F.binary_cross_entropy_with_logits(c.float(), t.float(), reduction="none") * tm.float()).sum(1).mean()
