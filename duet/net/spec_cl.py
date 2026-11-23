"""Per-circuit tables of the Clifford network: everything NetCL and its trainer need for one circuit of duet.clifford_circuits on K
blocks (K = len(prep)), as numpy arrays. Imports no torch (sampler processes fork early).

build(d, layers, prep, meas, p, noise, products=None, slots=None) -> dict:
  m, T, H, W, K detectors, layers t = 0.. L (L the final data measurement), grid (d + 1)^2 of check cells, blocks
  n_obs, slots reliable products (the targets), product slots S = max(2, K, n_obs) unless given: at least two,
                      so two-block circuits keep their arithmetic; slots past n_obs are empty
  targets per product its measurement events (block, time, reset instance): (b, L, 0) is the final
                      measurement of block b after its first preparation; the target count is separate from K
  flat (m,) detector -> index in the (K, T, H, W) grid: (block, round, cell); each block in its own frame
  mx, mz (K, T, H, W) the check type of every site that holds a detector (frames: a cell flips type at an H)
  gate (K, T, 2, 4) [block, layer, side (0 the gate before the layer, 1 the gate after), role]: roles control, target,
                      S, H; layer l's gate sits between detector layers l - 1 and l, as a CNOT does in today's networks
  pcode (K, T, 2) the exchange partner of a site at that side: 0 none, 1 the partner block at the same cell, 2 the
                      partner block at R90(cell) (receiver in frame 1), 3 at R270(cell) (receiver in frame 0), 4 / 5 the
                      same block at the S partner cell (frame 0 / 1; mirror tables below)
  pblk (K, T, 2) the partner's block: the other block of the layer's CNOT pair for codes 1-3, the block itself
                      otherwise (two blocks: b XOR 1 for codes 1-3, NetCL's rule before K blocks)
  bound (K, T, 4) per block and detector layer [physical preparation before round t, physical readout at t,
                      artificial cut before t, artificial cut after t]: a whole circuit has preparation at t = 0,
                      readout at t = L and no cuts; windows set the cut flags
  parts (S, T, K, 2) per product slot: [layer, block, X part, Z part] of its logical back-propagated to the start of
                      round t (duet.clifford_circuits.observable_parts)
  lines (S, T, K, 2, d) uint16 bit masks and orient (S, T, K, 2) int8 (0 rows y = 2k + 1, 1 columns x = 2k + 1, -1
                      none): the data lines that representative j occupies on each block and part, back-propagated
                      through the transversal gates (H keeps them, a CNOT copies them through its pairing, the fold
                      adds the mirrored lines to the Z part)
  A (m, S d) representative j of product k = the final data records of line j (lines and orient at t = L: row
                      y = 2j + 1 or column x = 2j + 1 in the final frame) on every block of the product, NetRO's rule, so line j's labels are its own flips; column k d + j holds the
                      detectors whose records XOR to the representative's and the product's, solved over GF(2) on all
                      records as geometry does (they span the time columns of the checks between line j and the
                      product's line, duet.clifford_circuits.line0): y_{k,j} = y_k + (D A)_{k d + j} (mod 2). Where such
                      a column starts at a random first value (uniform model, a check of the other type at
                      preparation), the representative also reads that first record (never in CNOT-only circuits or
                      under the Lukin model, whose noiseless encoder fixes every first-round check)
  labelled the circuit with one observable per representative (n_obs + k d + j), for labels_ro.extended_dem
products: the reliable-product basis to use (duet.clifford_circuits.make_circuit), e.g. carried through a relabelling.
Static tables per d: grid_maps(d) -> (g90, g270) cell gathers; mirror_maps(d) -> (2, H W) S partner cells (-1 none).
routes(s, dem) -> per detector the detectors that share an error mechanism with it."""
import functools

import numpy as np
import stim

from duet import clifford_circuits as cc
from duet.circuits import block_offset, rotated_patch
from duet.net.geometry import solve_gf2

ROLES = ("control", "target", "S", "H")


def cell_index(d, xy):
    x, y = xy
    return (x // 2) * (d + 1) + (y // 2)


@functools.lru_cache(maxsize=None)
def grid_maps(d):
    """(g90, g270) (H W,) int64: the cell a receiver at cell q reads, R90(q) and R270(q)."""
    H = d + 1
    g90 = np.zeros(H * H, np.int64)
    g270 = np.zeros(H * H, np.int64)
    for i in range(H):
        for j in range(H):
            q = (2 * i, 2 * j)
            g90[cell_index(d, q)] = cell_index(d, cc.rot90(d, q))
            g270[cell_index(d, q)] = cell_index(d, cc.rot270(d, q))
    return g90, g270


@functools.lru_cache(maxsize=None)
def mirror_maps(d):
    """(2, H W) int64 per frame: an X check's S partner is the first other check in its flow through the fold round
    (a Z check), and that Z check's partner is the X check; -1 where none."""
    patch = rotated_patch(d)
    keys = [(t, s) for t in "XZ" for s in range(len(patch[t]))]
    out = np.full((2, (d + 1) ** 2), -1, np.int64)
    for f in (0, 1):
        fl = cc.round_flows(d, f, True)
        for key in keys:
            others = [k for k in fl[key][0] if keys[k] != key]
            if cc.cur_type(key[0], f) == "X" and others:
                q, z = cell_index(d, patch[key[0]][key[1]]), cell_index(d, patch[keys[others[0]][0]][keys[others[0]][1]])
                if out[f, q] < 0:
                    out[f, q] = z
                if out[f, z] < 0:
                    out[f, z] = q
    return out


def _rot_line(orient, k, d, rot):
    """A data line (orient 0 row y = 2k + 1, 1 column x = 2k + 1) under rot in {"id", "r90", "r270", "fold"}."""
    if rot == "id":
        return orient, k
    if rot == "r90":            # (x, y) -> (2d - y, x): column k -> row k, row k -> column d - 1 - k
        return (0, k) if orient == 1 else (1, d - 1 - k)
    if rot == "r270":           # (x, y) -> (y, 2d - x): row k -> column k, column k -> row d - 1 - k
        return (1, k) if orient == 0 else (0, d - 1 - k)
    return (1 - orient, d - 1 - k)   # fold (x, y) -> (2d - y, 2d - x)


def _map_masks(orient, masks, d, rot):
    if orient < 0:
        return -1, masks.copy()
    out = np.zeros_like(masks)
    o2 = orient
    for k in range(d):
        o2, k2 = _rot_line(orient, k, d, rot)
        sel = (masks >> k) & 1
        out ^= (sel << k2).astype(masks.dtype)
    return o2, out


def _xor_into(state, b, part, orient, masks):
    o0, m0 = state[b][part]
    if orient < 0 or not masks.any():
        return
    if o0 < 0 or not m0.any():
        state[b][part] = (orient, masks.copy())
        return
    assert o0 == orient, "two parts of one type with different orientations"
    state[b][part] = (o0, m0 ^ masks)


def line_tables(d, layers, blocks, meas, frames):
    """(lines (T, K, 2, d) uint16, orient (T, K, 2) int8) of the product of `blocks` (module docstring); K = len(meas)
    blocks."""
    K = len(meas)
    ops = cc.parse_k(layers, K)
    L = len(ops)
    T = L + 1
    f_end = frames[-1]
    state = [[(-1, np.zeros(d, np.uint16)) for _ in range(2)] for _ in range(K)]   # [block][part]: (orient, masks)
    for b in blocks:
        part = 0 if meas[b] == "X" else 1
        orient = 0 if (meas[b] == "Z") == (f_end[b] == 0) else 1
        state[b][part] = (orient, (1 << np.arange(d)).astype(np.uint16))
    lines = np.zeros((T, K, 2, d), np.uint16)
    orient = np.full((T, K, 2), -1, np.int8)

    def put(t):
        for b in range(K):
            for part in range(2):
                o, m = state[b][part]
                if o >= 0 and m.any():
                    lines[t, b, part], orient[t, b, part] = m, o

    put(L)
    for t in range(L - 1, -1, -1):
        pairs, gates = ops[t]
        for b, g in enumerate(gates):
            if g == "S":
                o, m = _map_masks(*state[b][0], d, "fold")
                _xor_into(state, b, 1, o, m)
        put(t)
        for b, g in enumerate(gates):
            if g == "H":
                state[b][0], state[b][1] = state[b][1], state[b][0]
        for c, tg in pairs:                       # disjoint pairs: each back-propagates on its own
            fc, ft = frames[t][c], frames[t][tg]
            fwd = "id" if fc == ft else ("r270" if fc == 0 else "r90")          # control position -> target position
            bwd = {"id": "id", "r90": "r270", "r270": "r90"}[fwd]
            xo, xm = _map_masks(*state[c][0], d, fwd)
            zo, zm = _map_masks(*state[tg][1], d, bwd)
            _xor_into(state, tg, 0, xo, xm)
            _xor_into(state, c, 1, zo, zm)
    return lines, orient


ROUTE_CAP = 24            # neighbours kept per detector: at L = 1 at most 18, none dropped
N_REL = 8                 # relations: 4 [other block] + 2 [other slice] + [other check type]


def routes(s, dem, cap=ROUTE_CAP):
    """The routing of the single-layer residual: per detector i of build's table s
    the detectors j != i that share an error mechanism with it in dem (the circuit's model with errors not decomposed,
    labels_ro.extended_dem), any block and slice. At most cap per detector: first those the trunk's 3 x 3 x 3
    convolution cannot reach (another block, or a cell more than one step away in x or y), then by the summed
    probability of the shared mechanisms, then by index. -> nbr (m, cap) int16 neighbour detectors (m: none), rel (m,
    cap) uint8 relations (N_REL: none), n (m,) every neighbour before the cap."""
    m = s["m"]
    tc = s["tij"]
    blk, ty = s["block"].astype(np.int64), s["mz"].reshape(-1)[s["flat"]].astype(np.int64)
    rows, cols, ps = [], [], []
    for e, inst in enumerate(inst for inst in dem.flattened() if inst.type == "error"):
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        rows += [e] * len(ds)
        cols += ds
        ps.append(inst.args_copy()[0])
    E = np.zeros((len(ps), m))
    E[rows, cols] = 1
    w = (E * np.array(ps)[:, None]).T @ E                                    # (m, m) summed shared probability
    np.fill_diagonal(w, 0)
    far = (blk[:, None] != blk[None]) | (np.abs(tc[:, None, 1:] - tc[None, :, 1:]).max(-1) > 1)
    rel = 4 * (blk[:, None] != blk[None]) + 2 * (tc[:, None, 0] != tc[None, :, 0]) + (ty[:, None] != ty[None])
    nbr = np.full((m, cap), m, np.int16)
    out = np.full((m, cap), N_REL, np.uint8)
    n = (w > 0).sum(1)
    for i in range(m):
        js = np.flatnonzero(w[i] > 0)
        js = js[np.lexsort((js, -w[i, js], ~far[i, js]))][:cap]
        nbr[i, :len(js)], out[i, :len(js)] = js, rel[i, js]
    return nbr, out, n


def _det_records(c):
    out, n = [], 0
    for inst in c.flattened():
        if inst.name == "DETECTOR":
            out.append({n + t.value for t in inst.targets_copy()})
        elif inst.name == "OBSERVABLE_INCLUDE":
            pass
        elif stim.gate_data(inst.name).produces_measurements:
            n += len(inst.targets_copy())
    return out, n


def build(d, layers, prep, meas, p, noise="uniform", products=None, slots=None, pauli="frame"):
    K = len(prep)
    c, meta = cc.make_circuit(d, layers, p, tuple(prep), tuple(meas), meta=True, noise=noise, products=products,
                              pauli=pauli)                # pauli="gate": noisy physical Paulis
    ops = cc.parse_k(layers, K)
    L = len(ops)
    T, H = L + 1, d + 1
    m, n_obs = c.num_detectors, c.num_observables
    assert n_obs >= 1, "a circuit without a reliable product"
    S = max(2, K, n_obs) if slots is None else slots
    assert S >= n_obs, "fewer product slots than products"
    blk, t, xy, ty = meta["det_block"], meta["det_t"], meta["det_xy"], meta["det_type"]
    cells = (xy[:, 0] // 2) * H + xy[:, 1] // 2
    flat = (blk * T + t) * H * H + cells
    assert len(set(flat.tolist())) == m, "two detectors share a site"
    mx = np.zeros(K * T * H * H, np.uint8)
    mz = np.zeros(K * T * H * H, np.uint8)
    mx[flat[ty == 0]] = 1
    mz[flat[ty == 1]] = 1
    frames = meta["frames"]
    gate = np.zeros((K, T, 2, 4), np.uint8)
    pcode = np.zeros((K, T, 2), np.uint8)
    pblk = np.repeat(np.arange(K, dtype=np.uint8)[:, None, None], T * 2, 1).reshape(K, T, 2)
    for l, (pairs, gates) in enumerate(ops):
        sides = [(l, 0)] + ([(l - 1, 1)] if l >= 1 else [])
        for lay, side in sides:
            for c_, t_ in pairs:
                same = frames[l][c_] == frames[l][t_]
                for b in (c_, t_):
                    gate[b, lay, side, 0 if b == c_ else 1] = 1
                    pcode[b, lay, side] = 1 if same else (2 if frames[l][b] == 1 else 3)
                    pblk[b, lay, side] = t_ if b == c_ else c_
            for b, g in enumerate(gates):
                if g == "S":
                    gate[b, lay, side, 2] = 1
                    pcode[b, lay, side] = 4 + frames[l][b]
                elif g == "H":
                    gate[b, lay, side, 3] = 1
    bound = np.zeros((K, T, 4), np.uint8)
    bound[:, 0, 0] = 1
    bound[:, T - 1, 1] = 1
    targets = [tuple((b, L, 0) for b in sub) for sub, _ in meta["products"]]
    parts = np.zeros((S, T, K, 2), np.uint8)
    lines = np.zeros((S, T, K, 2, d), np.uint16)
    orient = np.full((S, T, K, 2), -1, np.int8)
    for k, (sub, _) in enumerate(meta["products"]):
        parts[k] = cc.observable_parts(layers, sub, meas)
        lines[k], orient[k] = line_tables(d, layers, sub, meas, frames + [meta["final_frames"]])
    # representatives: the final data records of line j on every block of the product, in the final
    # frame; A by GF(2) over all records, as duet.net.geometry solves NetRO's
    dets, n_rec = _det_records(c)
    obs = {}
    for inst in c.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            obs.setdefault(int(inst.gate_args_copy()[0]), set()).update(n_rec + tt.value for tt in inst.targets_copy())
    mq = [tt.value for inst in c.flattened() if stim.gate_data(inst.name).produces_measurements
          for tt in inst.targets_copy()]
    qxy = c.get_final_qubit_coordinates()
    on_line = {}                                         # (block, orient, coordinate) -> final data records
    for r in range(n_rec - K * d * d, n_rec):            # the final data measurement: the last K d^2 records
        x, y = (int(v) for v in qxy[mq[r]])
        b = x // block_offset(d, 1)
        on_line.setdefault((b, 0, y), []).append(r)
        on_line.setdefault((b, 1, x - block_offset(d, b)), []).append(r)
    f_end = meta["final_frames"]
    rep_recs = []
    for k, (sub, _) in enumerate(meta["products"]):
        assert min(obs[k]) >= n_rec - K * d * d, f"product {k} reads records other than the final data"
        for j in range(d):
            recs = [r for b in sub for r in on_line[(b, 0 if (meas[b] == "Z") == (f_end[b] == 0) else 1, 2 * j + 1)]]
            assert len(recs) == d * len(sub)
            rep_recs.append(sorted(recs))
    # line j on every block differs from the product (line0) by checks whose time columns may start at a random first
    # value, a record but no detector (the uniform model's checks of the other type at preparation, reached through a
    # CNOT between blocks in different frames): the representative then also reads those first records. solve_gf2
    # takes the detectors first, so a first record enters only where no detector combination fixes it (a random first
    # value), and the solution is unique
    syn = n_rec - K * d * d
    first = {}
    for r in range(syn):
        first.setdefault(mq[r], r)
    free = sorted(set(first.values()) - {min(rs) for rs in dets if len(rs) == 1})
    Mr = np.zeros((n_rec, m + len(free)), np.uint8)
    for i, recs in enumerate(dets):
        Mr[sorted(recs), i] = 1
    Mr[free, m + np.arange(len(free))] = 1
    Tg = np.zeros((n_rec, S * d), np.uint8)
    for col, recs in enumerate(rep_recs):
        Tg[recs, col] ^= 1
        Tg[sorted(obs[col // d]), col] ^= 1
    X = solve_gf2(Mr, Tg)
    A = X[:m]
    for col in range(len(rep_recs)):
        rep_recs[col] = sorted(set(rep_recs[col]) ^ {free[i] for i in np.flatnonzero(X[m:, col])})
    lab = c.copy()
    for col, recs in enumerate(rep_recs):
        lab.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - n_rec) for r in recs], [n_obs + col])
    tij = np.stack([t, xy[:, 0] // 2, xy[:, 1] // 2], 1)
    return dict(d=d, m=m, T=T, H=H, W=H, K=K, n_obs=n_obs, slots=S, targets=targets, layers=list(layers),
                prep="".join(prep), meas="".join(meas), flat=flat, mx=mx.reshape(K, T, H, H), mz=mz.reshape(K, T, H, H),
                gate=gate, pcode=pcode, pblk=pblk, bound=bound, parts=parts, lines=lines, orient=orient, A=A,
                labelled=str(lab), tij=tij, block=blk, circuit=str(c))
