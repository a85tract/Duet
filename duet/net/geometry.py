"""Geometry, labels and GF(2) relations of a two-block transversal-CNOT circuit, read from the stim circuit alone (`duet/circuits.py`); the idea follows CoDecoder's
`training/convlines_train/surface.py` (a85tract/CoDecoder, dec0ee6, Apache-2.0), generalised to two blocks.

analyse(circuit) -> dict:
  blocks the qubits split into two clusters by the widest gap between qubit coordinates along x or y. Data qubits are
            the qubits measured exactly once; each block holds d^2 of them, all measured in one basis (basis[b]).
  grid detector k sits at (block, t, i, j): (i, j) = its (x, y) over the lattice spacing from its block's corner,
            t = its time coordinate from 0. Both blocks share T x H x W and their sites. Type (0 X, 1 Z): a site with a
            detector that reads data records has its block's measured basis, every other site the other one; the two
            blocks must agree on every site (a transversal CNOT pairs identical layouts).
  gates every transversal CNOT between the blocks: (layer before, layer after, control block); the layer before is
            the latest detector layer declared ahead of the gate (-1 if none), the layer after the next one. At most
            one gate between two layers; every pair joins the same (i, j) of the two blocks.
  products one per observable of the circuit: the blocks whose data records it reads, and per involved block the axis
            its data qubits share (0: constant x, a column; 1: constant y, a row), so the block's d lines along that
            axis, sorted by position. Representative j = line j of every involved block. off (m, d) = signed distance
            of a detector of an involved block to line j in half-line units (CoDecoder's line_off; 0 elsewhere).
  labels the circuit with one observable more per representative, index n_obs + P d + j (`labelled`).
  A (m, n_obs d) uint8, column P d + j: the detectors whose records XOR to representative j's records plus
            product P's, solved over GF(2); so y_{P,j} = y_P + (D A)_{P d + j} (mod 2) on every shot, D the detection
            events (stim reports detectors and observables relative to the noiseless reference, so no constant).
check_faults(spec) verifies A H = F_j + F_P over every fault of the detector error model; neighbourhood(spec) lists
the faults whose detectors do not fit one coupled neighbourhood of the network."""
import numpy as np
import stim

BASIS = {"M": "Z", "MZ": "Z", "MR": "Z", "MRZ": "Z", "MX": "X", "MRX": "X", "MY": "Y", "MRY": "Y"}
CNOT = {"CX", "CNOT", "ZCX"}


def _scan(c):
    """One pass over the flattened circuit: per record its qubit and basis; detectors (absolute records, coordinates);
    observables {k: set of records}; CNOTs (records so far, detectors so far, [(control, target)]); per record the
    index of the measurement instruction that made it."""
    rq, rb, dets, obs, cx, ri = [], [], [], {}, [], []
    for inst in c:
        name = inst.name
        if name == "DETECTOR":
            dets.append(([len(rq) + t.value for t in inst.targets_copy()], inst.gate_args_copy()))
        elif name == "OBSERVABLE_INCLUDE":
            s = obs.setdefault(int(inst.gate_args_copy()[0]), set())
            for t in inst.targets_copy():
                if not t.is_measurement_record_target:
                    raise ValueError("observables must be measurement records")
                s ^= {len(rq) + t.value}
        elif name in CNOT:
            v = [t.value for t in inst.targets_copy()]
            cx.append((len(rq), len(dets), list(zip(v[::2], v[1::2]))))
        elif stim.gate_data(name).produces_measurements:
            if name not in BASIS:
                raise ValueError(f"{name}: only single-qubit measurements are supported")
            k = ri[-1] + 1 if ri else 0
            for t in inst.targets_copy():
                rq.append(t.value)
                rb.append(BASIS[name])
                ri.append(k)
    return np.array(rq, np.int64), rb, dets, obs, cx, np.array(ri, np.int64)


def data_records(rq, ri, obs):
    """Mask over the records: the final data measurements, i.e. the records of qubits measured once that were made by a
    measurement instruction an observable reads (the transversal measurement). With two or more rounds every ancilla is
    measured at least twice, so this is every record of a qubit measured once; with one round (L = 1) the ancillas are
    measured once too, by their round's instruction, which no observable reads."""
    counts = np.bincount(rq)
    final = {int(ri[r]) for recs in obs.values() for r in recs}
    return np.array([counts[q] == 1 and (not final or int(ri[r]) in final) for r, q in enumerate(rq)], bool)


def _split(xy):
    """(axis, threshold) of the widest gap between the coordinates (the two blocks lie on either side)."""
    best = (0.0, 0, 0.0)
    for ax in (0, 1):
        u = np.unique(np.round(xy[:, ax], 6))
        if len(u) > 1:
            g = np.diff(u)
            k = int(np.argmax(g))
            if g[k] > best[0] and g[k] > np.median(g):
                best = (float(g[k]), ax, float(u[k] + u[k + 1]) / 2)
    if best[0] == 0:
        raise ValueError("the qubits do not form two blocks separated by a gap")
    return best[1], best[2]


def _lattice(v, origin, s):
    pos = (np.round(v, 6) - origin) / s
    assert np.allclose(pos, np.round(pos), atol=1e-6), "coordinates are not on a lattice"
    return np.round(pos).astype(np.int64)


def _spacing(v):
    u = np.unique(np.round(v, 6))
    return float(np.min(np.diff(u))) if len(u) > 1 else 1.0


def _bits(a):
    """(r, n) 0/1 -> (r, ceil(n / 64)) uint64 words, column c at bit c % 64 of word c // 64."""
    a = np.asarray(a, np.uint8)
    p = np.zeros((a.shape[0], -(-a.shape[1] // 64) * 8), np.uint8)
    p[:, :(a.shape[1] + 7) // 8] = np.packbits(a, axis=1, bitorder="little")
    return p.view("<u8")


def solve_gf2(M, T):
    """X (m, k) uint8 with M X = T over GF(2), M (r, m), T (r, k); free variables 0. ValueError if inconsistent."""
    r, m = M.shape
    k = T.shape[1]
    W = _bits(np.concatenate([M, T], 1))
    one = np.uint64(1)
    rank, piv = 0, []
    for col in range(m):
        w, b = divmod(col, 64)
        bit = (W[:, w] >> np.uint64(b)) & one
        rows = np.flatnonzero(bit[rank:])
        if len(rows) == 0:
            continue
        p = rank + rows[0]
        if p != rank:
            W[[rank, p]] = W[[p, rank]]
            bit[[rank, p]] = bit[[p, rank]]
        hit = np.flatnonzero(bit)
        hit = hit[hit != rank]
        W[hit] ^= W[rank]
        piv.append(col)
        rank += 1
        if rank == r:
            break
    full = np.unpackbits(W.view(np.uint8), axis=1, bitorder="little")[:, :m + k]
    if full[rank:, m:].any():
        bad = sorted(set(np.flatnonzero(full[rank:, m:].any(0)).tolist()))
        raise ValueError(f"no GF(2) solution for target columns {bad[:8]}")
    X = np.zeros((m, k), np.uint8)
    X[piv] = full[:rank, m:]
    return X


def analyse(circuit):
    c = circuit.flattened()
    rq, rb, dets, obs, cx, ri = _scan(c)
    m, n_rec = len(dets), len(rq)
    coords = c.get_final_qubit_coordinates()
    counts = np.bincount(rq, minlength=c.num_qubits)
    isdata = data_records(rq, ri, obs)
    data = sorted({int(rq[r]) for r in np.flatnonzero(isdata)})
    missing = [q for q in coords if len(coords[q]) < 2] + [q for q in range(c.num_qubits) if counts[q] and q not in coords]
    if missing:
        raise ValueError(f"qubits without (x, y) coordinates: {missing[:5]}")
    allq = sorted(q for q in coords if counts[q])
    xy = {q: np.array(coords[q][:2], float) for q in allq}
    ax, thr = _split(np.array([xy[q] for q in allq]))
    qblock = {q: int(xy[q][ax] > thr) for q in allq}
    nd = [sum(qblock[q] == b for q in data) for b in (0, 1)]
    d = int(round(np.sqrt(nd[0])))
    if nd[0] != nd[1] or d * d != nd[0] or d % 2 == 0:
        raise ValueError(f"the blocks hold {nd} qubits measured once; expected d^2 data qubits each, d odd")
    rec_of = {int(rq[r]): r for r in range(n_rec) if isdata[r]}
    basis = []
    for b in (0, 1):
        bs = {rb[rec_of[q]] for q in data if qblock[q] == b}
        if len(bs) != 1 or bs <= {"Y"}:
            raise ValueError(f"block {b}: data measured in bases {bs}; expected one of X, Z")
        basis.append(bs.pop())

    # grid
    if any(len(cc) < 3 for _, cc in dets):
        raise ValueError("every detector needs (x, y, t) coordinates")
    site = np.array([cc[:3] for _, cc in dets], float)
    blk = (site[:, ax] > thr).astype(np.int64)
    tij = np.zeros((m, 3), np.int64)
    tt = site[:, 2] - site[:, 2].min()
    assert np.allclose(tt, np.round(tt)), "detector time coordinates are not integers"
    tij[:, 0] = np.round(tt)
    # The lattice of each block (origin, spacing, extent) from its detectors' sites and its ancillas (the measured qubits
    # that are not data). With two or more rounds these are the same sites; with one round only one
    # check type has detectors, and the ancillas keep the lattice of every other depth.
    dset = set(data)
    ext = [0, 0]
    for b in (0, 1):
        sel = blk == b
        anc = np.array([xy[q] for q in allq if qblock[q] == b and q not in dset]).reshape(-1, 2)
        for a in (0, 1):
            ref = np.concatenate([site[sel, a], anc[:, a]])
            org, sp = np.round(ref.min(), 6), _spacing(ref)
            tij[sel, 1 + a] = _lattice(site[sel, a], org, sp)
            ext[a] = max(ext[a], int(_lattice(ref, org, sp).max()) + 1)
    sites = [{(int(i), int(j)) for i, j in tij[blk == b, 1:]} for b in (0, 1)]
    if sites[0] != sites[1]:
        raise ValueError("the two blocks have different check layouts")
    reads_data = np.array([any(isdata[r] for r in recs) for recs, _ in dets])
    typ_of = {}
    for b in (0, 1):
        fin = {(int(i), int(j)) for i, j in tij[(blk == b) & reads_data, 1:]}
        for s in sites[b]:
            t = int((s in fin) == (basis[b] == "Z"))            # 1 Z, 0 X
            if typ_of.setdefault(s, t) != t:
                raise ValueError(f"site {s}: the blocks disagree on the check type")
    typ = np.array([typ_of[(int(i), int(j))] for i, j in tij[:, 1:]], np.int64)
    if len({(int(b), *map(int, r)) for b, r in zip(blk, tij)}) != m:
        raise ValueError("two detectors share a grid site")
    T, H, W = int(tij[:, 0].max()) + 1, *ext

    # gates
    org = [np.array([site[blk == b, a].min() for a in (0, 1)]) for b in (0, 1)]
    gates, last = [], None
    for n_r, n_d, pairs in cx:
        inter = [(a, b) for a, b in pairs if a in qblock and b in qblock and qblock[a] != qblock[b]]
        if not inter:
            continue
        if len(inter) != len(pairs) or len({qblock[a] for a, _ in inter}) != 1:
            raise ValueError("a CNOT instruction mixes directions or intra-block pairs with inter-block ones")
        for a, b in inter:
            if a not in dset or b not in dset or not np.allclose(xy[a] - org[qblock[a]], xy[b] - org[qblock[b]]):
                raise ValueError("a transversal CNOT pair does not join the same data qubit of the two blocks")
        before = int(tij[:n_d, 0].max()) if n_d else -1
        key = (n_r, before)
        ctl = qblock[inter[0][0]]
        if last is not None and last[0] == key:                  # the same layer, split over instructions
            if last[1] != ctl:
                raise ValueError("one CNOT layer with two directions")
            continue
        gates.append((before, before + 1, ctl))
        last = (key, ctl)
    if len({g[0] for g in gates}) != len(gates):
        raise ValueError("two CNOT layers between the same two detector layers")

    # products, representatives, labels
    n_obs = circuit.num_observables
    if sorted(obs) != list(range(n_obs)):
        raise ValueError("observables must be numbered 0 .. n - 1")
    dq = {b: [q for q in data if qblock[q] == b] for b in (0, 1)}
    prods, rep_recs = [], []
    for k in range(n_obs):
        qs = [int(rq[r]) for r in sorted(obs[k])]
        if not all(isdata[r] for r in obs[k]):
            raise ValueError(f"observable {k} reads records that are not final data measurements")
        P = {"blocks": [], "axis": {}, "lines": {}, "line_pos": {}}
        off = np.zeros((m, d), np.int64)
        for b in (0, 1):
            mine = [q for q in qs if qblock[q] == b]
            if not mine:
                continue
            pts = np.array([xy[q] for q in mine])
            shared = [a for a in (0, 1) if np.allclose(pts[:, a], pts[0, a])]
            if len(mine) != d or len(shared) != 1:
                raise ValueError(f"observable {k}: its block-{b} qubits are not one line of d data qubits")
            a = shared[0]
            vals = np.unique(np.round([xy[q][a] for q in dq[b]], 6))
            lines = [sorted(q for q in dq[b] if np.isclose(xy[q][a], v)) for v in vals]
            if len(lines) != d or any(len(li) != d for li in lines):
                raise ValueError(f"block {b}: the data qubits do not form d lines of d along axis {a}")
            s = float(np.diff(vals).min())
            sel = blk == b
            off[sel] = np.rint(2 * (site[sel, a][:, None] - vals[None, :]) / s).astype(np.int64)
            P["blocks"].append(b)
            P["axis"][b] = a
            P["lines"][b] = lines
            P["line_pos"][b] = vals.tolist()
        P["off"] = off
        prods.append(P)
        for j in range(d):
            rep_recs.append(sorted(rec_of[q] for b in P["blocks"] for q in P["lines"][b][j]))

    # A: detectors whose records XOR to (representative j) + (product P)
    M = np.zeros((n_rec, m), np.uint8)
    for k, (recs, _) in enumerate(dets):
        for r in recs:
            M[r, k] ^= 1
    Tg = np.zeros((n_rec, n_obs * d), np.uint8)
    for P in range(n_obs):
        for j in range(d):
            col = P * d + j
            Tg[rep_recs[col], col] ^= 1
            Tg[sorted(obs[P]), col] ^= 1
    A = solve_gf2(M, Tg)

    lab = circuit.copy()
    for col, recs in enumerate(rep_recs):
        lab.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - n_rec) for r in recs], [n_obs + col])
    return dict(d=d, m=m, n_obs=n_obs, T=T, H=H, W=W, basis=basis, block=blk, tij=tij, type=typ, gates=gates,
                products=prods, A=A, labelled=str(lab), axis=ax)


STATIC = ("d", "m", "n_obs", "T", "H", "W", "basis", "axis")


def same_geometry(a, b):
    """True if two analyses (two patterns of one circuit family) share everything but the gates, A and labels."""
    return all(a[k] == b[k] for k in STATIC) and all(np.array_equal(a[k], b[k]) for k in ("block", "tij", "type")) \
        and all(p["blocks"] == q["blocks"] and np.array_equal(p["off"], q["off"]) for p, q in zip(a["products"], b["products"]))


def gate_table(spec):
    """(2, T, 2, 2) float32: [block, layer, side (0 the gate before the layer, 1 the gate after), role (0 control,
    1 target)] = 1 where the block plays that role in that gate."""
    g = np.zeros((2, spec["T"], 2, 2), np.float32)
    for before, after, ctl in spec["gates"]:
        for b in (0, 1):
            role = 0 if b == ctl else 1
            if 0 <= after < spec["T"]:
                g[b, after, 0, role] = 1
            if 0 <= before < spec["T"]:
                g[b, before, 1, role] = 1
    return g


def _faults(spec):
    """Per fault of the labelled circuit's detector error model: (detectors, observables) as lists."""
    dem = stim.Circuit(spec["labelled"]).detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    out = []
    for inst in dem.flattened():
        if inst.type == "error":
            ts = inst.targets_copy()
            out.append(([t.val for t in ts if t.is_relative_detector_id()],
                        [t.val for t in ts if t.is_logical_observable_id()]))
    return out


def check_faults(spec, faults=None):
    """Number of (fault, representative) pairs violating A H = F_j + F_P (0 means the relation is exact)."""
    faults = _faults(spec) if faults is None else faults
    A, n_obs, d = spec["A"], spec["n_obs"], spec["d"]
    bad = 0
    for dts, obs in faults:
        g = A[dts].sum(0) % 2 if dts else np.zeros(A.shape[1], np.int64)
        f = np.zeros(A.shape[1], np.int64)
        for o in obs:
            if o < n_obs:
                f[o * d:(o + 1) * d] ^= 1
            else:
                f[o - n_obs] ^= 1
        bad += int((g != f).sum())
    return bad


def neighbourhood(spec, faults=None, kernel=(3, 5, 5)):
    """Faults whose detectors do not fit one coupled neighbourhood: inside one block a stem window (kernel); across
    the blocks, the layers of one gate and a window of the kernel's (i, j) size with the blocks overlaid."""
    faults = _faults(spec) if faults is None else faults
    pairs = {(g[0], g[1]) for g in spec["gates"]}
    out = []
    for dts, _ in faults:
        if not dts:
            continue
        b, tij = spec["block"][dts], spec["tij"][dts]
        span = tij.max(0) - tij.min(0)
        if (b == b[0]).all():
            ok = all(span[k] < kernel[k] for k in range(3))
        else:
            ts = set(tij[:, 0].tolist())
            ok = span[1] < kernel[1] and span[2] < kernel[2] and any(ts <= {lo, hi} for lo, hi in pairs)
        if not ok:
            out.append(dts)
    return out
