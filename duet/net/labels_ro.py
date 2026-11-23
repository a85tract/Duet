"""Per-layer labels and propagation tables of the read-out with explicit propagation. Imports no torch (the sampler processes fork before the trainer touches a GPU).

Layers t = 0.. L (T = L + 1): t < L the syndrome round after transversal CNOT layer t, t = L the final data
measurement. The frame of layer t: after CNOT layer t, before CNOT layer t + 1 (L - 1 and L share one). Over GF(2) on the
logical X frame (x_A, x_B), the CNOT between layers t and t + 1 is C[t] (x_target += x_control; identity if none) and
M(t -> L) = C[L - 1]... C[t]. r_P(t) = e_P M(t -> L): the blocks P's representatives occupy at layer t.

Every error e of the labelled circuit's detector error model (decompose_errors=False, approximate_disjoint_errors=True,
as geometry.check_faults) is assigned to t(e) = the layer of its first detector (detector-free: L). Its flips o(e) of
the representatives (P, j) are its per-layer label at t(e): u_{P,t,j} = XOR of o_{P,j}(e) over the fired errors with
t(e) = t. In the extended model each error also flips observable n_obs (d + 1) + (t(e) n_obs + P) d + j for every
representative it flips, so stim's DEM sampler returns detectors, final labels and per-layer labels in one draw.

check_encoding / check_locality / check_circuit_faults are the checks of the plan."""
import numpy as np
import stim


def gate_maps(spec):
    """(L, 2, 2) uint8: C[t] = the CNOT between layers t and t + 1 on (x_A, x_B) (identity without one)."""
    L = spec["T"] - 1
    C = np.tile(np.eye(2, dtype=np.uint8), (L, 1, 1))
    befores = sorted(g[0] for g in spec["gates"])
    assert befores == list(range(-1, L - 1)), f"expected CNOT layers before detector layers 0 .. L - 1: {befores}"
    for before, after, ctl in spec["gates"]:
        if before >= 0:                         # the CNOT before layer 0 lies inside layer 0's frame
            C[before][1 - ctl, ctl] = 1
    return C


def suffix_maps(spec):
    """(T, 2, 2) uint8: M(t -> L), composed over GF(2) from the end."""
    C = gate_maps(spec)
    T = spec["T"]
    M = np.zeros((T, 2, 2), np.uint8)
    M[T - 1] = np.eye(2, dtype=np.uint8)
    for t in range(T - 2, -1, -1):
        M[t] = (M[t + 1].astype(np.int64) @ C[t]) % 2
    return M


def e_vec(spec):
    """(n_obs, 2) uint8: the blocks each product reads."""
    e = np.zeros((spec["n_obs"], 2), np.uint8)
    for k, P in enumerate(spec["products"]):
        e[k, P["blocks"]] = 1
    return e


def supports(spec):
    """(n_obs, T, 2) uint8: r_P(t) = e_P M(t -> L)."""
    return (np.einsum("pb,tbc->ptc", e_vec(spec).astype(np.int64), suffix_maps(spec).astype(np.int64)) % 2).astype(np.uint8)


def supports_from_bits(pattern, L, n_obs_blocks):
    """The same table from the pattern's bits alone (an independent route for the checks): bit l of the pattern is CNOT
    layer l's direction, 0 = control A."""
    r = np.array(n_obs_blocks, np.int64)                     # (n_obs, 2): r_P(L)
    out = np.zeros((len(r), L + 1, 2), np.uint8)
    out[:, L] = r
    out[:, L - 1] = r
    for t in range(L - 2, -1, -1):
        ctl = (pattern >> (t + 1)) & 1                       # CNOT layer t + 1 sits between layers t and t + 1
        C = np.eye(2, dtype=np.int64)
        C[1 - ctl, ctl] = 1
        r = (r @ C) % 2
        out[:, t] = r
    return out


def n_lab(spec):
    return spec["n_obs"] * (spec["d"] + 1)


def n_total(spec):
    """Observables of the extended model: products, representatives, per-layer representatives."""
    return n_lab(spec) + spec["T"] * spec["n_obs"] * spec["d"]


def dem_of(spec):
    return stim.Circuit(spec["labelled"]).detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)


def errors(spec, dem=None):
    """The model's errors: [(p, detectors, observables)] in order, and per error its layer t(e)."""
    dem = dem_of(spec) if dem is None else dem
    tdet = spec["tij"][:, 0]
    L = spec["T"] - 1
    out, tt = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ts = inst.targets_copy()
        dts = [t.val for t in ts if t.is_relative_detector_id()]
        obs = [t.val for t in ts if t.is_logical_observable_id()]
        out.append((inst.args_copy()[0], dts, obs))
        tt.append(int(tdet[dts].min()) if dts else L)
    return out, np.array(tt, np.int64)


def extended_dem(spec, dem=None):
    """The labelled circuit's model with the per-layer observables (module docstring); (model, stats)."""
    dem = dem_of(spec) if dem is None else dem
    n_obs, d, T = spec["n_obs"], spec["d"], spec["T"]
    base = n_lab(spec) - n_obs                               # observable n_obs + P d + j -> per-layer index - this
    tdet = spec["tij"][:, 0]
    out = stim.DetectorErrorModel()
    n_err = n_free = 0
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        ts = inst.targets_copy()
        dts = [t.val for t in ts if t.is_relative_detector_id()]
        t0 = int(tdet[dts].min()) if dts else T - 1
        reps = [t.val for t in ts if t.is_logical_observable_id() and t.val >= n_obs]
        n_free += not dts and bool(reps)
        out.append("error", inst.args_copy(),
                   ts + [stim.target_logical_observable_id(base + t0 * n_obs * d + o) for o in reps])
        n_err += 1
    out.append("logical_observable", [], [stim.target_logical_observable_id(n_total(spec) - 1)])
    assert out.num_detectors == spec["m"] and out.num_observables == n_total(spec), (out.num_detectors, out.num_observables)
    return out, dict(errors=n_err, detector_free_flipping=n_free)


def split_labels(o, spec):
    """o (N, n_total) 0/1 -> y (N, n_obs), yl (N, n_obs, d), u (N, n_obs, d, T)."""
    n_obs, d, T = spec["n_obs"], spec["d"], spec["T"]
    nl = n_lab(spec)
    u = o[:, nl:].reshape(len(o), T, n_obs, d).transpose(0, 2, 3, 1)
    return o[:, :n_obs], o[:, n_obs:nl].reshape(len(o), n_obs, d), u


# ---------------------------------------------------------------- checks


def check_encoding(spec, ext=None):
    """Every error's per-layer observables lie on its layer t(e) and XOR over the layers to its representative flips;
    its product flips are untouched. Returns the number of violating errors."""
    ext = extended_dem(spec)[0] if ext is None else ext
    n_obs, d, T = spec["n_obs"], spec["d"], spec["T"]
    nl, tdet = n_lab(spec), spec["tij"][:, 0]
    bad = 0
    for inst in ext.flattened():
        if inst.type != "error":
            continue
        ts = inst.targets_copy()
        dts = [t.val for t in ts if t.is_relative_detector_id()]
        obs = np.zeros(n_total(spec), np.uint8)
        for t in ts:
            if t.is_logical_observable_id():
                obs[t.val] ^= 1
        t0 = int(tdet[dts].min()) if dts else T - 1
        per = obs[nl:].reshape(T, n_obs * d)
        ok = not per[np.arange(T) != t0].any() and np.array_equal(per.sum(0) % 2, obs[n_obs:nl])
        bad += not ok
    return bad


def local_flips(spec, errs, tt):
    """(E, 2, d) uint8: each error's flips of line j on each block in layer t(e)'s frame, M(t(e) -> L)^-1 F(e), with
    F(e) the final frame (one line family shared by the products, whose block rows form an invertible matrix)."""
    n_obs, d = spec["n_obs"], spec["d"]
    E_ = e_vec(spec).astype(np.int64)
    assert n_obs == 2 and round(abs(np.linalg.det(E_))) % 2 == 1, "local_flips needs two products over invertible blocks"
    Einv = np.round(np.linalg.inv(E_)).astype(np.int64) % 2
    M = suffix_maps(spec).astype(np.int64)
    Minv = np.array([np.round(np.linalg.inv(m)).astype(np.int64) % 2 for m in M])
    assert all(np.array_equal((M[t] @ Minv[t]) % 2, np.eye(2)) for t in range(len(M)))
    out = np.zeros((len(errs), 2, d), np.uint8)
    for k, (_, _, obs) in enumerate(errs):
        o = np.zeros((n_obs, d), np.int64)
        for v in obs:
            if v >= n_obs:
                o[(v - n_obs) // d, (v - n_obs) % d] ^= 1
        if o.any():
            out[k] = (Minv[tt[k]] @ ((Einv @ o) % 2)) % 2
    return out


def check_locality(spec, errs, tt, f=None):
    """Errors whose detectors all lie in one block must flip nothing on the other block in layer t(e)'s frame (an
    error seen in one block only cannot sit on the other one in its own layer's frame). Counts of violating errors and
    of errors checked."""
    f = local_flips(spec, errs, tt) if f is None else f
    blk = spec["block"]
    bad = n = 0
    for k, (_, dts, _) in enumerate(errs):
        if not dts:
            continue
        bs = set(blk[dts].tolist())
        if len(bs) == 1:
            n += 1
            other = 1 - bs.pop()
            bad += int(f[k, other].any())
    return bad, n


def _lines(circuit, spec):
    """(qubit -> (block, line j)) for the data qubits of both blocks in the products' shared line family (lines ordered
    by position, as geometry's representatives), the final record of each data qubit, and the data qubits."""
    from duet.net.geometry import _scan, _split, data_records
    c = circuit.flattened()
    rq, _, _, obs, _, ri = _scan(c)
    counts = np.bincount(rq, minlength=c.num_qubits)
    isdata = data_records(rq, ri, obs)
    coords = c.get_final_qubit_coordinates()
    data = sorted({int(rq[r]) for r in np.flatnonzero(isdata)})
    allq = sorted(q for q in coords if counts[q])
    xy = {q: np.array(coords[q][:2], float) for q in allq}
    ax, thr = _split(np.array([xy[q] for q in allq]))
    qb = {q: int(xy[q][ax] > thr) for q in data}
    rec = {int(rq[r]): r for r in range(len(rq)) if isdata[r]}
    axes = {P["axis"][b] for P in spec["products"] for b in P["blocks"]}
    assert len(axes) == 1, f"the products use different line families: {axes}"
    a = axes.pop()
    lines = {}
    for b in (0, 1):
        qs = [q for q in data if qb[q] == b]
        vals = np.unique(np.round([xy[q][a] for q in qs], 6))
        assert len(vals) == spec["d"]
        for q in qs:
            lines[q] = (b, int(np.searchsorted(vals, np.round(xy[q][a], 6))))
    for P in spec["products"]:
        for b in P["blocks"]:
            for j, li in enumerate(P["lines"][b]):
                assert all(lines[q] == (b, j) for q in li), "line order differs from geometry's representatives"
    return lines, rec, data


def _cnot_layer_starts(c, spec):
    """Instruction indices (in the flattened circuit) of the first instruction of each transversal CNOT layer 0.. L-1."""
    from duet.net.geometry import CNOT
    lines, _, data = _lines(c, spec)
    blk = {q: lines[q][0] for q in data}
    starts, last_layer, n_det = [], None, 0
    for i, inst in enumerate(c):
        if inst.name == "DETECTOR":
            n_det += 1
        elif inst.name in CNOT:
            v = [t.value for t in inst.targets_copy()]
            pairs = list(zip(v[::2], v[1::2]))
            if all(a in blk and b in blk and blk[a] != blk[b] for a, b in pairs):
                if last_layer != n_det:                 # a new transversal layer (detectors declared since the last)
                    starts.append(i)
                    last_layer = n_det
    assert len(starts) == spec["T"] - 1, (len(starts), spec["T"])
    return starts


def check_circuit_faults(spec):
    """Labels against circuit faults: stim's representative circuit fault of every error of the labelled circuit's
    model, simulated forward through the noiseless circuit (FlipSimulator, one lane per error) to layer t(e)'s frame
    (just before CNOT layer t(e) + 1; for t(e) >= L - 1 the final data records). Its line flips f on both blocks must
    satisfy M(t(e) -> L) f = o(e) on every representative, and must equal local_flips. Also the delay t(e) - (the layer
    whose frame the fault occurs in) and the layers spanned by each error's detectors."""
    c = stim.Circuit(spec["labelled"]).flattened()
    dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    errs, tt = errors(spec, dem)
    n_obs, d, T = spec["n_obs"], spec["d"], spec["T"]
    L = T - 1
    key = {}
    for k, (_, dts, obs) in enumerate(errs):
        key[(tuple(sorted(dts)), tuple(sorted(obs)))] = k
    expl = c.explain_detector_error_model_errors(dem_filter=dem, reduce_to_one_representative_error=True)
    loc = [None] * len(errs)
    for ex in expl:
        dts = sorted(t.dem_target.val for t in ex.dem_error_terms if t.dem_target.is_relative_detector_id())
        obs = sorted(t.dem_target.val for t in ex.dem_error_terms if t.dem_target.is_logical_observable_id())
        loc[key[(tuple(dts), tuple(obs))]] = ex.circuit_error_locations[0]
    assert all(x is not None for x in loc), "an error without a circuit fault"
    lines, rec, data = _lines(c, spec)
    starts = _cnot_layer_starts(c, spec)
    tau = {starts[t + 1]: t for t in range(L - 1)}             # read layer t's frame before CNOT layer t + 1
    E, nq = len(errs), c.num_qubits
    slab = np.full(E, -1, np.int64)
    layer_of_instr = np.searchsorted(np.array(starts[1:] + [len(c) + 1]), np.arange(len(c)), side="right")
    at = {}
    meas_flip = []
    for k, x in enumerate(loc):
        i = x.stack_frames[0].instruction_offset
        if x.flipped_measurement is not None and x.flipped_measurement.record_index is not None \
                and not x.flipped_pauli_product:
            meas_flip.append((k, x.flipped_measurement.record_index))
            slab[k] = layer_of_instr[i]
            continue
        at.setdefault(i, []).append(k)
    sim = stim.FlipSimulator(batch_size=E, disable_stabilizer_randomization=True, num_qubits=nq)
    xs_at = np.zeros((E, nq), bool)
    for i, inst in enumerate(c):
        if i in tau:
            t = tau[i]
            sel = np.flatnonzero(tt == t)
            if len(sel):
                ps = sim.peek_pauli_flips()
                for k in sel:
                    xs_at[k] = ps[k].to_numpy()[0]
        gd = stim.gate_data(inst.name)
        if gd.produces_measurements:
            sim.do(stim.CircuitInstruction(inst.name, inst.targets_copy()))
        elif not gd.is_noisy_gate:
            sim.do(inst)
        for k in at.get(i, ()):
            slab[k] = layer_of_instr[i]
        if i in at:
            masks = {p: np.zeros((nq, E), bool) for p in "XYZ"}
            for k in at[i]:
                for g in loc[k].flipped_pauli_product:
                    gt = g.gate_target
                    p = "X" if gt.is_x_target else "Y" if gt.is_y_target else "Z"
                    masks[p][gt.value, k] = True
            for p, mk in masks.items():
                if mk.any():
                    sim.broadcast_pauli_errors(pauli=p, mask=mk)
    mf = sim.get_measurement_flips(bit_packed=False)                     # (records, E)
    for k, r in meas_flip:
        mf[r, k] ^= True
    f = np.zeros((E, 2, d), np.uint8)
    late = tt >= L - 1
    for q in data:
        b, j = lines[q]
        f[late, b, j] ^= mf[rec[q], late].astype(np.uint8)
        f[~late, b, j] ^= xs_at[~late, q].astype(np.uint8)
    M = suffix_maps(spec)
    e = e_vec(spec).astype(np.int64)
    bad = 0
    for k, (_, _, obs) in enumerate(errs):
        o = np.zeros((n_obs, d), np.int64)
        for v in obs:
            if v >= n_obs:
                o[(v - n_obs) // d, (v - n_obs) % d] ^= 1
        pred = (e @ M[tt[k]].astype(np.int64) @ f[k].astype(np.int64)) % 2
        bad += int(not np.array_equal(pred, o))
    fl = local_flips(spec, errs, tt)
    mism = int((fl != f).any((1, 2)).sum())
    spans = np.array([int(np.ptp(spec["tij"][dts, 0])) if dts else -1 for _, dts, _ in errs])
    delay = tt - slab
    return dict(errors=E, bad=bad, local_flips_mismatch=mism,
                delay={int(v): int((delay == v).sum()) for v in np.unique(delay)},
                span={int(v): int((spans == v).sum()) for v in np.unique(spans)},
                measurement_faults=len(meas_flip)), f


def correlated_mass(spec, errs, tt, f=None):
    """The 00/11 case: probability of the errors that flip line j on both blocks in layer t(e)'s frame, against all
    errors that flip some line; and the part of it in layers where some product's support is both blocks."""
    f = local_flips(spec, errs, tt) if f is None else f
    p = np.array([e[0] for e in errs])
    any_ = f.any((1, 2))
    both = (f[:, 0] & f[:, 1]).any(1)
    sup = supports(spec)                                                   # (n_obs, T, 2)
    both_sup = (sup[:, :, 0] & sup[:, :, 1]).any(0)[tt].astype(bool)
    return dict(p_flip=float(p[any_].sum()), p_both=float(p[both].sum()), p_both_in_both_support=float(p[both & both_sup].sum()),
                n_flip=int(any_.sum()), n_both=int(both.sum()))
