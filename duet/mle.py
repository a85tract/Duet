"""Exact most-likely-error (MLE) decoding of a stim detector error model.

The problem is the one Cain et al. (arXiv 2403.03272, supplement "MLE decoder") solve with Gurobi and Bonilla Ataides et
al. (arXiv 2509.11370) use for their Fig. 1b: stim's detector error model (DEM) of the circuit, not decomposed, gives
mechanisms j with probability p_j, detector set D_j and observable set L_j. For a syndrome s (the set of fired
detectors) the MLE is the assignment x in {0, 1}^n of largest probability prod_j p_j^x_j (1 - p_j)^(1 - x_j) with
sum_{j: i in D_j} x_j = s_i (mod 2) for every detector i, i.e. of least weight W(x) = sum_j w_j x_j with
w_j = ln((1 - p_j) / p_j); the decoder predicts the observable flips sum_j x_j L_j (mod 2).

Exact reductions applied to the DEM before solving (``Dem``); none changes the optimal weight, and each optimum of the
reduced problem is an optimum of the original:
- mechanisms with no detector are dropped (they only add weight);
- of mechanisms with the same detector set (stim already merged equal detector-and-observable sets), only the lightest
  is kept: an optimum never holds two of them (dropping both lowers the weight and keeps the syndrome) and swapping a
  heavier one for the lightest keeps the syndrome and lowers the weight. Exact ties (equal weights, different
  observables) are true MLE ambiguities; the kept one is the first in stim's order. Their count is recorded.

Solvers (open source; Gurobi needs a licence):
- ``CpSat``: OR-Tools CP-SAT, one Boolean per mechanism and one XOR constraint per detector; weights scaled to integers
  (``SCALE``), so the proven optimum is exact for the rounded weights and within n_active * 0.5 / SCALE of the true
  optimum;
- ``Highs``: HiGHS MILP, the formulation of Cain et al. (sum_j x_j - 2 k_i = s_i, k_i integer), real weights, gaps 0
  (absolute gap ``ABS_GAP``).
Both keep one model per DEM and change only the syndrome per shot. ``exhaustive`` is a meet-in-the-middle search for
tiny DEMs (validation). Every solution is checked against its syndrome (``Dem.syndrome``).
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import stim

SCALE = 10 ** 6       # CP-SAT integer weights: round(w * SCALE)
ABS_GAP = 1e-7        # HiGHS absolute MIP gap (relative gap 0)


class Dem:
    """A stim DEM as arrays: ``p``, ``w`` (weights), ``H`` (detectors x mechanisms, CSC, uint8), ``L`` (observable
    bit masks per mechanism, int64), after the exact reductions in the module docstring (``reduce=False``: none but
    the merge of equal detector-and-observable sets, for checking the reductions). ``orig`` maps each kept mechanism
    to its index among the DEM's error instructions."""

    def __init__(self, dem: stim.DetectorErrorModel, reduce: bool = True):
        self.n_det, self.n_obs = dem.num_detectors, dem.num_observables
        groups: dict[tuple[int, ...], list[tuple[float, int, int]]] = {}
        k = 0
        self.n_raw = self.n_no_det = 0
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            dets, obs = [], 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs ^= 1 << t.val
                elif t.is_separator():
                    raise ValueError("decomposed DEM: build it with decompose_errors=False")
            self.n_raw += 1
            if p <= 0:
                k += 1
                continue
            if not dets:
                self.n_no_det += 1
            else:
                key = tuple(sorted(set(d for d in dets if dets.count(d) % 2)))
                if key:
                    groups.setdefault(key if reduce else (key, obs), []).append((p, obs, k))
                else:
                    self.n_no_det += 1
            k += 1
        keys = list(groups)
        p, obs, orig = np.empty(len(keys)), np.empty(len(keys), np.int64), np.empty(len(keys), np.int64)
        self.n_dominated = self.n_ties = self.n_merged = 0
        for m, key in enumerate(keys):
            g: dict[int, list] = {}
            for p_, o_, k_ in groups[key]:      # equal detectors and observables: one mechanism (stim merges these)
                if o_ in g:
                    q = g[o_][0]
                    g[o_][0] = q * (1 - p_) + p_ * (1 - q)
                    self.n_merged += 1
                else:
                    g[o_] = [p_, o_, k_]
            g = sorted(g.values(), key=lambda e: (-e[0], e[2]))
            self.n_dominated += len(g) - 1
            self.n_ties += sum(1 for e in g[1:] if e[0] == g[0][0])
            p[m], obs[m], orig[m] = g[0]
        if np.any(p >= 0.5):
            raise ValueError("a mechanism with p >= 1/2 has non-positive weight")
        self.p, self.L, self.orig = p, obs, orig
        self.w = np.log((1 - p) / p)
        if not reduce:                 # (detectors, observables) keys: only exact duplicates were merged
            keys = [k_[0] for k_ in keys]
        rows = np.concatenate([np.array(k_, np.int64) for k_ in keys]) if keys else np.zeros(0, np.int64)
        cols = np.repeat(np.arange(len(keys)), [len(k_) for k_ in keys])
        self.H = sp.csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(self.n_det, len(keys)))
        self.Hr = self.H.tocsr()
        self.n_mech = len(keys)
        self.keys = keys
        self.index = {key: m for m, key in enumerate(keys)}   # with reduce=False the last of equal keys wins

    def syndrome(self, x_idx) -> np.ndarray:
        """Fired detectors (sorted indices) of the mechanisms ``x_idx``."""
        x = np.zeros(self.n_mech, np.uint8)
        x[np.asarray(x_idx, np.int64)] = 1
        return np.flatnonzero((self.H @ x) % 2)

    def weight(self, x_idx) -> float:
        return float(self.w[np.asarray(x_idx, np.int64)].sum())

    def observables(self, x_idx) -> int:
        out = 0
        for m in np.asarray(x_idx, np.int64):
            out ^= int(self.L[m])
        return out

    def check(self, x_idx, fired) -> bool:
        return np.array_equal(self.syndrome(x_idx), np.sort(np.asarray(fired, np.int64)))


def fired_of(row) -> np.ndarray:
    """Fired detector indices of one shot (a bool / uint8 row)."""
    return np.flatnonzero(np.asarray(row))


class CpSat:
    """CP-SAT model of one DEM; ``solve(fired, hint)`` per shot."""

    def __init__(self, dem: Dem, workers: int = 1, time_limit: float = 3600.0, params: dict | None = None):
        from ortools.sat.python import cp_model
        self.cp_model, self.dem = cp_model, dem
        m = cp_model.CpModel()
        self.x = [m.new_bool_var(f"x{j}") for j in range(dem.n_mech)]
        self.y = [m.new_bool_var(f"y{i}") for i in range(dem.n_det)]   # fixed per shot to 1 - s_i
        Hr = dem.Hr
        for i in range(dem.n_det):
            lits = [self.x[j] for j in Hr.indices[Hr.indptr[i]:Hr.indptr[i + 1]]]
            m.add_bool_xor(lits + [self.y[i]])
        self.wi = np.rint(dem.w * SCALE).astype(np.int64)
        m.minimize(sum(int(c) * v for c, v in zip(self.wi, self.x)))
        self.model = m
        self.y_index = [v.index for v in self.y]
        self.x_index = [v.index for v in self.x]
        self.params = dict(num_search_workers=workers, max_time_in_seconds=time_limit, log_search_progress=False)
        self.params.update(params or {})

    def solve(self, fired, hint=None) -> dict:
        cp_model, dem = self.cp_model, self.dem
        s = np.zeros(dem.n_det, np.int64)
        s[np.asarray(fired, np.int64)] = 1
        proto = self.model.proto
        for i, v in zip(self.y_index, 1 - s):
            dom = proto.variables[i].domain
            dom[0] = dom[1] = int(v)
        self.model.clear_hints()
        if hint is not None:
            hs = set(int(j) for j in hint)
            for j, v in enumerate(self.x):
                self.model.add_hint(v, j in hs)
        solver = cp_model.CpSolver()
        for k, v in self.params.items():
            setattr(solver.parameters, k, v)
        t0 = time.perf_counter()
        st = solver.solve(self.model)
        secs = time.perf_counter() - t0
        name = solver.status_name(st)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"status": name, "optimal": False, "x": None, "secs": secs}
        vals = np.array([solver.boolean_value(v) for v in self.x])
        x = np.flatnonzero(vals)
        return {"status": name, "optimal": st == cp_model.OPTIMAL, "x": x, "secs": secs,
                "obj_int": int(round(solver.objective_value)), "bound": solver.best_objective_bound / SCALE,
                "weight": dem.weight(x)}


class Highs:
    """HiGHS MILP of one DEM (Cain et al.'s formulation); ``solve(fired, hint)`` per shot."""

    def __init__(self, dem: Dem, time_limit: float = 3600.0, options: dict | None = None):
        import highspy
        self.highspy, self.dem = highspy, dem
        nx, nz = dem.n_mech, dem.n_det
        deg = np.diff(dem.Hr.indptr)
        self.deg = deg
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nx + nz, nz
        lp.col_cost_ = np.concatenate([dem.w, np.zeros(nz)])
        lp.col_lower_ = np.zeros(nx + nz)
        lp.col_upper_ = np.concatenate([np.ones(nx), np.floor(deg / 2)])
        lp.row_lower_ = np.zeros(nz)
        lp.row_upper_ = np.zeros(nz)
        A = sp.hstack([dem.H.astype(np.float64), sp.diags(-2.0 * np.ones(nz))]).tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        lp.integrality_ = [highspy.HighsVarType.kInteger] * (nx + nz)
        self.lp = lp
        self.opts = dict(output_flag=False, threads=1, mip_rel_gap=0.0, mip_abs_gap=ABS_GAP, time_limit=time_limit,
                         random_seed=0)
        self.opts.update(options or {})

    def solve(self, fired, hint=None) -> dict:
        highspy, dem = self.highspy, self.dem
        nx, nz = dem.n_mech, dem.n_det
        s = np.zeros(nz)
        s[np.asarray(fired, np.int64)] = 1
        lp = self.lp
        lp.row_lower_ = s
        lp.row_upper_ = s
        lp.col_upper_ = np.concatenate([np.ones(nx), np.floor((self.deg - s) / 2)])
        h = highspy.Highs()
        for k, v in self.opts.items():
            h.setOptionValue(k, v)
        h.passModel(lp)
        if hint is not None:
            x0 = np.zeros(nx)
            x0[np.asarray(hint, np.int64)] = 1
            z0 = (dem.Hr @ x0 - s) / 2
            sol = highspy.HighsSolution()
            sol.col_value = list(np.concatenate([x0, z0]))
            sol.value_valid = True
            h.setSolution(sol)
        t0 = time.perf_counter()
        h.run()
        secs = time.perf_counter() - t0
        ms = h.getModelStatus()
        name = h.modelStatusToString(ms)
        info = h.getInfo()
        if info.primal_solution_status != 2:   # kSolutionStatusFeasible
            return {"status": name, "optimal": False, "x": None, "secs": secs}
        xv = np.asarray(h.getSolution().col_value[:nx])
        x = np.flatnonzero(xv > 0.5)
        return {"status": name, "optimal": ms == highspy.HighsModelStatus.kOptimal, "x": x, "secs": secs,
                "obj": float(info.objective_function_value), "bound": float(info.mip_dual_bound),
                "gap": float(info.mip_gap), "nodes": int(info.mip_node_count), "weight": dem.weight(x)}


class Scip:
    """SCIP with one XOR constraint per detector, XOR(x_j: j touches i, y_i) = 1, where the Boolean y_i is fixed to
    1 - s_i per shot; the model is built once per DEM and only y's bounds change between shots (``freeTransform``).
    SCIP handles XOR constraints natively (parity propagation, Gaussian elimination in presolving)."""

    def __init__(self, dem: Dem, time_limit: float = 3600.0, params: dict | None = None, heuristics: bool = True,
                 obs_constraint: bool = False):
        """``obs_constraint``: one more XOR fixes the parity of observable 0 per shot (``solve(..., obs=q)``): the
        least weight of the other class."""
        import pyscipopt
        self.pyscipopt, self.dem = pyscipopt, dem
        m = pyscipopt.Model()
        m.hideOutput()
        if not heuristics:            # the start solution (long-beam Tesseract) replaces SCIP's primal heuristics
            m.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        # constraints/xor/presolpairwise off: SCIP 10's pairwise XOR presolving returned a wrong "infeasible" and a
        p = {"limits/time": time_limit, "limits/gap": 0.0, "limits/absgap": ABS_GAP,
             "randomization/randomseedshift": 0, "constraints/xor/presolpairwise": False}
        p.update(params or {})
        for k, v in p.items():
            m.setParam(k, v)
        self.x = [m.addVar(vtype="B", obj=float(w)) for w in dem.w]
        self.y = [m.addVar(vtype="B", obj=0.0) for _ in range(dem.n_det)]
        Hr = dem.Hr
        for i in range(dem.n_det):
            m.addConsXor([self.x[j] for j in Hr.indices[Hr.indptr[i]:Hr.indptr[i + 1]]] + [self.y[i]], True)
        self.z = None
        if obs_constraint:
            self.z = m.addVar(vtype="B", obj=0.0)
            m.addConsXor([self.x[j] for j in np.flatnonzero(dem.L & 1)] + [self.z], True)
        self.m = m

    def solve(self, fired, hint=None, obs: int | None = None, objlimit: float | None = None) -> dict:
        """``objlimit``: only solutions lighter than it count; with none SCIP ends "infeasible" (result "x" None).
        SCIP keeps earlier shots' solutions across ``freeTransform``, so a returned solution is used only if it was
        found in this solve (status optimal / time limit) and reproduces this shot's syndrome (callers check)."""
        dem, m = self.dem, self.m
        t0 = time.perf_counter()
        m.freeTransform()
        m.setObjlimit(m.infinity() if objlimit is None else float(objlimit))
        s = np.zeros(dem.n_det, np.int64)
        s[np.asarray(fired, np.int64)] = 1
        for v, val in zip(self.y, 1 - s):
            m.chgVarLb(v, float(val))
            m.chgVarUb(v, float(val))
        if self.z is not None:             # observable parity forced to obs: z = 1 - obs
            assert obs in (0, 1)
            m.chgVarLb(self.z, float(1 - obs))
            m.chgVarUb(self.z, float(1 - obs))
        if hint is not None:
            sol = m.createSol()
            hs = set(int(j) for j in hint)
            for j in hs:
                m.setSolVal(sol, self.x[j], 1.0)
            for v, val in zip(self.y, 1 - s):
                m.setSolVal(sol, v, float(val))
            m.addSol(sol, free=True)
        t1 = time.perf_counter()
        m.optimize()
        secs = time.perf_counter() - t1
        st = m.getStatus()
        if m.getNSols() == 0 or st == "infeasible":
            return {"status": st, "optimal": False, "x": None, "secs": secs, "prep_s": t1 - t0}
        best = m.getBestSol()
        xs = np.flatnonzero(np.array([m.getSolVal(best, v) for v in self.x]) > 0.5)
        return {"status": st, "optimal": st == "optimal", "x": xs, "secs": secs, "prep_s": t1 - t0,
                "bound": float(m.getDualbound()), "nodes": int(m.getNNodes()), "weight": dem.weight(xs)}


def make_solver(name: str, dem: Dem, time_limit: float = 3600.0):
    """The solvers by name: "scip" (the read's setting: SCIP, primal heuristics off, started from long-beam
    bundle different HiGHS builds and cannot share a process: one solver family per process."""
    if name == "scip":
        return Scip(dem, time_limit=time_limit, heuristics=False)
    if name == "scip_default":
        return Scip(dem, time_limit=time_limit)
    if name == "highs":
        return Highs(dem, time_limit=time_limit)
    if name in ("cpsat", "cpsat8"):
        return CpSat(dem, workers=8 if name == "cpsat8" else 1, time_limit=time_limit)
    raise ValueError(name)


def exhaustive(dem: Dem, fired) -> tuple[float, list[np.ndarray]]:
    """Least weight and every optimal assignment, by meet in the middle over all 2^n assignments (n <= 40)."""
    n = dem.n_mech
    if n > 40:
        raise ValueError(f"{n} mechanisms: too many for exhaustive search")
    syn = np.zeros(n, object)
    for j in range(n):
        v = 0
        for i in dem.H.indices[dem.H.indptr[j]:dem.H.indptr[j + 1]]:
            v |= 1 << int(i)
        syn[j] = v
    target = 0
    for i in np.asarray(fired, np.int64):
        target |= 1 << int(i)
    a, b = n // 2, n - n // 2

    def table(lo, k):
        syns, ws = [0], [0.0]
        for j in range(lo, lo + k):
            syns += [s_ ^ syn[j] for s_ in syns]
            ws += [w_ + dem.w[j] for w_ in ws]
        return syns, ws

    s1, w1 = table(0, a)
    s2, w2 = table(a, b)
    best2: dict[int, list[int]] = {}
    for m, (s_, w_) in enumerate(zip(s2, w2)):
        best2.setdefault(s_, []).append(m)
    best, sols = math.inf, []
    for m1, (s_, w_) in enumerate(zip(s1, w1)):
        for m2 in best2.get(s_ ^ target, ()):
            tot = w_ + w2[m2]
            if tot < best - 1e-12:
                best, sols = tot, [(m1, m2)]
            elif abs(tot - best) <= 1e-12:
                sols.append((m1, m2))
    out = []
    for m1, m2 in sols:
        bits = [j for j in range(a) if m1 >> j & 1] + [a + j for j in range(b) if m2 >> j & 1]
        out.append(np.array(bits, np.int64))
    return best, out
