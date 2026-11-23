import numpy as np
import pytest
import stim

from duet.lukin_bench import _Code, make_circuit, random_logical
from duet.mle import Dem, exhaustive, make_solver

# two mechanisms with equal detectors and different observables (the lighter one is kept), one without detectors
DEM = stim.DetectorErrorModel("""
error(0.1) D0 D1
error(0.01) D0 D1 L0
error(0.05) D1 D2
error(0.02) D2
error(0.03) D0
error(0.2) L0
""")


def test_dem_reductions():
    dem = Dem(DEM)
    assert dem.n_raw == 6 and dem.n_no_det == 1 and dem.n_dominated == 1 and dem.n_mech == 4
    m = dem.index[(0, 1)]
    assert dem.p[m] == pytest.approx(0.1) and dem.L[m] == 0
    assert np.allclose(dem.w, np.log((1 - dem.p) / dem.p))
    raw = Dem(DEM, reduce=False)
    assert raw.n_mech == 5


def test_exhaustive_matches_reductions():
    dem, raw = Dem(DEM), Dem(DEM, reduce=False)
    for fired in ([0], [1, 2], [0, 2], [0, 1, 2], []):
        w, sols = exhaustive(dem, fired)
        assert w == pytest.approx(exhaustive(raw, fired)[0])
        assert all(dem.check(x, fired) and dem.weight(x) == pytest.approx(w) for x in sols)
    assert exhaustive(dem, [0])[0] == pytest.approx(np.log(0.97 / 0.03))   # D0 alone beats D0D1 + D1D2 + D2


@pytest.mark.parametrize("name", ["scip", "highs"])
def test_solver_equals_exhaustive(name):
    pytest.importorskip({"scip": "pyscipopt", "highs": "highspy"}[name])
    circ = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.02,
                                  before_measure_flip_probability=0.01, after_reset_flip_probability=0.01)
    dem = Dem(circ.detector_error_model(decompose_errors=False))
    sol = make_solver(name, dem, 60.0)
    dets = circ.compile_detector_sampler(seed=5).sample(40)
    for row in dets:
        fired = np.flatnonzero(row)
        r = sol.solve(fired)
        assert r["optimal"] and dem.check(r["x"], fired)
        assert r["weight"] == pytest.approx(exhaustive(dem, fired)[0], abs=1e-6)


def test_ccw_rotation_variant_is_valid():
    for d in (3, 5):
        _Code(d, ccw=True)                   # asserts that supports and logicals map onto the other type
    lg = random_logical(4, 1)
    c = make_circuit(3, lg, 0.0, round_after_1q_layer=True, h_rotation="ccw")
    dets, obs = c.compile_detector_sampler(seed=1).sample(32, separate_observables=True)
    assert not dets.any() and not obs.any()
    make_circuit(3, lg, 3e-3, round_after_1q_layer=True, h_rotation="ccw").detector_error_model()


def test_scip_objective_limit_other_class():
    """The margin test's objective limit : above the other class's least weight SCIP
    returns it, below it "infeasible" and no solution, also after earlier shots' solutions. Tolerance contract: SCIP
    compares the objective with the limit only up to its relative feasibility tolerance (numerics/feastol 1e-6, times
    max(1, |weight|)): with the limit 1e-6 to 1e-5 below the other class's weight it still returned that solution on
    10 of these 12 shots, with feastol 1e-9 on none. The margin test's limits lie at least 1e-4 (W + C)
    from the MLE's weight, 100 times that tolerance, so the test puts the limit 1e-4 x max(1, weight) above and below."""
    pytest.importorskip("pyscipopt")
    from duet.mle import Scip
    circ = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.02,
                                  before_measure_flip_probability=0.01, after_reset_flip_probability=0.01)
    dem = Dem(circ.detector_error_model(decompose_errors=False))
    assert dem.n_mech <= 16
    allx = (np.arange(2 ** dem.n_mech)[:, None] >> np.arange(dem.n_mech)) & 1
    syn = (allx @ dem.H.T.toarray()) % 2
    obs, w = (allx @ (dem.L & 1)) % 2, allx @ dem.w
    sol = Scip(dem, obs_constraint=True)
    for row in circ.compile_detector_sampler(seed=7).sample(12):
        fired = np.flatnonzero(row)
        same = np.all(syn == row.astype(np.int64), axis=1)
        q = int(obs[same][np.argmin(w[same])])            # the MLE's class; the other class is 1 - q
        w_other = w[same & (obs == 1 - q)].min()
        off = 1e-4 * max(1.0, w_other)
        r = sol.solve(fired, obs=1 - q, objlimit=w_other + off)
        assert r["optimal"] and dem.check(r["x"], fired) and r["weight"] == pytest.approx(w_other, abs=1e-6)
        r = sol.solve(fired, obs=1 - q, objlimit=w_other - off)
        assert r["status"] == "infeasible" and r["x"] is None


