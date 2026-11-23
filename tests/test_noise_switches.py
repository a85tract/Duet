"""noise switches (duet.clifford_circuits.make_circuit, noise "<model>/<switch>..."): every default call
gives the same stim circuit text, bit for bit, as the code before the switches (commit ae4b946), and each switch removes
exactly its component."""
import hashlib

import numpy as np
import pytest

from duet import clifford_circuits as cc

# sha256 of the joined circuit texts of default_calls, computed with commit ae4b946 (before the switches)
BEFORE = "a36831b995f29097"


def default_calls():
    """A fixed spread of default calls: four noise models, two to four blocks, every gate, both Pauli modes."""
    rng = np.random.default_rng([2026, 10, 31])
    out = []
    for noise in ("uniform", "lukin", "si1000", "cain"):
        for d in (3, 5):
            for L in (1, 2, 4):
                for _ in range(3):
                    lay, prep, meas = cc.sample_circuit(rng, L, 0.4, "IHSXYZ")
                    out.append((d, lay, prep, meas, dict(noise=noise)))
                lay, prep, meas = cc.sample_circuit_k(rng, 3, L, 0.5, "IHS")
                out.append((d, lay, prep, meas, dict(noise=noise)))
                lay, prep, meas = cc.sample_circuit_k(rng, 4, L, 0.5, "IHSX")
                out.append((d, lay, prep, meas, dict(noise=noise, pauli="gate")))
    out.append((3, ["C0", "II"], ("X", "Z"), ("Z", "Z"), {}))
    return out


def combined_sha(calls=None):
    h = hashlib.sha256()
    for d, lay, prep, meas, kw in calls or default_calls():
        h.update(str(cc.make_circuit(d, lay, 0.003, prep, meas, **kw)).encode())
    return h.hexdigest()[:16]


def test_default_circuits_unchanged():
    assert combined_sha() == BEFORE


def _mass(c):
    """Summed noise probability over every noisy target (channels and noisy measurements)."""
    return sum(i.gate_args_copy()[0] * len(i.targets_copy()) for i in c.flattened()
               if i.gate_args_copy() and i.name not in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"))


@pytest.mark.parametrize("sw", ["dreset", "areset", "dmeas", "ameas", "idle", "g1", "g2"])
def test_switch_removes_its_component(sw):
    lay, prep, meas = ["C0", "HS"], ("X", "Z"), ("Z", "X")
    base = cc.make_circuit(5, lay, 0.003, prep, meas, products=None)
    off = cc.make_circuit(5, lay, 0.003, prep, meas, noise=f"uniform/{sw}")
    assert off.num_detectors == base.num_detectors and off.num_observables == base.num_observables
    assert _mass(off) < _mass(base)
    assert str(off.without_noise()) == str(base.without_noise())
    assert str(cc.make_circuit(5, lay, 0.003, prep, meas, noise="uniform/")) == str(base)


def test_encoded_preparation():
    lay, prep, meas = ["C0"], ("X", "Z"), ("Z", "Z")
    enc = cc.make_circuit(5, lay, 0.003, prep, meas, noise="uniform/enc")
    encu = cc.make_circuit(5, lay, 0.003, prep, meas, noise="uniform/encu")
    base = cc.make_circuit(5, lay, 0.003, prep, meas)
    # encu: the uniform layout's detectors; enc: every first-round check a detector (two blocks, d^2 - 1 checks each)
    assert encu.num_detectors == base.num_detectors < enc.num_detectors
    t0 = lambda c: sum(1 for v in c.get_detector_coordinates().values() if v[2] == 0)   # noqa: E731
    assert t0(enc) == 2 * (5 * 5 - 1)
    for c in (enc, encu):                     # deterministic without noise
        c.without_noise().detector_error_model()
        assert not np.any(c.without_noise().compile_detector_sampler().sample(64))


def test_draw_default_stream_and_ent_k2():
    """train_cl.draw: ent_k2 = 0 gives the two-block stream of before (its code copied here); ent_k2 > 0 adds Bell pairs
    that idle to the depth, at every depth from 1."""
    from duet import kblock_sets
    from duet.net import train_cl

    def old(rng, L, p_cnot, gates, cnot_share, p_set):
        if cnot_share and rng.random() < cnot_share:
            return cc.cnot_layers(rng.integers(0, 2, size=L).tolist()), ("Z", "Z"), ("Z", "Z")
        q = float(p_set[rng.integers(len(p_set))]) if len(p_set) else p_cnot
        return cc.sample_circuit(rng, L, q, gates)

    p_set = (0.0, 1 / 3, 2 / 3, 1.0)
    for k in range(200):
        L = (1, 2, 4, 6, 8)[k % 5]
        r1, r2 = np.random.default_rng([7, k]), np.random.default_rng([7, k])
        assert train_cl.draw(r1, L, 1 / 3, "IHS", 0.25, p_set, 2, 0.1) == old(r2, L, 1 / 3, "IHS", 0.25, p_set)
    bell = 0
    for k in range(400):
        L = (1, 2, 4)[k % 3]
        lay, prep, meas = train_cl.draw(np.random.default_rng([8, k]), L, 1 / 3, "IHS", 0.25, p_set, 2, 0.1, 0.1)
        assert len(lay) == L and cc.logical_products(lay, prep, meas)
        r = np.random.default_rng([8, k])
        u = r.random()
        if 0.25 <= u < 0.35:
            assert (lay, prep, meas) == kblock_sets.ent(r, 2, L) and meas in (("Z", "Z"), ("X", "X"))
            bell += 1
    assert 20 < bell < 70


if __name__ == "__main__":
    print(combined_sha())
