"""Opponents on the Clifford stage's circuits: Cain et al. and LoMatching with Definition-1 regions
layers) and longer mixed circuits. The two use the same subgraphs, detector for detector, and agree on almost every
shot: on circuits with S a few shots in 10^4 get different corrections (12 and 38 of 20,000 at d = 3 and 5 on SI C0 IS C1
with preparation XZ and readout ZX) with the same failure count to within a few shots."""
import itertools

import numpy as np
import pytest

from duet import clifford_circuits as cc
from duet.decoders.cain2025 import CainDecoder
from duet.decoders.lomatching_dec import LoMatchingRegions

CASES = [(["C0"], "ZZ", "ZZ"), (["C1"], "ZZ", "ZZ"), (["C0"], "XZ", "ZZ"), (["C0"], "ZZ", "XX"), (["HI"], "ZZ", "XZ"),
         (["SI"], "XZ", "XZ"), (["IS"], "ZZ", "ZZ"), (["II"], "ZX", "ZX"), (["C0", "SH"], "XZ", "ZZ"),
         (["HS", "C0", "SH", "C1"], "ZX", "ZX"), (["SI", "C0", "IS", "C1"], "XZ", "ZX")]


@pytest.mark.parametrize("d", [3, 5])
def test_cain_and_lomatching_agree_on_short_and_mixed_circuits(d):
    checked = 0
    for lay, prep, meas in CASES:
        c, meta = cc.make_circuit(d, lay, 0.003, tuple(prep), tuple(meas), meta=True)
        if c.num_observables == 0:
            continue
        checked += 1
        dets, obs = c.compile_detector_sampler(seed=11).sample(2000, separate_observables=True)
        cd = CainDecoder(c)
        regs = cc.regions(meta, lay, meas)
        assert [sorted(map(int, v[0])) for v in cd.subgraphs] == [sorted(map(int, r)) for r in regs], (lay, prep, meas)
        cain = np.asarray(cd.decode_batch(dets), bool)
        lom = np.asarray(LoMatchingRegions(c, regs).decode_batch(dets), bool)
        assert cain.shape == obs.shape
        assert (cain != lom).any(1).mean() <= 0.005, (lay, prep, meas)
        fc, fl = (cain != obs).any(1).sum(), (lom != obs).any(1).sum()
        assert abs(int(fc) - int(fl)) <= 0.002 * len(obs), (lay, prep, meas, fc, fl)
        assert fc < 0.2 * len(obs), (lay, prep, meas)                       # decodes, far from a coin toss
    assert checked >= 8


def test_every_depth_one_layer_and_basis_builds():
    d = 3
    for lay in (["C0"], ["C1"], ["II"], ["HI"], ["SI"], ["HS"], ["SS"]):
        for prep in itertools.product("ZX", repeat=2):
            for meas in itertools.product("ZX", repeat=2):
                c = cc.make_circuit(d, lay, 0.003, prep, meas)
                if c.num_observables:
                    CainDecoder(c)
