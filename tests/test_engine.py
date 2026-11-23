"""The Triton engine of NetCL (duet.net.engine_cl, GPU only) and the faster table build (duet.net.spec_fast): on random
weights (every parameter drawn) and shots of mixed-gate circuits (two blocks, and three blocks with pblk and bound),
the engine's decisions equal the eager fp32 read path's (train_cl.collate and forward, TF32 off) on every shot, with
logits within 1e-3 (mode precise) or 0.1 (onepass) at tiles of 16 and 64 sites and in chunks; its decision bits match
its own logits; and spec_fast.build's tables equal spec_cl.build's."""
import numpy as np
import pytest
import torch

from duet import clifford_circuits as cc
from duet.net import model_cl as M
from duet.net import spec_cl, spec_fast, train_cl

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine runs on a GPU")


def circuits(K, n, d=3, L=4, seed=3):
    rng = np.random.default_rng(seed)
    if K == 2:
        return [cc.sample_circuit(rng, L, 0.5, "IHS") for _ in range(n)]
    return [cc.sample_circuit_k(rng, K, L, 0.5, "IHS") for _ in range(n)]


def random_net(d, seed=1, **arch):
    torch.manual_seed(seed)
    net = M.NetCL(d, **arch)
    g = torch.Generator().manual_seed(seed)
    rn = lambda p, mean, std: p.copy_(mean + std * torch.randn(p.shape, generator=g))   # noqa: E731
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv3d)):
                rn(mod.weight, 0.0, mod.weight[0].numel() ** -0.5)
                if mod.bias is not None:
                    rn(mod.bias, 0.0, 0.1)
            elif isinstance(mod, torch.nn.LayerNorm):
                rn(mod.weight, 1.0, 0.1)
                rn(mod.bias, 0.0, 0.1)
        for cp in net.couple:
            rn(cp.maps, 0.0, cp.maps.shape[-1] ** -0.5)
        hd = net.head
        rn(hd.q0, 0.0, 0.5)
        rn(hd.e, 0.0, 0.5)
        hd.b_line.add_(0.3 * torch.randn(hd.b_line.shape, generator=g))
        rn(hd.part, 0.0, 0.5)
        net.w.add_(0.3 * torch.randn(net.w.shape, generator=g))
        rn(net.b, 0.0, 0.3)
    return net


def test_spec_fast_equals_spec_cl():
    for K, d in ((2, 3), (2, 5), (3, 3)):
        for lay, prep, meas in circuits(K, 3, d):
            for noise in ("uniform", "lukin"):
                s0 = spec_cl.build(d, lay, prep, meas, 0.003, noise)
                s1 = spec_fast.build(d, lay, prep, meas, 0.003, noise)
                assert s0.keys() == s1.keys()
                for k, v in s0.items():
                    if isinstance(v, np.ndarray):
                        assert v.dtype == s1[k].dtype and np.array_equal(v, s1[k]), k
                    else:
                        assert type(v) is type(s1[k]) and v == s1[k], k


@gpu
@pytest.mark.parametrize("K", [2, 3])
def test_engine_cl_matches_eager(K):
    from duet.net import engine_cl
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    d, n = 3, 256
    net = random_net(d, width=64, bott=16, blocks=2, heads=4).cuda().eval()
    engines = {(md, BM, ch): engine_cl.EngineCL(net, mode=md, BM=BM, chunk=ch, nlp=8)
               for md in ("precise", "onepass") for BM, ch in ((16, None), (64, 96))}
    for i, (lay, prep, meas) in enumerate(circuits(K, 3, d)):
        tab = train_cl.tables(d, lay, prep, meas, 0.01, "uniform")[0]
        circ = cc.make_circuit(d, lay, 0.01, prep, meas)
        det = circ.compile_detector_sampler(seed=11 + i).sample(n, bit_packed=True)
        zero = np.zeros((n, (tab["n_total"] + 7) // 8), np.uint8)
        with torch.no_grad():
            c_ref = train_cl.forward(net, train_cl.collate([(tab, det, zero)], d, torch.device("cuda"), train=False),
                                     train=False)[0][:, :tab["n_obs"]].cpu().numpy()
        s = spec_fast.build(d, lay, prep, meas, 0.01)
        for (md, BM, ch), eng in engines.items():
            info = eng.circuit(lay, prep, meas, spec=s)
            c, conf, dec = eng.decode(info, det, B=128)
            tol = 1e-3 if md == "precise" else 0.1
            assert np.abs(c - c_ref).max() < tol, (K, i, md, BM, float(np.abs(c - c_ref).max()))
            assert np.array_equal(dec, c > 0) and np.allclose(conf, np.abs(c).min(1))
            if md == "precise":
                assert np.array_equal(dec, c_ref > 0), (K, i, BM)
