""": a NetCL with more trunk blocks, warm-started from a checkpoint with fewer
(duet.net.model_cl.load_grown), computes the checkpoint's function: equal outputs on a batch of a real circuit."""
import numpy as np
import torch

from duet.net import model_cl as M
from duet.net import train_cl


def _batch(d=3):
    tab, ext = train_cl.tables(d, ["HI", "C0", "II"], ("Z", "Z"), ("X", "X"), 0.003, "uniform")
    dp, op, _ = ext.compile_sampler(seed=5).sample(64, bit_packed=True)
    return train_cl.collate([(tab, dp, op)], d, torch.device("cpu"), train=False)


def test_grown_network_computes_the_same_function():
    torch.manual_seed(0)
    small = M.NetCL(3, width=16, bott=8, blocks=2, heads=2)
    with torch.no_grad():                     # every weight away from its initialisation, the zero projections too
        for q in small.parameters():
            q.add_(0.05 * torch.randn_like(q))
    ck = dict(cfg=small.cfg, model=small.state_dict())
    big = M.NetCL(3, width=16, bott=8, blocks=4, heads=2)
    assert M.load_grown(big, ck) == 2
    assert all(float(big.blocks[i].up.weight.abs().sum()) == 0 for i in (2, 3))
    assert all(float(big.couple[i].up.weight.abs().sum()) == 0 for i in (3, 4))
    b = _batch()
    small.eval(), big.eval()
    with torch.no_grad():
        out_s = train_cl.forward(small, b, train=False)
        out_b = train_cl.forward(big, b, train=False)
    for x, y in zip(out_s, out_b):
        assert torch.equal(x, y)
    assert float(out_s[0].abs().sum()) > 0
    # the same configuration loads as before; another width is refused
    same = M.NetCL(3, width=16, bott=8, blocks=2, heads=2)
    assert M.load_grown(same, ck) == 0
    try:
        M.load_grown(M.NetCL(3, width=32, bott=8, blocks=4, heads=2), ck)
        raise RuntimeError("accepted another width")
    except AssertionError:
        pass
    assert np.isfinite(out_b[0].numpy()).all()
