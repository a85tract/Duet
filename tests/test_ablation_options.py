"""ablation options of NetCL (data/training/TRAIN.md): --couple late / none, --head pool, and --accum above
the chunks per batch. CPU, small networks."""
import copy
import os

import numpy as np
import pytest
import torch

from duet.net import model_cl as M
from duet.net import train_cl as T

D = 3


def _item(K, L, seed, n=8, cnot_share=0.0):
    """One drawn K-block circuit of depth L (train_cl.draw): (tables, packed detectors, packed observables) of n shots."""
    rng = np.random.default_rng(seed)
    lay, prep, meas = T.draw(rng, L, 0.6, "IHS", cnot_share, (), K)
    tab, ext = T.tables(D, lay, prep, meas, 0.003, "lukin")
    dp, op, _ = ext.compile_sampler(seed=seed).sample(n, bit_packed=True)
    return tab, dp, op


def _batch(K, L, seed, n=8, cnot_share=0.0):
    it = _item(K, L, seed, n, cnot_share)
    return it[0], T.collate([it], D, torch.device("cpu"), train=False)


def _randomise(net, seed=0):
    """Every zero-initialised output layer random, so the exchange and the blocks act from the start."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, q in net.named_parameters():
            if name.endswith("up.weight") or name.endswith("up.bias"):
                q.copy_(torch.randn(q.shape, generator=g) * 0.2)
    return net.eval()


def _flip_block_bits(b, blk, frac=0.5, seed=1):
    """A copy of batch b with random detector bits of block blk flipped (gidx // (T H W) is a detector's block)."""
    T_, H = b["mx"].shape[2], b["mx"].shape[3]
    of = (b["gidx"] // (T_ * H * H)) == blk
    g = torch.Generator().manual_seed(seed)
    flip = (torch.rand(b["bits"].shape, generator=g) < frac) & of
    out = dict(b)
    out["bits"] = b["bits"] ^ flip.to(b["bits"].dtype)
    assert bool(flip.any())
    return out


def _nets(**kw):
    torch.manual_seed(0)
    base = _randomise(M.NetCL(D, 32, 8, 2, 4, **kw))
    return base


def test_defaults_unchanged():
    net = M.NetCL(D, 32, 8, 2, 4)
    assert "couple" not in net.cfg and "head" not in net.cfg
    keys = set(net.state_dict())
    assert set(M.NetCL(D, 32, 8, 2, 4, couple="late").state_dict()) == keys
    assert set(M.NetCL(D, 32, 8, 2, 4, couple="none").state_dict()) == keys
    assert M.NetCL(D, 32, 8, 2, 4, couple="none").cfg["couple"] == "none"
    pool = M.NetCL(D, 32, 8, 2, 4, head="pool")
    assert pool.cfg["head"] == "pool" and isinstance(pool.head, M.PoolCL)
    assert M.NetCL(**pool.cfg).cfg == pool.cfg
    with pytest.raises(AssertionError):
        M.NetCL(D, 32, 8, 2, 4, couple="none", head="pool")


def test_late_and_none_trunk_keep_blocks_apart():
    """With the exchange active (random up layers), block 0's trunk features ignore block 1's bits under late and none
    and depend on them under cross (a circuit with CNOTs between the blocks)."""
    for seed in range(20):
        tab, b = _batch(2, 4, seed, cnot_share=1.0)
        if (tab["pcode"] >= 1).any() and ((tab["pcode"] >= 1) & (tab["pcode"] <= 3)).any():
            break
    b2 = _flip_block_bits(b, 1)
    cross = _nets()
    for mode in ("late", "none"):
        net = M.NetCL(D, 32, 8, 2, 4, couple=mode).eval()
        net.load_state_dict(cross.state_dict())
        args = lambda x: (x["bits"], x["gidx"], x["mx"], x["mz"], x["gate"], x["pcode"], x.get("pblk"),  # noqa: E731
                          x.get("bound"))
        with torch.no_grad():
            f1, f2 = net.features(*args(b)), net.features(*args(b2))
            c1, c2 = cross.features(*args(b)), cross.features(*args(b2))
        assert torch.equal(f1[:, 0], f2[:, 0]), mode
        assert not torch.equal(f1[:, 1], f2[:, 1])
        assert not torch.allclose(c1[:, 0], c2[:, 0]), "cross must carry block 1 into block 0"


def test_none_readout_ignores_uninvolved_blocks():
    """couple none: a product slot's outputs ignore the bits of a block that holds no part of it at any layer; late
    (the joint read-out) does not (its outside bins only damp those sites)."""
    found = False
    for seed in range(200):
        tab, b = _batch(3, 4, seed)
        inv = (tab["offb"] != 4 * D - 1).any((3, 4))                       # (S, T, K): the block holds a part
        for k in range(tab["n_obs"]):
            free = [blk for blk in range(3) if not inv[k, :, blk].any()]
            if free:
                found = True
                break
        if found:
            break
    assert found
    b2 = _flip_block_bits(b, free[0])
    late = _nets(couple="late")
    none = M.NetCL(D, 32, 8, 2, 4, couple="none").eval()
    none.load_state_dict(late.state_dict())
    with torch.no_grad():
        n1, n2 = T.forward(none, b, train=False), T.forward(none, b2, train=False)
        l1, l2 = T.forward(late, b, train=False), T.forward(late, b2, train=False)
    for x, y in zip(n1, n2):
        assert torch.equal(x[:, k], y[:, k])
    assert not torch.equal(l1[0][:, k], l2[0][:, k])


def test_none_involvement_covers_parts():
    """The per-block read-out's support test (per block, layer and representative j: some offset bin is not outside,
    the joint read-out's own notion of support) keeps every block and layer where spec_cl's parts hold an X or Z part
    (it may also keep representatives' lines on a block without a part)."""
    from duet.net import spec_cl
    for seed in range(5):
        rng = np.random.default_rng(seed)
        lay, prep, meas = T.draw(rng, 6, 0.6, "IHS", 0.0, (), 4)
        s = spec_cl.build(D, lay, prep, meas, 0.003, "lukin")
        offb = M.offset_bins(D, s["lines"], s["orient"], s["mx"], s["mz"])
        inv = (offb != 4 * D - 1).any(3)                                     # (S, T, K, d)
        part = s["parts"].any(-1)                                            # (S, T, K)
        assert inv[np.broadcast_to(part[..., None], inv.shape)].all()


def test_pool_forward_backward():
    net = M.NetCL(D, 32, 8, 2, 4, head="pool").train()
    yb = T.collate([_item(3, 4, 3, 16)], D, torch.device("cpu"))
    c, z, l = T.forward(net, yb)
    assert c.shape == yb["y"].shape and z is None and l is None
    loss, lf, ll = M.loss_cl(c, z, l, yb["y"], yb["yl"], yb["u"], yb["slot"], 1.0)
    assert torch.equal(loss, lf) and float(ll) == 0.0
    loss.backward()
    assert net.head.q0.grad is not None and float(net.head.b_sup.grad.abs().sum()) > 0
    ref = (torch.nn.functional.binary_cross_entropy_with_logits(c, yb["y"].float(), reduction="none")
           * yb["slot"].float()).sum(1).mean()
    assert torch.allclose(lf, ref)


def test_accum_split_same_gradient():
    """--accum 8 at 4 chunks per batch (each chunk in two shot slices) gives the gradient of --accum 4 (fp32, CPU)."""
    items = [_item(2, 4, 10 + i, 16) for i in range(4)]
    torch.manual_seed(0)
    net = _randomise(M.NetCL(D, 32, 8, 2, 4)).train()
    grads = []
    for accum in (4, 8):
        net.zero_grad(set_to_none=True)
        per_batch, chunk = 4, 16
        if accum <= per_batch:
            part = per_batch // accum
            mbs = [items[j * part:(j + 1) * part] for j in range(accum)]
        else:
            h = chunk // (accum // per_batch)
            mbs = [[(tab, dp[i:i + h], op[i:i + h])] for tab, dp, op in items for i in range(0, chunk, h)]
        assert len(mbs) == accum
        for mb in mbs:
            bb = T.collate(mb, D, torch.device("cpu"))
            c, z, l = T.forward(net, bb)
            loss, _, _ = M.loss_cl(c, z, l, bb["y"], bb["yl"], bb["u"], bb["slot"], 1.0)
            (loss / accum).backward()
        grads.append(torch.cat([q.grad.flatten() for q in net.parameters() if q.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("opts", [["--couple", "late"], ["--couple", "none"], ["--head", "pool"],
                                  ["--accum", "8"], []])
def test_train_smoke(tmp_path, opts):
    """A few steps of each arm's options through train_cl.main (CPU, tiny), lukin and K = 2, 3, 4 batches."""
    out = str(tmp_path / "run")
    args = ["--out", out, "--d", "3", "--width", "16", "--bott", "8", "--blocks", "1", "--batch", "64", "--chunk", "16",
            "--samples", "640", "--noise", "lukin", "--block-counts", "2,3,4", "--lukin-share", "0.5", "--p-tri", "0.2",
            "--p-cnot-set", "0,0.3333333,0.6666667,1", "--cnot-share", "0.25", "--ent-share", "0.1",
            "--eval-circuits", "1", "--eval-cnot", "1", "--eval-per", "8", "--eval-every", "320", "--workers", "2",
            "--prefetch", "4", "--lookup", "gather", "--seed", "9205"] + opts
    assert T.main(args) == 0
    assert os.path.exists(os.path.join(out, "run.json")) and os.path.exists(os.path.join(out, "final_ema.pt"))
    ck = torch.load(os.path.join(out, "final_ema.pt"), map_location="cpu", weights_only=False)
    net = M.NetCL(**ck["cfg"])
    net.load_state_dict(ck["model"])
    if "--couple" in opts:
        assert net.couple_mode == opts[1]
    if "--head" in opts:
        assert isinstance(net.head, M.PoolCL)
    # an exact resume of a stopped run with the option (max-steps 1, then the rest)
    out2 = str(tmp_path / "run2")
    assert T.main([v if v != out else out2 for v in args] + ["--max-steps", "4"]) == 3
    assert T.main([v if v != out else out2 for v in args]) == 0
    ck2 = torch.load(os.path.join(out2, "final_ema.pt"), map_location="cpu", weights_only=False)
    for k_ in ck["model"]:
        assert torch.allclose(ck["model"][k_].float(), ck2["model"][k_].float(), atol=1e-4), k_
