""": the development bank's seeds and circuits (scripts/dev_bank.py)."""
import glob
import importlib.util
import os

import pytest

from seed_literals import seed_values
from duet import bench_sets as bs
from duet import lukin_bench as lb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bank = _mod("scripts/dev_bank.py", "lukin_bank")




def test_seed_literals_in_tree():
    """No executable seed literal in another module falls in this run's range."""
    lo, hi = bank.BANK_SEED_BASE, bank.BANK_SEED_BASE + bank.BANK_SEED_SPAN
    bad = []
    for top in ("duet", "ops", "tests"):
        for dp, _, fs in os.walk(os.path.join(ROOT, top)):
            for f in fs:
                if f.endswith(".py") and os.path.join(dp, f) != os.path.join(ROOT, "scripts", "dev_bank.py"):
                    bad += [(f, v) for v in seed_values(open(os.path.join(dp, f), encoding="utf-8").read())
                            if lo <= v < hi]
    assert not bad, bad


def test_seed_layout():
    seen = {bank.seed(g, d, D, i) for g in (1, 2, 3, 9) for d in (3, 5, 7, 99) for D in (1, 10, 999)
            for i in (0, 39, 10 ** 9 - 1)}
    assert len(seen) == 4 * 4 * 3 * 3
    for bad in ((4, 5, 1, 0), (1, 100, 1, 0), (1, 5, 1000, 0), (1, 5, 0, 0), (1, 5, 1, 10 ** 9)):
        with pytest.raises(AssertionError):
            bank.seed(*bad)


def test_bank_points():
    """The bank's circuits are the benchmark builder's (duet.bench_sets.lukin) for random_logical's indices 20-39, the
    test set's 0-19 excluded; one seed per circuit, all distinct and in the run's range."""
    pts = bank.points()
    assert [(p["d"], p["depth"]) for p in pts] == [(d, D) for d in (3, 5, 7) for D in range(1, 11)]
    test = {(D, c["layers"], c["meas"]) for D in range(1, 11) for c in
            [bs._c(*bs.lukin(D, i)) for i in range(bs.LUKIN_CIRCUITS)]}
    seeds = []
    for p in pts:
        assert p["shots"] == bank.SHOTS[p["d"]][p["depth"] - 1] and p["noise"] == "lukin" and p["p"] == 0.003
        assert [c["index"] for c in p["circuits"]] == list(range(20, 40))
        for c in p["circuits"]:
            g = lb.random_logical(p["depth"], c["index"])
            lay, prep, meas, prods = bs.lukin(p["depth"], c["index"])
            assert c["layers"] == " ".join(lay) and c["prep"] == "XXXX" and c["meas"] == g["bases"] == meas
            assert c["products"][0] == [b for b, ch in enumerate(g["target"]) if ch != "I"] and c["pauli"] == "gate"
            assert c["products"] == [list(s) for s in prods] and c["target_slot"] == 0
            seeds.append(int(c["seed"]))
        if p["depth"] >= 3:      # D = 1, 2 circuits repeat among 20 draws of a small set; deeper ones are new
            assert not any((p["depth"], c["layers"], c["meas"]) in test for c in p["circuits"])
    assert len(set(seeds)) == len(seeds) == 600
    assert all(bank.BANK_SEED_BASE <= s < bank.BANK_SEED_BASE + bank.BANK_SEED_SPAN for s in seeds)
    bench = {int(c["seed"]) for p in bs.external_points(families=("lukin",), noises=("lukin",)) for c in p["circuits"]}
    assert not bench & set(seeds)


def _args(**kw):
    from duet.net import train_cl
    return train_cl.parser().parse_args(["--out", "/nonexistent", "--d", "3", "--noise", "lukin", "--seed", "8461",
                                         "--Ls", "4,1,6", "--block-counts", "2,3"] +
                                        [x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", str(v))])


def test_lukin_chunks_are_the_benchmark_circuits():
    """A lukin chunk's circuit, products and labels are the benchmark builder's for the same logical circuit
    (random_logical(D, k, seed)), with physical Paulis; its stim circuit equals bench_sets.circuit's."""
    import stim
    from duet.net import spec_cl, train_cl
    a = _args(lukin_share=0.5)
    plan = train_cl.batch_plan(a, 100)
    for k in (1, 7, 12):
        kind = plan(2 * k + 1)
        assert kind[2] == "lukin" and kind[0] == 4
        D = kind[1] // 2
        job = train_cl.chunk_job(a, k, kind, (), ())
        lay, prep, meas, p, prods, pauli = train_cl._draw_chunk(job)
        g = lb.random_logical(D, k, seed=8461)
        assert (" ".join(lay), "".join(prep), "".join(meas), pauli) == (
            " ".join(bs.lukin(D, k, seed=8461)[0]), "XXXX", g["bases"], "gate")
        assert prods[0] == [b for b, ch in enumerate(g["target"]) if ch != "I"]
        for d in (3, 7) if k == 1 else (3,):          # d = 7: A3's chunks
            entry = dict(d=d, p=0.003, noise="lukin")
            c_ = bs._c(*bs.lukin(D, k, seed=8461))
            ref = bs.circuit(entry, c_)
            s = spec_cl.build(d, lay, prep, meas, p, "lukin", prods, pauli=pauli)
            # (ref through its text: the chunk's circuit is text, a text round trip moves the floats in the last bits)
            assert stim.Circuit(s["circuit"]) == stim.Circuit(str(ref)) and s["labelled"].startswith(s["circuit"].rstrip())
            tab, ext = train_cl.tables(d, lay, prep, meas, p, "lukin", prods, pauli)
            assert tab["n_obs"] == ref.num_observables == len(prods) and tab["m"] == ref.num_detectors
        # physical Paulis carry noise that frame Paulis do not: the models differ when a Pauli acts
        if any(ch in "XYZ" for t in lay for ch in t):
            assert spec_cl.build(3, lay, prep, meas, p, "lukin", prods)["circuit"] != s["circuit"]


def test_lukin_plan_and_p_ramp():
    from duet.net import train_cl
    a0, a = _args(), _args(lukin_share=0.5, lukin_depths="1,2,3")
    p0, p1 = train_cl.batch_plan(a0, 100), train_cl.batch_plan(a, 100)
    kinds = [p1(i) for i in range(12)]
    assert [x[2] for x in kinds] == [False, "lukin"] * 6
    assert [x[1] for x in kinds[1::2]] == [2, 4, 6, 2, 4, 6]
    assert [p1(2 * i)[:2] for i in range(6)] == [p0(i)[:2] for i in range(6)]       # the mix's own order kept
    b = _args(p_tri=0.2, samples=1000, chunk=10)
    ps = [train_cl.chunk_job(b, k, (2, 4, False, False), (), ())[1] for k in range(100)]
    assert ps[:80] == [0.003] * 80 and abs(ps[90] - 0.006) < 1e-12 and max(ps) <= 0.006 + 1e-12
    assert ps[85] == pytest.approx(0.0045) and ps[95] == pytest.approx(0.0045)


START = os.environ.get("BANK_START") or next(iter(sorted(glob.glob(
    "networks/lukin/d5_ft4_s*/final_ema.pt"))), "")


@pytest.mark.skipif(not START, reason="the d = 5 Lukin-noise final (BANK_START or ft/lukin)")
def test_grown_arm_starts_with_the_start():
    """Arm C9: the start grown to 9 blocks (model_cl.load_grown, the two new blocks zero output) computes the start's
    outputs on a batch of lukin-family shots."""
    import torch
    from duet.net import model_cl as M
    from duet.net import train_cl
    ck = torch.load(START, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    small = M.NetCL(**ck["cfg"]).eval()
    small.load_state_dict(ck["model"])
    big = M.NetCL(**dict(ck["cfg"], blocks=9)).eval()
    assert M.load_grown(big, ck) == 9 - ck["cfg"]["blocks"] > 0
    lay, prep, meas, prods = bs.lukin(3, 20)
    tab, ext = train_cl.tables(d, lay, tuple(prep), tuple(meas), 0.003, "lukin", [list(s) for s in prods], "gate")
    dp, op, _ = ext.compile_sampler(seed=5).sample(64, bit_packed=True)
    b = train_cl.collate([(tab, dp, op)], d, torch.device("cpu"), train=False)
    with torch.no_grad():
        for x, y in zip(train_cl.forward(small, b, train=False), train_cl.forward(big, b, train=False)):
            assert torch.equal(x, y)




def test_fresh_points():
    """The fresh d = 7 test set (BANK_SET=fresh7): the first test set's points (circuits, shots, MLE prefix) with new
    seeds seed(3, 7, D, i), distinct from the bank's, the benchmark's and each other."""
    old, new = bank.test_points((7,)), bank.fresh_points((7,))
    assert [p["name"] for p in new] == [p["name"] for p in old] and len(new) == 10
    seeds = []
    for p, q in zip(new, old):
        assert {k: v for k, v in p.items() if k not in ("circuits", "seeds")} == \
            {k: v for k, v in q.items() if k not in ("circuits", "seeds")}
        assert p["seeds"] == [str(bank.seed(3, 7, p["depth"], 0)), str(bank.seed(3, 7, p["depth"], 19) + 1)]
        for c, e in zip(p["circuits"], q["circuits"]):
            assert {k: v for k, v in c.items() if k != "seed"} == {k: v for k, v in e.items() if k != "seed"}
            assert int(c["seed"]) == bank.seed(3, 7, p["depth"], c["index"])
            seeds.append(int(c["seed"]))
    used = {int(c["seed"]) for p in bank.points() + old for c in p["circuits"]}
    assert len(set(seeds)) == len(seeds) == 200 and not set(seeds) & used
