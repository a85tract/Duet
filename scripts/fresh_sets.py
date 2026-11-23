"""The fresh test sets' manifests
and the check that their seeds are new. Everything downstream (generation, reads, opponents, trials, scores) runs on
these manifests with tools (scripts/eval_pipeline.py, route.py, score.py; scripts/dev_bank.py with BANK_SET=fresh_bench2).

Seeds: range [FRESH_SEED_BASE, FRESH_SEED_BASE + 10^14) = [1.84461e19, 1.84462e19), directly above run
[1.8446e19, 1.84461e19) and every earlier range, below 2^64; a git grep
  main result 1 (purpose test, g = 2): circuit i of point (d, K, L, p) samples all its shots with stim seed
      FRESH_SEED_BASE + 2 10^13 + d 10^11 + K 10^10 + L 10^8 + pcode 10^6 + i (pcode 1, 2, 3 for p = 0.1, 0.2, 0.3 %);
  main result 2 (purpose bench2, g = 4): the open benchmark's fixed lukin-family circuits at d = 3, 5, 7, new noise only:
      FRESH_SEED_BASE + 4 10^13 + d 10^11 + D 10^8 + i;

  python scripts/fresh_sets.py manifest --purpose test|bench2|dry --out F the points (duet.bench_sets' manifest format)
  python scripts/fresh_sets.py seedcheck --new F... --out J [--earlier DIR] every new seed in its purpose's range, the
        ranges disjoint from every earlier one, and no new seed equal to any seed of an earlier manifest
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pipeline as ep  # noqa: E402

ROOT = ep.ROOT
FRESH_SEED_BASE, FRESH_SEED_SPAN = 18_446_100_000_000_000_000, 10 ** 14
FRESH_TAG = 20261111
PURPOSE = {"test": 2, "bench2": 4, "dry": 9}
DRY = dict(Ks=(2,), ds=(3, 5), pts=((1, 0.003), (8, 0.003)), shots={(1, 0.003): 200, (8, 0.003): 50}, ncirc=4)


def seed(purpose, d, K, L, p, i):
    g = PURPOSE[purpose]
    assert purpose != "bench2" and 0 < d < 100 and 0 < K < 10 and 0 < L < 100 and 0 <= i < 10 ** 6, (purpose, d, K, L, i)
    s = FRESH_SEED_BASE + g * 10 ** 13 + d * 10 ** 11 + K * 10 ** 10 + L * 10 ** 8 + ep.PCODE[p] * 10 ** 6 + i
    assert FRESH_SEED_BASE + g * 10 ** 13 <= s < FRESH_SEED_BASE + (g + 1) * 10 ** 13 <= FRESH_SEED_BASE + FRESH_SEED_SPAN < 2 ** 64
    return s


def seed_bench2(d, D, i):
    assert d in (3, 5, 7) and 0 < D <= 10 and 0 <= i < 20, (d, D, i)
    s = FRESH_SEED_BASE + PURPOSE["bench2"] * 10 ** 13 + d * 10 ** 11 + D * 10 ** 8 + i
    assert FRESH_SEED_BASE + 4 * 10 ** 13 <= s < FRESH_SEED_BASE + 5 * 10 ** 13 <= FRESH_SEED_BASE + FRESH_SEED_SPAN < 2 ** 64
    return s


def circuits(purpose, K, L, n):
    """The n logical circuits of (purpose, K, L), ep.circuits' law with tag (new circuits)."""
    rng = ep.np.random.default_rng([FRESH_TAG, PURPOSE[purpose], K, L])
    return [(*ep.cc.sample_circuit_k(rng, K, L, ep.P_SET[i % 4], ep.GATES), ep.P_SET[i % 4]) for i in range(n)]


def point_name(purpose, d, K, L, p):
    return f"fresh{purpose[0] if purpose == 'test' else 'y'}_K{K}_d{d}_L{L}_p{ep.PCODE[p]}"


def points(purpose):
    """ep.points' entries (the same fields) with seeds, circuits and names."""
    if purpose == "test":
        Ks, ds, pts, shots, ncirc = ep.KS, ep.DS, ep.POINTS, ep.SIZE["test"], ep.NCIRC
    else:
        Ks, ds, pts, shots, ncirc = DRY["Ks"], DRY["ds"], DRY["pts"], DRY["shots"], DRY["ncirc"]
    out = []
    for K in Ks:
        for L, p in pts:
            circs = circuits(purpose, K, L, ncirc)
            for d in ds:
                cs = []
                for i, (lay, prep, meas, q) in enumerate(circs):
                    prods = ep.cc.logical_products(lay, prep, meas)
                    cs.append(dict(layers=" ".join(lay), prep="".join(prep), meas="".join(meas), pauli="frame",
                                   cnot_layers=sum(len(pr) > 0 for pr, _ in ep.cc.parse_k(lay, K)), q=round(q, 7),
                                   products=[list(s) for s, _ in prods], values=[v for _, v in prods],
                                   seed=str(seed(purpose, d, K, L, p, i))))
                n = int(shots[(L, p)])
                out.append(dict(name=point_name(purpose, d, K, L, p), family="general_mix", sub="mix", purpose=purpose,
                                d=d, noise="lukin", p=p, depth=L, L=L, K=K, shots=n, bplomr_shots=min(ep.BPLOMR, n),
                                seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], circuits=cs))
    return out


def bench2_points():
    """ep.bench2_points with noise seeds (the benchmark's circuits, shots and MLE prefix unchanged)."""
    out = []
    for pt in ep.bench2_points():
        cs = [dict(c, seed=str(seed_bench2(pt["d"], pt["depth"], int(c["index"])))) for c in pt["circuits"]]
        out.append(dict(pt, circuits=cs, seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], purpose="bench2"))
    return out


def cmd_manifest(a):
    g = PURPOSE[a.purpose]
    lo, hi = FRESH_SEED_BASE + g * 10 ** 13, FRESH_SEED_BASE + (g + 1) * 10 ** 13
    common = dict(version=1, run="", purpose=a.purpose, seed_base=str(FRESH_SEED_BASE),
                  seed_range=[str(lo), str(hi)], noise="lukin",
                  plan="fresh test sets: new seeds, the same circuit laws as the first test")
    if a.purpose == "bench2":
        src = os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")
        pts = bench2_points()
        m = dict(common, family="lukin",
                 seed_rule="FRESH_SEED_BASE + 4 10^13 + d 10^11 + D 10^8 + i (circuit i = 0-19 of depth D at distance d)",
                 circuit_rule="the open benchmark's fixed lukin-family circuits (data/open_benchmark/MANIFEST.json, family and "
                              "noise lukin, circuits i = 0-19; duet.bench_sets.lukin), unchanged; the same circuits, "
                              "shots per circuit and mle_shots prefix as MANIFEST_bench2.json",
                 source_manifest_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest(), points=pts)
    else:
        pts = points(a.purpose)
        m = dict(common, family="general_mix",
                 seed_rule=f"FRESH_SEED_BASE + {g} 10^13 + d 10^11 + K 10^10 + L 10^8 + pcode 10^6 + i",
                 circuit_rule=f"numpy default_rng([{FRESH_TAG}, {g}, K, L]); circuit i: sample_circuit_k(rng, K, L, "
                              f"P_SET[i % 4], '{ep.GATES}'), the same circuits at every d and p (the first test's law, new tag)",
                 generator="duet.clifford_circuits.sample_circuit_k (K-block mix law; train_cl.draw's mix)",
                 sizes="data/first_test/ design B (test sizes)" if a.purpose == "test" else "a tiny dry-run set",
                 points=pts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(m, open(a.out, "w"), indent=0)
    shots = int(sum(pt["shots"] * len(pt["circuits"]) for pt in pts))
    extra = dict(mle=int(sum(ep.bs.prefix(pt, "mle") * len(pt["circuits"]) for pt in pts))) if a.purpose == "bench2" else {}
    print(json.dumps(dict(purpose=a.purpose, points=len(pts), shots=shots, **extra)))


def seeds_of(m):
    return [int(c["seed"]) for pt in m["points"] for c in pt["circuits"]]


def cmd_seedcheck(a):
    """The new manifests' seeds: inside their purpose's range, distinct, the fresh range disjoint from the first test's, and
    every EARLIER range, and none equal to a seed of any earlier manifest of the repository."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lukin_bank", os.path.join(ROOT, "scripts", "dev_bank.py"))
    bank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank)
    ranges = dict(bank.EARLIER)
    ranges["(scripts/dev_bank.py)"] = (bank.BANK_SEED_BASE, bank.BANK_SEED_BASE + bank.BANK_SEED_SPAN)
    ranges["(scripts/eval_pipeline.py)"] = (ep.FIRST_SEED_BASE, ep.FIRST_SEED_BASE + ep.FIRST_SEED_SPAN)
    rt = (FRESH_SEED_BASE, FRESH_SEED_BASE + FRESH_SEED_SPAN)
    overlap = [k for k, (lo, hi) in ranges.items() if lo < rt[1] and rt[0] < hi]
    assert not overlap, ("the fresh range overlaps", overlap)
    new, out = {}, dict(fresh_range=[str(rt[0]), str(rt[1])], earlier_ranges=len(ranges), new={}, earlier={})
    for f in a.new.split(","):
        m = json.load(open(f))
        g = PURPOSE[m["purpose"]]
        s = seeds_of(m)
        lo, hi = FRESH_SEED_BASE + g * 10 ** 13, FRESH_SEED_BASE + (g + 1) * 10 ** 13
        assert all(lo <= x < hi for x in s), (f, "a seed outside its purpose's range")
        assert len(set(s)) == len(s), (f, "repeated seeds")
        assert not any(set(s) & v for v in new.values()), (f, "seeds shared between new manifests")
        new[f] = set(s)
        out["new"][os.path.relpath(f, ROOT)] = dict(purpose=m["purpose"], points=len(m["points"]), circuits=len(s),
                                                    min=str(min(s)), max=str(max(s)),
                                                    sha256=hashlib.sha256(open(f, "rb").read()).hexdigest())
    allnew = set().union(*new.values())
    base = a.earlier or os.path.join(ROOT, "results")
    olds = sorted(set(glob.glob(os.path.join(base, "**", "*MANIFEST*.json"), recursive=True))
                  - {os.path.abspath(f) for f in new})
    for f in olds:
        try:
            m = json.load(open(f))
        except (OSError, ValueError):
            continue
        s = set()
        for pt in m.get("points", []) if isinstance(m, dict) else []:
            for c in pt.get("circuits", []) if isinstance(pt, dict) else []:
                if isinstance(c, dict) and "seed" in c:
                    s.add(int(c["seed"]))
        if not s:
            continue
        both = allnew & s
        assert not both, (f, "shares seeds with a new manifest", sorted(both)[:5])
        out["earlier"][os.path.relpath(f, base)] = dict(circuits=len(s), min=str(min(s)), max=str(max(s)), shared=0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(new={k: v["circuits"] for k, v in out["new"].items()}, earlier=len(out["earlier"]),
                          earlier_seeds=sum(v["circuits"] for v in out["earlier"].values()), shared=0,
                          ranges_checked=len(ranges))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "seedcheck"])
    ap.add_argument("--purpose", choices=list(PURPOSE), default="test")
    ap.add_argument("--new", default="")
    ap.add_argument("--earlier", default="", help="seedcheck: a directory of earlier manifests (default: results/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(manifest=cmd_manifest, seedcheck=cmd_seedcheck)[a.mode](a)


if __name__ == "__main__":
    main()
