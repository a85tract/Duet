"""test-set reads: a lukin point without the benchmark's exact MLE
(the manifest's prefix 0: d = 7, D = 9, 10) is converted without MLE parts, and rep's (b) runs over the depths where
the MLE ran (d = 7: D = 1-8), checked against the manifest on the test set. Synthetic shots only."""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _t():
    if "lukin_test" not in sys.modules:
        spec = importlib.util.spec_from_file_location("lukin_test", os.path.join(ROOT, "scripts/benchmark2_reads.py"))
        m = sys.modules["lukin_test"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    return sys.modules["lukin_test"]


def _manifest():
    return {p["name"]: p for p in json.load(open(os.path.join(ROOT, "data/open_benchmark/MANIFEST.json")))["points"]}


def test_manifest_mle_depths():
    t = _t()
    for d, want in ((3, list(range(1, 11))), (5, list(range(1, 11))), (7, list(range(1, 9)))):
        assert [p["depth"] for p in t.bank.test_points((d,)) if t.bs.prefix(p, "mle")] == want


def test_conv_point_without_mle(tmp_path, monkeypatch):
    t = _t()
    full, rng = _manifest(), np.random.default_rng(1)
    C, n, M, K = 2, 16, 20, 2
    rel, bench, test = tmp_path / "rel", tmp_path / "pred", tmp_path / "test"
    monkeypatch.setattr(t.bs, "circuit", lambda pt, c: SimpleNamespace(num_detectors=M, num_observables=K))
    pts, sums = {}, {}
    for name in ("lukin_D8_d7_lukin_p3", "lukin_D9_d7_lukin_p3"):
        pt = pts[name] = dict(full[name], circuits=full[name]["circuits"][:C], shots=n)
        dd = rel / "points" / name
        dd.mkdir(parents=True)
        json.dump(dict(entry=pt, circuits=[dict(detectors=M, observables=K)] * C), open(dd / "meta.json", "w"))
        for i in range(C):
            for kind, w in (("dets", (M + 7) // 8), ("obs", 1)):
                f = dd / f"c{i}.{kind}.b8"
                rng.integers(0, 256, (n, w), dtype=np.uint8).tofile(f)
                sums[f"points/{name}/c{i}.{kind}.b8"] = t.sha256_file(f)
        for src in ("lomr", "net", "mle"):
            if src == "mle" and not t.bs.prefix(pt, "mle"):
                continue
            os.makedirs(bench / src, exist_ok=True)
            nd = n // 2 if src == "mle" else n
            np.savez(bench / src / f"{name}.npz", pred=rng.integers(0, 4, (C, n, 1), dtype=np.uint8),
                     ok=np.ones((C, n), bool), decoded=np.full(C, nd))
    a = SimpleNamespace(release=str(rel), bench=str(bench), test=str(test))
    job = lambda name: (a, pts[name], {r: h for r, h in sums.items() if r.startswith(f"points/{name}/")})
    r9 = t._conv_point(job("lukin_D9_d7_lukin_p3"))
    assert r9["mle_shots"] == 0 and not (test / "dec" / "mle" / "lukin_D9_d7_lukin_p3").exists()
    for dec in ("lomr", "start"):
        assert len(os.listdir(test / "dec" / dec / "lukin_D9_d7_lukin_p3")) == C
    assert (test / "sets" / "lukin_D9_d7_lukin_p3.npz").exists()
    r8 = t._conv_point(job("lukin_D8_d7_lukin_p3"))
    assert r8["mle_shots"] == full["lukin_D8_d7_lukin_p3"]["mle_shots"] > 0
    z = np.load(test / "dec" / "mle" / "lukin_D8_d7_lukin_p3" / "c1_0.npz")
    assert z["idx"].tolist() == list(range(n // 2))
    os.remove(bench / "mle" / "lukin_D8_d7_lukin_p3.npz")      # a point with an MLE prefix still needs its file
    with pytest.raises(FileNotFoundError):
        t._conv_point(job("lukin_D8_d7_lukin_p3"))


def _test_set(t, root, mle, ok_false=()):
    """A synthetic directory at d = 7 (bank.points' names and depths: the test set's with BANK_SET=test; 2 circuits of 64
    shots each): the network's logits, LoMatching and the start on every shot, the MLE's parts on the depths in `mle`
    (uncertified on ok_false)."""
    rng, C, n, net, sha = np.random.default_rng(7), 2, 64, "MH7_q4", "ab" * 32
    for pt in t.bank.points((7,)):
        name = pt["name"]
        for sub in ("sets", f"net/{net}"):
            os.makedirs(root / sub, exist_ok=True)
        np.savez(root / "sets" / f"{name}.npz", dets=np.zeros((C, n, 1), np.uint8),
                 obs=rng.integers(0, 2, (C, n, 1), dtype=np.uint8), m=np.full(C, 8), k=np.full(C, 1))
        np.savez(root / "net" / net / f"{name}.npz", logits=rng.normal(0, 3, (C, n, 4)).astype(np.float32),
                 secs=np.ones(1), batch=n, info=json.dumps(dict(sha256=sha)))
        for dec in ["lomr", "start"] + (["mle"] if pt["depth"] in mle else []):
            os.makedirs(root / "dec" / dec / name, exist_ok=True)
            for i in range(C):
                np.savez(root / "dec" / dec / name / f"c{i}_0.npz", idx=np.arange(n),
                         sec=np.full(n, np.nan, np.float32), pred=rng.integers(0, 2, n, dtype=np.uint8),
                         ok=np.full(n, not (dec == "mle" and pt["depth"] in ok_false)))
    thr = root / "thr.json"
    json.dump(dict(nets={"7": dict(name=net, sha256=sha)}, t={"7": {t.fkey(f): [-1.0] * 10 for f in t.FS}}),
              open(thr, "w"))
    return SimpleNamespace(test=str(root), thr=str(thr), ds="7", mle="mle", start="start", start_net="", extra=[],
                           out=str(root / "rep.json"))


def test_rep_b_over_the_mle_depths(tmp_path, monkeypatch):
    t = _t()
    monkeypatch.setattr(t, "BOOT", 200)
    monkeypatch.setattr(t.bank, "TEST", True)
    a = _test_set(t, tmp_path / "ok", mle=range(1, 9))
    t.rep(a)
    rec = json.load(open(a.out))["7"]["systems"]
    for lab, v in rec.items():
        assert v["b"]["depths"] == "1-8", lab
        assert [q["D"] for q in v["depths"] if "mle_shots" in q] == list(range(1, 9)), lab
        assert v["a"]["depths"] == "1-10", lab
    # on the test set the MLE must match the manifest: none at D = 9, every depth 1-8 with certified shots
    for root, kw in (("extra", dict(mle=range(1, 10))), ("short", dict(mle=range(1, 8))),
                     ("uncert", dict(mle=range(1, 9), ok_false=(8,)))):
        with pytest.raises(AssertionError):
            t.rep(_test_set(t, tmp_path / root, **kw))
    monkeypatch.setattr(t.bank, "TEST", False)        # the development path (dry runs) keeps (b) over every MLE depth
    a = _test_set(t, tmp_path / "dev", mle=range(1, 11))
    t.rep(a)
    assert json.load(open(a.out))["7"]["systems"]["network"]["b"]["depths"] == "1-10"


def test_load_dec_skips_unfinished_parts(tmp_path):
    t = _t()
    pt = dict(name="p")
    d = tmp_path / "dec" / "tess" / "p"
    d.mkdir(parents=True)
    part = dict(idx=np.arange(3), pred=np.zeros(3, np.uint8), ok=np.ones(3, bool), sec=np.zeros(3, np.float32))
    np.savez(d / "c0_0.npz", **part)
    np.savez(d / "c0_3.npz.tmp.npz", **part)            # left by a killed job
    got = t.bank.load_dec(str(tmp_path), "tess", pt)
    assert list(got) == [0] and got[0][0].tolist() == [0, 1, 2]
