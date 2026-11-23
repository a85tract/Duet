"""Paired failure ratios of two decoders' reads from per-shot results on the same shots (where X or Y is the MLE,
only shots whose MLE is proven optimal count).
Per point: the shots both decoded (the shorter prefix of each circuit), only the certified ones where X or Y is the
MLE; failure rate = the fraction of shots with any reliable product wrong, equal weight over circuits (patterns);
X / Y with a 95 % interval from 10,000 bootstrap draws, the same draws for X and Y, the seed fixed by the point's
one-circuit points (family 3) and at points whose circuits are the complete family of CNOT directions (2^L patterns),
where they redraw each circuit's shots, paired (a multinomial over the four outcomes of a shot). Zero counts: when
either failure count is zero, 0.5 is added to both (equivalently 0.5 / shots to both equal-weight rates), in the point
estimate and in every draw where it happens; the point is flagged (zero_count). Per family, d and p: the geometric
mean of the ratios over the depths both read, with its interval from the per-draw mean of the log ratios, every
point kept and the flagged ones counted; family 3 pools its patterns and depths per d. --buildable PRED/<dec>: only
the circuits where that matching decoder dropped no error.
         [--buildable PRED/lomr] [--out JSON]"""
import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

from duet import bench_sets as bs

BOOT = 10_000


def table(fx, fy):
    """Per circuit the paired counts [shots, both fail, X only fails, Y only fails]."""
    return np.array([[len(x), int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum())] for x, y in zip(fx, fy)],
                    np.int64)


def rates(n, fx, fy):
    """Equal-weight rates of X and Y over the last axis, 0.5 added to both failure counts where either is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (fx.sum(-1) == 0) | (fy.sum(-1) == 0)
        add = np.where(z, 0.5 / n.sum(-1), 0.0)
    return (fx / n).mean(-1) + add, (fy / n).mean(-1) + add, z


def paired(tab, seed, mode, boot=BOOT):
    """X / Y at one point from its per-circuit table: rates, ratio, 95 % interval, the zero-count flag, and the per-draw
    log ratios. mode "circuits" redraws circuits, "shots" redraws each circuit's shots (paired)."""
    n, a_, b_, c_ = tab.T.astype(np.float64)
    fx, fy = a_ + b_, a_ + c_
    rx, ry, z = rates(n, fx, fy)
    rng = np.random.default_rng(seed)
    if mode == "circuits":
        rows = rng.integers(len(n), size=(boot, len(n)))
        bx, by, zb = rates(n[rows], fx[rows], fy[rows])
    else:
        cells = np.stack([a_, b_, c_, n - a_ - b_ - c_], 1) / n[:, None]
        dr = np.stack([rng.multinomial(int(k), q, size=boot) for k, q in zip(n, cells)], 1).astype(np.float64)
        bx, by, zb = rates(np.broadcast_to(n, dr.shape[:2]), dr[..., 0] + dr[..., 1], dr[..., 0] + dr[..., 2])
    logs = np.log(bx / by)
    return dict(rate_x=float((fx / n).mean()), rate_y=float((fy / n).mean()), ratio=float(rx / ry),
                ratio95=np.exp(np.percentile(logs, [2.5, 97.5])).tolist(), fails_x=int(fx.sum()), fails_y=int(fy.sum()),
                shots=int(n.sum()), bootstrap=mode, zero_count=bool(z), draws_zero_count=float(zb.mean())), logs


def observables(a, name):
    """Per circuit the stored observable flips (shots, k) bool."""
    f = os.path.join(a.sets, name + ".npz")
    z = np.load(f)
    O = bs.packed(f, "obs")
    return [bs.unpack(O[i], int(z["k"][i])) for i in range(len(z["k"]))]


def fails(a, spec, name, obs):
    """Per circuit (failure vector, certified vector) of decoder spec at a point, or None if it has no result."""
    kind, path = spec.split(":", 1)
    f = os.path.join(path, name + ".npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    if kind == "net":
        n = z["shots"]
        F = [np.unpackbits(z["fail"][i], count=int(n[i]), bitorder="little").astype(bool) for i in range(len(n))]
        return [(x, np.ones(len(x), bool)) for x in F]
    out = []
    for i, n in enumerate(z["decoded"]):
        y = obs[i][:n]
        pred = np.unpackbits(z["pred"][i, :n], axis=1, count=y.shape[1], bitorder="little").astype(bool)
        out.append(((pred != y).any(1), z["ok"][i, :n].astype(bool)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--buildable", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.x.split(":", 1)[1], "*.npz")))
    out, groups = dict(x=a.x, y=a.y, points={}), {}
    for name in names:
        if a.only and not re.search(a.only, name):
            continue
        obs = observables(a, name)
        X, Y = fails(a, a.x, name, obs), fails(a, a.y, name, obs)
        if X is None or Y is None:
            continue
        keep = range(len(X))
        if a.buildable:
            pc = json.load(open(os.path.join(a.buildable, name + ".json")))["per_circuit"]
            keep = [c["i"] for c in pc if c["dropped"] == 0]
        fx, fy, dropped, empty = [], [], 0, 0
        for (x, ox), (y, oy) in (z_ for i, z_ in enumerate(zip(X, Y)) if i in keep):
            n = min(len(x), len(y))
            ok = ox[:n] & oy[:n]
            dropped += int((~ok).sum())
            if not ok.any():                         # no shot both decoded: the circuit has no rate
                empty += 1
                continue
            fx.append(x[:n][ok])
            fy.append(y[:n][ok])
        if not fx:
            continue
        L = int(re.search(r"L(\d+)$", name).group(1))
        r, logs = paired(table(fx, fy), zlib.crc32(name.encode()) % 2 ** 31,
                         "shots" if len(X) == 1 else "circuits")
        r["uncertified_left_out"] = dropped
        r["circuits"], r["circuits_left_out"], r["circuits_empty"] = len(keep), len(X) - len(keep), empty
        out["points"][name] = r
        m = re.match(r"(mix|struct)_.*?d(\d+)_(?:p(\d)_)?L(\d+)$", name) or re.match(r"(d)(\d+)_p(\d)_L(\d+)$", name)
        if m:
            fam = {"d": "cnot"}.get(m.group(1), m.group(1))
            key = f"{fam}_d{m.group(2)}" + (f"_p{m.group(3)}" if fam != "struct" else "")
            groups.setdefault(key, []).append((int(m.group(4)), r, logs))
    out["geometric_means"] = {
        k: dict(ratio=float(np.exp(np.mean([np.log(r["ratio"]) for _, r, _ in v]))),
                ratio95=np.exp(np.percentile(np.mean([lg for _, _, lg in v], 0), [2.5, 97.5])).tolist(),
                depths=sorted({L for L, _, _ in v}), points=len(v), points_zero_count=sum(r["zero_count"] for _, r, _ in v))
        for k, v in groups.items()}
    print(json.dumps({k: {x: round(v[x], 5) if isinstance(v[x], float) else v[x] for x in
                          ("rate_x", "rate_y", "ratio", "ratio95", "fails_x", "fails_y", "shots", "zero_count")}
                      for k, v in out["points"].items()}, indent=0))
    print(json.dumps(out["geometric_means"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
