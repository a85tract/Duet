"""Scores of the ablation networks (d = 5) against the full network on the same shots, on CPU:

    python scripts/ablation_score.py --reads ABL_READS --arms abl_noexch... --dev-manifest M --lukin-manifest M
        --dev-sets DIR --lukin-sets DIR [--main-dev DIR --main-lukin DIR] [--ref abl_full] --out JSON --md MD

Every arm (and main_d5, the reference row) against the full arm on the same shots of d = 5, each read alone by
scripts/operating_points.py read. Development bank (main result 1's d = 5 points): a shot fails when any product is wrong; lukin
development bank (d = 5, D = 1-10): when the target product (slot 0) is wrong (route.py point_truth, --fail target).
Per point: X / Y (X the arm, Y the full arm) with a 95 % interval from 10,000 draws that redraw the circuits with their
observed paired counts, the seed fixed by the point's name, and the 0.5 zero rule; a point with no discordant shot is degenerate (1 [1, 1]). Pooled rows: the
geometric mean of the per-point ratios, the interval from each draw's mean log ratio, failure counts summed; per K over
the six depths at p = 0.3 %, over K = 2, 3, 4 (18 points), the p = 0.1 / 0.2 % points as their own row; lukin per depth
and over D = 1-10. Every point of a row must be read whole by one network file of each side, or the run stops."""
import argparse
import json
import os
import re
import sys
import zlib
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import operating_points as route  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from paired_stats import paired, table  # noqa: E402 (unchanged)

D = 5


def read_of(base, name, pt):
    """(logits (circuits, shots, 4), network SHA-256) of a whole read of point pt, or a stop."""
    f = os.path.join(base, name, pt["name"] + ".npz")
    assert os.path.exists(f), ("no read", name, pt["name"])
    with np.load(f) as z:
        lg = z["logits"]
        sha = json.loads(str(z["info"])).get("sha256")
    assert lg.shape[:2] == (len(pt["circuits"]), pt["shots"]), (name, pt["name"], lg.shape, "a partial read")
    return lg, sha


def fails(sets, fail, pt, lg):
    """(circuits, shots) bool: route.py point_truth's comparison (any product, or the target only)."""
    truth, netp = route.point_truth(SimpleNamespace(dir=sets, fail=fail), pt, lg)
    return (truth != netp).reshape(lg.shape[0], lg.shape[1])


def point_row(name, fx, fy):
    tab = table(list(fx), list(fy))
    r, logs = paired(tab, zlib.crc32(name.encode()) % 2 ** 31, "circuits")
    r["degenerate"] = bool(tab[:, 2].sum() == 0 and tab[:, 3].sum() == 0)
    if r["degenerate"]:
        r["ratio"], r["ratio95"], logs = 1.0, [1.0, 1.0], np.zeros_like(logs)
    return r, logs


def pool(rows):
    """[(row, logs)] -> the geometric mean of the ratios, its 95 % interval from the per-draw mean log ratio, sums."""
    lr = np.mean([lg for _, lg in rows], 0)
    return dict(ratio=float(np.exp(np.mean([np.log(r["ratio"]) for r, _ in rows]))),
                ratio95=np.exp(np.percentile(lr, [2.5, 97.5])).tolist(), points=len(rows),
                fails_x=sum(r["fails_x"] for r, _ in rows), fails_y=sum(r["fails_y"] for r, _ in rows),
                shots=sum(r["shots"] for r, _ in rows), zero_count=sum(r["zero_count"] for r, _ in rows),
                degenerate=sum(r["degenerate"] for r, _ in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", default="abl_full")
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--lukin-manifest", required=True)
    ap.add_argument("--dev-sets", required=True)
    ap.add_argument("--lukin-sets", required=True)
    ap.add_argument("--main-dev", default="", help="dev reads' directory (main_d5 as 'main')")
    ap.add_argument("--main-lukin", default="", help="lukin dev reads' directory")
    ap.add_argument("--main-sha", default="", help="main_d5's final_ema.pt SHA-256 (the reads' record must match)")
    ap.add_argument("--banks", default="dev,lukin", help="smoke tests: one bank only")
    ap.add_argument("--only", default="", help="smoke tests: the points whose names match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    banks = dict(dev=(json.load(open(a.dev_manifest)), a.dev_sets, "any", a.main_dev),
                 lukin=(json.load(open(a.lukin_manifest)), a.lukin_sets, "target", a.main_lukin))
    arms = [x for x in a.arms.split(",") if x]
    out = dict(ref=a.ref, d=D, boot=10000, sha256={}, banks={})
    for bank, (m, sets, fail, main_dir) in banks.items():
        if bank not in a.banks.split(","):
            continue
        pts = [pt for pt in m["points"] if pt["d"] == D and (not a.only or re.search(a.only, pt["name"]))]
        assert pts, bank
        ref = {}
        for pt in pts:
            lg, sha = read_of(a.reads, a.ref, pt)
            ref[pt["name"]] = fails(sets, fail, pt, lg)
            out["sha256"].setdefault(a.ref, set()).add(sha)
        res = out["banks"][bank] = {}
        for arm in arms:
            if arm == "main_d5":
                if not main_dir:
                    continue
                base, nm = main_dir, "main"
            else:
                base, nm = a.reads, arm
            if bank == "lukin" and arm in ("abl_k2", "abl_k3"):
                continue
            kk = re.match(r"abl_k(\d)$", arm)
            mine = [pt for pt in pts if not (kk and bank == "dev" and pt["K"] != int(kk.group(1)))]
            if not os.path.isdir(os.path.join(base, nm)) or not all(
                    os.path.exists(os.path.join(base, nm, pt["name"] + ".npz")) for pt in mine):
                print(json.dumps(dict(bank=bank, arm=arm, skipped="not every point read yet")), flush=True)
                continue
            per = {}
            for pt in mine:
                lg, sha = read_of(base, nm, pt)
                out["sha256"].setdefault(arm, set()).add(sha)
                per[pt["name"]] = (pt, *point_row(pt["name"], fails(sets, fail, pt, lg), ref[pt["name"]]))
            rows = {}
            if bank == "dev":
                p3 = [v for v in per.values() if v[0]["p"] == 0.003]
                for K in (2, 3, 4):
                    v = [x for x in p3 if x[0]["K"] == K]
                    if v:
                        rows[f"K{K}"] = pool([(r, lg) for _, r, lg in v])
                if len({x[0]["K"] for x in p3}) == 3:
                    rows["pooled"] = pool([(r, lg) for _, r, lg in p3])
                lo = [v for v in per.values() if v[0]["p"] != 0.003]
                if lo:
                    rows["p1p2"] = pool([(r, lg) for _, r, lg in lo])
            else:
                for name, (pt, r, lg) in sorted(per.items(), key=lambda x: x[1][0]["depth"]):
                    rows[f"D{pt['depth']}"] = pool([(r, lg)])
                rows["pooled"] = pool([(r, lg) for _, r, lg in per.values()])
            res[arm] = dict(points={k: r for k, (_, r, _) in per.items()}, rows=rows)
            print(json.dumps(dict(bank=bank, arm=arm, **{k: [round(v["ratio"], 4), [round(x, 4) for x in v["ratio95"]],
                                                              v["fails_x"], v["fails_y"]] for k, v in rows.items()})),
                  flush=True)
    for k, v in out["sha256"].items():
        assert len(v) == 1, ("reads of one system from different network files", k, v)
        out["sha256"][k] = v.pop()
    if a.main_sha and "main_d5" in out["sha256"]:
        assert out["sha256"]["main_d5"] == a.main_sha, ("main_d5 reads are not of main_d5", out["sha256"])
    json.dump(out, open(a.out, "w"), indent=1)
    open(a.md, "w").write(markdown(out, arms))


def cell(v):
    lo, hi = v["ratio95"]
    flag = "" if not (v["zero_count"] or v["degenerate"]) else f" z{v['zero_count']}d{v['degenerate']}"
    return f"{v['ratio']:.3f} [{lo:.3f}, {hi:.3f}] ({v['fails_x']} / {v['fails_y']}){flag}"


def markdown(out, arms):
    L = []
    dev, lukin = out["banks"].get("dev", {}), out["banks"].get("lukin", {})
    L.append("| arm | dev K = 2 | dev K = 3 | dev K = 4 | dev pooled (18 points) | dev p = 0.1 / 0.2 % | lukin pooled (D = 1-10) |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in arms:
        if arm not in dev and arm not in lukin:
            continue
        r = dev.get(arm, {}).get("rows", {})
        q = lukin.get(arm, {}).get("rows", {})
        L.append(f"| {arm} | " + " | ".join(cell(r[k]) if k in r else "-" for k in ("K2", "K3", "K4", "pooled", "p1p2"))
                 + f" | {cell(q['pooled']) if 'pooled' in q else '-'} |")
    L.append("")
    L.append("lukin development bank per depth (target product; ratio [95 %] (arm / full failures)):")
    L.append("")
    L.append("| arm | " + " | ".join(f"D = {D_}" for D_ in range(1, 11)) + " |")
    L.append("|---|" + "---|" * 10)
    for arm in arms:
        q = lukin.get(arm, {}).get("rows", {})
        if q:
            L.append(f"| {arm} | " + " | ".join(cell(q[f'D{D_}']) if f"D{D_}" in q else "-" for D_ in range(1, 11)) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
