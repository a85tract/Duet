"""Summarise the live rows (scripts/paced_stream.py cmp, one json per row) as one Markdown table: per row the points and
shots streamed, routed shots live and simulated, answers changed, failures live and simulated, the live time per layer, the live time against
the simulator's replay of the same live trial durations (this CPU's seconds), and the live time converted to the basis
against the simulation of the same batches from the stored trials at the live N, each an equal-weight average over the points. Prints the totals and the rows' median and range of both ratios.
  python scripts/live_summary.py --dir DIR --rows ROWS_FILE --out OUT.md"""
import argparse
import json
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = ["| row | result | d | f | cap | points | shots | routed live = sim | answers changed (to fail / to correct) "
       "| failures live = sim | us per layer live (mean over points; worst point) | live / replay (this CPU) "
       "| live in the basis / sim (N, same batches; points' range) |", "|---" * 13 + "|"]
tot = dict(rows=0, changed=0, routed_diff=0)
ratios = dict(basis_sim=[], replay=[])
for ln in open(a.rows):
    k, res, d, f, cap, R, N = ln.split()
    p = os.path.join(a.dir, k + ".cmp.json")
    if not os.path.exists(p):
        out.append(f"| {k} | {res} | {d} | {f} | {cap} | not run |" + " |" * 7)
        continue
    rs = json.load(open(p))["rows"]
    pool = lambda key, sel: statistics.mean(r[key] for r, k_ in zip(rs, sel) if k_)
    both = [bool(r["us_per_layer_live_basis_estimate"] and r["us_per_layer_sim_basis"]) for r in rs]
    live, rep = (pool(x, [True] * len(rs)) for x in ("us_per_layer_live", "us_per_layer_replay_live"))
    est, sim = (pool(x, both) if any(both) else None for x in ("us_per_layer_live_basis_estimate", "us_per_layer_sim_basis"))
    worst = max(rs, key=lambda r: r["us_per_layer_live"])
    rr = [r["us_per_layer_live_basis_estimate"] / r["us_per_layer_sim_basis"] for r in rs
          if r["us_per_layer_live_basis_estimate"] and r["us_per_layer_sim_basis"]]
    s = lambda key: sum(r[key] for r in rs)
    tot["rows"] += 1
    tot["changed"] += s("answers_changed")
    tot["routed_diff"] += s("routed_live") != s("routed_sim")
    ratios["replay"].append(live / rep)
    if est and sim:
        ratios["basis_sim"].append(est / sim)
    eq = lambda x, y: f"{s(x):,} {'=' if s(x) == s(y) else '!='} {s(y):,}"
    out.append(f"| {k} | {res} | {d} | {float(f):g} | {cap} | {len(rs)} | {s('shots'):,} | "
               f"{eq('routed_live', 'routed_sim')} | {s('answers_changed')} ({s('changed_to_fail')} / "
               f"{s('changed_to_correct')}) | {eq('fails_live', 'fails_sim')} | {live:.1f}; {worst['us_per_layer_live']:.1f} at "
               f"{worst['point']} | {live / rep:.3f} | "
               + (f"{est / sim:.3f} (N = {rs[0]['N']}; {sum(both)} of {len(rs)} points; {min(rr):.3f}-{max(rr):.3f})"
                  if est and sim else "- (f = 0: no trials; the live / replay column is the check)" if s("routed_live") == 0
                  else "- (no paired uncapped trials)") + " |")
open(a.out, "w").write("\n".join(out) + "\n")
for k_, v in ratios.items():
    tot[k_] = dict(rows=len(v), median=statistics.median(v), min=min(v), max=max(v), above_1=sum(x > 1 for x in v))
print(json.dumps(tot))
