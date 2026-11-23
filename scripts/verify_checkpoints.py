"""Verifier of the trained networks (CPU):

    python scripts/verify_checkpoints.py RUNS_DIR NAME...

Per network NAME in RUNS_DIR: run.json present and not stopped; samples trained = the budget (5e7 for main_*, 1e8 for
abl_*: whole 1,024-sample steps, so 48,828 or 97,656 steps); final_ema.pt loads into NetCL(**cfg) with every weight
(strict) and its cfg is the run's (blocks, width, couple, head); the last monitor rows of log.jsonl (the final row and
the one before); for main_*, the step-0 row equals the start's final EMA row (the --init checkpoint's run directory,
log.jsonl), every family and depth. Prints one JSON line per network and a summary line; exit 0 when all pass."""
import json
import os
import sys

import torch

from duet.net import model_cl as M

BUDGET = {"main": 5e7, "abl": 1e8}


def check(runs, name):
    out = dict(run=name, ok=False)
    d = os.path.join(runs, name)
    rj = os.path.join(d, "run.json")
    if not os.path.exists(rj):
        out["error"] = "no run.json"
        return out
    r = json.load(open(rj))
    a = r["args"]
    budget = BUDGET[name.split("_")[0]]
    steps = int(budget // a["batch"])
    out.update(samples=r["samples"], steps=r["step"], budget=budget, stopped=r.get("stopped"),
               minutes=round(r.get("minutes", 0), 1), d=a["d"], blocks=a["blocks"], seed=a["seed"])
    errs = []
    if r.get("stopped") or r["step"] != steps or r["samples"] != steps * a["batch"] or a["samples"] != budget:
        errs.append(f"samples {r['samples']} / steps {r['step']} against budget {budget:g} ({steps} steps)")
    ck = torch.load(os.path.join(d, "final_ema.pt"), map_location="cpu", weights_only=False)
    net = M.NetCL(**ck["cfg"])
    net.load_state_dict(ck["model"], strict=True)
    cfg = ck["cfg"]
    want = dict(blocks=a["blocks"], width=a["width"], couple=a.get("couple", "cross"), head=a.get("head", "lines"))
    got = dict(blocks=cfg["blocks"], width=cfg["width"], couple=cfg.get("couple", "cross"), head=cfg.get("head", "lines"))
    if want != got:
        errs.append(f"cfg {got} against args {want}")
    out.update(params=sum(q.numel() for q in net.parameters()), cfg=got, ckpt_step=ck.get("step"))
    rows = [json.loads(line) for line in open(os.path.join(d, "log.jsonl"))]
    out["rows"] = len(rows)
    out["last_rows"] = [dict(step=x["step"], final=x.get("final"), ema=x["ema_failures"]) for x in rows[-2:]]
    if not rows or not rows[-1].get("final") or rows[-1]["step"] != steps:
        errs.append("no final monitor row at the last step")
    if name.startswith("main_"):
        start_log = os.path.join(os.path.dirname(a["init"]), "log.jsonl")
        s = [json.loads(line) for line in open(start_log)][-1]
        r0 = rows[0]
        same = r0["step"] == 0 and r0["ema_failures"] == s["ema_failures"] and r0["ema_per_depth"] == s["ema_per_depth"]
        out["step0"] = dict(run=r0["ema_failures"], start=s["ema_failures"], equal=same)
        if not same:
            errs.append("step-0 row differs from the start's final row")
    out["errors"] = errs
    out["ok"] = not errs
    return out


def main():
    runs, names = sys.argv[1], sys.argv[2:]
    res = [check(runs, n) for n in names]
    for x in res:
        print(json.dumps(x), flush=True)
    bad = [x["run"] for x in res if not x["ok"]]
    print(json.dumps(dict(verified=len(res) - len(bad), failed=bad)), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
