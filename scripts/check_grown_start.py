"""Check that the grown d = 3 start computes the start's outputs.

    python scripts/check_grown_start.py START BLOCKS NBATCH TRAIN_CL_ARGS...

Builds the start as saved (NetCL(**cfg), its blocks) and the start grown to BLOCKS trunk blocks, then reads the first
NBATCH training batches of the run whose train_cl arguments follow (the run's own chunk stream: train_cl.batch_plan,
chunk_job, _chunk; lukin-family and production-mix batches alike) with both, in float32 and under the training's bf16
autocast, and prints per batch (K, L, kind) and whether every output (c, z, l) is equal (torch.equal) and the largest
absolute difference. Exit 0 when every batch is equal in both precisions."""
import argparse
import json
import sys

import numpy as np
import torch

from duet.net import model_cl as M
from duet.net import train_cl as T


def main():
    start, blocks, nb, rest = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    a = T.parser().parse_args(rest + ["--out", "/nonexistent"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(start, map_location="cpu", weights_only=False)
    small = M.NetCL(**ck["cfg"]).to(dev).eval()
    small.load_state_dict(ck["model"])
    big = M.NetCL(**dict(ck["cfg"], blocks=blocks)).to(dev).eval()
    added = M.load_grown(big, ck)
    big = big.to(dev).eval()
    for net in (small, big):
        net.head.lookup = a.lookup
    steps = int(a.samples // a.batch)
    plan = T.batch_plan(a, steps)
    p_set = tuple(float(v) for v in a.p_cnot_set.split(",")) if a.p_cnot_set else ()
    p_l1 = T.p_weights(a.p_l1)
    per = a.batch // a.chunk
    print(json.dumps(dict(start=start, start_blocks=ck["cfg"]["blocks"], blocks=blocks, added=added,
                          params_small=sum(q.numel() for q in small.parameters()),
                          params_big=sum(q.numel() for q in big.parameters()), device=str(dev))), flush=True)
    ok = True
    for i in range(nb):
        kind = plan(i)
        items = [T._chunk(T.chunk_job(a, k, kind, p_set, p_l1))[1:] for k in range(i * per, (i + 1) * per)]
        b = T.collate(items, a.d, dev, train=False)
        row = dict(batch=i, K=kind[0], L=kind[1], kind=str(kind[2]), shots=sum(len(x[1]) for x in items))
        for prec in ("fp32", "bf16"):
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=prec == "bf16"):
                xs = T.forward(small, b, train=False)
                ys = T.forward(big, b, train=False)
            eq = all(torch.equal(x, y) for x, y in zip(xs, ys))
            diff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(xs, ys))
            row[prec] = dict(equal=eq, max_abs_diff=diff)
            ok &= eq
        print(json.dumps(row), flush=True)
    print("GROW_CHECK", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
