"""Exact resume of NetCL training: a run stopped after 3 of 6 steps and restarted from its ckpt.pt ends
with the same weights, EMA and losses as the uninterrupted run (CPU, fp32), and a finished run exits at once."""
import json

import torch

from duet.net import train_cl

ARGS = ["--d", "3", "--width", "16", "--bott", "8", "--blocks", "1", "--heads", "2", "--batch", "64", "--chunk", "16",
        "--samples", "384", "--warmup", "2", "--eval-circuits", "2", "--eval-cnot", "1", "--eval-per", "16",
        "--eval-every", "128", "--workers", "2", "--prefetch", "4", "--device", "cpu", "--cnot-share", "0.25",
        "--p-cnot-set", "0,0.3333333,0.6666667,1", "--accum", "2"]


def test_resume_is_exact(tmp_path):
    torch.set_num_threads(1)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert train_cl.main(ARGS + ["--out", a]) == 0
    assert train_cl.main(ARGS + ["--out", b, "--max-steps", "3"]) == 3
    with open(f"{b}/log.jsonl", "a") as f:                       # a kill after a row, before its checkpoint
        f.write(json.dumps(dict(step=4, train_loss=[0, 0, 0])) + "\n" + '{"step": 5, "trai')
    assert train_cl.main(ARGS + ["--out", b]) == 0
    fa, fb = torch.load(f"{a}/final_ema.pt", weights_only=False), torch.load(f"{b}/final_ema.pt", weights_only=False)
    assert fa["step"] == fb["step"] == 6
    for k in fa["model"]:
        assert torch.equal(fa["model"][k], fb["model"][k]), k
    ca, cb = torch.load(f"{a}/ckpt.pt", weights_only=False), torch.load(f"{b}/ckpt.pt", weights_only=False)
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), k
    rows = lambda p: [json.loads(x) for x in open(f"{p}/log.jsonl")]                        # noqa: E731
    ra, rb = rows(a), rows(b)
    assert [r["step"] for r in ra] == [r["step"] for r in rb] == [2, 4, 6]
    assert [r["train_loss"] for r in ra] == [r["train_loss"] for r in rb]
    assert [r["ema_failures"] for r in ra] == [r["ema_failures"] for r in rb]
    assert set(ra[-1]["ema_per_depth"]) == {"mix", "cnot"}
    assert train_cl.main(ARGS + ["--out", a]) == 0                       # finished: exits at once
