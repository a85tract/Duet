"""Two-block reference of the Clifford network: NetCL's tables,
outputs, loss and gradients on stored two-block shots with stored weights, recorded with the code of the earlier two-block code
22b1d21 (NetCL format 2, before the K-block change) in tests/data/kblock_net_k2.pt; tests/test_net_multi.py recomputes
them with today's code (eager fp32 on the CPU) and requires them equal bit for bit. Circuits: train_cl.draw's
production mix (a 25 % CNOT-only share, a CNOT probability from {0, 1/3, 2/3, 1}, I / H / S), depths 4 and 6, d = 3
and 5, both noise models, 8 circuits per distance and model, 16 shots each from the extended model; weights: NetCL(width 32, bott 8, blocks 2, heads 4), torch seed 7, every parameter
moved by 0.1 N(0, 1) so that no path is zero. A v2 reference (--ckpt, tests/data/kblock_net_k2_v2.pt) holds a trained
two-block network's checkpoint by path and SHA-256 instead of weights, only its distance's circuits, and a digest of
every tensor (the trained network's gradients would make the file large); check and the test read the checkpoint from

  PYTHONPATH=<the earlier two-block code code> python tests/kblock_net_k2_ref.py make --out tests/data/kblock_net_k2.pt
  PYTHONPATH=<the earlier two-block code code> python tests/kblock_net_k2_ref.py make --ckpt CKPT --out tests/data/kblock_net_k2_v2.pt
  PYTHONPATH=<code> python tests/kblock_net_k2_ref.py check --ref tests/data/kblock_net_k2.pt
On a GPU (deterministic algorithms), each code dumps its outputs on the same device and the two dumps are compared:
  PYTHONPATH=<code> python tests/kblock_net_k2_ref.py dump --ref tests/data/kblock_net_k2.pt --device cuda --out X.pt
  python tests/kblock_net_k2_ref.py compare --ref OLD.pt --out NEW.pt"""
import argparse
import hashlib
import io
import os

import numpy as np
import torch

from duet import clifford_circuits as cc
from duet.net import model_cl as M
from duet.net import spec_cl
from duet.net import train_cl

SEED0 = 18_430_000_000_000_000_000 + 2 * 10 ** 15
SPEC_KEYS = ("flat", "mx", "mz", "gate", "pcode", "parts", "lines", "orient", "A", "labelled", "tij", "block",
             "circuit")
TAB_KEYS = ("m", "T", "n_obs", "flat", "mx", "mz", "gate", "pcode", "offb", "cnt", "A", "n_total")
SHAPE = dict(width=32, bott=8, blocks=2, heads=4)


def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def cases():
    out = []
    for d in (3, 5):
        for noise in ("uniform", "lukin"):
            rng = np.random.default_rng([20261029, d, int(noise == "lukin")])
            for i in range(8):
                lay, prep, meas = train_cl.draw(rng, (4, 6)[i % 2], cc.P_CNOT, "IHS", 0.25, (0.0, 1 / 3, 2 / 3, 1.0))
                out.append((d, noise, list(lay), tuple(prep), tuple(meas)))
    return out


def digest(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(x.dtype.str.encode() + str(x.shape).encode() + x.tobytes()).hexdigest()[:16]
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def table_digests(cs):
    out = []
    for d, noise, lay, prep, meas in cs:
        s = spec_cl.build(d, lay, prep, meas, 0.003, noise)
        tab, ext = train_cl.tables(d, lay, prep, meas, 0.003, noise)
        row = {f"spec.{k}": digest(s[k]) for k in SPEC_KEYS}
        row.update({f"tab.{k}": digest(tab[k]) for k in TAB_KEYS}, ext=digest(str(ext)))
        out.append(row)
    return out


def ckpt_path(ref):
    return os.environ.get("DUET_K2_CKPT", ref["ckpt"]["path"])


def nets(ref):
    """Per distance the network's configuration and weights: the stored ones, or a v2 reference's checkpoint."""
    if "ckpt" not in ref:
        return {d: (dict(d=d, **SHAPE), w) for d, w in ref["weights"].items()}
    raw = open(ckpt_path(ref), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == ref["ckpt"]["sha256"], f"{ckpt_path(ref)}: not the reference's checkpoint"
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return {ck["cfg"]["d"]: (ck["cfg"], ck["model"])}


def outputs(ref, device="cpu"):
    """Everything the test compares, recomputed from ref's circuits, shots and weights with the code on the path."""
    dev = torch.device(device)
    out = {}
    for d, (cfg, w) in sorted(nets(ref).items()):
        net = M.NetCL(**cfg).to(dev)
        net.load_state_dict(w)
        groups = {}
        for i, (d_, noise, lay, prep, meas) in enumerate(ref["cases"]):
            if d_ == d:
                groups.setdefault((noise, len(lay)), []).append(i)
        total = 0
        for (noise, L), idx in sorted(groups.items()):
            items = []
            for i in idx:
                _, _, lay, prep, meas = ref["cases"][i]
                tab, _ = train_cl.tables(d, lay, prep, meas, 0.003, noise)
                items.append((tab, ref["dets"][i], ref["obs"][i]))
            b = train_cl.collate(items, d, dev)
            c, z, l = train_cl.forward(net, b, train=True)
            loss, _, _ = M.loss_cl(c, z, l, b["y"], b["yl"], b["u"], b["slot"], 1.0)
            with torch.no_grad():
                ce, ze, le = train_cl.forward(net, b, train=False)
            key = f"d{d}_{noise}_L{L}"
            for name, v in (("c", c), ("z", z), ("l", l), ("loss", loss), ("c_eval", ce), ("z_eval", ze),
                            ("l_eval", le)):
                out[f"{key}.{name}"] = v.detach().cpu()
            total = total + loss
        net.zero_grad()
        total.backward()
        for name, q in net.named_parameters():
            out[f"d{d}.grad.{name}"] = q.grad.detach().cpu()
    return out


def make(path, ckpt=None):
    cs = cases()
    if ckpt:                                    # v2: one trained network and its distance's circuits
        raw = open(ckpt, "rb").read()
        d0 = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)["cfg"]["d"]
        cs = [c for c in cs if c[0] == d0]
    dets, obs = [], []
    for i, (d, noise, lay, prep, meas) in enumerate(cs):
        _, ext = train_cl.tables(d, lay, prep, meas, 0.003, noise)
        dp, op, _ = ext.compile_sampler(seed=SEED0 + i).sample(16, bit_packed=True)
        dets.append(dp)
        obs.append(op)
    ref = dict(cases=cs, dets=dets, obs=obs)
    if ckpt:
        ref["ckpt"] = dict(path=os.path.abspath(ckpt), sha256=hashlib.sha256(raw).hexdigest())
    else:
        ref["weights"] = {}
        for d in (3, 5):
            torch.manual_seed(7)
            net = M.NetCL(d, **SHAPE)
            with torch.no_grad():
                for q in net.parameters():
                    q.add_(0.1 * torch.randn_like(q))
            ref["weights"][d] = net.state_dict()
    ref["digests"] = table_digests(cs)
    out = outputs(ref)
    if ckpt:
        ref["out_digest"] = {k: digest(v.numpy()) for k, v in out.items()}
    else:
        ref["out"] = out
    code = os.path.dirname(os.path.dirname(os.path.abspath(cc.__file__)))
    commit = open(os.path.join(code, "COMMIT")).read().strip() if os.path.exists(os.path.join(code, "COMMIT")) else "?"
    ref.update(cpu=cpu_model(), torch=torch.__version__, commit=commit)
    torch.save(ref, path)
    print(len(cs), "circuits;", len(out), "tensors; code", commit, "; cpu", ref["cpu"])


def compare(old, now):
    diff = {k: float((now[k].double() - v.double()).abs().max()) for k, v in old.items() if not torch.equal(now[k], v)}
    print(dict(tensors=len(now), unequal=len(diff), max_diff=max(diff.values(), default=0.0), first=sorted(diff)[:5]))
    return not diff and set(old) == set(now)


def unequal(ref, now):
    """Names of the tensors in which now differs from ref (its tensors, or their digests in a v2 reference)."""
    want = ref["out_digest"] if "out_digest" in ref else ref["out"]
    if set(want) != set(now):
        return sorted(set(want) ^ set(now))
    if "out_digest" in ref:
        return sorted(k for k, v in want.items() if digest(now[k].numpy()) != v)
    return sorted(k for k, v in want.items() if not torch.equal(now[k], v))


def check(path):
    ref = torch.load(path, weights_only=False)
    bad = [i for i, (a, b) in enumerate(zip(table_digests(ref["cases"]), ref["digests"])) if a != b]
    now = outputs(ref)
    diff = unequal(ref, now)
    print(dict(tables_differ=bad, tensors=len(now), unequal=len(diff), first=diff[:5]))
    return not bad and not diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["make", "check", "dump", "compare"])
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", help="make: a trained two-block network (v2 reference) in place of random weights")
    a = ap.parse_args()
    if a.what == "make":
        make(a.out, a.ckpt)
    elif a.what == "check":
        raise SystemExit(0 if check(a.ref) else 1)
    elif a.what == "dump":
        if a.device != "cpu":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        torch.save(outputs(torch.load(a.ref, weights_only=False), a.device), a.out)
    else:
        raise SystemExit(0 if compare(torch.load(a.ref), torch.load(a.out)) else 1)


if __name__ == "__main__":
    main()
