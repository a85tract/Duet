"""Network reads on the open benchmark's stored points.

  read one network (--model, --d) on its stored points: per circuit the tables with the stored product basis, then
            the engine's precise mode (duet.net.engine_cl) or, with --path frozen, the frozen forward of
            the logits. Writes decode.py's part files PRED/net/parts/<point>/c<i>_<a>.npz, one per circuit and range of
            at most --part shots, and skips parts that exist, so a job resumes without reading a shot twice. Circuits
            are grouped by shape (blocks, layers, product slots); the engine's chunk of shots comes from the bytes of
            every buffer it allocates for the shape, against --budget-gb, and its buffers are freed between shapes. --k1 idle: K = 1 circuits read with an idle second block.
  dev 13.3-13.4 on development shots (never a test shot), for the cells --cells model:d...: the decision
            check (the engine against the frozen forward on a sample of benchmark circuits, with the detectors of the
            software- and physical-Pauli circuits compared), the K = 1 forms (native, idle second block) and the
            twin's network reads (physical and software Paulis). Records in --out.
  devmle the twin's exact MLE (env_mle) on the same development shots.
  devstats the outcomes and their rules: the decision check, the K = 1 form, the twin's penalty per cell.

  python scripts/bench_net.py read --manifest M --sets SETS --pred PRED --model lukin --d 3 [--only REGEX] [--path frozen]
  python scripts/bench_net.py read... --release R --net-path P --net-sha H the shots
            from the release tree R and another network of --model, --d
            (its SHA-256 asserted; the record states whether the network code equals 1ad51f7's); dev takes --net-path too
  python scripts/bench_net.py dev --manifest M --out DIR --cells lukin:3,uniform:3 [--check] [--k1] [--twin]
  python scripts/bench_net.py devmle --manifest M --out DIR --workers 32 (env_mle)
  python scripts/bench_net.py devstats --manifest M --out DIR"""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np

from duet import bench_decode as bd
from duet import bench_sets as bs
from duet import clifford_circuits as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.environ.get("DUET_NETWORKS", "networks")        # directory of the network files (see networks/NETWORKS.md)
FINAL = {(m, d): f"{m}/d{d}_ft4_s{(8420 if m == 'lukin' else 7420) + d}/final_ema.pt"
         for m in ("lukin", "uniform") for d in (3, 5, 7, 9)}
SHA = {  # the networks of reads of record; d = 9: the development job's record
    ("lukin", 3): "388def833b586967d359dcfb28ca97d647bc25e6ad9f7767cfcebdec68187e27",
    ("uniform", 3): "1bbee27f53d93f87765620ac66692c423994756f339e226fe50674324e4b333a",
    ("lukin", 5): "34f37ad6767170dc6183cdb307b44435d84fe78869b5ea4e2151d43fd40f2a33",
    ("uniform", 5): "b2230be0c3c7ccce49a8d20af9f33ea6ac804938f560261c9bdef42ee90550b6",
    ("lukin", 7): "75e371765ecdf6bd15491d8db6b0ef3544ab225fd0b6ddd6b75f0f6b9fabd4d7",
    ("uniform", 7): "f925906f88fbb7a9d7fc48e1f44da766fe711968da46f4c8f921d8a993e2e727",
    ("lukin", 9): "bd06a84fd264815f62493638042e2b631d0a3db5ea4dc008f525f404c8af0d00",       # job 35133729
    ("uniform", 9): "06fd9a01efffae0eae9bd55b3ba29e803eeaa5d83eb5caea33124621dbf89f43"}
FROZEN = {  # network code at commit 1ad51f7, the frozen read path's
    "duet/net/model_cl.py": "36f2e3ada212964f9cd427cc9d8afe1c8fbe12556d472ef04f7a34327223dfb7",
    "duet/net/train_cl.py": "55188a8776f2b9c9790bf3a2baf8f8f31df6f2642b80a157dbe8a467d3b31c70",
    "duet/net/spec_cl.py": "a2ac195fae5a7173d214e8239712fa4afee240b539bb8bff4abc5f46780e8a70"}
BATCH = {3: 4096, 5: 2048, 7: 1024, 9: 512}          # the frozen forward's batches (eval_net's)
DEV0 = 5 * 10 ** 8                                     # development circuit index DEV0 + 10^7 c + 10^4 q + i
DEV_C = dict(check=1, k1=2, gate=3, frame=4, idle=5)
IDLE_NOISE = 9                                         # the idle block's stim seeds at test points
CHECK_SHOTS = 8192
K1_SHOTS = {("lukin", 3): 50000, ("lukin", 5): 300000, ("uniform", 3): 5000, ("uniform", 5): 10000}
TWIN = {("lukin", 3): (1100, 600), ("uniform", 3): (110, 70), ("uniform", 5): (300, 180)}  # shots: lukin, mccd
BOOT = 10_000


# ---------------------------------------------------------------------------------------------------- circuits
def args_of(c_):
    """(layers, prep, meas, products) of a manifest circuit; products None: make_circuit's default."""
    prods = c_.get("products")
    return c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]), None if prods is None else [list(s) for s in prods]


def variant(pt):
    """The seed's variant field v of a point, as the pilot's."""
    if pt["family"] == "ours_k":
        return 300 * (pt["K"] - 3) + 10 * pt["L"] + bs.ms.FAMILY[pt["sub"]]
    return int(pt["circuits"][0]["seed"]) // 10 ** 9 % 1000


def dev_seed(pt, c, i):
    """Development stim seed of circuit i for purpose c (DEV_C): circuit index DEV0 + 10^7 c + 10^4 q + i, q ours_k's
    pattern; the probe and the pilot used indices below 2·10^8."""
    q = int(pt["circuits"][0]["seed"]) // 10 ** 6 % 1000 if pt["family"] == "ours_k" else 0
    assert 0 <= q < 100 and 0 <= i < 10 ** 4
    return bs.dev_seed(pt["noise"], pt["family"], pt["d"], variant(pt), DEV0 + 10 ** 7 * DEV_C[c] + 10 ** 4 * q + i)


def idle_seed(pt, i, part):
    """Test points read in the idle-block form: the stim seed of block 1's events for part `part` of circuit i."""
    nz = ("uniform", "lukin", "si1000", "cain").index(pt["noise"]) + 1
    assert i < 1000 and part < 10 ** 6
    return (bs.BENCH_SEED_BASE + IDLE_NOISE * 10 ** 15 + bs.FAMILY[pt["family"]] * 10 ** 14 + nz * 10 ** 13
            + pt["d"] * 10 ** 12 + variant(pt) * 10 ** 9 + i + 1000 * part)


def idle_pair(pt, c_):
    """The idle-block form of a K = 1 circuit: ((layers, prep, meas) of the two-block circuit, the two-block detector
    index of each of the K = 1 circuit's detectors, block 1's detector indices, the two-block stim circuit). The blocks
    share no gate, so block 0's events have the K = 1 circuit's distribution and block 1's are independent of them."""
    lay, prep, meas, _ = args_of(c_)
    lay2, prep2, meas2 = [g + "I" for g in lay], prep + ("Z",), meas + ("Z",)
    c1, m1 = bs.circuit(pt, c_, meta=True)
    c2, m2 = cc.make_circuit(pt["d"], lay2, pt["p"], prep2, meas2, meta=True, noise=pt["noise"],
                             pauli=c_.get("pauli", "frame"))
    key = lambda m, j: (int(m["det_t"][j]), *map(int, m["det_xy"][j]), int(m["det_type"][j]))   # noqa: E731
    blk = np.asarray(m2["det_block"])
    pos = {key(m2, j): j for j in np.flatnonzero(blk == 0)}
    i0 = np.array([pos[key(m1, j)] for j in range(c1.num_detectors)], np.int64)
    i1 = np.flatnonzero(blk == 1)
    assert len(pos) == len(set(i0)) == c1.num_detectors and len(i0) + len(i1) == c2.num_detectors
    assert list(m2["products"][0][0]) == [0] and c1.num_observables == 1
    return (lay2, prep2, meas2), i0, i1, c2


def assemble(dets1, dets2, i0, i1):
    """Packed two-block input: the K = 1 circuit's events (bool) on block 0, block 1's from the two-block sample."""
    x = np.zeros((len(dets1), len(i0) + len(i1)), bool)
    x[:, i0] = dets1
    x[:, i1] = dets2[:, i1]
    return np.packbits(x, axis=1, bitorder="little")


# ---------------------------------------------------------------------------------------------------- networks
def load_net(model, d, device="cuda", path=None):
    """(network, record): final network of (model, d), EMA weights, with the file's SHA-256 and step."""
    import torch
    from duet.net import model_cl, train_cl
    path = path or os.path.join(M, FINAL[(model, d)])
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert train_cl.format_ok(ck["format"]), ck["format"]
    net = model_cl.NetCL(**ck["cfg"])
    net.load_state_dict(ck["model"])
    net = net.to(device).eval()
    assert net.d == d, (net.d, d)
    return net, dict(path=path, sha256=sha, step=ck.get("step"), cfg=net.cfg)


def frozen_code():
    """{file: SHA-256} of the network code this process imports, which must equal 1ad51f7's."""
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest() for f in FROZEN}


def named_net(a, model, d):
    """(network, record) of a read: network of record (its SHA-256 and 1ad51f7's network code asserted), or with
    --net-path another network of the same model and distance,
    whose file must have the SHA-256 --net-sha; its record states whether the network code equals 1ad51f7's."""
    if a.net_path:
        net, info = load_net(model, d, path=a.net_path)
        assert info["sha256"] == a.net_sha, ("not the named network", info["sha256"])
        info["frozen_code"] = frozen_code() == FROZEN
        return net, info
    assert frozen_code() == FROZEN, "the network code differs from 1ad51f7's"
    net, info = load_net(model, d)
    assert info["sha256"] == SHA.get((model, d)), ("not the network of record", info["sha256"])
    return net, info


def release_meta(root, name):
    """points/<name>/meta.json of the release duet-bench-v1 (its manifest entry; per circuit detectors, observables)."""
    with open(os.path.join(root, "points", name, "meta.json")) as f:
        return json.load(f)


def release_rows(root, name, i, m, k):
    """Read-only memmaps (shots, bytes) of circuit i's detection events and observable flips in the release (stim's b8
    rows: ceil(m / 8) and ceil(k / 8) bytes per shot), so a part reads only its own rows."""
    d = os.path.join(root, "points", name)
    return (np.memmap(os.path.join(d, f"c{i}.dets.b8"), np.uint8, "r").reshape(-1, (m + 7) // 8),
            np.memmap(os.path.join(d, f"c{i}.obs.b8"), np.uint8, "r").reshape(-1, (k + 7) // 8))


class Reader:
    """One network's logits and decisions, circuit by circuit: the engine's precise mode (one engine per circuit shape,
    its buffers freed when the shape changes) or the frozen forward."""

    def __init__(self, net, d, noise, p, path="engine", budget_gb=None):
        """budget_gb: the engine's buffers per shape; half the GPU's memory by default (40 GB on an A100 80 GB, 10 GB on
        an A100 MIG 3g slice)."""
        import torch
        self.net, self.d, self.noise, self.p, self.path = net, d, noise, p, path
        self.budget = (budget_gb * 2 ** 30 if budget_gb else
                       torch.cuda.get_device_properties(0).total_memory / 2 if torch.cuda.is_available() else 2 ** 33)
        self.e, self.shape = None, None

    def free(self):
        import torch
        self.e, self.shape = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def load(self, lay, prep, meas, products):
        """Load a circuit; its sizes (m, n_obs, nb) and the setup seconds."""
        t0 = time.time()
        if self.path == "frozen":
            from duet.net import train_cl
            self.tab = train_cl.tables(self.d, lay, prep, meas, self.p, self.noise, products)[0]
            info = dict(m=self.tab["m"], n_obs=self.tab["n_obs"], nb=(self.tab["m"] + 7) // 8)
        else:
            from duet.net import engine_cl, spec_fast
            s = spec_fast.build(self.d, lay, prep, meas, self.p, self.noise, products)
            shape = (s["K"], s["T"], s["slots"])
            if shape != self.shape:
                self.free()
                self.e = engine_cl.EngineCL(self.net, "precise", BM=16 if self.d == 3 else 64, chunk=4096)
                self.shape = shape
                info = self.e.circuit(lay, prep, meas, self.noise, self.p, spec=s)
                ch = 4096
                while ch > 1 and ch * shot_bytes(self.e) > self.budget:
                    ch //= 2
                self.e.cfg["chunk"] = ch
            else:
                info = self.e.circuit(lay, prep, meas, self.noise, self.p, spec=s)
            self.info = info
        info = dict(m=int(info["m"]), n_obs=int(info["n_obs"]), nb=int(info.get("nb", (info["m"] + 7) // 8)))
        return info, time.time() - t0

    def logits(self, packed):
        """ of packed shots of the loaded circuit."""
        if self.path == "frozen":
            c, self.batch = frozen(self.net, self.tab, packed, self.d)
            return c, c > 0
        c, _, dec = self.e.decode(self.info, np.ascontiguousarray(packed[:, :self.info["nb"]]), B=4096)
        return c, dec

    def settings(self):
        return dict(path=self.path, mode="precise" if self.path == "engine" else "fp32 eager, TF32 off",
                    engine=None if self.e is None else dict(self.e.cfg), shot_bytes=None if self.e is None else
                    shot_bytes(self.e), budget_gb=self.budget / 2 ** 30,
                    batch=getattr(self, "batch", None) if self.path == "frozen" else 4096)


def shot_bytes(e):
    """Bytes per shot of every buffer EngineCL.buffers allocates for the loaded shape (x, uc, ub, pa, l) and of a
    graph's input and output."""
    return (4 * e.MR * e.C + 2 * e.MR * e.UW + 2 * e.NG * e.UW + 4 * e.S * e.NQ * e.d * e.C + 4 * e.S * e.d * e.T
            + e.NBYM + 4 * (e.S + 1) + 1)


def frozen(net, tab, packed, d):
    """(logits (n, n_obs), the batch used) by the frozen forward (train_cl.collate and forward, eval_net's batches; the
    batch halves on a CUDA out-of-memory error, so a circuit too large for the GPU at eval_net's batch runs at a smaller
    one)."""
    import torch
    from duet.net import train_cl
    B, s, out = BATCH[d], 0, []
    dev = next(net.parameters()).device
    while s < len(packed):
        b = min(B, len(packed) - s)
        zero = np.zeros((b, (tab["n_total"] + 7) // 8), np.uint8)
        try:
            with torch.no_grad():
                c, _, _ = train_cl.forward(net, train_cl.collate([(tab, packed[s:s + b], zero)], d, dev, train=False),
                                           train=False)
        except torch.cuda.OutOfMemoryError:
            c = None                                     # freed with the exception, outside the handler
        if c is None:
            assert B > 1, "one shot does not fit"
            B //= 2
            torch.cuda.empty_cache()
            continue
        out.append(c[:, :tab["n_obs"]].float().cpu().numpy())
        s += b
    torch.cuda.empty_cache()                             # the engine's buffers come next in the development job
    return np.concatenate(out), B


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def commit():
    f = os.path.join(ROOT, "COMMIT")
    return open(f).read().strip() if os.path.exists(f) else None


# ---------------------------------------------------------------------------------------------------- read
def part_path(pred, name, i, s):
    return os.path.join(pred, "net", "parts", name, f"c{i}_{s}.npz")


def write_part(path, pred, obs, target_slot, secs, build, settings):
    """A part in decode.py's format (every shot counted: the network has no certification)."""
    ok = np.ones(len(pred), bool)
    tal = bd.tally(pred, obs, ok, target_slot)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", pred=np.packbits(pred, axis=1, bitorder="little"), ok=np.packbits(ok, bitorder="little"),
             n=len(pred), cpu=secs, build=build, dropped=np.array([0]), tally=json.dumps(tal),
             settings=json.dumps(settings, default=str))
    os.replace(path + ".tmp.npz", path)
    return tal


def read(a, net=None, info=None):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if net is None:
        net, info = named_net(a, a.model, a.d)
    pts = [pt for pt in json.load(open(a.manifest))["points"] if pt["noise"] == a.model and pt["d"] == a.d
           and (not a.only or re.search(a.only, pt["name"]))]
    metas = {pt["name"]: release_meta(a.release, pt["name"]) for pt in pts} if a.release else {}
    groups = {}                                           # by the engine's shape (K, T, slots = max(2, K, products))
    for pt in pts:
        ks = ([c["observables"] for c in metas[pt["name"]]["circuits"]] if a.release else
              np.load(os.path.join(a.sets, pt["name"] + ".npz"))["k"])
        for i in range(len(pt["circuits"])):
            K, k = (2, 2) if pt["K"] == 1 and a.k1 == "idle" else (pt["K"], int(ks[i]))
            groups.setdefault((K, pt["L"] + 1, max(2, K, k)), []).append((pt, i))
    rd = Reader(net, a.d, a.model, 0.003, a.path, a.budget_gb)
    base = dict(network=info, model=a.model, d=a.d, k1=a.k1, commit=commit(), gpu=gpu_name())
    t0, done = time.time(), 0
    for shape in sorted(groups):
        for pt, i in groups[shape]:
            c_, name = pt["circuits"][i], pt["name"]
            assert pt["p"] == 0.003, name
            if a.release:                     # the release's b8 files
                cm = metas[name]["circuits"][i]
                m, k = int(cm["detectors"]), int(cm["observables"])
                Di, Oi = release_rows(a.release, name, i, m, k)
                assert len(Di) == len(Oi) == pt["shots"], (name, i, len(Di), len(Oi))
            else:
                path = os.path.join(a.sets, name + ".npz")
                z = np.load(path)
                m, k = int(z["m"][i]), int(z["k"][i])
                Di, Oi = bs.packed(path, "dets")[i], bs.packed(path, "obs")[i]
            todo = [s for s in range(0, pt["shots"], a.part) if not os.path.exists(part_path(a.pred, name, i, s))]
            if not todo:
                continue
            idle = pt["K"] == 1 and a.k1 == "idle"
            if idle:
                (lay, prep, meas), i0, i1, c2 = idle_pair(pt, c_)
                prods = None
            else:
                lay, prep, meas, prods = args_of(c_)
            sizes, build = rd.load(lay, prep, meas, prods)
            assert sizes["n_obs"] == (2 if idle else k) and sizes["m"] == (len(i0) + len(i1) if idle else m), name
            for s in todo:
                e = min(pt["shots"], s + a.part)
                det, obs = np.asarray(Di[s:e]), bs.unpack(np.asarray(Oi[s:e]), k)
                if idle:
                    d2 = c2.compile_detector_sampler(seed=idle_seed(pt, i, s // a.part)).sample(e - s)
                    det = assemble(bs.unpack(det, m), d2, i0, i1)
                tg = time.time()
                _, dec = rd.logits(det)
                secs = time.time() - tg
                pred = dec[:, :1] if idle else dec
                set_ = dict(base, **rd.settings(), idle_seed=idle_seed(pt, i, s // a.part) if idle else None)
                tal = write_part(part_path(a.pred, name, i, s), pred, obs, c_.get("target_slot"), secs, build, set_)
                done += e - s
                print(f"{name} c{i} [{s}, {e}) fails {tal['fails']} {secs:.1f} s", flush=True)
    print(f"{done} shots in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------- dev
def check_circuits(pts, model, d):
    """ 13.4's decision-check sample of one cell: [(point, circuit index)]."""
    P = {pt["name"]: pt for pt in pts if pt["noise"] == model and pt["d"] == d}
    tag = f"d{d}_{model}_p3"
    want = [(f"lom_rep_{g}_{tag}", 0) for g in ("I_Z", "H_X", "CNOT_alt_X")]
    for b in "ZX":
        pt = P.get(f"lom_2q_{b}_{tag}")
        if pt:
            want.append((pt["name"], int(np.argmax([c_["cnot_layers"] for c_ in pt["circuits"]]))))
    want += [(f"lom_deep_deep32_{tag}", 0), (f"lukin_D1_{tag}", 0), (f"lukin_D10_{tag}", 0), (f"mccd_t1_D18_{tag}", 0),
             (f"mccd_t2_D36_{tag}", 0), (f"cain_ghz_ghz3_{tag}", 0)]
    want += [(f"ours_k_{s}_d{d}_L{L}_{model}_p3", 0) for s, L in (("mix_K3", 16), ("f1b_K4", 20), ("cnot_K4", 10),
                                                                    ("struct_ghz_chain_z_K3", 16))]
    return [(P[n], i) for n, i in want if n in P]


def twin_circuits(pts, model, d):
    """The twin's circuits of one cell: lukin circuits 0-3 at every depth, mccd circuits 0-3 of every point."""
    return [(pt, i) for pt in pts if pt["noise"] == model and pt["d"] == d and pt["family"] in ("lukin", "mccd")
            for i in range(4)]


def k1_circuits(pts, model, d):
    return [(pt, 0) for pt in pts if pt["noise"] == model and pt["d"] == d and pt["family"] == "lom_rep" and pt["K"] == 1]


def sample(pt, c_, seed, n):
    """Packed (dets, obs) of n development shots of circuit c_ of point pt."""
    return bs.circuit(pt, c_).compile_detector_sampler(seed=seed).sample(n, separate_observables=True, bit_packed=True)


def twin_file(out, model, d):
    return os.path.join(out, f"twin_shots_{model}_d{d}.npz")


def twin_sample(out, pts, model, d):
    """Sample the twin's development shots of one cell once and store them (dets and obs packed, per circuit and arm), so
    that the network and the MLE read the same shots; devstats compares their checksums."""
    f = twin_file(out, model, d)
    if os.path.exists(f):
        return
    arrs = {}
    for pt, i in twin_circuits(pts, model, d):
        n = TWIN[(model, d)][pt["family"] == "mccd"]
        for arm in ("gate", "frame"):
            dets, obs = sample(pt, dict(pt["circuits"][i], pauli=arm), dev_seed(pt, arm, i), n)
            arrs[f"{pt['name']}|{i}|{arm}|dets"], arrs[f"{pt['name']}|{i}|{arm}|obs"] = dets, obs
    np.savez(f + f".{os.getpid()}.tmp.npz", **arrs)
    os.replace(f + f".{os.getpid()}.tmp.npz", f)


def sha_of(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def versions():
    import stim
    return dict(stim=stim.__version__, numpy=np.__version__)


def dev(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pts = json.load(open(a.manifest))["points"]
    os.makedirs(a.out, exist_ok=True)
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")]
    if a.twin:
        for cell in cells:
            if cell in TWIN:
                twin_sample(a.out, pts, *cell)           # first, so the MLE job finds them
    for model, d in cells:
        net, info = named_net(a, model, d) if a.net_path else load_net(model, d)
        rec = dict(network=info, frozen_code=frozen_code() == FROZEN, gpu=gpu_name(), commit=commit(), **versions())
        eng = Reader(net, d, model, 0.003, "engine", a.budget_gb)
        frz = Reader(net, d, model, 0.003, "frozen")
        if a.check:
            rows = []
            for pt, i in check_circuits(pts, model, d):
                if a.only and not re.search(a.only, pt["name"]):
                    continue
                c_ = pt["circuits"][i]
                lay, prep, meas, prods = args_of(c_)
                _, mg = bs.circuit(pt, c_, meta=True)
                _, mf = bs.circuit(pt, dict(c_, pauli="frame"), meta=True)
                same = all(np.array_equal(np.asarray(mg[x]), np.asarray(mf[x]))
                           for x in ("det_block", "det_t", "det_xy", "det_type"))
                dets, obs = sample(pt, c_, dev_seed(pt, "check", i), CHECK_SHOTS)
                ti, _ = frz.load(lay, prep, meas, prods)
                t0 = time.time()
                c32, d32 = frz.logits(dets)
                t1 = time.time()
                si, _ = eng.load(lay, prep, meas, prods)
                assert si == ti, (pt["name"], si, ti)
                ce, de = eng.logits(dets)
                t2 = time.time()
                y = bs.unpack(obs, si["n_obs"])
                bad = np.flatnonzero((de != d32).any(1))
                rows.append(dict(point=pt["name"], i=i, K=pt["K"], L=pt["L"], pauli=c_["pauli"], shots=CHECK_SHOTS,
                                 detectors_equal=bool(same), fail_frozen=int((d32 != y).any(1).sum()),
                                 fail_engine=int((de != y).any(1).sum()), differ=int(len(bad)),
                                 differ_c=[(float(c32[j].min()), float(ce[j].min())) for j in bad[:20]],
                                 max_abs_dc=float(np.abs(ce - c32).max()),
                                 c_at_max_dc=float(np.abs(c32).flat[np.abs(ce - c32).argmax()]),
                                 max_abs_c=float(np.abs(c32).max()),
                                 max_rel_dc=float((np.abs(ce - c32) / np.maximum(np.abs(c32), 1.0)).max()),
                                 chunk=eng.e.cfg["chunk"],
                                 shot_bytes=shot_bytes(eng.e), secs_frozen=round(t1 - t0, 2),
                                 secs_engine=round(t2 - t1, 2)))
                print(json.dumps(rows[-1]), flush=True)
            rec["check"] = rows
        if a.k1f and (model, d) in K1_SHOTS:
            n, rows, data = K1_SHOTS[(model, d)], [], []
            for pt, i in k1_circuits(pts, model, d):     # the native form, then the idle-block form: two shapes
                c_ = pt["circuits"][i]
                dets, obs = sample(pt, c_, dev_seed(pt, "k1", i), n)
                eng.load(*args_of(c_))
                data.append((pt, i, dets, bs.unpack(obs, 1)[:, 0], eng.logits(dets)[1][:, 0]))
            for pt, i, dets, y, dn in data:
                (lay2, prep2, meas2), i0, i1, c2 = idle_pair(pt, pt["circuits"][i])
                d2 = c2.compile_detector_sampler(seed=dev_seed(pt, "idle", i)).sample(n)
                eng.load(lay2, prep2, meas2, None)
                di = eng.logits(assemble(bs.unpack(dets, len(i0)), d2, i0, i1))[1][:, 0]
                fn, fi = dn != y, di != y
                rows.append(dict(point=pt["name"], i=i, shots=n, both=int((fn & fi).sum()),
                                 native_only=int((fn & ~fi).sum()), idle_only=int((~fn & fi).sum())))
                print(json.dumps(rows[-1]), flush=True)
            rec["k1"] = rows
        if a.twin and (model, d) in TWIN:
            z, fails, sums = np.load(twin_file(a.out, model, d)), {}, {}
            for pt, i in twin_circuits(pts, model, d):
                eng.load(*args_of(pt["circuits"][i]))
                for arm in ("gate", "frame"):
                    key = f"{pt['name']}|{i}|{arm}"
                    dets, obs = z[key + "|dets"], z[key + "|obs"]
                    dec = eng.logits(dets)[1]
                    fails[key] = np.packbits((dec != bs.unpack(obs, dec.shape[1])).any(1), bitorder="little")
                    sums[key] = sha_of(dets)
            np.savez(os.path.join(a.out, f"twin_net_{model}_d{d}.npz"), **fails)
            rec["twin_shots_sha"] = sums
        eng.free()
        del net
        json.dump(rec, open(os.path.join(a.out, f"dev_{model}_d{d}.json"), "w"), indent=1)
        print(f"cell {model} d = {d} done", flush=True)


# ---------------------------------------------------------------------------------------------------- devmle
PTS = {}


def _mle_job(job):
    out, model, d, name, i, arm = job
    pt, key = PTS[name], f"{name}|{i}|{arm}"
    z = np.load(twin_file(out, model, d))
    dets, obs = z[key + "|dets"], z[key + "|obs"]
    c, dec_ = bd.build("mle", pt, dict(pt["circuits"][i], pauli=arm))
    pred, ok, cpu = bd.decode("mle", dec_, bs.unpack(dets, c.num_detectors))
    fail = (pred != bs.unpack(obs, c.num_observables)).any(1)
    return key, np.packbits(fail, bitorder="little"), np.packbits(ok, bitorder="little"), sha_of(dets), cpu


def devmle(a):
    PTS.update({pt["name"]: pt for pt in json.load(open(a.manifest))["points"]})
    pts = list(PTS.values())
    os.makedirs(a.out, exist_ok=True)
    for model, d in TWIN:
        twin_sample(a.out, pts, model, d)                # here if the GPU job has not stored them yet
    jobs = [(a.out, model, d, pt["name"], i, arm) for model, d in TWIN for pt, i in twin_circuits(pts, model, d)
            for arm in ("gate", "frame")]
    jobs.sort(key=lambda j: -(PTS[j[3]]["K"] * PTS[j[3]]["L"] * PTS[j[3]]["d"] ** 3))   # the slowest first
    res = {cell: {} for cell in TWIN}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for job, r in zip(jobs, pool.imap(_mle_job, jobs)):
            res[(job[1], job[2])][r[0]] = r[1:]
            print(r[0], f"{r[4]:.0f} s", flush=True)
    for (model, d), rr in res.items():
        np.savez(os.path.join(a.out, f"twin_mle_{model}_d{d}.npz"),
                 **{f"{k}|fail": v[0] for k, v in rr.items()}, **{f"{k}|ok": v[1] for k, v in rr.items()})
        json.dump(dict(versions(), circuits={k: dict(sha=v[2], cpu=v[3]) for k, v in rr.items()}),
                  open(os.path.join(a.out, f"twin_mle_{model}_d{d}.json"), "w"), indent=0)


# ---------------------------------------------------------------------------------------------------- devstats
def pct(x, q):
    return [float(v) for v in np.percentile(x, q)]


def k1_rule(rows, seed=1671):
    """Pooled idle-block / native failures of the K = 1 forms with a 95 % interval from shots redrawn paired within
    each circuit; the idle-block form is chosen when the interval's upper end is below 1."""
    rng = np.random.default_rng(seed)
    fi = sum(r["both"] + r["idle_only"] for r in rows)
    fn = sum(r["both"] + r["native_only"] for r in rows)
    bi, bn = np.zeros(BOOT), np.zeros(BOOT)
    for r in rows:
        q = np.array([r["both"], r["native_only"], r["idle_only"],
                      r["shots"] - r["both"] - r["native_only"] - r["idle_only"]], float) / r["shots"]
        x = rng.multinomial(r["shots"], q, size=BOOT)
        bi += x[:, 0] + x[:, 2]
        bn += x[:, 0] + x[:, 1]
    lo, hi = pct(np.log(rr(bi, bn)), [2.5, 97.5])
    return dict(idle_fails=int(fi), native_fails=int(fn), ratio=fi / fn if fn else None,
                ratio95=[float(np.exp(lo)), float(np.exp(hi))], form="idle" if np.exp(hi) < 1 else "native")


def rr(n, m):
    """Failures / failures with rule: 0.5 added to both where either is zero."""
    z = 0.5 * ((n == 0) | (m == 0))
    return (n + z) / (m + z)


def penalty(cell_rows, seed=1673):
    """The twin's penalty of one cell: (network / MLE failures, physical Paulis) / (the same, software Paulis) on the MLE's
    certified shots, with a 95 % interval from 10,000 draws of circuits redrawn within family and depth, the same circuit
    indices in both arms and the network's and the MLE's counts of a circuit kept together."""
    rng = np.random.default_rng(seed)
    keys = sorted({(r["point"], r["i"]) for r in cell_rows})
    by = {(r["point"], r["i"], r["arm"]): r for r in cell_rows}
    N = np.array([[by[(p, i, arm)]["net"] for arm in ("gate", "frame")] for p, i in keys], float)
    Mx = np.array([[by[(p, i, arm)]["mle"] for arm in ("gate", "frame")] for p, i in keys], float)
    pen = lambda n, m: rr(n[..., 0], m[..., 0]) / rr(n[..., 1], m[..., 1])                 # noqa: E731
    groups = {}
    for j, (p, _) in enumerate(keys):
        groups.setdefault(p, []).append(j)               # a point is one family and depth
    idx = np.concatenate([rng.choice(g, size=(BOOT, len(g))) for g in groups.values()], axis=1)
    bN, bM = N[idx].sum(1), Mx[idx].sum(1)
    est = float(pen(N.sum(0), Mx.sum(0)))
    lo, hi = pct(pen(bN, bM), [2.5, 97.5])
    return dict(circuits=len(keys), shots_gate=int(sum(by[(p, i, "gate")]["shots"] for p, i in keys)),
                shots_frame=int(sum(by[(p, i, "frame")]["shots"] for p, i in keys)), net_gate=int(N[:, 0].sum()),
                mle_gate=int(Mx[:, 0].sum()), net_frame=int(N[:, 1].sum()), mle_frame=int(Mx[:, 1].sum()),
                uncertified=int(sum(r["uncertified"] for r in cell_rows)), penalty=est, penalty95=[lo, hi],
                fine_tune_called=bool(est >= 1.10 and lo > 1.0))


def twin_rows(out, model, d, sums):
    """Per circuit and arm: the MLE's certified shots and the network's and the MLE's failures on them."""
    zn = np.load(os.path.join(out, f"twin_net_{model}_d{d}.npz"))
    zm = np.load(os.path.join(out, f"twin_mle_{model}_d{d}.npz"))
    msha = json.load(open(os.path.join(out, f"twin_mle_{model}_d{d}.json")))["circuits"]
    rows = []
    for key, s in sums.items():
        assert msha[key]["sha"] == s, ("the network and the MLE read other shots", key)
        p, i, arm = key.split("|")
        n = TWIN[(model, d)][p.startswith("mccd")]
        ok, fn, fm = (np.unpackbits(x, count=n, bitorder="little").astype(bool)
                      for x in (zm[key + "|ok"], zn[key], zm[key + "|fail"]))
        rows.append(dict(point=p, i=int(i), arm=arm, shots=int(ok.sum()), uncertified=int((~ok).sum()),
                         net=int((fn & ok).sum()), mle=int((fm & ok).sum())))
    return rows


def devstats(a):
    out, k1rows = dict(check={}, k1=None, twin={}), []
    for f in sorted(os.listdir(a.out)):
        if not (f.startswith("dev_") and f.endswith(".json")):
            continue
        r, cell = json.load(open(os.path.join(a.out, f))), f[4:-5]
        model, d = cell.split("_d")[0], int(cell.split("_d")[1])
        if "check" in r:
            rows = r["check"]
            out["check"][cell] = dict(circuits=len(rows), shots=sum(x["shots"] for x in rows),
                                      differ=sum(x["differ"] for x in rows),
                                      max_abs_dc=max(x["max_abs_dc"] for x in rows),
                                      max_rel_dc=max(x.get("max_rel_dc", 0.0) for x in rows) or None,
                                      detectors_equal=all(x["detectors_equal"] for x in rows),
                                      passed=all(x["differ"] == 0 and x["max_abs_dc"] <= 1e-3 and x["detectors_equal"]
                                                 for x in rows), network=r["network"]["sha256"],
                                      step=r["network"]["step"], frozen_code=r["frozen_code"])
        k1rows += [dict(x, cell=cell) for x in r.get("k1", [])]
        if "twin_shots_sha" in r and os.path.exists(os.path.join(a.out, f"twin_mle_{model}_d{d}.json")):
            rows = twin_rows(a.out, model, d, r["twin_shots_sha"])
            out["twin"][cell] = dict(penalty(rows), rows=rows)    # the rows: the penalty again from the record
    if k1rows:
        out["k1"] = dict(k1_rule(k1rows), per_cell={c: k1_rule([x for x in k1rows if x["cell"] == c])
                                                    for c in sorted({x["cell"] for x in k1rows})})
    out["checks_passed"] = all(c["passed"] for c in out["check"].values()) if out["check"] else None
    json.dump(out, open(os.path.join(a.out, "devstats.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "dev", "devmle", "devstats"])
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/open_benchmark/MANIFEST.json"))
    ap.add_argument("--sets")
    ap.add_argument("--pred")
    ap.add_argument("--out")
    ap.add_argument("--model", choices=["lukin", "uniform"])
    ap.add_argument("--d", type=int)
    ap.add_argument("--only", default="", help="read: the points; dev: the decision check's circuits")
    ap.add_argument("--path", default="engine", choices=["engine", "frozen"])
    ap.add_argument("--k1", default="native", choices=["native", "idle"], help="read: the K = 1 form")
    ap.add_argument("--part", type=int, default=2 ** 20, help="read: shots per part file")
    ap.add_argument("--budget-gb", type=float, default=None, help="the engine's buffers (default half the GPU)")
    ap.add_argument("--cells", default="", help="dev: model:d,...")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--k1-forms", dest="k1f", action="store_true")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--release", default="", help="read: the shots from this duet-bench-v1 tree instead of --sets")
    ap.add_argument("--net-path", default="", help="read, dev: another network of --model, --d (one cell)")
    ap.add_argument("--net-sha", default="", help="the SHA-256 the --net-path file must have")
    a = ap.parse_args()
    assert not a.net_path or (a.net_sha and (a.cmd != "dev" or len(a.cells.split(",")) == 1)), "--net-sha, one cell"
    {"read": read, "dev": dev, "devmle": devmle, "devstats": devstats}[a.cmd](a)


if __name__ == "__main__":
    main()
