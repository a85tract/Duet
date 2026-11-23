"""The pair end to end on one machine: a batch of B shots of one circuit through the
network (the Triton engine, precise mode, one CUDA graph per batch size, host memory to host memory) on the GPU, then
its routed shots through Tesseract on N cores in arrival order, each to the first free core;
time per shot = the batch's wall time / B. A routed shot whose decode passes the cap takes the network's answer: its
timed). Low confidence (no solution within Tesseract's limits) also keeps the network's answer. Every decision is

                                                    first 65,536 shots of each), the threshold, the reference reads
                                                    the Tesseract workers instead of the local stim build's DEM
  e2e.py check --pack DIR --nets DIR a few shots of two points through both decoders (about a minute)
  e2e.py run --pack DIR --nets DIR --out F.jsonl the timing, passes A (B = 256, the Lukin group's batch), L
                                                    (batch-1 latency), C (B = 4,096), E (1,024, 16,384), X (65,536,
                                                    circuit i = 20 only: --xpos), in this order,
                                                    N = 20, 10 fast cores, 1 (--cores: the Spark's layout, or auto)
  e2e.py sum --runs F.jsonl the tables (each depth weighted equally, as the selection score)

E2E_HEAD=1 runs the headline pairs instead: the long beam (no overrides) with no cap on the shots each
d's network routes (d = 3 ft4 f = 10 %, d = 5 MH_q4 f = 5 %, d = 7 MH7_q4 f = 10 %); pack then takes d = 3 too, the
"""
import argparse
import json
import os
import pickle
import select
import signal
import socket
import struct
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

F_ROUTE, CAP, SETTING = 0.005, 0.1, "o1nc"
OPTS = dict(num_det_orders=1, beam_climbing=False)       # speed.py SETTINGS["o1nc"]
HEAD = os.environ.get("E2E_HEAD") == "1"   #: the headline pairs: the long beam, no cap, per d a network, f
HEAD_D = {3: ("ft4", 0.10), 5: ("MH_q4", 0.05), 7: ("MH7_q4", 0.10)}
if HEAD:
    CAP, SETTING, OPTS = 1e5, "long", {}                  # no cap (1e5 s: no decode comes near it)


def froute(d):
    return HEAD_D[d][1] if HEAD else F_ROUTE
POS = (0, 1)                                              # circuits i = 20, 21 (the network timing's circuits)
NMAX = 65536
SPARK_CORES = {1: [19], 10: [5, 6, 7, 8, 9, 15, 16, 17, 18, 19], 20: list(range(20))}   # 5-9, 15-19: Cortex-X925
PASSES = {"A": [256], "C": [4096], "X": [65536], "E": [1024, 16384], "Ea": [1024], "Eb": [16384]}   #: 
REPS = {256: 32, 1024: 8, 4096: 2, 16384: 1, 65536: 1}    # batches per (circuit, B, N)


def _lk():
    """scripts/bank_report.py as a module (it loads scripts/dev_bank.py as lukin_bank) -> (bank, report)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lukin_report", os.path.join(ROOT, "scripts", "bank_report.py"))
    m = sys.modules["lukin_report"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.bank, m


def build_circuit(meta, z, pos):
    """The stim circuit of a packed circuit, built by duet.bench_sets as on the reference host; same_text: equal to the stored one."""
    from duet import bench_sets as bs
    circ = bs.circuit(dict(d=meta["d"], p=meta["p"], noise=meta["noise"]), meta["circuits"][pos]["c"])
    return circ, str(circ) == str(z[f"circ{pos}"])


# ------------------------------------------------------------------------------------------------------------ pack

def pack(a):
    bank, report = _lk()
    from duet import bench_sets as bs
    os.makedirs(a.out, exist_ok=True)
    for d in (int(x) for x in a.ds.split(",")):
        part = "tess" if HEAD else "tess_" + SETTING
        for pt in bank.points((d,)):
            r = report.load(a.bank, pt, HEAD_D[d][0] if HEAD else "ft4", [part])
            n, s = r["n"], np.abs(r["logits"][:, 0])
            o, thr = report.routed(s, froute(d))
            path = bank.set_path(a.bank, pt)
            z = np.load(path)
            entry, m_, k_ = json.loads(str(z["entry"])), z["m"].astype(int), z["k"].astype(int)
            Dp, Op = bs.packed(path, "dets"), bs.packed(path, "obs")
            t = r[part]
            out, meta = {}, dict(name=pt["name"], d=d, D=pt["depth"], p=pt["p"], noise=pt["noise"], thr=thr, f=froute(d),
                                 net=HEAD_D[d][0] if HEAD else "ft4", part=part,
                                 n_routed_point=len(o), shots_point=len(s), n_per_circuit=n, circuits=[])
            for pos in POS:
                nn = min(n, a.nmax)
                f0 = pos * n
                nb = (int(m_[pos]) + 7) // 8
                rr = np.zeros(nn, bool)
                oo = o[(o >= f0) & (o < f0 + nn)] - f0
                rr[oo] = True
                assert t["cov"][f0:f0 + nn][rr].all(), "a routed shot without a Tesseract read"
                c_ = entry["circuits"][pos]
                out.update({f"dets{pos}": np.ascontiguousarray(Dp[pos, :nn, :nb]), f"obs{pos}": Op[pos, :nn, 0],
                            f"lg{pos}": r["logits"][f0:f0 + nn, 0].astype(np.float32),
                            f"dnet{pos}": r["dec_net"][f0:f0 + nn], f"rref{pos}": rr,
                            f"tdec{pos}": t["dec"][f0:f0 + nn], f"tok{pos}": t["ok"][f0:f0 + nn],
                            f"tsec{pos}": t["sec"][f0:f0 + nn].astype(np.float32),
                            f"circ{pos}": np.array(str(bs.circuit(entry, c_)))})
                meta["circuits"].append(dict(pos=pos, i=c_["index"] if "index" in c_ else 20 + pos, m=int(m_[pos]),
                                             k=int(k_[pos]), nb=nb, n=nn, routed=int(rr.sum()), c=c_))
            np.savez(os.path.join(a.out, pt["name"] + ".npz"), meta=np.array(json.dumps(meta)), **out)
            print(json.dumps(dict(point=pt["name"], thr=thr, routed=[c["routed"] for c in meta["circuits"]],
                                  n=[c["n"] for c in meta["circuits"]])), flush=True)


def dems(a):
    """Write each packed circuit's detector error model as text: DIR/dem/<point>_c<pos>.dem."""
    import glob
    os.makedirs(os.path.join(a.pack, "dem"), exist_ok=True)
    for path in sorted(glob.glob(os.path.join(a.pack, "lukin_*.npz"))):
        meta, z = load_pack(path)
        for pos in POS:
            circ, same = build_circuit(meta, z, pos)
            assert same
            txt = str(circ.detector_error_model(decompose_errors=False))
            with open(dem_path(path, meta, pos), "w") as f:
                f.write(txt)
            print(json.dumps(dict(point=meta["name"], pos=pos, chars=len(txt))), flush=True)


def dem_path(path, meta, pos):
    return os.path.join(os.path.dirname(path), "dem", f"{meta['name']}_c{pos}.dem")


class TessDem:
    """duet.decoders.tesseract_dec.Tesseract on a given detector error model: the reference DEM text, so that another stim
    build (the Spark's, compiled from source for aarch64) cannot change a probability's last bits and so a tie."""

    def __init__(self, dem, **overrides):
        from importlib.metadata import version
        from duet.decoders import tesseract_dec as td
        s = {**td.SETTINGS, **overrides}
        orders = td.utils.build_det_orders(dem, s["num_det_orders"], getattr(td.utils.DetOrder, s["det_order"]),
                                           s["det_order_seed"])
        cfg = td.tesseract.TesseractConfig(dem=dem, det_beam=s["det_beam"], beam_climbing=s["beam_climbing"],
                                           no_revisit_dets=s["no_revisit_dets"], pqlimit=s["pqlimit"], det_orders=orders)
        self.dec = cfg.compile_decoder()
        self.settings = {"package": f"tesseract-decoder {version('tesseract-decoder')}", "dem": "reference text", **s}


def obs_masks(dem):
    """uint8 per error of the flattened DEM: its observables as bits (bit j = L_j, repeats cancel)."""
    out = []
    for ins in dem.flattened():
        if ins.type == "error":
            m = 0
            for t in ins.targets_copy():
                if t.is_logical_observable_id():
                    m ^= 1 << t.val
            out.append(m)
    return np.array(out, np.uint8)


def predict(dec, dets):
    """Tesseract on one shot -> (observable bits from its predicted errors and the DEM, decode's own bits, ok, s).
    The first is the decision: the aarch64 source build's decode returns wrong observable bits (bit 1 of the lukin
    circuits) although its predicted errors equal the x86 wheel's."""
    t0 = time.perf_counter_ns()
    raw = np.asarray(dec.dec.decode(dets), bool)
    sec = (time.perf_counter_ns() - t0) / 1e9
    ok = not bool(getattr(dec.dec, "low_confidence_flag", False))
    errs = np.asarray(dec.dec.predicted_errors_buffer, np.int64)
    pb = int(np.bitwise_xor.reduce(dec.obsmask[errs])) if len(errs) else 0
    praw = int(np.packbits(raw, bitorder="little")[0]) if len(raw) else 0
    return pb, praw, ok, sec


def make_tess(path, meta, z, pos, opts):
    """(decoder, info): from the reference DEM text when the pack has it (round trip checked), else from the circuit."""
    import stim
    from duet.decoders.tesseract_dec import Tesseract
    circ, same = build_circuit(meta, z, pos)
    dp = dem_path(path, meta, pos)
    if os.path.exists(dp):
        txt = open(dp).read()
        dem = stim.DetectorErrorModel(txt)
        info = dict(same_text=same, dem="reference", dem_roundtrip=str(dem) == txt,
                    dem_same_here=str(circ.detector_error_model(decompose_errors=False)) == txt)
        dec = TessDem(dem, **opts)
    else:
        dem = circ.detector_error_model(decompose_errors=False)
        dec, info = Tesseract(circ, **opts), dict(same_text=same, dem="local")
    dec.obsmask = obs_masks(dem)
    return dec, info


def load_pack(path):
    z = np.load(path)
    return json.loads(str(z["meta"])), z


# ------------------------------------------------------------------------------------------- Tesseract workers

def _recv_msg(s):
    return pickle.loads(s.recv(1 << 20))


def _serve(sock, dec, m):
    """A worker: decode the shots it is sent, one at a time (tag int64, packed detectors) -> (tag, pred, raw, ok, s)."""
    while True:
        msg = sock.recv(1 << 16)
        if not msg:
            os._exit(0)
        tag = struct.unpack_from("<q", msg)[0]
        row = np.frombuffer(msg, np.uint8, offset=8)
        dets = np.unpackbits(row, count=m, bitorder="little").astype(bool)
        pb, praw, ok, sec = predict(dec, dets)
        sock.send(struct.pack("<qBBBd", tag, pb, praw, int(ok), sec))


def _zygote(ctrl):
    """Holds the built decoders; forks pinned workers on request and passes their sockets back (SCM_RIGHTS)."""
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)                    # forked workers are reaped by the kernel
    decs = {}
    while True:
        try:
            msg = _recv_msg(ctrl)
        except (EOFError, OSError):
            os._exit(0)
        if msg[0] == "build":
            _, key, path, pos, opts = msg
            meta, z = load_pack(path)
            t0 = time.time()
            dec, info = make_tess(path, meta, z, pos, opts)
            decs[key] = (dec, meta["circuits"][pos]["m"])
            ctrl.send(pickle.dumps(dict(build_s=time.time() - t0, settings=dec.settings, **info)))
        elif msg[0] == "fork":
            _, key, core = msg
            a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            pid = os.fork()
            if pid == 0:
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                ctrl.close()
                a.close()
                if core is not None:
                    os.sched_setaffinity(0, {core})
                _serve(b, *decs[key])
            b.close()
            socket.send_fds(ctrl, [struct.pack("<q", pid)], [a.fileno()])
            a.close()
        elif msg[0] == "drop":
            decs.pop(msg[1], None)
            ctrl.send(pickle.dumps("ok"))
        else:
            os._exit(0)


class Zygote:
    def __init__(self):
        import multiprocessing as mp
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.p = mp.get_context("fork").Process(target=_zygote, args=(b,), daemon=True)
        self.p.start()
        b.close()
        self.s = a

    def build(self, key, path, pos, opts=OPTS):
        self.s.send(pickle.dumps(("build", key, path, pos, opts)))
        return _recv_msg(self.s)

    def fork(self, key, core):
        self.s.send(pickle.dumps(("fork", key, core)))
        msg, fds, _, _ = socket.recv_fds(self.s, 64, 1)
        return [struct.unpack("<q", msg)[0], socket.socket(fileno=fds[0]), core]

    def drop(self, key):
        self.s.send(pickle.dumps(("drop", key)))
        return _recv_msg(self.s)

    def close(self):
        try:
            self.s.send(pickle.dumps(("exit",)))
        except OSError:
            pass
        self.p.join(5)


def kill(workers):
    for w in workers:
        try:
            os.kill(w[0], signal.SIGKILL)
        except ProcessLookupError:
            pass
        w[1].close()


def dispatch(zyg, key, workers, rows, cap=CAP):
    """rows: packed detector rows (bytes) in arrival order -> per row (pred byte or None, ok, worker s, seen s,
    its worker is killed and a fresh fork takes its core."""
    res, refork = [None] * len(rows), 0
    nxt, free, infl = 0, list(range(len(workers)))[::-1], {}
    clock = time.perf_counter
    while nxt < len(rows) or infl:
        while free and nxt < len(rows):
            w = free.pop()
            workers[w][1].send(struct.pack("<q", nxt) + rows[nxt])
            infl[w] = (nxt, clock())
            nxt += 1
        tmo = max(0.0, min(t0 for _, t0 in infl.values()) + cap - clock())
        socks = {workers[w][1]: w for w in infl}
        rd, _, _ = select.select(list(socks), [], [], tmo)
        now = clock()
        for s in rd:
            w = socks[s]
            msg = s.recv(64)
            if not msg:
                raise RuntimeError(f"worker {workers[w][0]} died")
            q2, pb, praw, ok, sec = struct.unpack("<qBBBd", msg)
            q, t0 = infl.pop(w)
            assert q2 == q, (q2, q)
            res[q] = (pb, bool(ok), sec, now - t0, False, praw)
            free.append(w)
        for w, (q, t0) in list(infl.items()):
            if now - t0 >= cap:
                kill([workers[w]])
                workers[w] = zyg.fork(key, workers[w][2])
                refork += 1
                del infl[w]
                res[q] = (None, False, None, now - t0, True, None)
                free.append(w)
    return res, refork


# ------------------------------------------------------------------------------------------------- the network

class Net:
    """The engine (precise) on one circuit: graphs per batch size, pinned host buffers, timed host-to-host calls."""

    def __init__(self, model, d, meta, z, pos, path="engine"):
        import torch
        from duet.net import engine_cl, spec_fast
        self.torch = torch
        c_ = meta["circuits"][pos]["c"]
        lay, prep, meas = c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"])
        prods = [list(s) for s in c_["products"]]
        t0 = time.perf_counter()
        s = spec_fast.build(d, lay, prep, meas, meta["p"], meta["noise"], prods)
        self.eng = engine_cl.EngineCL(model, "precise")
        self.circ = self.eng.circuit(lay, prep, meas, meta["noise"], meta["p"], spec=s)
        torch.cuda.synchronize()
        self.setup_s = time.perf_counter() - t0
        self.nb = int(self.circ.get("nb", (self.circ["m"] + 7) // 8))
        assert self.nb == meta["circuits"][pos]["nb"], (self.nb, meta["circuits"][pos]["nb"])
        self.g = {}

    def prep(self, B):
        torch = self.torch
        Bg = min(B, 16384)
        if Bg not in self.g:
            g, xin, out, dec = self.eng.graph(Bg)
            self.g[Bg] = (g, xin[:Bg * self.nb].view(Bg, self.nb), out, dec)
        self.hin = torch.zeros(B, self.nb, dtype=torch.uint8).pin_memory()
        self.hdec = torch.zeros(B, dtype=torch.uint8).pin_memory()
        self.hlog = torch.zeros(B, dtype=torch.float32).pin_memory()
        self.B, self.Bg = B, Bg

    def fill(self, rows):
        self.hin.numpy()[:] = rows

    def call(self):
        g, x2, out, dec = self.g[self.Bg]
        Bg = self.Bg
        for q in range(self.B // Bg):
            x2.copy_(self.hin[q * Bg:(q + 1) * Bg], non_blocking=True)
            g.replay()
            self.hdec[q * Bg:(q + 1) * Bg].copy_(dec, non_blocking=True)
            self.hlog[q * Bg:(q + 1) * Bg].copy_(out[:, 0], non_blocking=True)
        self.torch.cuda.synchronize()

    def free(self):
        del self.eng, self.g
        self.torch.cuda.empty_cache()


def cores_spec(s):
    if s == "spark":
        return SPARK_CORES
    allowed = sorted(os.sched_getaffinity(0))
    if s == "auto":
        return {1: allowed[-1:], len(allowed) // 2: allowed[len(allowed) // 2:], len(allowed): allowed}
    return {int(k): [int(c) for c in v.split("+")] for k, v in (x.split(":") for x in s.split(","))}


def env_record(a, cores):
    import platform
    import torch
    import triton
    from importlib.metadata import version
    pr = torch.cuda.get_device_properties(0)
    cpu = {}
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("CPU part") or ln.startswith("model name"):
                cpu[ln.split(":")[1].strip()] = cpu.get(ln.split(":")[1].strip(), 0) + 1
    except OSError:
        pass
    return dict(kind="env", host=platform.node(), gpu=torch.cuda.get_device_name(0), sm=pr.multi_processor_count,
                torch=torch.__version__, triton=triton.__version__, stim=version("stim"),
                tesseract=version("tesseract-decoder"), cpu=cpu, cores={str(k): v for k, v in cores.items()},
                cap_s=CAP, f={d: froute(d) for d in (3, 5, 7)} if HEAD else F_ROUTE, setting=SETTING, opts=OPTS,
                head=HEAD, argv=sys.argv, start=time.strftime("%Y-%m-%d %H:%M:%S %Z"))


def check_batch(z, pos, idx, lg, dnet, routed, res, thr, mask):
    """Counts of differences from the reference reads on the batch's shots idx (positions in the circuit)."""
    rref = z[f"rref{pos}"][idx]
    rr_thr = np.abs(z[f"lg{pos}"][idx]) <= thr
    pair = dnet.copy()
    tdiff = tcap_al = tcap = low = rawdiff = 0
    for j, r in zip(routed, res):
        pb, ok, sec, seen, capped, praw = r
        tcap += capped
        if pb is not None and ok:
            pair[j] = pb
            tdiff += int(((pb ^ int(z[f"tdec{pos}"][idx[j]])) & mask) != 0)
            rawdiff += int(((praw ^ int(z[f"tdec{pos}"][idx[j]])) & mask) != 0)
        low += (pb is not None and not ok)
    al = z[f"dnet{pos}"][idx].copy()
    use = rref & z[f"tok{pos}"][idx] & (z[f"tsec{pos}"][idx] <= CAP)
    al[use] = z[f"tdec{pos}"][idx][use]
    tcap_al = int((rref & (z[f"tsec{pos}"][idx] > CAP)).sum())
    obs = z[f"obs{pos}"][idx]
    rt = np.zeros(len(idx), bool)
    rt[routed] = True
    return dict(route_vs_reference=int((rt != rref).sum()), route_vs_thr=int((rt != rr_thr).sum()),
                net_diff=int((((dnet ^ z[f"dnet{pos}"][idx]) & mask) != 0).sum()),
                max_dlogit=float(np.abs(lg - z[f"lg{pos}"][idx]).max()), tess_diff=tdiff, tess_raw_diff=rawdiff, capped=tcap,
                capped_alpine=tcap_al, low_conf=low, pair_diff=int((((pair ^ al) & mask) != 0).sum()),
                fails_pair=int(((pair ^ obs) & 1).sum()), fails_alpine=int(((al ^ obs) & 1).sum()),
                fails_net=int(((dnet ^ obs) & 1).sum()))


def run(a):
    zyg = Zygote()                                             # before CUDA: the zygote forks CPU-only workers
    import torch
    bank, _ = _lk()
    bn = bank.bench_net()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    cores = cores_spec(a.cores)
    Ns = [int(x) for x in a.ns.split(",")] if a.ns else sorted(cores, reverse=True)
    fo = open(a.out, "a")

    def emit(rec):
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, (list, dict))}), flush=True)

    emit(env_record(a, cores))
    t_end = time.time() + 60 * a.minutes
    ds = [int(x) for x in a.ds.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    nets = {d: bn.load_net("lukin", d, "cuda", path=os.path.join(a.nets, f"d{d}.pt")) for d in ds}
    for d, (_, info) in nets.items():
        emit(dict(kind="net", d=d, path=info["path"], sha256=info["sha256"], step=info["step"]))
    for ps in a.passes.split(","):
        for d in ds:
            for D in depths:
                path = os.path.join(a.pack, f"lukin_D{D}_d{d}.npz")
                meta, z = load_pack(path)
                for pos in [int(x) for x in (a.xpos if ps == "X" else a.pos).split(",")]:
                    if time.time() > t_end:
                        emit(dict(kind="stop", reason="budget", passes=ps, d=d, D=D, pos=pos))
                        fo.close()
                        zyg.close()
                        return
                    block(a, zyg, nets[d][0], d, D, meta, z, pos, path, ps, cores, Ns, emit)
    emit(dict(kind="end", time=time.strftime("%Y-%m-%d %H:%M:%S %Z")))
    fo.close()
    zyg.close()


def block(a, zyg, model, d, D, meta, z, pos, path, ps, cores, Ns, emit):
    key = (meta["name"], pos)
    bld = zyg.build(key, path, pos)
    net = Net(model, d, meta, z, pos)
    c = meta["circuits"][pos]
    n, thr, mask = c["n"], meta["thr"], (1 << c["k"]) - 1
    dets = z[f"dets{pos}"]
    base = dict(d=d, D=D, pos=pos, L=2 * D, thr=thr, n=n, build_s=round(bld["build_s"], 4), tess_info={k: bld[k] for k in bld if k not in ("build_s", "settings")},
                net_setup_s=round(net.setup_s, 4), passes=ps)
    if ps == "L":
        latency(a, zyg, key, net, z, pos, dets, thr, mask, cores, base, emit)
    else:
        for B in PASSES[ps]:
            net.prep(B)
            R = REPS[B] if not a.reps else a.reps
            for _ in range(3 if B <= 4096 else 1):             # warm-up: the network only
                net.fill(dets[np.arange(B) % n])
                net.call()
            for N in Ns:
                workers = [zyg.fork(key, cr) for cr in cores[N]]
                walls, nets_, routed_n, refork, chk = [], [], [], 0, {}
                for r in range(R):
                    idx = (r * B + np.arange(B)) % n
                    net.fill(dets[idx])
                    rows_np = net.hin.numpy()
                    t0 = time.perf_counter()
                    net.call()
                    t1 = time.perf_counter()
                    lg = net.hlog.numpy()
                    rt = np.flatnonzero(np.abs(lg) <= thr)
                    rows = [rows_np[j].tobytes() for j in rt]
                    res, rf = dispatch(zyg, key, workers, rows)
                    t2 = time.perf_counter()
                    dnet = net.hdec.numpy().copy()
                    walls.append(t2 - t0)
                    nets_.append(t1 - t0)
                    routed_n.append(len(rt))
                    refork += rf
                    ck = check_batch(z, pos, idx, lg.copy(), dnet, rt, res, thr, mask)
                    ck["tess_s"] = float(sum(x[2] for x in res if x[2] is not None))
                    ck["seen_max_s"] = float(max([x[3] for x in res], default=0.0))
                    for kk, v in ck.items():
                        chk[kk] = max(chk.get(kk, 0), v) if kk in ("max_dlogit", "seen_max_s") else chk.get(kk, 0) + v
                kill(workers)
                emit(dict(kind="batch", **base, B=B, N=N, cores=cores[N], reps=R, us_per_shot=1e6 * float(np.mean(walls)) / B,
                          net_us_per_shot=1e6 * float(np.mean(nets_)) / B, wall_s=[round(w, 5) for w in walls],
                          routed=routed_n, refork=refork, **chk))
    zyg.drop(key)
    net.free()


def latency(a, zyg, key, net, z, pos, dets, thr, mask, cores, base, emit):
    """Batch 1: each shot alone through the network and, when routed, Tesseract on one core (the fastest)."""
    net.prep(1)
    w = [zyg.fork(key, cores[1][0])]
    rref = np.flatnonzero(z[f"rref{pos}"])
    idx = np.concatenate([np.arange(min(a.lat_shots, len(dets))), rref[:a.lat_routed]])
    for _ in range(20):
        net.fill(dets[:1])
        net.call()
    tn, tr, kinds, chk = [], [], [], dict(net_diff=0, tess_diff=0, capped=0, route_vs_reference=0)
    for j in idx:
        net.fill(dets[j:j + 1])
        t0 = time.perf_counter()
        net.call()
        lg = float(net.hlog.numpy()[0])
        routed = abs(lg) <= thr
        res = None
        if routed:
            res, _ = dispatch(zyg, key, w, [net.hin.numpy()[0].tobytes()])
        t1 = time.perf_counter()
        (tr if routed else tn).append(t1 - t0)
        dn = int(net.hdec.numpy()[0])
        chk["net_diff"] += int(((dn ^ int(z[f"dnet{pos}"][j])) & mask) != 0)
        chk["route_vs_reference"] += int(routed != bool(z[f"rref{pos}"][j]))
        if res:
            pb, ok, sec, seen, capped, praw = res[0]
            chk["capped"] += capped
            if pb is not None and ok:
                chk["tess_diff"] += int(((pb ^ int(z[f"tdec{pos}"][j])) & mask) != 0)
    kill(w)
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None             # noqa: E731
    emit(dict(kind="latency", **base, shots=len(tn), routed_shots=len(tr),
              net_us=dict(mean=1e6 * float(np.mean(tn)), p50=1e6 * q(tn, 50), p99=1e6 * q(tn, 99), max=1e6 * max(tn)),
              routed_ms=dict(mean=1e3 * float(np.mean(tr)) if tr else None, p50=1e3 * q(tr, 50) if tr else None,
                             max=1e3 * max(tr) if tr else None), routed_s=[round(x, 5) for x in tr], **chk))


# ----------------------------------------------------------------------------------------------------- check

def check(a):
    """A few shots of two points through the engine and Tesseract in this process; decisions against the pack."""
    import torch
    bank, _ = _lk()
    bn = bank.bench_net()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    t_start = time.time()
    for name in a.points.split(","):
        meta, z = load_pack(os.path.join(a.pack, name + ".npz"))
        d, pos, c = meta["d"], 0, meta["circuits"][0]
        model, info = bn.load_net("lukin", d, "cuda", path=os.path.join(a.nets, f"d{d}.pt"))
        net = Net(model, d, meta, z, pos)
        B = a.shots
        net.prep(B)
        net.fill(z[f"dets{pos}"][:B])
        t0 = time.time()
        net.call()
        t_net = time.time() - t0
        mask = (1 << c["k"]) - 1
        dn, lg = net.hdec.numpy(), net.hlog.numpy()
        nd = int((((dn ^ z[f"dnet{pos}"][:B]) & mask) != 0).sum())
        t0 = time.time()
        dec, tinfo = make_tess(os.path.join(a.pack, name + ".npz"), meta, z, pos, OPTS)
        t_build = time.time() - t0
        rr = np.flatnonzero(z[f"rref{pos}"])[:a.routed]
        td, tdr, tt, ts = 0, 0, 0, []
        for j in rr:
            dets = np.unpackbits(z[f"dets{pos}"][j], count=c["m"], bitorder="little").astype(bool)
            pb, praw, ok, sec = predict(dec, dets)
            ts.append(sec)
            td += int(((pb ^ int(z[f"tdec{pos}"][j])) & mask) != 0)
            tdr += int(((praw ^ int(z[f"tdec{pos}"][j])) & mask) != 0)
            tt += int(((pb ^ int(z[f"tdec{pos}"][j])) & 1) != 0)
        print(json.dumps(dict(point=name, gpu=torch.cuda.get_device_name(0), shots=B, net_diff=nd,
                              max_dlogit=float(np.abs(lg - z[f"lg{pos}"][:B]).max()), first_call_s=round(t_net, 2),
                              net_setup_s=round(net.setup_s, 2), routed_checked=len(rr), tess_diff=td, tess_raw_diff=tdr, tess_target_diff=tt,
                              tess_s_mean=round(float(np.mean(ts)), 5), alpine_s_mean=round(float(z[f"tsec{pos}"][rr].mean()), 5),
                              tess_build_s=round(t_build, 2), **tinfo, elapsed_s=round(time.time() - t_start, 1))), flush=True)
        net.free()


# ------------------------------------------------------------------------------------------------------- sum

def summarize(a):
    recs = [json.loads(ln) for f in a.runs.split(",") for ln in open(f)]
    env = [r for r in recs if r["kind"] == "env"]
    print("### Environment\n")
    for e in env:
        print(f"- {e['host']}: {e['gpu']} ({e['sm']} SMs), torch {e['torch']}, triton {e['triton']}, stim {e['stim']}, "
              f"tesseract {e['tesseract']}, CPU {e['cpu']}, start {e['start']}")
    bt = [r for r in recs if r["kind"] == "batch"]
    print("\n### Batch mode: µs per shot (wall / B; each depth weighted equally, circuits averaged); pair failures\n")
    print("| d | B | N | pair µs per shot | network µs per shot | routed per batch | capped (reference) | re-forks | "
          "decision differences: net / routing / Tesseract / pair | fails pair / reference pair / net |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted({r["d"] for r in bt}):
        for B in sorted({r["B"] for r in bt if r["d"] == d}):
            for N in sorted({r["N"] for r in bt if r["d"] == d and r["B"] == B}):
                rs = [r for r in bt if r["d"] == d and r["B"] == B and r["N"] == N]
                Ds = sorted({r["D"] for r in rs})
                pm = np.mean([np.mean([r["us_per_shot"] for r in rs if r["D"] == D]) for D in Ds])
                nm = np.mean([np.mean([r["net_us_per_shot"] for r in rs if r["D"] == D]) for D in Ds])
                s = lambda k: sum(r[k] for r in rs)                       # noqa: E731
                print(f"| {d} | {B:,} | {N} | {pm:.1f} | {nm:.1f} | {np.mean([np.mean(r['routed']) for r in rs]):.1f} | "
                      f"{s('capped')} ({s('capped_alpine')}) | {s('refork')} | {s('net_diff')} / {s('route_vs_reference')} / "
                      f"{s('tess_diff')} / {s('pair_diff')} | {s('fails_pair')} / {s('fails_alpine')} / {s('fails_net')} |"
                      + ("" if len(Ds) == 10 else f" depths {Ds}"))
    b256 = [r for r in bt if r["B"] == 256]
    if b256:
        print("\n### Per layer at B = 256 (the Lukin group's setting; L = 2D layers per circuit): µs per shot per layer\n")
        print("| d | N | " + " | ".join(f"D = {D}" for D in range(1, 11)) + " | mean |")
        print("|---|---|" + "---|" * 11)
        for d in sorted({r["d"] for r in b256}):
            for N in [None] + sorted({r["N"] for r in b256}, reverse=True):
                v = []
                for D in range(1, 11):
                    rs = [r for r in b256 if r["d"] == d and r["D"] == D and (N is None or r["N"] == N)]
                    v.append(np.mean([(r["net_us_per_shot"] if N is None else r["us_per_shot"]) / r["L"] for r in rs])
                             if rs else np.nan)
                print(f"| {d} | {'network' if N is None else f'pair, N = {N}'} | " + " | ".join(f"{x:.1f}" for x in v)
                      + f" | {np.nanmean(v):.1f} |")
    lt = [r for r in recs if r["kind"] == "latency"]
    if lt:
        print("\n### Batch 1 (one shot at a time; routed shots on one core)\n")
        for d in sorted({r["d"] for r in lt}):
            rs = [r for r in lt if r["d"] == d]
            allr = np.concatenate([r["routed_s"] for r in rs]) if rs else np.array([])
            print(f"- d = {d}: network-only shots µs mean / p99 / max (depths averaged): "
                  f"{np.mean([r['net_us']['mean'] for r in rs]):.0f} / {np.mean([r['net_us']['p99'] for r in rs]):.0f} / "
                  f"{max(r['net_us']['max'] for r in rs):.0f}; routed shots ms mean / p99 / max: "
                  f"{1e3 * allr.mean():.1f} / {1e3 * np.percentile(allr, 99):.1f} / {1e3 * allr.max():.1f} "
                  f"({len(allr)} shots); stream mean at f = {froute(d):.1%}: "
                  f"{np.mean([r['net_us']['mean'] for r in rs]) / 1e3 * (1 - froute(d)) + 1e3 * allr.mean() * froute(d):.3f} ms;"
                  f" differences net / routing / Tesseract: {sum(r['net_diff'] for r in rs)} / "
                  f"{sum(r['route_vs_reference'] for r in rs)} / {sum(r['tess_diff'] for r in rs)}, capped "
                  f"{sum(r['capped'] for r in rs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode_", choices=["pack", "dems", "check", "run", "sum"])
    ap.add_argument("--bank")
    ap.add_argument("--out")
    ap.add_argument("--pack")
    ap.add_argument("--nets")
    ap.add_argument("--runs")
    ap.add_argument("--ds", default="7,5")
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--pos", default="0,1")
    ap.add_argument("--passes", default="A,L,C,E,X")
    ap.add_argument("--xpos", default="0", help="the circuits of pass X (B = 65,536)")
    ap.add_argument("--cores", default="spark")
    ap.add_argument("--ns", default="")
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=100)
    ap.add_argument("--lat-shots", type=int, default=1000)
    ap.add_argument("--lat-routed", type=int, default=32)
    ap.add_argument("--points", default="lukin_D1_d5,lukin_D10_d7")
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--routed", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=NMAX, help="pack: the first shots of each circuit")
    a = ap.parse_args()
    {"pack": pack, "dems": dems, "check": check, "run": run, "sum": summarize}[a.mode_](a)


if __name__ == "__main__":
    main()
