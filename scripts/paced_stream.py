"""The live paced stream on development shots (Benchmarks I and II): batches of 256 shots through the network on the
GPU, the routed shots through Tesseract on N cores, each trial stopped at its cap in that CPU's time.
Nothing here reads a test set.

  live --manifest DEV --dir SETS --net NET --name N --thr THR --f F --cap c|none --N N --out OUT [--fac R]
       [--only RE] [--circuits C] [--batches R] [--ckpt CK [--grow 7] | --fake-net US] [--cores LIST] [--fail target]
    Per point one stream of C circuits x R batches of B = 256 shots by route.py batch_plan (circuit-major: circuit 0's
    first R batches, then circuit 1's...; the plan curve simulates). The network on batch k + 1 starts when the GPU is
    free and batch k's first trial has started (paced: at most one batch waits whole in the queue). The network is the
    engine, precise, host to host (scripts/end_to_end.py's Net: a CUDA graph per circuit, pinned buffers; here with every
    product's logit), or with --fake-net US a sleep of US microseconds per shot with the stored fp32 development read
    (NET/N) in its place. A shot is routed when its score (the smallest |logit| over the products) is at most route.py
    thr's t_f of its depth. Its 21 trials (trial j: order j, beam j) join one first-in-first-out queue, largest beam
    stops c x L x R seconds after its start. A trial's cost and observables are read before its end
    time; its flag and its deadline status are recorded apart, and a trial flagged or past its deadline gives no answer.
     rule: the lowest cost among the answers, the lower trial on a tie; with none the network's answer. A batch's
    answers are assembled when its last trial ends, and its completion time is taken after that; the clock runs from the
    first network call to the last batch's answers, so dispatch, the queue's drain, the answer assembly and every
    synchronisation are inside it.
    OUT/<point>.npz: per batch the network's start and end, its first trial's start and its completion; per trial its
    composition's answer; per routed shot its flat index (circuit x shots + shot) and its trials' answer. OUT.jsonl: per
    point the time per layer (wall / (batches x B) / L), the latency median and p99, the busy share, and the failures
    (any product, --fail target: slot 0) of the network and of the stream.
  cmp --manifest DEV --dir SETS --net NET --name N --thr THR --live OUT --tess TESS --cap c|none --out J
       [--basis grace --cal DIR] [--fac-spec...] [--live-fac F]
    (CPU) the two complete compositions on every sampled shot of a live run: answer
    changes by cause and their effect on failures; the live time beside the paced replay
    of its own trial and GPU durations, and (an estimate) the replay converted to the basis beside route.py's simulation
    of the same batches on the basis."""
import argparse
import collections
import importlib.util
import itertools
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operating_points as route  # noqa: E402
import eval_pipeline as ep  # noqa: E402

from duet import bench_sets as bs  # noqa: E402

K = ep.K_TRIALS
ROOT = ep.ROOT


class Pool:
    """N worker threads, worker w pinned to cores[w]; each owns its decoders (one per circuit of the current point,
    compiled by the worker itself; none shared). run(fn) has every worker call fn(w) once."""

    def __init__(self, cores):
        self.cores, self.N = list(cores), len(cores)
        self.cv, self.gen, self.fn, self.left = threading.Condition(), 0, None, 0
        self.done = threading.Event()
        self.decs = [None] * self.N
        self.th = [threading.Thread(target=self._loop, args=(w,), daemon=True) for w in range(self.N)]
        for t in self.th:
            t.start()

    def _loop(self, w):
        os.sched_setaffinity(0, {self.cores[w]})
        seen = 0
        while True:
            with self.cv:
                while self.gen == seen:
                    self.cv.wait()
                seen, fn = self.gen, self.fn
            if fn is None:
                return
            try:
                fn(w)
            finally:
                with self.cv:
                    self.left -= 1
                    if self.left == 0:
                        self.done.set()

    def run(self, fn):
        self.done.clear()
        with self.cv:
            self.fn, self.left = fn, self.N
            self.gen += 1
            self.cv.notify_all()
        self.done.wait()

    def build(self, cfgs):
        def fn(w):
            self.decs[w] = [c.compile_decoder() for c in cfgs]
        self.run(fn)

    def close(self):
        with self.cv:
            self.fn = None
            self.gen += 1
            self.cv.notify_all()
        for t in self.th:
            t.join()


def phys(cpus):
    """One logical CPU per physical core among cpus (the lowest of its hyperthread siblings in the set)."""
    out, seen = [], set()
    for c in sorted(cpus):
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        if sib not in seen:
            seen.add(sib)
            out.append(c)
    return out


def _e2e():
    spec = importlib.util.spec_from_file_location("lukin_e2e", os.path.join(ROOT, "scripts", "end_to_end.py"))
    mod = sys.modules["lukin_e2e"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(ckpt, grow):
    """(network on cuda, record): the checkpoint's EMA weights, or grown to `grow` trunk blocks with zero-output new
    blocks."""
    import hashlib
    import torch
    from duet.net import model_cl
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"], blocks=grow) if grow else dict(ck["cfg"])
    net = model_cl.NetCL(**cfg)
    if grow:
        model_cl.load_grown(net, ck)
    else:
        net.load_state_dict(ck["model"])
    rec = dict(path=ckpt, sha256=hashlib.sha256(open(ckpt, "rb").read()).hexdigest(), cfg=net.cfg, grown=bool(grow))
    return net.to("cuda").eval(), rec


def net_all(e2e):
    class NetAll(e2e.Net):
        """e2e.Net with every product's logit copied back (the routing score needs all of them)."""

        def __init__(self, model, d, pt, c_, nb):
            meta = dict(p=pt["p"], noise=pt["noise"], circuits={0: dict(c=c_, nb=nb)})
            super().__init__(model, d, meta, None, 0)
            self.k = max(1, len(c_["products"]))

        def prep(self, B):
            super().prep(B)
            self.hlog = self.torch.zeros(B, self.k, dtype=self.torch.float32).pin_memory()

        def call(self):
            g, x2, out, dec = self.g[self.Bg]
            Bg = self.Bg
            for q in range(self.B // Bg):
                x2.copy_(self.hin[q * Bg:(q + 1) * Bg], non_blocking=True)
                g.replay()
                self.hdec[q * Bg:(q + 1) * Bg].copy_(dec, non_blocking=True)
                self.hlog[q * Bg:(q + 1) * Bg].copy_(out[:, :self.k], non_blocking=True)
            self.torch.cuda.synchronize()
    return NetAll


def truth_bytes(path, i, n, k):
    O = bs.packed(path, "obs")
    return np.packbits(bs.unpack(O[i][:n], k), axis=1, bitorder="little")[:, 0] if k else np.zeros(n, np.uint8)


def cmd_live(a):
    fake = a.fake_net is not None
    m = ep.load_manifest(a.manifest)
    thr = json.load(open(a.thr))
    tmod, tt, _ = ep.tess_mod()
    if not fake:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model, mrec = load_model(a.ckpt, a.grow)
        NetAll = net_all(_e2e())
    cores = [int(x) for x in a.cores.split(",")] if a.cores else phys(os.sched_getaffinity(0))
    assert len(cores) >= a.N, (len(cores), a.N)
    N, B = a.N, a.B
    pool = Pool(cores[:N])
    os.makedirs(a.out, exist_ok=True)
    fo = open(a.out + ".jsonl", "a")

    def emit(rec):
        fo.write(json.dumps(rec, default=float) + "\n")
        fo.flush()
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, (list, dict))}, default=float), flush=True)

    emit(dict(kind="env", cpu=ep.cpu(), cores=cores[:N], N=N, B=B, f=a.f, cap=a.cap, fac=a.fac, fake_net=a.fake_net,
              module=tmod.__file__, net=None if fake else mrec, gpu=None if fake else torch.cuda.get_device_name(0),
              code=os.environ.get("CODE"), host=os.uname().nodename, fail=a.fail))
    clock = time.perf_counter
    for pt in ep.select(m, a.only):
        out = os.path.join(a.out, pt["name"] + ".npz")
        fnet = os.path.join(a.net, a.name, pt["name"] + ".npz")
        if os.path.exists(out) or not os.path.exists(fnet):
            continue
        t0 = time.time()
        path = ep.set_path(a.dir, pt)
        z = np.load(path)
        entry, m_, k_ = json.loads(str(z["entry"])), z["m"].astype(int), z["k"].astype(int)
        lg_all = route.load_logits(a.net, a.name, pt)
        n = lg_all.shape[1]
        C = min(a.circuits or len(pt["circuits"]), len(pt["circuits"]))
        R = min(a.batches or n // B, n // B)
        L, d = pt["L"], pt["d"]
        t_f = thr["t"][route.group_key(pt, thr["by"])][route.fkey(a.f)]
        cap = None if a.cap in (None, "none") else float(a.cap) * L * a.fac
        Dp = bs.packed(path, "dets")
        cfgs = [ep.tess_config(bs.circuit(entry, entry["circuits"][i])) for i in range(C)]
        w0 = cfgs[0].compile_decoder()                 # the main thread first (pybind11's statics), as sstream
        w0.decode_to_errors(np.zeros(w0.num_detectors, bool), 0, 0) if cap is None else \
            w0.decode_to_errors(np.zeros(w0.num_detectors, bool), 0, 0, 1.0)
        assert cap is None or hasattr(w0, "deadline_exceeded"), "a cap needs stoppable build"
        del w0
        pool.build(cfgs)
        nets = []
        if not fake:
            for i in range(C):
                c_ = pt["circuits"][i]
                net = NetAll(model, d, pt, c_, (int(m_[i]) + 7) // 8)
                net.prep(B)
                for _ in range(3):                     # warm-up: the network only
                    net.fill(np.ascontiguousarray(Dp[i, :B, :net.nb]))
                    net.call()
                nets.append(net)
        kk = [len(pt["circuits"][i]["products"]) for i in range(C)]
        truth = [truth_bytes(path, i, R * B, int(k_[i])) for i in range(C)]
        setup = time.time() - t0
        plan, order = route.batch_plan(C, R, n, B)    # circuit-major, as route.py curve
        nb = len(plan)
        bat = dict(net0=np.zeros(nb), net1=np.zeros(nb), first=np.full(nb, np.nan), last=np.zeros(nb),
                   nr=np.zeros(nb, np.int64))
        fin = np.zeros((nb, B), np.uint8)              # every sampled shot's answer (the composition's)
        fresh_all = np.zeros((nb, B), bool)               # the live routing
        dnet_all = np.zeros((nb, B), np.uint8)         # the live network's decisions
        left = np.zeros(nb, np.int64)
        started = [threading.Event() for _ in range(nb)]
        recs = [None] * nb
        q, cv, st8 = collections.deque(), threading.Condition(), dict(end=False)
        T0 = clock()

        def finish(b):
            """Batch b's answers, then its completion time: the clock stops at the
            batch's last answer."""
            x = recs[b]
            n_ = int(bat["nr"][b])
            f_ = x["dnet"].copy()
            if n_:
                ok = ~x["flag"] & ~x["drop"]
                c_ = np.where(ok, x["cost"], np.inf).reshape(n_, K)[:, ::-1]    # trial order (the queue ran 20 first)
                j = c_.argmin(1)
                h = np.isfinite(c_[np.arange(n_), j])
                an = x["pred"].reshape(n_, K)[:, ::-1][np.arange(n_), j]
                f_[x["routed"][h]] = an[h]
                x["has"], x["ans"] = h, np.where(h, an, 0).astype(np.uint8)
            fin[b] = f_
            bat["last"][b] = clock() - T0

        def worker(w):
            decs = pool.decs[w]
            while True:
                with cv:
                    while not q and not st8["end"]:
                        cv.wait()
                    if not q:
                        return
                    b, i = q.popleft()
                x = recs[b]
                k, j = i // K, K - 1 - i % K                # each shot's trials largest beam first
                dec = decs[plan[b][0]]
                t1 = clock()
                if i == 0:                                 # the batch's first trial: the next network may start
                    bat["first"][b] = t1 - T0
                    started[b].set()
                e = dec.decode_to_errors(x["rows"][k], j, j) if cap is None else \
                    dec.decode_to_errors(x["rows"][k], j, j, cap)
                fl = bool(dec.low_confidence_flag)
                dr = bool(getattr(dec, "deadline_exceeded", False))      # recorded apart from the flag
                if not (fl or dr):
                    x["cost"][i] = dec.cost_from_errors(e)
                    x["pred"][i] = ep.pk(dec.get_observables_from_errors(e))
                t2 = clock()                               # after the cost and the observables
                x["st"][i], x["en"][i], x["wk"][i], x["flag"][i], x["drop"][i] = t1 - T0, t2 - T0, w, fl, dr
                with cv:
                    left[b] -= 1
                    last = left[b] == 0
                if last:
                    finish(b)

        th = threading.Thread(target=pool.run, args=(worker,))
        th.start()
        for b, (i, r) in enumerate(plan):
            if b and bat["nr"][b - 1]:
                started[b - 1].wait()                      # paced: the previous batch's first trial has started
            idx = np.arange(r * B, (r + 1) * B)
            nbytes = (int(m_[i]) + 7) // 8
            dets = np.ascontiguousarray(Dp[i, r * B:(r + 1) * B, :nbytes])
            t1 = clock()
            if fake:
                time.sleep(a.fake_net * 1e-6 * B)
                lg = lg_all[i, idx, :max(1, kk[i])]
                dnet = np.packbits(lg[:, :kk[i]] > 0, axis=1, bitorder="little")[:, 0] if kk[i] else \
                    np.zeros(B, np.uint8)
            else:
                net = nets[i]
                net.fill(dets)
                net.call()
                lg, dnet = net.hlog.numpy().copy(), net.hdec.numpy().copy()
            rt = np.flatnonzero(np.min(np.abs(lg[:,:max(1, kk[i])]), axis=1) <= t_f)
            t2 = clock()
            n_ = len(rt)
            recs[b] = dict(rows=np.unpackbits(dets[rt], axis=1, count=int(m_[i]), bitorder="little").astype(bool),
                           routed=rt, flat=i * n + idx[rt], dnet=dnet.astype(np.uint8), lg=lg, st=np.zeros(n_ * K),
                           en=np.zeros(n_ * K), wk=np.zeros(n_ * K, np.int16), flag=np.zeros(n_ * K, bool),
                           drop=np.zeros(n_ * K, bool), cost=np.full(n_ * K, np.inf), pred=np.zeros(n_ * K, np.uint8),
                           has=np.zeros(n_, bool), ans=np.zeros(n_, np.uint8))
            fresh_all[b, rt], dnet_all[b] = True, dnet
            bat["net0"][b], bat["net1"][b], bat["nr"][b] = t1 - T0, t2 - T0, n_
            if n_:
                with cv:
                    left[b] = n_ * K
                    q.extend((b, x_) for x_ in range(n_ * K))
                    cv.notify_all()
            else:
                finish(b)
                started[b].set()
        with cv:
            st8["end"] = True
            cv.notify_all()
        th.join()
        assert not q and (left == 0).all(), (len(q), int(left.sum()))
        wall = float(bat["last"].max() - bat["net0"][0])        # first network call to the last answer
        tr = np.stack([truth[i][r * B:(r + 1) * B] for i, r in plan])
        dn, fz = dnet_all.copy(), fin.copy()
        if a.fail == "target":
            tr, dn, fz = tr & 1, dn & 1, fz & 1
        fl_net, fl_str = int((dn != tr).sum()), int((fz != tr).sum())
        lgd = []
        for b, (i, r) in enumerate(plan):
            if not fake and kk[i]:
                ref = lg_all[i, r * B:(r + 1) * B, :kk[i]]
                lgd.append(float(np.nanmax(np.abs(recs[b]["lg"][:, :kk[i]] - ref))))
        cat = lambda k_: np.concatenate([x[k_] for x in recs])
        lat = bat["last"] - bat["net0"]
        sec = cat("en") - cat("st")
        np.savez(out + ".tmp.npz", N=N, B=B, cap=np.nan if cap is None else cap, fac=a.fac, f=a.f, t_f=t_f, C=C, R=R,
                 plan=np.array(plan), order=order, **bat, fin=fin, routed=fresh_all, dnet=dnet_all,
                 has=cat("has"), ans=cat("ans"), flat=cat("flat"),
                 **{k_: cat(k_) for k_ in ("st", "en", "wk", "flag", "drop", "cost", "pred")},
                 lg_differ=np.array(lgd), setup=setup, n=n, cpu=ep.cpu(), fail=a.fail)
        os.replace(out + ".tmp.npz", out)
        emit(dict(kind="point", point=pt["name"], d=d, K=pt["K"], L=L, p=pt["p"], batches=nb, circuits=C,
                  per_circuit=R, us_per_layer=1e6 * wall / (nb * B) / L, lat_med_ms=1e3 * float(np.median(lat)),
                  lat_p99_ms=1e3 * float(np.percentile(lat, 99)), shots_per_s=nb * B / wall,
                  routed=int(bat["nr"].sum()), dropped=int(cat("drop").sum()), flagged=int(cat("flag").sum()),
                  no_answer=int((~cat("has")).sum()), busy=float(sec.sum() / (N * wall)) if wall else 0.0,
                  net_us_per_batch=1e6 * float(np.mean(bat["net1"] - bat["net0"])), fails_net=fl_net,
                  fails_stream=fl_str, lg_differ_max=float(max(lgd)) if lgd else 0.0, setup_s=round(setup, 1),
                  minutes=round((time.time() - t0) / 60, 2)))
        pool.decs = [None] * pool.N
        for net in nets:
            net.free()
    emit(dict(kind="end", time=time.strftime("%Y-%m-%d %H:%M:%S %Z")))
    pool.close()


def cmd_cmp(a):
    """Per point of a live run: the two complete compositions on every sampled shot. The
    simulated one: the stored development read's routing (score <= t_f) and decisions, each routed shot's answer by
    route.answers on its stored trial records at cap c x L (the basis's seconds, --basis / --cal as route.py sim);
    the live one: the live network's routing and decisions and the live trials' answers. Answer changes are counted
    with their causes and their
    effect on failures. Time: the live wall clock (this machine); the paced replay of the
    live run's own trial durations (queue order) and recorded per-batch GPU durations (the simulation's error here);
    and, labelled an estimate, the same replay with the trial durations converted to the basis (GPU durations
    unchanged; the live CPU's factor from --live-fac, else the median ratio of the stored to the live seconds of the
    same uncapped trials x the stored CPU's factor) beside route.py's simulation of the same batches on the basis
    (stored trial seconds, the live run's GPU durations)."""
    m = ep.load_manifest(a.manifest)
    thr = json.load(open(a.thr))
    rows = []
    for pt in ep.select(m, a.only):
        f = os.path.join(a.live, pt["name"] + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=False)
        n, L, B, N = int(z["n"]), pt["L"], int(z["B"]), int(z["N"])
        fac = route.basis_fac(argparse.Namespace(basis=a.basis, cal=a.cal, fac=a.fac_spec), pt["d"])
        t_f = thr["t"][route.group_key(pt, thr["by"])][route.fkey(float(z["f"]))]
        tl = float(z["t_f"])
        assert tl == t_f or abs(tl - t_f) <= 1e-12 * max(1.0, abs(t_f)), ("the live run used other thresholds", tl, t_f)
        order = z["order"].astype(np.int64)
        nb = len(z["plan"])
        lg = route.load_logits(a.net, a.name, pt)
        k_ = np.load(ep.set_path(a.dir, pt))["k"].astype(int)
        s_ = route.score_of(lg).reshape(-1)[order]
        fresh_sim = s_ <= t_f
        circ = order // n
        dsim = np.zeros(len(order), np.uint8)
        for c in np.unique(circ):
            mk = circ == c
            if k_[c]:
                dsim[mk] = np.packbits(lg[c, order[mk] % n, :k_[c]] > 0, axis=1, bitorder="little")[:, 0]
        T = route.load_trials(a.tess, pt, n, fac)
        lim = None if a.cap == "none" else float(a.cap) * L
        q = order[fresh_sim]
        have, pos = route.lookup(T, q)
        have[have] = route.valid_rows(T, pos[have], lim)
        # the simulated composition must be complete: a routed shot without its stored trials rejects the comparison
        assert have.all(), (pt["name"], int((~have).sum()), "routed development shots without a usable trial record")
        sa, sh = route.answers(T, pos[have], lim) if have.any() else (np.zeros(0, np.uint8), np.zeros(0, bool))
        fin_sim = dsim.copy()
        ii = np.flatnonzero(fresh_sim)[have]
        fin_sim[ii[sh]] = sa[sh]
        fin_live = z["fin"].reshape(-1).astype(np.uint8)
        fresh_live = z["routed"].reshape(-1)
        dlive = z["dnet"].reshape(-1).astype(np.uint8)
        path = ep.set_path(a.dir, pt)
        tb = {int(c): truth_bytes(path, int(c), n, int(k_[c])) for c in np.unique(circ)}
        tr = np.array([tb[int(c)][int(x % n)] for c, x in zip(circ, order)], np.uint8)
        if str(z["fail"]) == "target":
            fin_sim, fin_live, tr, dsim, dlive = fin_sim & 1, fin_live & 1, tr & 1, dsim & 1, dlive & 1
        ch = fin_sim != fin_live
        both_r, both_k = fresh_sim & fresh_live, ~fresh_sim & ~fresh_live
        # the live machine's trial seconds against the stored ones on the same uncapped trials
        fl = z["flat"].astype(np.int64)
        st, en = z["st"], z["en"]
        dur_live = (en - st).reshape(-1, K)[:, ::-1]              # (routed shots, 21) in trial order
        drop_live = z["drop"].reshape(-1, K)[:, ::-1]
        T0 = route.load_trials(a.tess, pt, n, fac, convert=False)   # as recorded (their CPU), T's rows
        h0, p0 = route.lookup(T0, fl)
        rat, live_fac = np.zeros(0), None
        if h0.any():
            sl = T0["sec"][p0[h0]]
            ok = ~drop_live[h0] & ~T0["drop"][p0[h0]] & np.isinf(T0["dl"][p0[h0]])[:, None] & (sl > 1e-5)
            rat = dur_live[h0][ok] / sl[ok]                       # live seconds / stored seconds
            if len(rat) and fac is not None:
                # one population: per eligible trial its basis seconds (stored x its CPU's factor) / its live seconds
                f_st = np.broadcast_to(route.row_factors(T0, fac, pt["d"])[p0[h0]][:, None], sl.shape)
                live_fac = float(np.median(sl[ok] * f_st[ok] / dur_live[h0][ok]))
        if a.live_fac:
            live_fac = a.live_fac
        nr = z["nr"].astype(np.int64)
        gpu = z["net1"] - z["net0"]
        wall_live = float(z["last"].max() - z["net0"][0])
        _, e_own = route.paced(nr, (en - st).tolist(), gpu, N)
        est = None
        if live_fac is not None:
            _, e_g = route.paced(nr, ((en - st) * live_fac).tolist(), gpu, N)
            est = 1e6 * float(e_g.max()) / (nb * B) / L
        # route.py's simulation of the same batches on the basis: stored trial seconds (capped), the live GPU times
        b_of = np.repeat(np.arange(nb), B)[fresh_sim]
        cnt = np.bincount(b_of, minlength=nb)
        secq = np.zeros((len(q), K))
        hv, pv = route.lookup(T, q)
        hv[hv] = route.valid_rows(T, pv[hv], lim)
        if hv.any():
            v = T["sec"][pv[hv]]
            secq[hv] = (np.minimum(v, lim) if lim is not None else v)[:, ::-1]
        _, e_sim = route.paced(cnt, secq.reshape(-1).tolist(), gpu, N)
        per = lambda w: 1e6 * w / (nb * B) / L
        rows.append(dict(point=pt["name"], shots=int(len(order)), routed_sim=int(fresh_sim.sum()),
                         routed_live=int(fresh_live.sum()), routed_sim_without_records=int((~hv).sum()),
                         answers_changed=int(ch.sum()), changed_by_routing=int((ch & (fresh_sim != fresh_live)).sum()),
                         changed_kept_network=int((ch & both_k).sum()), changed_routed_trials=int((ch & both_r).sum()),
                         network_decisions_differ=int((dsim != dlive).sum()),
                         fails_sim=int((fin_sim != tr).sum()), fails_live=int((fin_live != tr).sum()),
                         changed_to_fail=int((ch & (fin_live != tr) & (fin_sim == tr)).sum()),
                         changed_to_correct=int((ch & (fin_live == tr) & (fin_sim != tr)).sum()),
                         us_per_layer_live=per(wall_live), us_per_layer_replay_live=per(float(e_own.max())),
                         us_per_layer_live_basis_estimate=est, live_cpu=str(z["cpu"]), live_fac=live_fac,
                         live_fac_source="--live-fac" if a.live_fac else "paired trials (median)",
                         paired_trials=int(len(rat)), us_per_layer_sim_basis=per(float(e_sim.max())), N=N,
                         basis=a.basis))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(cap=a.cap, basis=a.basis, cal=a.cal, live=a.live, rows=rows), open(a.out, "w"), indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "cmp"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dir", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--name", default="ft4")
    ap.add_argument("--thr", default="")
    ap.add_argument("--f", type=float, default=0.1)
    ap.add_argument("--cap", default="none")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--fac", type=float, default=1.0, help="live: the reference basis's seconds -> this machine's")
    ap.add_argument("--fac-spec", dest="fac_spec", default="", help="cmp: route.py's --fac entries")
    ap.add_argument("--cal", default="", help="cmp: the calibration's directory")
    ap.add_argument("--basis", default="grace", help="cmp: grace, spark or local (route.py sim)")
    ap.add_argument("--live-fac", type=float, default=0.0, help="cmp: the live CPU's seconds -> the basis's")
    ap.add_argument("--only", default="")
    ap.add_argument("--circuits", type=int, default=4)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--fake-net", type=float, default=None, help="live: microseconds per shot instead of a GPU")
    ap.add_argument("--cores", default="")
    ap.add_argument("--fail", default="any", choices=["any", "target"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--live", default="")
    ap.add_argument("--tess", default="")
    a = ap.parse_args()
    dict(live=cmd_live, cmp=cmd_cmp)[a.mode](a)


if __name__ == "__main__":
    main()
