"""A faster duet.net.spec_cl.build with the same tables.

Where spec_cl.build's time went (py-spy --native, d = 9, in a process that has imported torch, as the decoders' have):
about 85 % in stim's Circuit.append inside clifford_circuits.make_circuit. stim's Python binding converts each target of
an appended instruction by trying it as a string, a Pauli string and a GateTarget before an integer, and every failed
try raises and catches a C++ exception; each costs microseconds of stack unwinding (more with many shared libraries
loaded). Python's profiler does not see this time.

build here runs spec_cl.build's own code (the same function, with other globals): clifford_circuits.make_circuit, also
its own code, appends to a recorder that writes each instruction as a line of stim's text format, and the circuit is
parsed once with stim.Circuit(text). stim fuses adjacent instructions of one gate and arguments in both paths, and the
arguments are written with Python's shortest round-trip repr, so the circuit is the same. With the appends gone, half
of the rest was geometry.solve_gf2 (a numpy pass over all rows per column); solve_gf2 here gives the same solution
table with np.array_equal. spec_cl, clifford_circuits, geometry and their callers are not changed.

    from duet.net import spec_fast
    s = spec_fast.build(d, layers, prep, meas, p, noise) # == spec_cl.build(...), faster"""
import types

import numpy as np
import stim

from duet import clifford_circuits as cc
from duet.net import spec_cl


class _Text:
    """make_circuit's stim.Circuit: append records the instruction as a line of stim's text format."""

    def __init__(self):
        self.lines = []

    def append(self, name, targets=(), arg=None):
        a = "" if arg is None else "(" + ", ".join(repr(float(v)) for v in
                                                    (arg if isinstance(arg, (list, tuple)) else (arg,))) + ")"
        self.lines.append(name + a + " " + " ".join(map(str, targets)))       # ints and rec[-k] texts


class _Stim:
    """make_circuit's stim: Circuit records text, target_rec gives the target's text; anything else is stim's."""
    Circuit = _Text

    @staticmethod
    def target_rec(k):
        return f"rec[{int(k)}]"

    def __getattr__(self, name):
        return getattr(stim, name)


def _rebound(fn, **names):
    """fn's code with some of its module's global names bound to other objects."""
    g = dict(fn.__globals__, **names)
    out = types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)
    out.__kwdefaults__ = fn.__kwdefaults__
    return out


_make_text = _rebound(cc.make_circuit, stim=_Stim())


def make_circuit(*args, **kw):
    """clifford_circuits.make_circuit (same arguments and result), the circuit parsed once from text."""
    out = _make_text(*args, **kw)
    rec, meta = out if isinstance(out, tuple) else (out, None)
    c = stim.Circuit("\n".join(rec.lines))
    return c if meta is None else (c, meta)


def solve_gf2(M, T):
    """geometry.solve_gf2: X (m, k) uint8 with M X = T over GF(2), M (r, m), T (r, k); free variables 0; ValueError if
    inconsistent. The columns of M, as bit sets of rows, enter a basis keyed by their highest row in column order (a
    detector's newest record is usually new, so most columns enter at once); a column that reduces to 0 is free. The
    basis columns are the columns independent of the earlier ones, geometry.solve_gf2's pivots, so the solution with
    the free variables 0 is the same."""
    m = M.shape[1]
    bits = lambda A: [int.from_bytes(v.tobytes(), "little") for v in np.packbits(A.T, axis=1, bitorder="little")]  # noqa
    basis = {}                                               # highest row -> (rows, columns of M combined)
    for c, v in enumerate(bits(M)):
        ids = 1 << c
        while v:
            b = basis.get(v.bit_length() - 1)
            if b is None:
                basis[v.bit_length() - 1] = (v, ids)
                break
            v ^= b[0]
            ids ^= b[1]
    X = np.zeros((m, T.shape[1]), np.uint8)
    bad = []
    for j, v in enumerate(bits(T)):
        ids = 0
        while v:
            b = basis.get(v.bit_length() - 1)
            if b is None:
                bad.append(j)
                break
            v ^= b[0]
            ids ^= b[1]
        X[:, j] = np.unpackbits(np.frombuffer(ids.to_bytes((m + 7) // 8, "little"), np.uint8), bitorder="little")[:m]
    if bad:
        raise ValueError(f"no GF(2) solution for target columns {bad[:8]}")
    return X


_cc = types.SimpleNamespace(**{k: v for k, v in vars(cc).items() if not k.startswith("__")})
_cc.make_circuit = make_circuit
build = _rebound(spec_cl.build, cc=_cc, solve_gf2=solve_gf2)
build.__doc__ = "spec_cl.build (same arguments and tables), its circuit built from text (module docstring)."
