"""A* ranked enumeration with admissible completion heuristics.

Repairs the leaf-starvation defect of naive best-first prefix-tree
enumeration: nodes are prioritized by f = (codelength so far) + h, where
h is a provable LOWER bound on the cheapest possible completion cost, so
f stays flat along optimal extensions and paths run to full-length
candidates instead of stalling on plateaus of cheap internal prefixes.
Ties break LIFO (depth-first), which with flat f carries paths to leaves.

Admissibility (h never exceeds true remaining cost) guarantees leaves are
still emitted in exact nonincreasing-probability order.
"""
import heapq, math
import numpy as np
from . import core

_LG = [float("inf")] + [math.lgamma(k * 0.5) for k in range(1, 4 * 4096 + 8)]  # lgamma(k/2)
LN2 = math.log(2)


def _lg(x2):  # lgamma of x where x2 = 2*x (integer half-steps)
    return _LG[x2]


def _kt_tail_bits(cmax, m):
    """-log2 of an upper bound on any m-step KT completion probability
    when the largest per-context majority count is cmax:
    prod_{j=0}^{m-1} (cmax+j+0.5)/(cmax+j+1)."""
    if m <= 0:
        return 0.0
    a = int(2 * cmax)
    return ((_lg(a + 2 + 2 * m) - _lg(a + 2)) - (_lg(a + 1 + 2 * m) - _lg(a + 1))) / LN2


def make_heuristic(pred, n):
    """Returns h(state, t) -> admissible lower bound (bits) on remaining
    codelength for the given predictor type."""
    if isinstance(pred, core.IIDOracle):
        c = -math.log2(max(pred.p, 1.0 - pred.p))
        return lambda st, t: (n - t) * c
    if isinstance(pred, core.MarkovOracle):
        pmax = max(1.0 - pred.b, pred.b, pred.a, 1.0 - pred.a)
        c = -math.log2(pmax)
        return lambda st, t: (n - t) * c
    if isinstance(pred, core.GEOracle):
        pmax = max(1.0 - pred.pg, pred.pg, 1.0 - pred.pb, pred.pb)
        c = -math.log2(pmax)
        return lambda st, t: (n - t) * c
    if isinstance(pred, core.KTHierarchy):
        def h(st, t):
            states, _ = st
            m = n - t
            best = None
            for s in states:
                cmax = max(s[1]) if s[1] else 0
                v = _kt_tail_bits(cmax, m)
                if best is None or v < best:
                    best = v
            return best or 0.0
        return h
    return lambda st, t: 0.0  # always admissible fallback


def astar_candidates(pred, n, max_leaves, max_pops, hfun=None):
    hfun = hfun or make_heuristic(pred, n)
    st0 = pred.init_state()
    heap = [(hfun(st0, 0), 0, 0, 0, 0.0, st0)]  # (f, tie, t, bits, nlp, state)
    tie = -1
    pops = 0
    leaves = 0
    while heap and pops < max_pops and leaves < max_leaves:
        f, _, t, bits, nlp, st = heapq.heappop(heap)
        pops += 1
        if t == n:
            leaves += 1
            yield bits, nlp
            continue
        p1 = pred.p1(st)
        for bit, p in ((0, 1.0 - p1), (1, p1)):
            if p > 1e-15:
                cnlp = nlp - math.log2(p)
                cst = pred.advance(st, bit)
                heapq.heappush(heap, (cnlp + hfun(cst, t + 1), tie, t + 1,
                                      bits | (bit << t), cnlp, cst))
                tie -= 1


def astar_grand_decode(pred, H, z_true, max_queries, chunk=256, max_pops=None):
    n = z_true.size
    if max_pops is None:
        max_pops = 60 * max_queries + 200000
    s = (H.astype(np.int64) @ z_true.astype(np.int64)) & 1
    z_int = 0
    for tt in range(n):
        z_int |= int(z_true[tt]) << tt
    buf = []
    q = 0
    Hi = H.astype(np.int64)
    for bits, _ in astar_candidates(pred, n, max_queries, max_pops):
        buf.append(bits)
        if len(buf) == chunk:
            ok, used, hit = core._check_chunk(buf, n, Hi, s, z_int)
            q += used
            if hit:
                return ok, q, False
            buf = []
    if buf:
        ok, used, hit = core._check_chunk(buf, n, Hi, s, z_int)
        q += used
        if hit:
            return ok, q, False
    return False, q, True


class KTHierarchyFast:
    """Bit-packed KT-hierarchy (orders 0,1,2), numerically identical
    predictive law to core.KTHierarchy, ~3-5x cheaper state updates.
    State: (packed_counts, ctx, l0, l1, l2); counts are 14 x 16-bit
    fields in one int: model0 cells 0-1, model1 cells 2-5, model2 6-13."""
    M = 16

    def init_state(self):
        return (0, 0, 0.0, 0.0, 0.0)

    @staticmethod
    def _cells(ctx):
        return (0 + 0, 2 + 2 * (ctx & 1), 6 + 2 * (ctx & 3))

    def _probs(self, st):
        packed, ctx, *_ = st
        out = []
        for base in self._cells(ctx):
            n0 = (packed >> (16 * base)) & 0xFFFF
            n1 = (packed >> (16 * (base + 1))) & 0xFFFF
            out.append((n1 + 0.5) / (n0 + n1 + 1.0))
        return out

    def p1(self, st):
        _, _, l0, l1, l2 = st
        mx = max(l0, l1, l2)
        w = (2.0 ** (l0 - mx), 2.0 ** (l1 - mx), 2.0 ** (l2 - mx))
        Z = w[0] + w[1] + w[2]
        p = self._probs(st)
        return (w[0] * p[0] + w[1] * p[1] + w[2] * p[2]) / Z

    def advance(self, st, bit):
        packed, ctx, l0, l1, l2 = st
        p = self._probs(st)
        q = [pi if bit else 1.0 - pi for pi in p]
        for base in self._cells(ctx):
            packed += 1 << (16 * (base + bit))
        return (packed, ((ctx << 1) | bit) & 3,
                l0 + math.log2(max(q[0], 1e-300)),
                l1 + math.log2(max(q[1], 1e-300)),
                l2 + math.log2(max(q[2], 1e-300)))


def _kt_tail_bits2(nmax, ntot, m):
    """-log2 upper bound: prod_j (nmax+j+0.5)/(ntot+j+1), admissible per
    model when all m future steps route through the most favorable
    context cell (tighter than the cmax/cmax version)."""
    if m <= 0:
        return 0.0
    a, b = int(2 * nmax), int(2 * ntot)
    v = ((_lg(b + 2 + 2 * m) - _lg(b + 2)) - (_lg(a + 1 + 2 * m) - _lg(a + 1))) / LN2
    return max(v, 0.0)


def kthfast_heuristic(n):
    spans = [(0, 1), (2, 2), (6, 4)]  # (base cell, #contexts) per model

    def h(st, t):
        m = n - t
        if m <= 0:
            return 0.0
        packed = st[0]
        best = None
        for base, nc in spans:
            for c in range(nc):
                n0 = (packed >> (16 * (base + 2 * c))) & 0xFFFF
                n1 = (packed >> (16 * (base + 2 * c + 1))) & 0xFFFF
                v = _kt_tail_bits2(max(n0, n1), n0 + n1, m)
                if best is None or v < best:
                    best = v
        return best or 0.0
    return h


def register_fast_heuristic():
    pass  # dispatch handled in enumerate_list below


def enumerate_list(pred, n, cap, max_pops, hfun=None):
    """One-time exact ranked enumeration -> (bits matrix (cap,n) uint8,
    nlp array).  Channel-independent for a fixed predictor, so the list
    is reusable across every block and operating point."""
    if hfun is None:
        hfun = (kthfast_heuristic(n) if isinstance(pred, KTHierarchyFast)
                else make_heuristic(pred, n))
    B = np.zeros((cap, n), dtype=np.uint8)
    nlps = np.zeros(cap)
    k = 0
    for bits, nlp in astar_candidates(pred, n, cap, max_pops, hfun=hfun):
        for t in range(n):
            B[k, t] = (bits >> t) & 1
        nlps[k] = nlp
        k += 1
    return B[:k], nlps[:k]


def make_list_decoder(B, H):
    """Precompute syndromes of the candidate list; returns decode(z) ->
    (ok, queries, capped)."""
    S = (B.astype(np.int64) @ H.T.astype(np.int64)) & 1
    Hi = H.astype(np.int64)

    def decode(z):
        s = (Hi @ z.astype(np.int64)) & 1
        match = np.all(S == s, axis=1)
        idx = np.flatnonzero(match)
        if idx.size == 0:
            return False, B.shape[0], True
        first = int(idx[0])
        return bool(np.array_equal(B[first], z)), first + 1, False
    return decode
