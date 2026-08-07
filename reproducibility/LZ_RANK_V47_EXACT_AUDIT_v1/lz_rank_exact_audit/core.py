from __future__ import annotations

import itertools
import math
from functools import lru_cache
from collections import defaultdict
from fractions import Fraction
from typing import Iterable

import numpy as np


def all_machines(s: int) -> list[np.ndarray]:
    """All labeled binary transition tables with canonical initial state 0."""
    return [np.asarray(x, dtype=np.int16).reshape(s, 2)
            for x in itertools.product(range(s), repeat=2 * s)]


def all_sequence_bits(n: int) -> np.ndarray:
    if n < 1 or n > 30:
        raise ValueError("n must be in 1..30")
    values = np.arange(1 << n, dtype=np.uint32)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint32)[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def counts_and_terminal(bits: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened [state,bit] counts and terminal state for every row."""
    if bits.ndim != 2:
        raise ValueError("bits must be a 2-D array")
    nseq, _ = bits.shape
    s = int(table.shape[0])
    rows = np.arange(nseq)
    state = np.zeros(nseq, dtype=np.int16)
    counts = np.zeros((nseq, 2 * s), dtype=np.uint8)
    for t in range(bits.shape[1]):
        b = bits[:, t].astype(np.int16, copy=False)
        idx = 2 * state + b
        counts[rows, idx] += 1
        state = table[state, b]
    return counts, state.astype(np.int16, copy=False)


@lru_cache(maxsize=None)
def _exact_ml_probability_key_tuple(row: tuple[int, ...]) -> Fraction:
    if len(row) % 2:
        raise ValueError("count row length must be even")
    numerator = 1
    denominator = 1
    for i in range(0, len(row), 2):
        a, c = row[i], row[i + 1]
        total = a + c
        if a:
            numerator *= a ** a
        if c:
            numerator *= c ** c
        if total:
            denominator *= total ** total
    return Fraction(numerator, denominator)


def exact_ml_probability_key(count_row: Iterable[int]) -> Fraction:
    """Exact W=2^{-n Hhat}; higher W means lower empirical entropy."""
    return _exact_ml_probability_key_tuple(tuple(map(int, count_row)))


def exact_pessimistic_ranks_from_counts(counts: np.ndarray) -> np.ndarray:
    """Exact pessimistic ranks, with Fraction ordering and no float pre-sort."""
    unique_rows, inverse, multiplicities = np.unique(
        counts, axis=0, return_inverse=True, return_counts=True
    )
    keys = [exact_ml_probability_key(row) for row in unique_rows]
    mass_by_key: dict[Fraction, int] = defaultdict(int)
    for key, mass in zip(keys, multiplicities.tolist()):
        mass_by_key[key] += int(mass)

    cumulative = 0
    rank_by_key: dict[Fraction, int] = {}
    for key in sorted(mass_by_key.keys(), reverse=True):
        cumulative += mass_by_key[key]
        rank_by_key[key] = cumulative

    rank_unique = np.fromiter((rank_by_key[k] for k in keys),
                              dtype=np.int64, count=len(keys))
    return rank_unique[inverse]


def exact_pessimistic_ranks(bits: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, terminal = counts_and_terminal(bits, table)
    return exact_pessimistic_ranks_from_counts(counts), counts, terminal


def exact_cstar(best_ranks: np.ndarray) -> dict[str, int | float]:
    ordered = np.sort(best_ranks.astype(np.int64, copy=False))
    best_num, best_den, best_k = 1, int(ordered[0]), 1
    for idx, den_value in enumerate(ordered, start=1):
        den = int(den_value)
        if idx * best_den > best_num * den:
            best_num, best_den, best_k = idx, den, idx
    g = math.gcd(best_num, best_den)
    return {
        "numerator": best_num // g,
        "denominator": best_den // g,
        "kstar": best_k,
        "b_at_kstar": int(ordered[best_k - 1]),
        "float": best_num / best_den,
        "log2": math.log2(best_num / best_den),
        "b_max": int(ordered[-1]),
        "b_median_low": int(ordered[(len(ordered) - 1) // 2]),
    }


def determinant_bareiss(matrix: list[list[int]]) -> int:
    n = len(matrix)
    if n == 0:
        return 1
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    a = [list(map(int, row)) for row in matrix]
    sign = 1
    prev = 1
    for k in range(n - 1):
        pivot_row = next((r for r in range(k, n) if a[r][k] != 0), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign *= -1
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                if k:
                    if numerator % prev:
                        raise ArithmeticError("non-exact Bareiss division")
                    numerator //= prev
                a[i][j] = numerator
        for i in range(k + 1, n):
            a[i][k] = 0
        prev = pivot
    return sign * a[n - 1][n - 1]


def arborescences_to_root(adjacency: list[list[int]], root: int) -> int:
    """Directed in-arborescences toward root via out-degree Laplacian."""
    m = len(adjacency)
    if not (0 <= root < m):
        raise ValueError("invalid root")
    if m == 1:
        return 1
    keep = [v for v in range(m) if v != root]
    lap_minor: list[list[int]] = []
    for v in keep:
        outdeg = sum(adjacency[v])
        row = []
        for w in keep:
            row.append((outdeg if v == w else 0) - adjacency[v][w])
        lap_minor.append(row)
    return abs(determinant_bareiss(lap_minor))


def best_type_size(count_row: Iterable[int], terminal: int,
                   table: np.ndarray, initial: int = 0) -> int:
    """Active-support Whittle/BEST type size with a distinguished auxiliary edge."""
    row = list(map(int, count_row))
    s = int(table.shape[0])
    if len(row) != 2 * s:
        raise ValueError("wrong count-row length")
    adjacency = [[0 for _ in range(s)] for _ in range(s)]
    for state in range(s):
        for bit in (0, 1):
            count = row[2 * state + bit]
            if count:
                adjacency[state][int(table[state, bit])] += count
    adjacency[int(terminal)][initial] += 1

    outdeg_full = [sum(r) for r in adjacency]
    indeg_full = [sum(adjacency[u][v] for u in range(s)) for v in range(s)]
    active = [v for v in range(s) if outdeg_full[v] + indeg_full[v] > 0]
    if not active:
        return 1 if sum(row) == 0 else 0
    if initial not in active:
        return 0
    remap = {v: i for i, v in enumerate(active)}
    adjacency_active = [[adjacency[v][w] for w in active] for v in active]
    arb = arborescences_to_root(adjacency_active, remap[initial])
    if arb == 0:
        return 0
    numerator = arb
    for v in active:
        d = outdeg_full[v]
        if d <= 0:
            raise AssertionError("active vertex with zero out-degree after closure")
        numerator *= math.factorial(d - 1)
    denominator = 1
    for value in row:
        denominator *= math.factorial(value)
    if numerator % denominator:
        raise ArithmeticError("BEST numerator is not divisible by label quotient")
    return numerator // denominator


def independent_best_ranks(bits: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    counts, terminal = counts_and_terminal(bits, table)
    typed = np.concatenate([counts, terminal[:, None].astype(np.uint8)], axis=1)
    unique_types, inverse, observed_masses = np.unique(
        typed, axis=0, return_inverse=True, return_counts=True
    )
    type_sizes: list[int] = []
    score_keys: list[Fraction] = []
    size_mismatches = 0
    for row, observed in zip(unique_types, observed_masses.tolist()):
        count_row = row[:-1]
        end = int(row[-1])
        calculated = best_type_size(count_row, end, table)
        type_sizes.append(calculated)
        score_keys.append(exact_ml_probability_key(count_row))
        if calculated != int(observed):
            size_mismatches += 1

    mass_by_score: dict[Fraction, int] = defaultdict(int)
    for key, size in zip(score_keys, type_sizes):
        mass_by_score[key] += size
    cumulative = 0
    rank_by_score: dict[Fraction, int] = {}
    for key in sorted(mass_by_score.keys(), reverse=True):
        cumulative += mass_by_score[key]
        rank_by_score[key] = cumulative
    rank_type = np.fromiter((rank_by_score[k] for k in score_keys),
                            dtype=np.int64, count=len(score_keys))
    return rank_type[inverse], {
        "n_types": int(len(unique_types)),
        "type_size_mismatches": int(size_mismatches),
    }


def active_support_regression_test() -> dict[str, int | bool]:
    table = np.array([[0, 0], [1, 1]], dtype=np.int16)
    bits = np.zeros((1, 4), dtype=np.uint8)
    counts, terminal = counts_and_terminal(bits, table)
    size = best_type_size(counts[0], int(terminal[0]), table)
    return {"computed_type_size": size, "expected_type_size": 1,
            "pass": bool(size == 1)}
