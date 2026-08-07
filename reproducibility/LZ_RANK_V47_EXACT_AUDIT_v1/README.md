# LZ Rank v47 Direct Exact-Arithmetic Audit

This package recomputes the finite-topology pessimistic-rank benchmark without using floating-point values for score ordering or tie decisions.

It addresses two review points:

1. the manuscript's type-count formula must use the active support, not all nominal machine states;
2. the earlier C1 implementation used a floating-point presort before exact tie refinement, which is not a proof of exact ordering for arbitrarily close unequal rational scores.

## Full run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_audit.py --scale full --output results_exact
```

The full grid is:

- all 16 canonical binary 2-state transition topologies at n = 8,10,12,14,16,18,20;
- all 729 canonical binary 3-state transition topologies at n = 8,10,12,14;
- an independent active-support Whittle/BEST and Matrix-Tree reconstruction for every 2-state topology at n <= 14.

## Smoke run

```bash
python run_audit.py --scale smoke --output results_smoke
```

## Exactness contract

- empirical-ML score keys are reduced `fractions.Fraction` objects;
- distinct score levels are sorted directly as exact rational numbers;
- no floating-point number participates in an ordering or tie test;
- the exact minimax ratio is maximized by integer cross-multiplication;
- the independent type route uses exact integer determinants and factorials on the active-support graph.

Expected full-run time depends on CPU and memory. The n=20 2-state stage materializes 2^20 binary strings and is the largest memory step.
