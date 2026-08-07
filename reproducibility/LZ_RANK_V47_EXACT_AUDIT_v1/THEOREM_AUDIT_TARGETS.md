# Theorem audit targets

The output is intended to validate numerical statements only. It does not prove the manuscript's conditional adversarial local lemma.

The paper should retain a C* numerical claim only when:

- `gate` is `PASS`;
- every independent 2-state check has zero rank and type-size mismatches;
- the corresponding row is present in `EXACT_AUDIT_SCORECARD.json`;
- the output hashes are preserved.

No asymptotic law should be inferred from the finite n values without a separate theorem.
