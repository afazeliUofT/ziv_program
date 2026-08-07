from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .core import (
    active_support_regression_test,
    all_machines,
    all_sequence_bits,
    exact_cstar,
    exact_pessimistic_ranks,
    independent_best_ranks,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run_family(s: int, ns: list[int], independent_ns: set[int],
               output: Path) -> tuple[list[dict], list[dict]]:
    machines = all_machines(s)
    rows: list[dict] = []
    independent_rows: list[dict] = []
    for n in ns:
        started = time.time()
        bits = all_sequence_bits(n)
        best = np.full(bits.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        rank_cache_for_independent: dict[int, np.ndarray] = {}
        print(f"[exact] s={s} n={n}: {len(machines)} topologies, {len(bits)} sequences", flush=True)
        for mi, table in enumerate(machines):
            ranks, _, _ = exact_pessimistic_ranks(bits, table)
            np.minimum(best, ranks, out=best)
            if n in independent_ns:
                rank_cache_for_independent[mi] = ranks
            if (mi + 1) % max(1, len(machines) // 10) == 0 or mi + 1 == len(machines):
                print(f"  topology {mi + 1}/{len(machines)}", flush=True)

        cstar = exact_cstar(best)
        row = {
            "s": s,
            "n": n,
            "n_topologies": len(machines),
            "n_sequences": int(len(bits)),
            "generic_factor_bound": len(machines),
            "generic_log2_bound": float(np.log2(len(machines))),
            "elapsed_s": round(time.time() - started, 3),
            **{f"Cstar_{k}": v for k, v in cstar.items()},
        }
        rows.append(row)
        print(f"  C*={cstar['numerator']}/{cstar['denominator']}={cstar['float']:.12g}", flush=True)

        if n in independent_ns:
            compared = 0
            rank_mismatches = 0
            type_size_mismatches = 0
            type_count = 0
            for mi, table in enumerate(machines):
                independent, stats = independent_best_ranks(bits, table)
                primary = rank_cache_for_independent[mi]
                rank_mismatches += int(np.count_nonzero(primary != independent))
                compared += int(len(primary))
                type_size_mismatches += int(stats["type_size_mismatches"])
                type_count += int(stats["n_types"])
            check = {
                "s": s,
                "n": n,
                "compared_sequence_topology_pairs": compared,
                "n_types": type_count,
                "rank_mismatches": rank_mismatches,
                "type_size_mismatches": type_size_mismatches,
                "pass": bool(rank_mismatches == 0 and type_size_mismatches == 0),
            }
            independent_rows.append(check)
            print(f"  independent BEST check: {'PASS' if check['pass'] else 'FAIL'}", flush=True)
    return rows, independent_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct exact-arithmetic rank audit")
    parser.add_argument("--scale", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", default="results_exact")
    parser.add_argument("--skip-s3", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    active_test = active_support_regression_test()
    if not active_test["pass"]:
        raise RuntimeError("active-support regression failed")

    if args.scale == "smoke":
        s2_ns = [8, 10]
        s3_ns = [8]
        independent_s2 = {8}
    else:
        s2_ns = [8, 10, 12, 14, 16, 18, 20]
        s3_ns = [8, 10, 12, 14]
        independent_s2 = {8, 10, 12, 14}

    rows, independent = run_family(2, s2_ns, independent_s2, output)
    if not args.skip_s3:
        rows3, independent3 = run_family(3, s3_ns, set(), output)
        rows.extend(rows3)
        independent.extend(independent3)

    all_independent_pass = all(item["pass"] for item in independent)
    all_generic_pass = all(item["Cstar_float"] <= item["generic_factor_bound"] + 1e-15
                           for item in rows)
    scorecard = {
        "package_version": __version__,
        "scale": args.scale,
        "active_support_regression": active_test,
        "rows": rows,
        "independent_checks": independent,
        "gate": "PASS" if all_independent_pass and all_generic_pass else "FAIL",
        "notes": [
            "All score ordering and tie grouping use Python Fraction objects directly.",
            "No floating-point value participates in an ordering or equality decision.",
            "The independent route uses active-support BEST/Matrix-Tree type sizes.",
        ],
    }
    write_json(output / "EXACT_AUDIT_SCORECARD.json", scorecard)

    with (output / "EXACT_CSTAR.csv").open("w", newline="") as f:
        fields = [
            "s", "n", "n_topologies", "n_sequences",
            "Cstar_numerator", "Cstar_denominator", "Cstar_float",
            "Cstar_log2", "Cstar_kstar", "Cstar_b_at_kstar",
            "generic_factor_bound", "generic_log2_bound", "elapsed_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    manifest = {
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "started_unix": start,
        "elapsed_s": round(time.time() - start, 3),
        "gate": scorecard["gate"],
    }
    write_json(output / "RUN_MANIFEST.json", manifest)

    summary_lines = [
        "# Exact rank-audit summary",
        "",
        f"Gate: **{scorecard['gate']}**",
        "",
        "| s | n | exact C* | log2 C* | generic bound |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(
            f"| {row['s']} | {row['n']} | "
            f"{row['Cstar_numerator']}/{row['Cstar_denominator']} "
            f"({row['Cstar_float']:.9g}) | {row['Cstar_log2']:.6f} | "
            f"{row['generic_factor_bound']} |"
        )
    summary_lines += ["", "## Independent BEST checks", ""]
    for item in independent:
        summary_lines.append(
            f"- s={item['s']}, n={item['n']}: "
            f"{'PASS' if item['pass'] else 'FAIL'}; "
            f"rank mismatches={item['rank_mismatches']}, "
            f"type-size mismatches={item['type_size_mismatches']}."
        )
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    hash_lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hash_lines.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(hash_lines) + "\n")
    print(f"[exact] gate={scorecard['gate']} output={output}", flush=True)
    return 0 if scorecard["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
