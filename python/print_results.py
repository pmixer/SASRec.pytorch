"""
Print a summary of results produced by run_eval.py.

Usage:
    python print_results.py results/ml-1m_20260220_151533.json
    python print_results.py results/ml-1m_20260220_151533.json --sort hr
"""

import json
import argparse
import numpy as np


def print_results(path, sort_by="ndcg"):
    with open(path) as f:
        data = json.load(f)

    ndcg_data        = data["ndcg"]
    hr_data          = data["hr"]
    ndcg_long_data   = data["ndcg_long"]
    hr_long_data     = data["hr_long"]
    n_users_data      = data["n_users"]
    n_users_long_data = data["n_users_long"]

    seq_lens = sorted(ndcg_data.keys(), key=int)
    all_methods = sorted({m for sl in ndcg_data.values() for m in sl})
    method_col = max(len(m) for m in all_methods)

    print(f"\n{path}")
    print("=" * (method_col + 62))

    for seq_len in seq_lens:
        ndcg_block        = ndcg_data[seq_len]
        hr_block          = hr_data[seq_len]
        ndcg_long_block   = ndcg_long_data[seq_len]
        hr_long_block     = hr_long_data[seq_len]
        n_users_block      = n_users_data[seq_len]
        n_users_long_block = n_users_long_data[seq_len]

        rows = []
        for method in all_methods:
            ndcg_vals = ndcg_block.get(method, [])
            hr_vals   = hr_block.get(method, [])
            if not ndcg_vals:
                continue
            rows.append((
                method,
                np.mean(ndcg_vals),
                np.mean(hr_vals),
                np.mean(ndcg_long_block.get(method, [])),
                np.mean(hr_long_block.get(method, [])),
                n_users_block.get(method),
                n_users_long_block.get(method),
            ))

        rows.sort(key=lambda r: r[1 if sort_by == "ndcg" else 2], reverse=True)
        best_ndcg      = rows[0][1]
        best_hr        = rows[0][2]
        best_ndcg_long = max(r[3] for r in rows)
        best_hr_long   = max(r[4] for r in rows)

        print(f"\n  Seq len {seq_len}  (n={len(next(iter(ndcg_block.values())))} repeats)")
        print(f"  {'Method':<{method_col}}  {'NDCG@10':>8}  {'HR@10':>8}  {'NDCG@10(long)':>14}  {'HR@10(long)':>11}  {'n':>7}  {'n_long':>7}")
        print(f"  {'-' * method_col}  {'--------':>8}  {'--------':>8}  {'-' * 14}  {'-' * 11}  {'-------':>7}  {'-------':>7}")

        for method, ndcg, hr, ndcg_long, hr_long, n, n_long in rows:
            ndcg_marker      = "*" if ndcg      == best_ndcg      else " "
            hr_marker        = "*" if hr        == best_hr        else " "
            ndcg_long_marker = "*" if ndcg_long == best_ndcg_long else " "
            hr_long_marker   = "*" if hr_long   == best_hr_long   else " "
            n_str      = str(n)      if n      is not None else "n/a"
            n_long_str = str(n_long) if n_long is not None else "n/a"
            print(
                f"  {method:<{method_col}}  {ndcg:.4f}{ndcg_marker}  {hr:.4f}{hr_marker}"
                f"  {ndcg_long:.4f}{ndcg_long_marker}         {hr_long:.4f}{hr_long_marker}"
                f"  {n_str:>7}  {n_long_str:>7}"
            )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="JSON file produced by run_eval.py")
    parser.add_argument("--sort", choices=["ndcg", "hr"], default="ndcg",
                        help="Metric to sort by (default: ndcg)")
    args = parser.parse_args()
    print_results(args.results_file, sort_by=args.sort)
