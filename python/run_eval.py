"""
Evaluate sequence filtering methods and write results to results/<dataset>_<datetime>.json.

Example usage:
python3 run_eval.py \
        --dataset ml-1m \
        --state_dict_path ml-1m_default/SASRec.epoch=920.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
        --seq_len_values 50 100 200 \
        --num_repeats 3
        
        
python3 run_eval.py \
        --dataset Electronics \
        --state_dict_path Electronics_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
        --seq_len_values 50 100 200 \
        --num_repeats 3
"""

import os
import json
import math
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from model import SASRec, PolicyNetwork
from utils import data_partition, evaluate_split_with_filter
from filters import (
    from_end,
    KMeansFilteringV2,
    FastMMRFiltering,
    FilterByDifficulty,
    uniform_random,
    PolicyFilter,
)


def build_methods(item_embeddings, seq_len_values, num_clusters=200, model=None, itemnum=None, policy_net=None):
    """
    Returns a list of (name, seq_len_or_None, filter_fn) triples.

    seq_len_or_None is set to a specific seq_len for methods that are
    parameterised per sequence length (e.g. bounded cluster filtering),
    and None for methods that apply to all sequence lengths.
    """
    methods = []

    # Baseline: most-recent items
    methods.append(("from_end", None, from_end))
    
#     methods.append(("uniform_random", None, uniform_random))

#     # K-Means: mixed unique + recent (seq-len agnostic)
#     kmeans_mixed = KMeansFilteringV2(item_embeddings, n_clusters=num_clusters)
#     methods.append((
#         f"kmeans_{num_clusters}_mixed_unique_and_recent",
#         None,
#         kmeans_mixed.mixed_unique_and_recent,
#     ))


#     # K-Means: bounded cluster filtering, max_per_cluster = multiplier * sqrt(seq_len)
#     for seq_len in seq_len_values:
#         for multiplier in [1, 2, 4]:
#             max_per_cluster = int(multiplier * math.sqrt(seq_len))
#             kmf = KMeansFilteringV2(
#                 item_embeddings,
#                 n_clusters=num_clusters,
#                 max_per_cluster=max_per_cluster,
#             )
#             methods.append((
#                 f"kmeans_{num_clusters}_bounded_{max_per_cluster}_per_cluster",
#                 seq_len,
#                 kmf.bounded_cluster_filtering,
#             ))

#     # MMR filtering (seq-len agnostic)
#     for lambda_recency in [0.5, 0.7, 0.9]:
#         mmr = FastMMRFiltering(item_embeddings, lambda_recency=lambda_recency)
#         methods.append((
#             f"mmr_lambda{lambda_recency}",
#             None,
#             mmr.mmr_filtering,
#         ))

#     # Difficulty-based filtering (requires model)
#     if model is not None and itemnum is not None:
#         for k_percent in [10, 20]:
#             diff = FilterByDifficulty(model, itemnum, k_percent=k_percent)
#             methods.append((
#                 f"difficulty_remove_easiest_{k_percent}pct",
#                 None,
#                 diff.filter_easiest_k_percent,
#             ))
#             methods.append((
#                 f"difficulty_remove_hardest_{k_percent}pct",
#                 None,
#                 diff.filter_hardest_k_percent,
#             ))

    # RL policy filter
    if policy_net is not None:
        pf = PolicyFilter(policy_net)
        methods.append(("rl_policy", None, lambda seq, ml: pf.filter(seq, ml)))

    return methods


def run_evaluation(model, dataset, seq_len_values, num_repeats, methods, split, args):
    ndcg_results      = {k: defaultdict(list) for k in seq_len_values}
    hr_results        = {k: defaultdict(list) for k in seq_len_values}
    ndcg_long_results = {k: defaultdict(list) for k in seq_len_values}
    hr_long_results   = {k: defaultdict(list) for k in seq_len_values}
    # Counts are constant across repeats; store a single int per (seq_len, method)
    n_users_results      = {k: {} for k in seq_len_values}
    n_users_long_results = {k: {} for k in seq_len_values}

    for seq_len in seq_len_values:
        for name, method_seq_len, fn in methods:
            # Skip per-seq_len methods that don't belong to this seq_len
            if method_seq_len is not None and method_seq_len != seq_len:
                continue

            for repeat in range(num_repeats):
                print(
                    f"  seq_len={seq_len}  method={name}  repeat={repeat + 1}/{num_repeats}",
                    flush=True,
                )
                ndcg, hr, ndcg_long, hr_long, n, n_long = evaluate_split_with_filter(
                    model, dataset,
                    None,       # args (unused inside the function)
                    seq_len,    # history_len
                    split=split,
                    filter_function=fn,
                    verbose=True,
                    user_start_idx=args.user_start_idx,
                    user_end_idx=args.user_end_idx,
                    long_users_only=args.long_users_only,
                )
                ndcg_results[seq_len][name].append(ndcg)
                hr_results[seq_len][name].append(hr)
                ndcg_long_results[seq_len][name].append(ndcg_long)
                hr_long_results[seq_len][name].append(hr_long)
                n_users_results[seq_len][name] = n
                n_users_long_results[seq_len][name] = n_long
                print(
                    f"    NDCG@10={ndcg:.4f}  HR@10={hr:.4f}"
                    f"  NDCG@10(long)={ndcg_long:.4f}  HR@10(long)={hr_long:.4f}"
                    f"  n={n}  n_long={n_long}",
                    flush=True,
                )

    return ndcg_results, hr_results, ndcg_long_results, hr_long_results, n_users_results, n_users_long_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--state_dict_path', required=True)
    parser.add_argument('--seq_len_values', type=int, nargs='+', default=[50, 100, 200])
    parser.add_argument('--num_repeats', type=int, default=3)
    parser.add_argument('--num_clusters', type=int, default=200)
    # Model architecture — must match the checkpoint
    parser.add_argument('--hidden_units', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--policy_path', type=str, default=None,
                        help='Path to a trained PolicyNetwork checkpoint (.pth) from train_rl_filter.py')
    parser.add_argument('--user_start_idx', type=int, default=0,
                        help='Start index into the sorted user list (inclusive, default: 0)')
    parser.add_argument('--user_end_idx', type=int, default=None,
                        help='End index into the sorted user list (exclusive, default: all users)')
    parser.add_argument('--long_users_only', action='store_true', default=False,
                        help='Only evaluate users whose history is longer than the sequence length')
    args = parser.parse_args()

    # ------------------------------------------------------------------ dataset
    print(f"Loading dataset '{args.dataset}' ...", flush=True)
    dataset = data_partition(args.dataset)
    [_, _, _, usernum, itemnum] = dataset

    # ------------------------------------------------------------------ model
    import argparse as _ap
    model_args = _ap.Namespace(
        device=args.device,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        maxlen=args.maxlen,
        norm_first=False,
    )

    print(f"Loading model from '{args.state_dict_path}' ...", flush=True)
    model = SASRec(usernum, itemnum, model_args).to(args.device)
    model.load_state_dict(
        torch.load(args.state_dict_path, map_location=torch.device(args.device))
    )
    model.eval()

    # ------------------------------------------------------------------ filters
    item_embeddings = model.item_emb.weight.data.cpu().numpy()

    policy_net = None
    if args.policy_path is not None:
        print(f"Loading RL policy from '{args.policy_path}' ...", flush=True)
        ckpt = torch.load(args.policy_path, map_location=torch.device(args.device))
        policy_saved_args = argparse.Namespace(**ckpt['args'])
        policy_args = argparse.Namespace(
            hidden_units=policy_saved_args.hidden_units,
            num_blocks=policy_saved_args.num_blocks,
            num_heads=policy_saved_args.num_heads,
            maxlen=policy_saved_args.policy_maxlen,
            dropout_rate=policy_saved_args.dropout_rate,
            device=args.device,
            norm_first=policy_saved_args.norm_first,
            attention=policy_saved_args.attention,
        )
        user_num = usernum if policy_saved_args.user_emb else 0
        policy_net = PolicyNetwork(user_num, itemnum, policy_args).to(args.device)
        policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        policy_net.eval()

    print("Building filter methods (K-Means fitting may take a moment) ...", flush=True)
    methods = build_methods(item_embeddings, args.seq_len_values, num_clusters=args.num_clusters, model=model, itemnum=itemnum, policy_net=policy_net)

    # ------------------------------------------------------------------ evaluate
    print("Starting evaluation ...", flush=True)
    ndcg_results, hr_results, ndcg_long_results, hr_long_results, n_users_results, n_users_long_results = run_evaluation(
        model, dataset, args.seq_len_values, args.num_repeats, methods, args.split, args
    )

    # ------------------------------------------------------------------ save
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"{args.dataset}_{timestamp}.json")

    output = {
        "ndcg":         {str(k): dict(v) for k, v in ndcg_results.items()},
        "hr":           {str(k): dict(v) for k, v in hr_results.items()},
        "ndcg_long":    {str(k): dict(v) for k, v in ndcg_long_results.items()},
        "hr_long":      {str(k): dict(v) for k, v in hr_long_results.items()},
        "n_users":      {str(k): dict(v) for k, v in n_users_results.items()},
        "n_users_long": {str(k): dict(v) for k, v in n_users_long_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
