"""eval_ndcg.py — per-user NDCG@10 on validation and test sets with a configurable filter.

Example
-------
# from_end baseline, first 10 users
python3 eval_ndcg.py \
  --dataset ml-1m \
  --sasrec_path ml-1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
  --max_length 50 \
  --filter from_end \
  --num_users 10

# trained policy filter
python3 eval_ndcg.py \
      --dataset ml-1m \
      --sasrec_path ml-1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
      --max_length 50 \
      --filter policy \
      --policy_path rl_policy_ml1m/policy.epoch=200.maxlen=50.ndcg=0.7597.pth \
      --num_users 10
"""

import argparse
import copy

import numpy as np
import torch

from model import SASRec, PolicyNetwork
from filters import from_end, uniform_random
from utils import data_partition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_negatives(valid_item, rated, itemnum, n=100, rng=None):
    """Sample n negatives from {1..itemnum} \\ rated."""
    if rng is None:
        rng = np.random.default_rng()
    rated_set = set(rated) | {0}
    item_idx = [valid_item]
    while len(item_idx) <= n:
        t = rng.integers(1, itemnum + 1)
        if t not in rated_set:
            item_idx.append(t)
    return item_idx


def _ndcg_at_10(model, u, seq, item_idx):
    """Return NDCG@10 for a single (user, padded-seq, candidate-list) triple."""
    with torch.no_grad():
        preds = -model.predict(np.array([u]), np.array([seq]), item_idx)
        preds = preds[0]
    rank = preds.argsort().argsort()[0].item()
    return 1 / np.log2(rank + 2)


def _build_seq(history, max_length):
    """Right-align history into a zero-padded array of length max_length."""
    seq = np.zeros(max_length, dtype=np.int32)
    idx = max_length - 1
    for item in reversed(history):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def eval_users(model, user_train, user_valid, user_test, usernum, itemnum,
               filter_fn, max_length, num_users, seed):
    """Evaluate the first num_users eligible users on both valid and test splits.

    Args:
        filter_fn: callable(sequence, max_length) -> filtered sequence.
                   For the policy filter with user conditioning, wrap it so
                   it already has user_id bound (or just ignore —
                   PolicyNetwork.filter without user_id uses 0 by default).
    """
    rng = np.random.default_rng(seed)

    users = [u for u in range(1, usernum + 1)
             if len(user_train[u]) >= 1 and len(user_valid[u]) >= 1 and len(user_test[u]) >= 1]
    users = users[:num_users]

    print(f"{'user':>6}  {'valid NDCG@10':>14}  {'test NDCG@10':>13}")
    print("-" * 40)

    valid_ndcgs, test_ndcgs = [], []
    for u in users:
        train_seq = user_train[u]
        valid_item = user_valid[u][0]
        test_item = user_test[u][0]
        rated = set(train_seq)

        # --- validation ---
        filtered_valid = filter_fn(train_seq, max_length)
        seq_valid = _build_seq(filtered_valid, max_length)
        neg_valid = _sample_negatives(valid_item, rated, itemnum, rng=rng)
        ndcg_valid = _ndcg_at_10(model, u, seq_valid, neg_valid)

        # --- test: valid item is part of history ---
        train_plus_valid = train_seq + [valid_item]
        filtered_test = filter_fn(train_plus_valid, max_length)
        seq_test = _build_seq(filtered_test, max_length)
        rated_test = rated | {valid_item}
        neg_test = _sample_negatives(test_item, rated_test, itemnum, rng=rng)
        ndcg_test = _ndcg_at_10(model, u, seq_test, neg_test)

        print(f"{u:>6}  {ndcg_valid:>14.4f}  {ndcg_test:>13.4f}")
        valid_ndcgs.append(ndcg_valid)
        test_ndcgs.append(ndcg_test)

    print("-" * 40)
    print(f"{'mean':>6}  {sum(valid_ndcgs)/len(valid_ndcgs):>14.4f}  {sum(test_ndcgs)/len(test_ndcgs):>13.4f}")
    return valid_ndcgs, test_ndcgs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-user NDCG@10 with configurable filter")

    # Dataset / model
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sasrec_path', required=True)
    parser.add_argument('--max_length', type=int, required=True,
                        help='Filter output length passed to SASRec')
    parser.add_argument('--device', default='cpu')

    # SASRec architecture (must match checkpoint)
    parser.add_argument('--sasrec_hidden_units', type=int, default=50)
    parser.add_argument('--sasrec_num_blocks', type=int, default=2)
    parser.add_argument('--sasrec_num_heads', type=int, default=1)
    parser.add_argument('--sasrec_maxlen', type=int, default=200)
    parser.add_argument('--sasrec_norm_first', action='store_true', default=False)

    # Filter
    parser.add_argument('--filter', dest='filter_name', default='from_end',
                        choices=['from_end', 'uniform_random', 'policy'],
                        help='Filtering method (default: from_end)')
    parser.add_argument('--policy_path', default=None,
                        help='Path to policy checkpoint (.pth); required when --filter policy')
    parser.add_argument('--user_emb', action='store_true', default=False,
                        help='Policy was trained with user embeddings (only relevant for --filter policy)')

    # Evaluation scope
    parser.add_argument('--num_users', type=int, default=10,
                        help='Number of users to evaluate (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for negative sampling (default: 42)')

    args = parser.parse_args()

    if args.filter_name == 'policy' and args.policy_path is None:
        parser.error('--policy_path is required when --filter policy')

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print(f"Dataset: {usernum} users, {itemnum} items")

    # ------------------------------------------------------------------
    # 2. SASRec
    # ------------------------------------------------------------------
    sasrec_args = argparse.Namespace(
        hidden_units=args.sasrec_hidden_units,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        maxlen=args.sasrec_maxlen,
        dropout_rate=0.0,
        device=args.device,
        norm_first=args.sasrec_norm_first,
    )
    model = SASRec(usernum, itemnum, sasrec_args).to(args.device)
    model.load_state_dict(torch.load(args.sasrec_path, map_location=args.device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded SASRec from {args.sasrec_path}")

    # ------------------------------------------------------------------
    # 3. Filter function
    # ------------------------------------------------------------------
    if args.filter_name == 'from_end':
        filter_fn = from_end
    elif args.filter_name == 'uniform_random':
        filter_fn = uniform_random
    else:  # policy
        ckpt = torch.load(args.policy_path, map_location=args.device)
        saved_args = argparse.Namespace(**ckpt['args'])
        policy_args = argparse.Namespace(
            hidden_units=saved_args.hidden_units,
            num_blocks=saved_args.num_blocks,
            num_heads=saved_args.num_heads,
            maxlen=saved_args.policy_maxlen,
            dropout_rate=0.0,
            device=args.device,
            norm_first=saved_args.norm_first,
            causal=saved_args.causal,
        )
        use_user_emb = getattr(saved_args, 'user_emb', False) or args.user_emb
        policy_net = PolicyNetwork(
            usernum if use_user_emb else 0, itemnum, policy_args
        ).to(args.device)
        policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        policy_net.eval()
        filter_fn = policy_net.filter

    print(f"Filter: {args.filter_name}")
    print()

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    if args.filter_name == 'policy' and (getattr(saved_args, 'user_emb', False) or args.user_emb):
        # Need to pass user_id per user; use a thin wrapper evaluated inline.
        rng = np.random.default_rng(args.seed)
        users = [u for u in range(1, usernum + 1)
                 if len(user_train[u]) >= 1 and len(user_valid[u]) >= 1 and len(user_test[u]) >= 1]
        users = users[:args.num_users]
        print(f"{'user':>6}  {'valid NDCG@10':>14}  {'test NDCG@10':>13}")
        print("-" * 40)
        valid_ndcgs, test_ndcgs = [], []
        for u in users:
            train_seq = user_train[u]
            valid_item = user_valid[u][0]
            test_item = user_test[u][0]
            rated = set(train_seq)

            filtered_valid = pf.filter(train_seq, args.max_length, user_id=u)
            seq_valid = _build_seq(filtered_valid, args.max_length)
            neg_valid = _sample_negatives(valid_item, rated, itemnum, rng=rng)
            ndcg_valid = _ndcg_at_10(model, u, seq_valid, neg_valid)

            train_plus_valid = train_seq + [valid_item]
            filtered_test = pf.filter(train_plus_valid, args.max_length, user_id=u)
            seq_test = _build_seq(filtered_test, args.max_length)
            neg_test = _sample_negatives(test_item, rated | {valid_item}, itemnum, rng=rng)
            ndcg_test = _ndcg_at_10(model, u, seq_test, neg_test)

            print(f"{u:>6}  {ndcg_valid:>14.4f}  {ndcg_test:>13.4f}")
            valid_ndcgs.append(ndcg_valid)
            test_ndcgs.append(ndcg_test)
        print("-" * 40)
        print(f"{'mean':>6}  {sum(valid_ndcgs)/len(valid_ndcgs):>14.4f}  {sum(test_ndcgs)/len(test_ndcgs):>13.4f}")
    else:
        eval_users(model, user_train, user_valid, user_test, usernum, itemnum,
                   filter_fn, args.max_length, args.num_users, args.seed)


if __name__ == '__main__':
    main()
