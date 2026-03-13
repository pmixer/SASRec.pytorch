"""train_rl_filter.py — REINFORCE training for PolicyNetwork sequence filter.

Loads a frozen SASRec checkpoint and trains a PolicyNetwork end-to-end to
select the max_length history items that maximise SASRec's validation NDCG.

Example
-------
python3 train_rl_filter.py \
--dataset ml-1m \
--sasrec_path ml-1m_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
--save_dir rl_policy_ml1m \
--max_length 50 \
--num_epochs 3 \
--num_samples 32 \
--dropout_rate 0.2 \
--baseline rloo \
--grad_clip 1.0 \
--gradient_accumulation_steps 16 \
--device cuda \
--entropy_coef 0.0 \
--seed 42 \
--reward reciprocal_rank \
--user_start_idx 0 --num_blocks 2 --attention full

python3 train_rl_filter.py \
--dataset Electronics \
--sasrec_path Electronics_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
--save_dir rl_policy_Electronics \
--max_length 50 \
--num_epochs 10 \
--num_samples 32 \
--dropout_rate 0.2 \
--baseline rloo \
--grad_clip 1.0 \
--gradient_accumulation_steps 16 \
--device cpu \
--seed 42 \
--reward reciprocal_rank \
--user_start_idx 0 --num_blocks 2 --user_end_idx 400000 --attention full


python3 train_rl_filter.py \
--dataset Electronics \
--sasrec_path Electronics_default/SASRec.epoch=20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
--save_dir rl_policy_Electronics \
--max_length 50 \
--num_epochs 10 \
--num_samples 32 \
--dropout_rate 0.2 \
--baseline rloo \
--grad_clip 1.0 \
--gradient_accumulation_steps 16 \
--device cpu \
--seed 42 \
--reward reciprocal_rank \
--user_start_idx 0 --num_blocks 1 --user_end_idx 400000 --attention full --policy_type encoder_decoder


python3 train_rl_filter.py \
--dataset Beauty \
--sasrec_path Beauty_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth \
--save_dir rl_policy_Beauty \
--max_length 50 \
--num_epochs 10 \
--num_samples 32 \
--dropout_rate 0.2 \
--baseline rloo \
--grad_clip 1.0 \
--gradient_accumulation_steps 16 \
--device cpu \
--entropy_coef 0.0 \
--seed 42 \
--reward reciprocal_rank \
--user_start_idx 0 --num_blocks 2 --attention full
"""

import os
import sys
import argparse

import numpy as np
import torch
import wandb

from model import SASRec, PolicyNetwork, EncoderDecoderPolicyNetwork
from filters import uniform_random, from_end
from utils import data_partition, evaluate_split_with_filter




# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(sasrec_model, u, filtered_seq, valid_item, rated, itemnum, max_length,
                   reward_type='ndcg10'):
    """Compute a reward for a single user given a filtered history sequence.

    Args:
        sasrec_model: frozen SASRec instance (eval mode).
        u:            user id (int).
        filtered_seq: list of item indices after filtering.
        valid_item:   the ground-truth validation item.
        rated:        set of items in user's training history.
        itemnum:      total number of items in the catalogue.
        max_length:   expected input length for SASRec (must be <= sasrec_maxlen).
        reward_type:  'ndcg10' (binary, sparse) or 'reciprocal_rank' (dense, always > 0).

    Returns:
        Reward (float).  NDCG@10 is in [0, 1]; reciprocal rank is in (0, 1].
    """
    # Right-align filtered_seq into a zero-padded array of length max_length
    seq = np.zeros([max_length], dtype=np.int32)
    idx = max_length - 1
    for item in reversed(filtered_seq):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break

    # Sample 100 negatives from {1..itemnum} \ rated
    rated_set = set(rated)
    rated_set.add(0)
    item_idx = [valid_item]

    for _ in range(100):
        t = np.random.randint(1, itemnum + 1)
        while t in rated_set:
            t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)

    with torch.no_grad():
        predictions = -sasrec_model.predict(np.array([u]), np.array([seq]), item_idx)
        predictions = predictions[0]

    rank = predictions.argsort().argsort()[0].item()

    if reward_type == 'reciprocal_rank':
        return 1.0 / (rank + 1)
    else:  # 'ndcg10'
        return 1 / np.log2(rank + 2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(policy_net, sasrec_model, dataset, args, optimizer, baseline_fn):
    """Run one epoch of REINFORCE training over a subset of users.

    Args:
        policy_net:    PolicyNetwork being trained.
        sasrec_model:  Frozen SASRec used as reward model.
        dataset:       Output of data_partition().
        args:          Parsed arguments.
        optimizer:     Adam optimizer over trainable policy parameters.
        baseline_fn:   Callable(sequence, max_length) -> list; baseline filter.

    Returns:
        Mean NDCG@10 over all collected rollouts (float).
    """
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    users = list(range(1, usernum + 1))

    total_ndcg = 0.0
    num_valid = 0
    optimizer.zero_grad()
    accumulated = 0

    for u in users[args.user_start_idx:args.user_end_idx]:
        seq = user_train[u]

        # Skip users whose history is already short enough (need target + max_length items)
        if len(seq) <= args.max_length + 1:
            continue

        target_item = seq[-1]
        input_seq = seq[:-1]
        rated = set(user_train[u])

        # ------------------------------------------------------------------
        # Baseline reward (no gradient) — skip expensive SASRec call for rloo
        # ------------------------------------------------------------------
        if args.baseline != 'rloo':
            baseline_seq = baseline_fn(input_seq, args.max_length)
            baseline_reward = compute_reward(
                sasrec_model, u, baseline_seq, target_item, rated, itemnum, args.max_length,
                reward_type=args.reward,
            )
        else:
            baseline_reward = 0.0

        # ------------------------------------------------------------------
        # Policy rollout (gradient flows through sample_log_probs_tensor)
        # ------------------------------------------------------------------
        result = policy_net.rollout(input_seq, args.max_length, args.num_samples)
        if result is None:
            continue
        batch_seqs, sample_log_probs_tensor = result

        # ------------------------------------------------------------------
        # Sample negatives once per user (reused across all rollouts)
        # ------------------------------------------------------------------
        rated_set = rated | {0}
        item_idx = [target_item]
        while len(item_idx) <= 100:
            t = np.random.randint(1, itemnum + 1)
            if t not in rated_set:
                item_idx.append(t)

        # Single batched SASRec forward pass
        with torch.no_grad():
            batch_preds = -sasrec_model.predict(
                np.array([u] * args.num_samples),
                batch_seqs,
                item_idx,
            )  # [num_samples, 101]

        # Compute rewards from batch predictions
        ranks = batch_preds.argsort(dim=1).argsort(dim=1)[:, 0]   # [num_samples]

        if args.reward == 'reciprocal_rank':
            rewards_tensor = 1.0 / (ranks.float() + 1)
        else:  # ndcg10
            rewards_tensor = 1.0 / torch.log2(ranks.float() + 2)

        total_ndcg += rewards_tensor.sum().item()
        num_valid += args.num_samples

        # ------------------------------------------------------------------
        # REINFORCE loss: -E[advantage * log_prob]
        # ------------------------------------------------------------------
        rewards_arr = rewards_tensor.cpu().numpy()

        if args.baseline == 'rloo':
            # Leave-One-Out baseline: per-sample baseline is mean of all other samples.
            # Advantages have zero mean by construction; no fixed external baseline needed.
            n = rewards_tensor.shape[0]
            rloo_baselines = (rewards_tensor.sum() - rewards_tensor) / max(n - 1, 1)
            advantages = rewards_tensor - rloo_baselines
            baseline_display = rewards_tensor.mean().item()
        else:
            advantages = rewards_tensor - baseline_reward
            baseline_display = baseline_reward

        if u <= 10:
            print(
                f"  u={u:5d}  baseline={baseline_display:.4f}"
                f"  reward_mean={rewards_arr.mean():.4f}"
                f"  reward_std={rewards_arr.std():.4f}"
                f"  reward_min={rewards_arr.min():.4f}"
                f"  reward_max={rewards_arr.max():.4f}"
            )
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        loss = -(advantages * sample_log_probs_tensor).mean()
        (loss / args.gradient_accumulation_steps).backward()

        accumulated += 1
        if accumulated >= args.gradient_accumulation_steps:
            grad_norm = sum(
                p.grad.norm() ** 2 for p in policy_net.parameters() if p.grad is not None
            ) ** 0.5
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            accumulated = 0
            if u <= 10:
                print(f"  grad_norm={grad_norm:.6f}")

    # Flush any remaining accumulated gradients
    if accumulated > 0:
        grad_norm = sum(
            p.grad.norm() ** 2 for p in policy_net.parameters() if p.grad is not None
        ) ** 0.5
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        print(f"  grad_norm={grad_norm:.6f}")

    return total_ndcg / max(num_valid, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE training for PolicyNetwork sequence filter"
    )

    # Dataset / paths
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (looks for data/<dataset>.txt)')
    parser.add_argument('--sasrec_path', required=True,
                        help='Path to frozen SASRec checkpoint (.pth)')
    parser.add_argument('--save_dir', required=True,
                        help='Directory for policy checkpoints')

    # Filter / sequence lengths
    parser.add_argument('--max_length', type=int, required=True,
                        help='Filter output length (must be <= sasrec_maxlen)')
    parser.add_argument('--policy_maxlen', type=int, default=500,
                        help='Max input length for PolicyNetwork (default 500)')

    # PolicyNetwork architecture
    parser.add_argument('--hidden_units', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--attention', type=str, default='full', choices=['full', 'left', 'right'],
                        help='Attention mask for policy: full (bidirectional), left (causal), right (forward-looking)')
    parser.add_argument('--user_emb', action='store_true', default=False,
                        help='Add a per-user embedding to the policy input to reduce gradient interference across users')
    parser.add_argument('--norm_first', action='store_true', default=False,
                        help='Use pre-norm (norm_first) variant')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Rollouts per user before gradient accumulation step')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help='Users to accumulate before optimizer step')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--user_start_idx', type=int, default=0,
                        help='Start index into the sorted user list (inclusive, default: 0)')
    parser.add_argument('--user_end_idx', type=int, default=None,
                        help='End index into the sorted user list (exclusive, default: all users)')

    # Baseline filter
    parser.add_argument('--baseline', type=str, default='rloo',
                        choices=['uniform_random', 'from_end', 'rloo'],
                        help='Baseline for advantage computation: rloo (leave-one-out, default), '
                             'from_end, or uniform_random')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 = disabled, default 1.0)')
    parser.add_argument('--reward', type=str, default='ndcg10',
                        choices=['ndcg10', 'reciprocal_rank'],
                        help='Training reward: ndcg10 (sparse, binary above rank 10) or '
                             'reciprocal_rank (dense, always > 0). '
                             'Greedy evaluation always uses ndcg10. (default: ndcg10)')

    # SASRec loading args (must match the checkpoint)
    parser.add_argument('--sasrec_hidden_units', type=int, default=50)
    parser.add_argument('--sasrec_num_blocks', type=int, default=2)
    parser.add_argument('--sasrec_num_heads', type=int, default=1)
    parser.add_argument('--sasrec_maxlen', type=int, default=200)
    parser.add_argument('--sasrec_norm_first', action='store_true', default=False,
                        help='Whether the SASRec checkpoint used pre-norm variant')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--policy_type', type=str, default='standard',
                        choices=['standard', 'encoder_decoder'],
                        help='Policy architecture: standard (parallel scoring) or '
                             'encoder_decoder (auto-regressive selection)')

    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    run = wandb.init(
        entity="lars-h-hertel-personal",
        project="ml-1m-sequence-selection",
        config=vars(args),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/mean_reward", step_metric="epoch")
    wandb.define_metric("eval/greedy_ndcg", step_metric="epoch")
    wandb.define_metric("test/ndcg", step_metric="epoch")
    wandb.define_metric("test/ndcg_from_end", step_metric="epoch")

    if args.max_length > args.sasrec_maxlen:
        print(
            f"WARNING: max_length ({args.max_length}) > sasrec_maxlen ({args.sasrec_maxlen}). "
            "Positional embeddings in SASRec may be out of bounds.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print(f"Dataset: {usernum} users, {itemnum} items")

    # ------------------------------------------------------------------
    # 2. Load frozen SASRec
    # ------------------------------------------------------------------
    sasrec_args = argparse.Namespace(
        hidden_units=args.sasrec_hidden_units,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        maxlen=args.sasrec_maxlen,
        dropout_rate=0.0,   # dropout off for eval
        device=args.device,
        norm_first=args.sasrec_norm_first,
    )
    sasrec_model = SASRec(usernum, itemnum, sasrec_args).to(args.device)
    sasrec_model.load_state_dict(
        torch.load(args.sasrec_path, map_location=args.device)
    )
    sasrec_model.eval()
    for param in sasrec_model.parameters():
        param.requires_grad_(False)
    print(f"Loaded SASRec from {args.sasrec_path}")

    # ------------------------------------------------------------------
    # 3. Construct PolicyNetwork
    # ------------------------------------------------------------------
    policy_args = argparse.Namespace(
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        maxlen=args.policy_maxlen,
        max_length=args.max_length,
        dropout_rate=args.dropout_rate,
        device=args.device,
        norm_first=args.norm_first,
        attention=args.attention,
    )
    if args.policy_type == 'encoder_decoder':
        policy_net = EncoderDecoderPolicyNetwork(itemnum, policy_args).to(args.device)
    else:
        policy_net = PolicyNetwork(usernum if args.user_emb else 0, itemnum, policy_args).to(args.device)

    # ------------------------------------------------------------------
    # 4. Copy and freeze SASRec item embeddings
    # ------------------------------------------------------------------
    policy_net.item_emb.weight.data.copy_(sasrec_model.item_emb.weight.data)
    policy_net.item_emb.weight.requires_grad_(False)

    # ------------------------------------------------------------------
    # 5. Xavier-init all non-embedding policy parameters
    #    user_emb is excluded: it is zero-initialised in __init__ so that
    #    training starts user-agnostic and personalises gradually.
    # ------------------------------------------------------------------
    for name, param in policy_net.named_parameters():
        if 'item_emb' not in name and 'user_emb' not in name and param.requires_grad:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except Exception:
                pass  # skip 1-D tensors (biases, layernorm weights)

    # ------------------------------------------------------------------
    # 6. Adam optimizer over trainable parameters only
    # ------------------------------------------------------------------
    trainable_params = [p for p in policy_net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    print(
        f"PolicyNetwork: {sum(p.numel() for p in trainable_params):,} trainable parameters"
    )

    # ------------------------------------------------------------------
    # 7. Resolve baseline filter
    # For 'rloo' the per-sample baseline is computed inside train_epoch;
    # baseline_fn is unused but kept for the non-rloo code path.
    # ------------------------------------------------------------------
    if args.baseline == 'from_end':
        baseline_fn = from_end
    elif args.baseline == 'rloo':
        baseline_fn = from_end
    else:
        baseline_fn = uniform_random

    # ------------------------------------------------------------------
    # 8. Epoch loop
    # ------------------------------------------------------------------
    epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        policy_net.train()
        mean_ndcg = train_epoch(
            policy_net, sasrec_model, dataset, args, optimizer, baseline_fn
        )
        print(f"Epoch {epoch:3d} | mean reward = {mean_ndcg:.4f}")
        run.log({"train/mean_reward": mean_ndcg, "epoch": epoch})

        # Checkpoint
        ckpt_name = (
            f"policy.epoch={epoch}.maxlen={args.max_length}.user_end={args.user_end_idx}.ndcg={mean_ndcg:.4f}.pth"
        )
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
        torch.save(
            {
                'epoch': epoch,
                'policy_net_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ndcg': mean_ndcg,
                'args': vars(args),
            },
            ckpt_path,
        )

        # Update 'policy_latest.pth' symlink for easy resuming
        latest_path = os.path.join(args.save_dir, 'policy_latest.pth')
        if os.path.islink(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.abspath(ckpt_path), latest_path)

        print(f"           | saved {ckpt_path}")

        policy_net.eval()
        _, _, eval_ndcg_long, _, _, _ = evaluate_split_with_filter(
            sasrec_model, dataset, None, args.max_length,
            split="valid",
            filter_function=policy_net.filter,
            user_start_idx=args.user_start_idx,
            user_end_idx=args.user_end_idx,
            long_users_only=True,
        )
        print(f"Epoch {epoch:3d} | greedy eval NDCG@10 (long users) = {eval_ndcg_long:.4f}")
        run.log({"eval/greedy_ndcg": eval_ndcg_long, "epoch": epoch})
        if eval_ndcg_long == 1.0:
            break
        policy_net.train()

    # Eval on test
    # Policy
    policy_net.eval()
    _, _, test_ndcg_long, _, _, n_long = evaluate_split_with_filter(
        sasrec_model, dataset, None, args.max_length,
        split="test",
        filter_function=policy_net.filter,
        user_start_idx=args.user_start_idx,
        user_end_idx=args.user_end_idx,
        long_users_only=True,
    )
    print(f"Test NDCG@10 (policy, long users) = {test_ndcg_long:.4f}  n={n_long}")
    run.log({"test/ndcg": test_ndcg_long, "epoch": epoch})
    # From end
    _, _, from_end_ndcg_long, _, _, n_long = evaluate_split_with_filter(
        sasrec_model, dataset, None, args.max_length,
        split="test",
        filter_function=lambda seq, ml: from_end(seq, ml),
        user_start_idx=args.user_start_idx,
        user_end_idx=args.user_end_idx,
        long_users_only=True,
    )
    print(f"Test NDCG@10 (from_end, long users) = {from_end_ndcg_long:.4f}  n={n_long}")
    run.log({"test/ndcg_from_end": from_end_ndcg_long, "epoch": epoch})

    run.finish()


if __name__ == '__main__':
    main()
