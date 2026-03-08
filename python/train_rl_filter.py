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
--num_epochs 1 \
--num_samples 32 \
--dropout_rate 0.2 \
--baseline rloo \
--grad_clip 1.0 \
--gradient_accumulation_steps 16 \
--device cpu \
--entropy_coef 0.0 \
--seed 42 \
--reward reciprocal_rank \
--user_start_idx 0 --num_blocks 1 --attention right
"""

import os
import sys
import copy
import random
import argparse

import numpy as np
import torch
import wandb

from model import SASRec, PolicyNetwork
from filters import uniform_random, from_end, PolicyFilter
from utils import data_partition




# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(sasrec_model, u, filtered_seq, valid_item, rated, itemnum, max_length,
                   reward_type='ndcg10', truncate_at_10=False):
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
        if truncate_at_10 and rank >= 10:
            return 0.0
        return 1 / np.log2(rank + 2)


# Keep old name as alias for backwards compatibility
def compute_ndcg(sasrec_model, u, filtered_seq, valid_item, rated, itemnum, max_length):
    return compute_reward(sasrec_model, u, filtered_seq, valid_item, rated, itemnum,
                          max_length, reward_type='ndcg10')


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

        # Skip users whose history is already short enough or has no validation item
        if len(seq) <= args.max_length or not user_valid[u]:
            continue

        valid_item = user_valid[u][0]
        rated = set(user_train[u])

        # ------------------------------------------------------------------
        # Baseline reward (no gradient)
        # ------------------------------------------------------------------
        baseline_seq = baseline_fn(seq, args.max_length)
        baseline_reward = compute_reward(
            sasrec_model, u, baseline_seq, valid_item, rated, itemnum, args.max_length,
            reward_type=args.reward,
        )

        # ------------------------------------------------------------------
        # Policy forward pass (gradient flows through log_probs_all)
        # ------------------------------------------------------------------
        log_probs_all = policy_net.get_log_probs(seq, user_id=u)   # [min(L, policy_maxlen)]
        probs = log_probs_all.detach().exp()             # detached for multinomial
#         import pdb; pdb.set_trace()

        num_to_sample = min(args.max_length, probs.shape[0])
        if num_to_sample < args.max_length:
            # Sequence visible to policy is shorter than max_length; skip.
            continue

        # Offset to map policy-window indices back to original sequence positions
        offset = max(0, len(seq) - args.policy_maxlen)

        # ------------------------------------------------------------------
        # Sample negatives once per user (reused across all rollouts)
        # ------------------------------------------------------------------
        rated_set = rated | {0}
        item_idx = [valid_item]
        while len(item_idx) <= 100:
            t = np.random.randint(1, itemnum + 1)
            if t not in rated_set:
                item_idx.append(t)

        # ------------------------------------------------------------------
        # Collect num_samples rollouts — fully batched
        # ------------------------------------------------------------------
        # [num_samples, max_length] — all rollout item selections at once
        all_selected = torch.multinomial(
            probs.unsqueeze(0).expand(args.num_samples, -1),
            args.max_length, replacement=False,
        )
        all_selected, _ = all_selected.sort(dim=1)   # chronological order

        # [num_samples] — sum of selected log-probs for each rollout (differentiable)
        sample_log_probs_tensor = log_probs_all[all_selected].sum(dim=1)

        # Build batched SASRec input sequences
        seq_window_arr = np.array(seq[offset:], dtype=np.int32)   # [L_window]
        selected_np = all_selected.cpu().numpy()                   # [num_samples, max_length]
        batch_seqs = seq_window_arr[selected_np]                   # [num_samples, max_length]

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

        # Entropy of the policy distribution over all L items.
        # H(π) = -Σ p_i log p_i  (uses non-detached log_probs_all → differentiable)
        entropy = -(log_probs_all.exp() * log_probs_all).sum()

        if u <= 10:
            print(
                f"  u={u:5d}  baseline={baseline_display:.4f}"
                f"  reward_mean={rewards_arr.mean():.4f}"
                f"  reward_std={rewards_arr.std():.4f}"
                f"  reward_min={rewards_arr.min():.4f}"
                f"  reward_max={rewards_arr.max():.4f}"
                f"  entropy={entropy.item():.3f}"
            )
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        loss = -(advantages * sample_log_probs_tensor).mean() - args.entropy_coef * entropy
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
# Greedy evaluation
# ---------------------------------------------------------------------------

def evaluate_greedy(policy_net, sasrec_model, dataset, args):
    """Evaluate the policy using deterministic top-k selection (one rollout per user).

    Switches the policy to eval mode, selects items greedily via PolicyFilter,
    and computes mean NDCG@10 over all eligible users (or a 10 000-user sample).

    Args:
        policy_net:   PolicyNetwork (will be set to eval mode internally).
        sasrec_model: Frozen SASRec reward model.
        dataset:      Output of data_partition().
        args:         Parsed arguments (uses max_length, policy_maxlen).

    Returns:
        Mean greedy NDCG@10 (float).
    """
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    policy_net.eval()
    pf = PolicyFilter(policy_net)

    users = list(range(1, usernum + 1))

    total_ndcg = 0.0
    num_valid = 0

    for u in users[args.user_start_idx:args.user_end_idx]:
        seq = user_train[u]
        if len(seq) <= args.max_length or not user_valid[u]:
            continue

        valid_item = user_valid[u][0]
        rated = set(user_train[u])

        filtered_seq = pf.filter(seq, args.max_length, user_id=u)
        ndcg = compute_reward(
            sasrec_model, u, filtered_seq, valid_item, rated, itemnum, args.max_length,
            reward_type='ndcg10',  # always evaluate with NDCG@10 regardless of training reward
        )
#         print(f"  greedy  u={u:5d}  ndcg={ndcg:.4f}")
        total_ndcg += ndcg
        num_valid += 1

    return total_ndcg / max(num_valid, 1)


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test_with_filter(filter_fn, sasrec_model, dataset, args):
    """Evaluate a filter on the test split using deterministic top-k selection.

    History = train[u] + [valid[u][0]]; target = test[u][0].

    Args:
        filter_fn:    Callable(sequence, max_length, user_id) -> filtered list.
        sasrec_model: Frozen SASRec reward model.
        dataset:      Output of data_partition().
        args:         Parsed arguments.

    Returns:
        Tuple (ndcg, ndcg_all_users) where ndcg is the mean NDCG@10 for users
        whose history is longer than args.max_length, and ndcg_all_users is the
        mean NDCG@10 across all valid users.
    """
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    users = list(range(1, usernum + 1))

    total_ndcg = 0.0
    num_valid = 0
    total_ndcg_all = 0.0
    num_valid_all = 0

    for u in users[args.user_start_idx:args.user_end_idx]:
        if not user_train[u] or not user_valid[u] or not user_test[u]:
            continue

        history = user_train[u] + [user_valid[u][0]]
        test_item = user_test[u][0]
        rated = set(user_train[u]) | {user_valid[u][0]}

        filtered_seq = filter_fn(history, args.max_length, u)
        reward = compute_reward(
            sasrec_model, u, filtered_seq, test_item, rated, itemnum, args.max_length,
            reward_type='ndcg10', truncate_at_10=True,
        )

        total_ndcg_all += reward
        num_valid_all += 1

        if len(history) > args.max_length:
            total_ndcg += reward
            num_valid += 1

    return total_ndcg / max(num_valid, 1), total_ndcg_all / max(num_valid_all, 1)


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
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy bonus coefficient to prevent policy collapse (default 0.01)')
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
        dropout_rate=args.dropout_rate,
        device=args.device,
        norm_first=args.norm_first,
        attention=args.attention,
    )
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

    # Zero-init out_proj so that at initialisation logits are driven entirely
    # by position_bias (cumsum), making the untrained policy equivalent to from_end.
#     torch.nn.init.zeros_(policy_net.out_proj.weight)
#     torch.nn.init.zeros_(policy_net.out_proj.bias)

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
    # baseline_fn is still called for the diagnostic print (baseline_display).
    # ------------------------------------------------------------------
    if args.baseline == 'from_end':
        baseline_fn = from_end
    elif args.baseline == 'rloo':
        baseline_fn = from_end   # used only for diagnostic comparison print
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
            f"policy.epoch={epoch}.maxlen={args.max_length}.ndcg={mean_ndcg:.4f}.pth"
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

        if epoch % 5 == 0:
            eval_ndcg = evaluate_greedy(policy_net, sasrec_model, dataset, args)
            print(f"Epoch {epoch:3d} | greedy eval NDCG@10 = {eval_ndcg:.4f}")
            run.log({"eval/greedy_ndcg": eval_ndcg, "epoch": epoch})
            if eval_ndcg == 1.0:
                break
            policy_net.train()

        if epoch % 1 == 0:
            policy_net.eval()
            pf = PolicyFilter(policy_net)
            test_ndcg, test_ndcg_all = evaluate_test_with_filter(lambda seq, ml, u: pf.filter(seq, ml, user_id=u),
                                      sasrec_model, dataset, args)
            print(f"Epoch {epoch:3d} | test NDCG@10 = {test_ndcg:.4f}  (all users: {test_ndcg_all:.4f})")
            run.log({"test/ndcg": test_ndcg, "test/ndcg_all_users": test_ndcg_all, "epoch": epoch})
            policy_net.train()

    policy_net.eval()
    pf = PolicyFilter(policy_net)
    test_ndcg, test_ndcg_all = evaluate_test_with_filter(lambda seq, ml, u: pf.filter(seq, ml, user_id=u),
                               sasrec_model, dataset, args)
    print(f"Test NDCG@10 (policy) = {test_ndcg:.4f}  (all users: {test_ndcg_all:.4f})")
    run.log({"test/ndcg": test_ndcg, "test/ndcg_all_users": test_ndcg_all, "epoch": epoch})

    from_end_ndcg, from_end_ndcg_all = evaluate_test_with_filter(lambda seq, ml, u: from_end(seq, ml),
                                   sasrec_model, dataset, args)
    print(f"Test NDCG@10 (from_end) = {from_end_ndcg:.4f}  (all users: {from_end_ndcg_all:.4f})")
    run.log({"test/ndcg_from_end": from_end_ndcg, "test/ndcg_from_end_all_users": from_end_ndcg_all, "epoch": epoch})

    run.finish()


if __name__ == '__main__':
    main()
