"""
Targeted-promotion data poisoning for SASRec (offline robustness study).

Usage (poisoning):
    python poison_sasrec.py poison --data data/Beauty.txt --out data/Beauty_p1.txt --budget 0.01 --target-tier tail --mode inject --seed 0

Then train the official SASRec repo on --out (same hyperparameters as the clean
run) and call exposure_rate() with the trained model's scoring function.

Input format: one "user item" pair per line, sorted by time, 1-indexed IDs
(this is what the official repo's preprocessed files look like).
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np


# ----------------------------------------------------------------- data utils
def load(path):
    seqs = defaultdict(list)
    with open(path) as f:
        for line in f:
            u, i = map(int, line.split())
            seqs[u].append(i)
    return seqs


def save(path, seqs):
    with open(path, "w") as f:
        for u in sorted(seqs):
            for i in seqs[u]:
                f.write(f"{u} {i}\n")


def train_part(seq):
    """SASRec split: last = test, second-to-last = validation, rest = train."""
    return seq[:-2]


# ----------------------------------------------------------- target / anchors
def pick_target(seqs, tier, rng):
    """Choose a target item by popularity tier: head, mid, or tail."""
    cnt = Counter(i for s in seqs.values() for i in s)
    ranked = [i for i, _ in cnt.most_common()]
    n = len(ranked)
    lo, hi = {"head": (0.0, 0.05), "mid": (0.4, 0.6), "tail": (0.8, 1.0)}[tier]
    return rng.choice(ranked[int(lo * n): max(int(hi * n), int(lo * n) + 1)])


def pick_anchors(seqs, target, k=20):
    """Items that most often appear as the LAST TRAINING item of real users
    (the context SASRec relies on most, per Fig. 4 / Sec. III-C)."""
    last_train = Counter()
    for s in seqs.values():
        tr = train_part(s)
        if tr and tr[-1] != target:
            last_train[tr[-1]] += 1
    return [i for i, _ in last_train.most_common(k)]


# ---------------------------------------------------------------- the attacks
def inject_fake_users(seqs, target, anchors, budget, seq_len, repeats, rng):
    """Create fake users whose training portion contains repeated
    'anchor -> target' transitions. Two filler items are appended at the end
    so the poison lands in TRAIN (SASRec holds out the last two items)."""
    cnt = Counter(i for s in seqs.values() for i in s)
    items, weights = zip(*cnt.items())
    n_fake = max(1, int(budget * len(seqs)))
    max_uid = max(seqs)
    out = dict(seqs)
    for k in range(1, n_fake + 1):
        n_pairs = repeats
        n_filler = max(seq_len - 2 * n_pairs, 1)
        seq = rng.choices(items, weights=weights, k=n_filler)  # popularity-like filler
        for _ in range(n_pairs):
            seq += [rng.choice(anchors), target]
        seq += rng.choices(items, weights=weights, k=2)  # becomes val/test, discarded
        # keep >=5 actions so fake users survive the paper's preprocessing filter
        out[max_uid + k] = seq
    return out, n_fake


def insert_into_real(seqs, target, anchors, budget, rng):
    """Alternative: insert target right after an anchor in a fraction of real
    histories (profile pollution)."""
    out = {u: list(s) for u, s in seqs.items()}
    users = rng.sample(list(out), max(1, int(budget * len(out))))
    anchor_set = set(anchors)
    touched = 0
    for u in users:
        tr_len = len(out[u]) - 2
        pos = [j for j in range(tr_len) if out[u][j] in anchor_set]
        if pos:
            out[u].insert(rng.choice(pos) + 1, target)
            touched += 1
    return out, touched


# ----------------------------------------------------------------- evaluation
def exposure_rate(score_fn, seqs, target, k=10, max_users=5000, seed=0):
    """Fraction of users with `target` in their top-k over all unseen items,
    plus the target's mean rank. Evaluate on ORIGINAL users only."""
    rng = random.Random(seed)
    users = list(seqs)
    if len(users) > max_users:
        users = rng.sample(users, max_users)
    hits, ranks = 0, []
    for u in users:
        hist = seqs[u][:-1]            # same input SASRec sees at test time
        if len(hist) == 0:             # nothing to recommend from
            continue
        if target in set(hist):        # user already has the target
            continue
        scores = score_fn(hist).astype(float)
        scores[0] = -np.inf                    # ignore padding
        scores[list(set(hist))] = -np.inf      # ignore items already seen
        rank = int((scores > scores[target]).sum()) + 1
        ranks.append(rank)
        hits += rank <= k
    return hits / max(len(ranks), 1), float(np.mean(ranks))


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poison")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=0.01, help="fraction of |U|")
    p.add_argument("--target-tier", choices=["head", "mid", "tail"], default="tail")
    p.add_argument("--target", type=int, default=None, help="override target item id")
    p.add_argument("--mode", choices=["inject", "insert"], default="inject")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3, help="anchor->target pairs per fake user")
    p.add_argument("--n-anchors", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    seqs = load(a.data)
    target = a.target or pick_target(seqs, a.target_tier, rng)
    anchors = pick_anchors(seqs, target, a.n_anchors)

    if a.mode == "inject":
        poisoned, n = inject_fake_users(seqs, target, anchors, a.budget,
                                        a.seq_len, a.repeats, rng)
        print(f"Injected {n} fake users ({a.budget:.2%} of {len(seqs)})")
    else:
        poisoned, n = insert_into_real(seqs, target, anchors, a.budget, rng)
        print(f"Modified {n} real users")

    save(a.out, poisoned)
    meta = {"target": target, "anchors": anchors, "n_clean_users": len(seqs),
            "args": vars(a)}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"target={target}  anchors[:5]={anchors[:5]}  -> {a.out}")


if __name__ == "__main__":
    main()
