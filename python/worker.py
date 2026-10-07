"""
worker.py  (called by run_experiments.py, you don't run it by hand)

1. Sets all random seeds.
2. Runs your main.py (training) unchanged, in this same process.
3. Grabs the trained `model`, `args`, `itemnum` from main.py's globals.
4. Computes exposure@10 / mean rank / median rank for several items in ONE pass
   over the users (each user is scored once, then every item is ranked).
5. Writes the numbers to a JSON file.

Usage (done for you by the driver):
    python worker.py SEED SPEC_JSON OUT_JSON MAX_USERS main.py --dataset=... [more main.py args]
"""
import json
import random
import runpy
import sys

import numpy as np
import torch

from poison_sasrec import load


def evaluate_items(g, clean, items, k=10, max_users=5000):
    model, args, itemnum = g["model"], g["args"], g["itemnum"]
    model.eval()

    rng = random.Random(0)                      # same users for every run
    users = list(clean)
    if len(users) > max_users:
        users = rng.sample(users, max_users)

    candidates = np.arange(1, itemnum + 1)
    ranks = {name: [] for name in items}

    with torch.no_grad():
        for u in users:
            hist = clean[u][:-1]                # same input SASRec sees at test time
            if len(hist) == 0:
                continue
            seq = np.zeros([args.maxlen], dtype=np.int32)
            h = hist[-args.maxlen:]
            seq[-len(h):] = h

            scores = model.predict(np.array([0]), np.array([seq]), candidates)
            scores = scores[0].detach().cpu().numpy().astype(np.float64)
            scores = np.concatenate([[-np.inf], scores])   # index == item id

            seen = set(hist)
            scores[list(seen)] = -np.inf        # never recommend seen items
            for name, item in items.items():
                if item in seen:                # user already has this item
                    continue
                ranks[name].append(int((scores > scores[item]).sum()) + 1)

    out = {}
    for name, r in ranks.items():
        r = np.array(r)
        out[name] = {
            "item": items[name],
            "exposure": float((r <= k).mean()) if len(r) else None,
            "mean_rank": float(r.mean()) if len(r) else None,
            "median_rank": float(np.median(r)) if len(r) else None,
            "n_users": int(len(r)),
        }
    return out


def main():
    seed = int(sys.argv[1])
    spec = json.load(open(sys.argv[2]))
    out_path = sys.argv[3]
    max_users = int(sys.argv[4])
    main_script = sys.argv[5]
    main_args = sys.argv[6:]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sys.argv = [main_script] + main_args       # main.py parses these
    g = runpy.run_path(main_script, run_name="__main__")

    clean = load(spec["clean_data"])           # ORIGINAL users only
    results = evaluate_items(g, clean, spec["items"], k=10, max_users=max_users)
    for name, r in results.items():
        print("EXPOSURE", name, r)
    json.dump(results, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
