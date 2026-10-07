"""
run_experiments.py - overnight batch runner for the SASRec targeted-promotion study.

For every seed it trains:
    * 1 clean model                          (baseline)
    * 1 poisoned model per (tier, budget)    (tail/mid/head  x  5%/10%/20%)
Each run is 200 epochs. Results are appended to ONE csv after every run, so
nothing is lost if the computer stops. Re-running the same command RESUMES:
finished runs are skipped.

Put this file, worker.py and poison_sasrec.py in the same folder as main.py, then:
    python run_experiments.py
"""
import csv
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time

from poison_sasrec import load, save, pick_target, pick_anchors, inject_fake_users

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# =========================== SETTINGS (edit these) ===========================
DATASET = "Beauty"                 # reads data/Beauty.txt
SEEDS = [0, 1, 2]                  # ~25 min per run -> 30 runs ~ 12 h. Use [0, 1] for ~8 h
BUDGETS = [0.05, 0.10, 0.20]
TIERS = ["tail", "mid", "head"]
NUM_EPOCHS = 200
MAX_USERS = 5000                   # users used to measure exposure
MAIN_ARGS = ["--maxlen=50", "--dropout_rate=0.5"]   # same args you used so far
#   add e.g. "--device=cuda" here if your main.py needs it
MASTER_SEED = 2024                 # picks the fixed targets/controls (same for all seeds)
SEQ_LEN, REPEATS, N_ANCHORS = 12, 3, 20            # poison shape (as in poison_sasrec.py)
RESULTS_CSV = "attack_results.csv"
# ============================================================================

TARGETS_FILE = "experiment_targets.json"
FIELDS = ["seed", "setting", "target_tier", "budget_pct", "n_fake_users",
          "target_id", "control_id",
          "target_exposure", "target_mean_rank", "target_median_rank", "target_n_users",
          "control_exposure", "control_mean_rank",
          "hit10", "ndcg10", "valid_hit10", "valid_ndcg10",
          "epochs", "train_minutes", "finished_at"]

os.makedirs("experiment_logs", exist_ok=True)
os.makedirs("experiment_results", exist_ok=True)


def choose_targets(clean):
    """One fixed target + one fixed (never poisoned) control per tier."""
    if os.path.exists(TARGETS_FILE):
        return json.load(open(TARGETS_FILE))
    rng = random.Random(MASTER_SEED)
    used, out = set(), {}
    for tier in TIERS:
        picks = []
        for _ in range(2):                       # target, control
            x = pick_target(clean, tier, rng)
            while x in used:
                x = pick_target(clean, tier, rng)
            used.add(x)
            picks.append(x)
        out[tier] = {"target": picks[0], "control": picks[1]}
    json.dump(out, open(TARGETS_FILE, "w"), indent=2)
    return out


def load_done():
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((int(r["seed"]), r["setting"], r["target_tier"],
                          int(round(float(r["budget_pct"])))))
    return done


def append_rows(rows):
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()


def train_and_eval(tag, dataset, train_dir, seed, items):
    """Runs worker.py (train 200 epochs + exposure eval). Returns dict or None."""
    spec_path = f"experiment_logs/{tag}.spec.json"
    out_path = f"experiment_results/{tag}.json"
    log_path = f"experiment_logs/{tag}.log"
    json.dump({"clean_data": f"data/{DATASET}.txt", "items": items}, open(spec_path, "w"))
    if os.path.exists(out_path):
        os.remove(out_path)

    cmd = [sys.executable, "worker.py", str(seed), spec_path, out_path, str(MAX_USERS),
           "main.py", f"--dataset={dataset}", f"--train_dir={train_dir}",
           f"--num_epochs={NUM_EPOCHS}"] + MAIN_ARGS
    t0 = time.time()
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    minutes = (time.time() - t0) / 60
    if rc != 0 or not os.path.exists(out_path):
        print(f"   !! FAILED ({tag}), see {log_path}")
        return None

    text = open(log_path, errors="ignore").read()
    m = re.findall(r"valid \(NDCG@10: ([\d.]+), HR@10: ([\d.]+)\), "
                   r"test \(NDCG@10: ([\d.]+), HR@10: ([\d.]+)\)", text)
    if not m:
        print(f"   !! could not find accuracy line in {log_path}")
        return None
    v_ndcg, v_hr, t_ndcg, t_hr = map(float, m[-1])
    return {"res": json.load(open(out_path)), "minutes": minutes,
            "valid_hit10": v_hr, "valid_ndcg10": v_ndcg,
            "hit10": t_hr, "ndcg10": t_ndcg}


def make_row(seed, setting, tier, pct, n_fake, tg, run):
    r = run["res"]
    t, c = r[f"{tier}_target"], r[f"{tier}_control"]
    return {
        "seed": seed, "setting": setting, "target_tier": tier, "budget_pct": pct,
        "n_fake_users": n_fake, "target_id": tg[tier]["target"], "control_id": tg[tier]["control"],
        "target_exposure": t["exposure"], "target_mean_rank": t["mean_rank"],
        "target_median_rank": t["median_rank"], "target_n_users": t["n_users"],
        "control_exposure": c["exposure"], "control_mean_rank": c["mean_rank"],
        "hit10": run["hit10"], "ndcg10": run["ndcg10"],
        "valid_hit10": run["valid_hit10"], "valid_ndcg10": run["valid_ndcg10"],
        "epochs": NUM_EPOCHS, "train_minutes": round(run["minutes"], 1),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    # ---- sanity checks ----
    if not os.path.exists(f"data/{DATASET}.txt"):
        sys.exit(f"data/{DATASET}.txt not found (run this from the folder that contains main.py)")
    if "EVAL BLOCK" in open("main.py", errors="ignore").read():
        sys.exit("Your main.py still contains the old exposure-evaluation block we added earlier.\n"
                 "Delete everything after print(\"Done\") in main.py (keep a backup copy), then rerun.\n"
                 "worker.py now does the evaluation for every run.")

    clean = load(f"data/{DATASET}.txt")
    tg = choose_targets(clean)
    print("Targets/controls:", json.dumps(tg))

    items = {}
    for tier in TIERS:
        items[f"{tier}_target"] = tg[tier]["target"]
        items[f"{tier}_control"] = tg[tier]["control"]

    done = load_done()
    start = time.time()
    n_run = 0

    for seed in SEEDS:
        # ---------- clean baseline (one training, evaluated on all 3 tiers) ----------
        if not all((seed, "clean", t, 0) in done for t in TIERS):
            print(f"\n[seed {seed}] CLEAN run")
            try:
                run = train_and_eval(f"{DATASET}_s{seed}_clean", DATASET, f"clean_s{seed}", seed, items)
            except Exception as e:
                print("   !! error:", e)
                run = None
            if run:
                append_rows([make_row(seed, "clean", t, 0, 0, tg, run) for t in TIERS])
                n_run += 1
                print(f"   done in {run['minutes']:.1f} min  hit@10={run['hit10']:.4f} ndcg@10={run['ndcg10']:.4f}")

        # ---------- poisoned runs ----------
        for tier in TIERS:
            for budget in BUDGETS:
                pct = int(round(budget * 100))
                if (seed, "poisoned", tier, pct) in done:
                    continue
                tag = f"{DATASET}_s{seed}_{tier}_{pct}"
                print(f"\n[seed {seed}] POISONED tier={tier} budget={pct}%")
                try:
                    anchors = pick_anchors(clean, tg[tier]["target"], N_ANCHORS)
                    rng = random.Random(f"poison-{seed}-{tier}-{pct}")
                    poisoned, n_fake = inject_fake_users(
                        clean, tg[tier]["target"], anchors, budget, SEQ_LEN, REPEATS, rng)
                    save(f"data/{tag}.txt", poisoned)
                    run = train_and_eval(tag, tag, "run", seed, items)
                except Exception as e:
                    print("   !! error:", e)
                    run = None
                if run:
                    append_rows([make_row(seed, "poisoned", tier, pct, n_fake, tg, run)])
                    n_run += 1
                    r = run["res"][f"{tier}_target"]
                    elapsed = (time.time() - start) / 3600
                    print(f"   done in {run['minutes']:.1f} min  target exposure={r['exposure']:.4f} "
                          f"mean rank={r['mean_rank']:.0f}  hit@10={run['hit10']:.4f}  "
                          f"(elapsed {elapsed:.1f} h, {n_run} runs this session)")

    print(f"\nAll finished. Results in {RESULTS_CSV}")
    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_CSV)
        cols = ["target_exposure", "target_mean_rank", "control_exposure", "hit10", "ndcg10"]
        print(df.groupby(["target_tier", "setting", "budget_pct"])[cols].mean().round(4))
    except Exception:
        pass


if __name__ == "__main__":
    main()
