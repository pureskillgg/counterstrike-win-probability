"""Round-2 paired test (pinned rule): per-second paired difference vs the GRU control, seed-matched,
match-cluster bootstrap (1,000 resamples of matches, RNG seed 0). Validation rows only.
Adopt if the pooled difference <= -0.0005 and every seed's 95% upper bound < 0."""
import json
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/final_results"
LOC = "G:/pureskill/csds-cs2/data/diagnostics"
VOL = SCR + "/vol_preds"

PATHS = {
    "gru": {
        0: f"{LOC}/pos24_screen_gru/gru_seed0_final_val_preds.parquet",
        1: f"{LOC}/pos24_gru_s12/gru_seed1_final_val_preds.parquet",
        2: f"{LOC}/pos24_gru_s12/gru_seed2_final_val_preds.parquet",
        3: f"{VOL}/pos24_gru_s34/gru_seed3_final_val_preds.parquet",
        4: f"{VOL}/pos24_gru_s34/gru_seed4_final_val_preds.parquet",
    },
    "xlstm_fast": {
        0: f"{VOL}/pos24_xlstmfast_s012/xlstm_fast_seed0_final_val_preds.parquet",
        1: f"{VOL}/pos24_xlstmfast_s012/xlstm_fast_seed1_final_val_preds.parquet",
        2: f"{VOL}/pos24_xlstmfast_s012/xlstm_fast_seed2_final_val_preds.parquet",
        3: f"{VOL}/pos24_xlstmfast_s3/xlstm_fast_seed3_final_val_preds.parquet",
        4: f"{LOC}/pos24_xlstm_fast_s4/xlstm_fast_seed4_final_val_preds.parquet",
    },
    "xlstm_blk": {
        0: f"{LOC}/pos24_round2_screen_laptop/xlstm_blk_seed0_final_val_preds.parquet",
        1: f"{VOL}/pos24_blk_s12/xlstm_blk_seed1_final_val_preds.parquet",
        2: f"{VOL}/pos24_blk_s12/xlstm_blk_seed2_final_val_preds.parquet",
        3: f"{VOL}/pos24_blk_s34/xlstm_blk_seed3_final_val_preds.parquet",
        4: f"{VOL}/pos24_blk_s34/xlstm_blk_seed4_final_val_preds.parquet",
    },
    "xlstm_sig": {
        0: f"{LOC}/pos24_round2_screen_laptop/xlstm_sig_seed0_final_val_preds.parquet",
        1: f"{LOC}/pos24_sig_s12_laptop/xlstm_sig_seed1_final_val_preds.parquet",
        2: f"{LOC}/pos24_sig_s12_laptop/xlstm_sig_seed2_final_val_preds.parquet",
        3: f"{VOL}/pos24_sig_s3/xlstm_sig_seed3_final_val_preds.parquet",
        4: f"{LOC}/pos24_sig_s4_laptop/xlstm_sig_seed4_final_val_preds.parquet",
    },
}
SEEDS = [0, 1, 2, 3, 4]
ARMS = ["xlstm_fast", "xlstm_blk", "xlstm_sig"]
EPS = 1e-7


def load(path):
    t = pq.read_table(path, columns=["match_key", "second", "round", "map_name", "label", "p"])
    return t


ref_t = load(PATHS["gru"][0])
keys_ref = ref_t.column("match_key")
uniq, inv = np.unique(np.asarray(keys_ref.to_pylist(), dtype=object), return_inverse=True)
M = len(uniq)
N_ROWS = ref_t.num_rows
print("validation rows", N_ROWS, "matches", M)


def loss_and_check(arm, seed):
    t = load(PATHS[arm][seed])
    # alignment: same rows in the same order as the GRU seed-0 reference
    same = (
        pc.all(pc.equal(t.column("match_key"), keys_ref)).as_py()
        and pc.all(pc.equal(t.column("second"), ref_t.column("second"))).as_py()
        and pc.all(pc.equal(t.column("label"), ref_t.column("label"))).as_py()
    )
    assert same, f"row alignment failed for {arm} seed {seed}"
    y = t.column("label").to_numpy().astype(np.float64)
    p = np.clip(t.column("p").to_numpy().astype(np.float64), EPS, 1 - EPS)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss


# per-match sums of the per-second difference and per-match row counts
match_counts = np.bincount(inv, minlength=M).astype(np.float64)
S = {arm: {} for arm in ARMS}
check = {}
gru_mean = {}
for s in SEEDS:
    g = loss_and_check("gru", s)
    gru_mean[s] = float(g.mean())
    for arm in ARMS:
        a = loss_and_check(arm, s)
        check[(arm, s)] = float(a.mean())
        d = a - g
        S[arm][s] = np.bincount(inv, weights=d, minlength=M)
    print("seed", s, "gru", round(gru_mean[s], 6), {arm: round(check[(arm, s)], 6) for arm in ARMS}, flush=True)

# sanity: recomputed means vs the trainer's final_val_log_loss in the metrics files
print("\ncheck: recomputed mean vs trainer final_val_log_loss (metrics files)")
for arm in ARMS + ["gru"]:
    for s in SEEDS:
        pth = PATHS[arm][s].replace("_final_val_preds.parquet", "_metrics.json")
        try:
            rec = json.load(open(pth))["final_val_log_loss"]
        except FileNotFoundError:
            rec = float("nan")
        mine = gru_mean[s] if arm == "gru" else check[(arm, s)]
        print(f"  {arm:<11} seed {s}: recomputed {mine:.6f}  trainer {rec:.6f}  diff {mine - rec:+.2e}")

# bootstrap: resample matches with replacement, 1,000 draws, RNG seed 0 (same draws for every seed/arm)
rng = np.random.default_rng(0)
B = 1000
draws = rng.integers(0, M, size=(B, M))
W = np.stack([np.bincount(draws[b], minlength=M) for b in range(B)]).astype(np.float64)
Nw = W @ match_counts  # resampled row counts per draw


def stats(sum_per_match):
    boot = (W @ sum_per_match) / Nw
    point = sum_per_match.sum() / match_counts.sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


results = {}
print("\n== Paired per-second difference vs GRU control (negative = arm better) ==")
for arm in ARMS:
    per_seed = {}
    for s in SEEDS:
        pt, lo, hi = stats(S[arm][s])
        per_seed[s] = {"diff": pt, "lo": lo, "hi": hi}
    pooled_sum = np.mean([S[arm][s] for s in SEEDS], axis=0)
    p_pt, p_lo, p_hi = stats(pooled_sum)
    all_upper_neg = all(per_seed[s]["hi"] < 0 for s in SEEDS)
    adopt = (p_pt <= -0.0005) and all_upper_neg
    results[arm] = {"pooled": {"diff": p_pt, "lo": p_lo, "hi": p_hi}, "per_seed": per_seed,
                    "all_seed_upper_below_0": all_upper_neg, "adopt": adopt}
    print(f"{arm}: pooled {p_pt:+.6f} [95% {p_lo:+.6f}, {p_hi:+.6f}]  "
          f"every seed upper<0: {all_upper_neg}  ADOPT: {adopt}")
    for s in SEEDS:
        r = per_seed[s]
        print(f"    seed {s}: {r['diff']:+.6f} [{r['lo']:+.6f}, {r['hi']:+.6f}]")

json.dump({"gru_mean": gru_mean, "arm_mean": {f"{a}|{s}": v for (a, s), v in check.items()},
           "results": results}, open(SCR + "/paired_round2.json", "w"), indent=1, default=float)
print("\nwrote", SCR + "/paired_round2.json")
