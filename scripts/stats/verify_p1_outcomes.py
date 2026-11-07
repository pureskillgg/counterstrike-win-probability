"""Independent verifier for P1 section 1 (round outcomes) from the 2026 tome.

Recomputes, from basic_round_end parquet only:
  - CT win rate in R1, R13 and R1-24 (reference)
  - P(win R2 | won R1), P(win R14 | won R13)
  - Base P(win r | lost r-1) and P(win r | won r-1), r in 3-12 and 15-24
  - Momentum minus base (after loss and after win)

Uncertainty: match-cluster bootstrap over ALL matches in round_end,
1000 draws, numpy default_rng(0). Each draw resamples matches, not rounds.

Log: G:\\pureskill\\csds-cs2\\logs\\stats\\verify_p1.log (truncated here, appended by verify_p1_sample.py)
"""
import glob
import os

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
LOG = os.path.join(ROOT, "logs", "stats", "verify_p1.log")
N_BOOT = 1000
SEED = 0
T_CODE, CT_CODE = 2, 3


def log(msg=""):
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ci_of(boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("")

    files = sorted(glob.glob(os.path.join(TOME, "basic_round_end.*", "dataframe_*")))
    log("== verify_p1_outcomes.py ==")
    log(f"round_end parquet files: {len(files)}")
    lf = pl.concat(
        [pl.scan_parquet(f).select(["match_key", "round", "winner_team_code"]) for f in files]
    )
    df = lf.collect()
    n_rows = df.height
    n_unique_pairs = df.select(["match_key", "round"]).unique().height
    keys = sorted(df["match_key"].unique().to_list())
    M = len(keys)
    rmax = int(df["round"].max())
    log(f"rows={n_rows}  unique(match,round)={n_unique_pairs}  matches={M}  max round={rmax}")
    codes = df["winner_team_code"].value_counts().sort("winner_team_code")
    log(f"winner_team_code counts: {codes.rows()}")
    rm_min = int(df["round"].min())
    log(f"round min={rm_min}")

    idx = {k: i for i, k in enumerate(keys)}
    W = np.zeros((M, rmax + 2), dtype=np.int8)  # 0 = missing
    for k, r, w in df.select(["match_key", "round", "winner_team_code"]).iter_rows():
        W[idx[k], r] = w
    has = W > 0
    ct = W == CT_CODE

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, M, size=(N_BOOT, M))

    def ratio_stat(num_vec, den_vec):
        num_vec = num_vec.astype(np.float64)
        den_vec = den_vec.astype(np.float64)
        est = num_vec.sum() / den_vec.sum()
        boot = num_vec[draws].sum(axis=1) / den_vec[draws].sum(axis=1)
        lo, hi = ci_of(boot)
        return est, lo, hi, int(den_vec.sum()), int(num_vec.sum()), boot

    results = {}

    # Section 1 headline items
    num = ct[:, 1]
    den = has[:, 1]
    results["ct_r1"] = ratio_stat(num & den, den)
    num = ct[:, 13]
    den = has[:, 13]
    results["ct_r13"] = ratio_stat(num & den, den)

    num_ct = np.zeros(M)
    den_ct = np.zeros(M)
    for r in range(1, 25):
        num_ct += (ct[:, r] & has[:, r]).astype(np.float64)
        den_ct += has[:, r].astype(np.float64)
    results["ct_r1_24"] = ratio_stat(num_ct, den_ct)

    def same_winner(a, b):
        m = has[:, a] & has[:, b]
        return (W[:, a] == W[:, b]) & m, m

    num, den = same_winner(1, 2)
    results["win2_given_win1"] = ratio_stat(num, den)
    num, den = same_winner(13, 14)
    results["win14_given_win13"] = ratio_stat(num, den)

    # Base, pooled over r in 3-12 and 15-24: pair (r-1, r) within the same half
    base_rounds = list(range(3, 13)) + list(range(15, 25))
    num_lost = np.zeros(M)
    den_base = np.zeros(M)
    num_won = np.zeros(M)
    for r in base_rounds:
        m = has[:, r - 1] & has[:, r]
        den_base += m.astype(np.float64)
        num_lost += ((W[:, r] != W[:, r - 1]) & m).astype(np.float64)
        num_won += ((W[:, r] == W[:, r - 1]) & m).astype(np.float64)
    results["base_after_loss"] = ratio_stat(num_lost, den_base)
    results["base_after_win"] = ratio_stat(num_won, den_base)

    def show(name, label):
        est, lo, hi, d, n, _ = results[name]
        log(f"{label:<44} n={d:>6}  events={n:>6}  rate={100*est:6.2f}%  CI {100*lo:6.2f} to {100*hi:6.2f}")

    log("")
    log("-- Round outcome rates (match-cluster bootstrap, 1000 draws, seed 0) --")
    show("ct_r1", "CT win, round 1")
    show("ct_r13", "CT win, round 13")
    show("ct_r1_24", "CT win, rounds 1-24 (reference)")
    show("win2_given_win1", "P(win R2 | won R1)")
    show("win14_given_win13", "P(win R14 | won R13)")
    show("base_after_loss", "Base P(win r | lost r-1), r in 3-12,15-24")
    show("base_after_win", "Base P(win r | won r-1), reference")

    log("")
    log("-- Momentum minus base (bootstrap CI on the same draws) --")
    m2 = results["win2_given_win1"]
    m14 = results["win14_given_win13"]
    bl = results["base_after_loss"]
    bw = results["base_after_win"]
    for label, mom, base_name in [
        ("R2 minus base after loss", m2, "bl"),
        ("R14 minus base after loss", m14, "bl"),
        ("R2 minus base after win", m2, "bw"),
        ("R14 minus base after win", m14, "bw"),
    ]:
        base = bl if base_name == "bl" else bw
        est = 100 * (mom[0] - base[0])
        boot = 100 * (mom[5] - base[5])
        lo, hi = ci_of(boot)
        log(f"{label:<32} {est:+6.2f} points  CI {lo:+6.2f} to {hi:+6.2f}")

    log("")
    log("-- R2 split by R1 winner side (n and rate) --")
    w1 = W[:, 1]
    w2 = W[:, 2]
    m = has[:, 1] & has[:, 2]
    for side, code in [("CT won R1", CT_CODE), ("T won R1", T_CODE)]:
        sel = m & (w1 == code)
        n_side = int(sel.sum())
        k_side = int((sel & (w2 == code)).sum())
        log(f"{side:<12} n={n_side}  R2 won by same side: {k_side} = {100*k_side/n_side:.2f}%")

    log("")
    log("-- R14 split by R13 winner side (n and rate) --")
    w13 = W[:, 13]
    w14 = W[:, 14]
    m = has[:, 13] & has[:, 14]
    for side, code in [("CT won R13", CT_CODE), ("T won R13", T_CODE)]:
        sel = m & (w13 == code)
        n_side = int(sel.sum())
        k_side = int((sel & (w14 == code)).sum())
        log(f"{side:<12} n={n_side}  R14 won by same side: {k_side} = {100*k_side/n_side:.2f}%")

    log("")
    log("-- Counts used in the claims --")
    log(f"matches with R1: {int(has[:,1].sum())}; R2: {int(has[:,2].sum())}; R13: {int(has[:,13].sum())}; R14: {int(has[:,14].sum())}")
    log(f"matches with R1 and R2: {int((has[:,1]&has[:,2]).sum())}; R13 and R14: {int((has[:,13]&has[:,14]).sum())}")
    missing13 = int((has[:, 12] & ~has[:, 13]).sum())
    log(f"matches with R12 but no R13: {missing13}")
    log(f"matches with R1 but no R2: {int((has[:,1]&~has[:,2]).sum())}")


if __name__ == "__main__":
    main()
