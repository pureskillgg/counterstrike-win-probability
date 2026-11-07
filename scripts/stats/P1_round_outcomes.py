"""P1 part 1: pistol-round outcomes from the 2026 tome (all matches in basic_round_end).

- CT win rate in round 1 and round 13, plus round counts.
- Momentum: P(win R2 | won R1) and P(win R14 | won R13), compared with the base rate
  P(win round r | lost round r-1) for r in 3..12 and 15..24.
- Match-cluster bootstrap: resample matches with replacement, 1000 draws, seed 0.
  The same match-level weight draws are used for every statistic, so differences are paired.
- Side labels: 2 = T, 3 = CT (winner_team_code). Pairs (r-1, r) used for momentum stay in one half,
  so the side that lost round r-1 and the side that won it refer to the same team identity.
"""
import glob
import os

import numpy as np
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
B = 1000
SEED = 0
T_CODE, CT_CODE = 2, 3
SET_S = list(range(3, 13)) + list(range(15, 25))  # base-rate rounds


def load_round_end():
    folder = glob.glob(os.path.join(TOME, "basic_round_end.*"))[0]
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return (pl.scan_parquet(files)
            .select(["match_key", "round", "winner_team_code", "tick"])
            .collect())


def fmt(x):
    return f"{x:.4f}" if x == x else "nan"


def main():
    re_df = load_round_end()
    print("== Source checks ==")
    print("round_end rows:", re_df.height)
    print("distinct matches with round_end:", re_df["match_key"].n_unique())
    dup = re_df.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("duplicate (match, round) groups:", dup.height)
    print("winner_team_code counts:", sorted(re_df["winner_team_code"].value_counts().rows()))
    print("round number min/max:", re_df["round"].min(), re_df["round"].max())
    per_match = (re_df.group_by("match_key")
                 .agg(pl.col("round").max().alias("max_round"), pl.len().alias("n_rounds")))
    print("matches where max round != number of round rows (gaps):",
          per_match.filter(pl.col("max_round") != pl.col("n_rounds")).height)
    nr = per_match["n_rounds"].to_numpy()
    print("rounds per match: min %d p25 %.0f median %.0f p75 %.0f max %d mean %.2f" % (
        nr.min(), np.percentile(nr, 25), np.median(nr), np.percentile(nr, 75), nr.max(), nr.mean()))
    has_r1 = re_df.filter(pl.col("round") == 1)["match_key"].n_unique()
    has_r13 = re_df.filter(pl.col("round") == 13)["match_key"].n_unique()
    has_r2 = re_df.filter(pl.col("round") == 2)["match_key"].n_unique()
    has_r14 = re_df.filter(pl.col("round") == 14)["match_key"].n_unique()
    print("matches with round 1:", has_r1, " round 2:", has_r2, " round 13:", has_r13, " round 14:", has_r14)

    keys = sorted(re_df["match_key"].unique().to_list())
    M = len(keys)
    idx_map = pl.DataFrame({"match_key": keys}).with_row_index("midx")
    j = re_df.join(idx_map, on="match_key", how="left")
    RMAX = int(re_df["round"].max())
    W = np.zeros((M, RMAX + 1), dtype=np.int16)  # 0 = missing
    W[j["midx"].to_numpy(), j["round"].to_numpy()] = j["winner_team_code"].to_numpy().astype(np.int16)
    valid = (W == T_CODE) | (W == CT_CODE)
    print("match universe M =", M, " valid round cells =", int(valid.sum()))

    # match-cluster bootstrap weights (same for all statistics)
    rng = np.random.default_rng(SEED)
    Wb = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        Wb[b] = np.bincount(rng.integers(0, M, M), minlength=M)

    def point_and_boot_rate(midx, num):
        pt = float(num.mean()) if len(num) else float("nan")
        samp = np.empty(B)
        for b in range(B):
            rw = Wb[b][midx]
            den = rw.sum()
            samp[b] = (rw * num).sum() / den if den > 0 else np.nan
        lo, hi = np.nanpercentile(samp, [2.5, 97.5])
        return pt, lo, hi, samp

    def report(label, midx, num):
        pt, lo, hi, _ = point_and_boot_rate(midx, num)
        n_m = len(np.unique(midx))
        print(f"{label:<62} n_rows={len(num):>6} n_matches={n_m:>5}  rate={fmt(pt)}  95% CI [{fmt(lo)}, {fmt(hi)}]")
        return pt, lo, hi, n_m, len(num)

    print("\n== 1. Pistol-round CT win rate (all matches) ==")
    res = {}
    for r in (1, 13):
        midx = np.where(valid[:, r])[0]
        num = (W[midx, r] == CT_CODE).astype(float)
        res[f"ct_r{r}"] = report(f"CT win rate, round {r}", midx, num)
    # reference: all rounds 1..24
    mm, rr = np.nonzero(valid[:, 1:25])
    rr = rr + 1
    num_all = (W[mm, rr] == CT_CODE).astype(float)
    res["ct_all"] = report("CT win rate, all rounds 1..24 (reference)", mm, num_all)
    for r in (1, 13):
        midx = np.where(valid[:, r])[0]
        print(f"  round {r} rounds: CT wins={int((W[midx, r] == CT_CODE).sum())}, "
              f"T wins={int((W[midx, r] == T_CODE).sum())}, total={len(midx)}")

    print("\n== 2. Momentum after pistol rounds ==")
    # R2 given R1 and R14 given R13
    for (a, b) in ((1, 2), (13, 14)):
        midx = np.where(valid[:, a] & valid[:, b])[0]
        same = (W[midx, b] == W[midx, a]).astype(float)  # side that won a also wins b
        report(f"P(win R{b} | won R{a})  [same side wins]", midx, same)
        for side, lab in ((CT_CODE, "CT"), (T_CODE, "T")):
            sel = W[midx, a] == side
            report(f"   ... of which {lab} won R{a}", midx[sel], same[sel])

    # base rate: side coming off a loss in rounds 3..12, 15..24
    base_m, base_loss, base_win = [], [], []
    per_round = []
    for r in SET_S:
        midx = np.where(valid[:, r - 1] & valid[:, r])[0]
        loss_win = (W[midx, r] != W[midx, r - 1]).astype(float)  # side that lost r-1 wins r
        base_m.append(midx)
        base_loss.append(loss_win)
        base_win.append(1.0 - loss_win)
        per_round.append((r, len(midx), float(loss_win.mean()) if len(midx) else float("nan")))
    base_m = np.concatenate(base_m)
    base_loss = np.concatenate(base_loss)
    base_win = np.concatenate(base_win)
    print("per-round base rate P(win r | lost r-1), point estimate:")
    print("  " + "; ".join(f"R{r}:{v:.3f}(n={n})" for r, n, v in per_round))
    report("Base: P(win r | lost r-1), r in 3..12,15..24 (pooled)", base_m, base_loss)
    report("Base: P(win r | won r-1), r in 3..12,15..24 (pooled, reference)", base_m, base_win)

    # paired differences: momentum minus base-after-loss, with the same bootstrap draws
    def paired_diff(label, mid_a, num_a, mid_b, num_b):
        pt_a = float(num_a.mean())
        pt_b = float(num_b.mean())
        samp = np.empty(B)
        for b in range(B):
            wa = Wb[b][mid_a]
            wb = Wb[b][mid_b]
            samp[b] = (wa * num_a).sum() / wa.sum() - (wb * num_b).sum() / wb.sum()
        lo, hi = np.percentile(samp, [2.5, 97.5])
        print(f"{label:<62} diff={pt_a - pt_b:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        return pt_a - pt_b, lo, hi

    m2 = np.where(valid[:, 1] & valid[:, 2])[0]
    s2 = (W[m2, 2] == W[m2, 1]).astype(float)
    m14 = np.where(valid[:, 13] & valid[:, 14])[0]
    s14 = (W[m14, 14] == W[m14, 13]).astype(float)
    print("paired differences (momentum minus base-after-loss):")
    paired_diff("R2 momentum minus base after loss", m2, s2, base_m, base_loss)
    paired_diff("R14 momentum minus base after loss", m14, s14, base_m, base_loss)
    # also: pooled momentum (R2 and R14 together) vs base
    mp = np.concatenate([m2, m14])
    sp = np.concatenate([s2, s14])
    paired_diff("pooled R2+R14 momentum minus base after loss", mp, sp, base_m, base_loss)
    # reference: the same momentum rates against the base for a side coming off a win (= 1 - base_loss)
    paired_diff("R2 momentum minus base after win (reference)", m2, s2, base_m, base_win)
    paired_diff("R14 momentum minus base after win (reference)", m14, s14, base_m, base_win)
    paired_diff("pooled R2+R14 momentum minus base after win (reference)", mp, sp, base_m, base_win)

    print("\n== 3. Sample-of-pistol-rounds weight check ==")
    print("pistol matches with both R1 and R13 valid:", int((valid[:, 1] & valid[:, 13]).sum()))


if __name__ == "__main__":
    main()
