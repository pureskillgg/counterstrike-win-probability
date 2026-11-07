"""P1 part 3: win rate by pistol type using within-team comparisons, plus sample-vs-tome selection check.

Because the default pistols identify the side (T = glock, CT = usp_silencer), a modal-type win rate is
mechanically the side's win rate. Here each pistol type X is compared within the same team:
side-rounds where at least one player held X versus side-rounds where nobody did.
Universe: the 150 matches of the 1 Hz sample. Pistol rounds 1 and 13. Match-cluster bootstrap, 1000 draws, seed 0.
"""
import glob
import os
from collections import Counter

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
SAMPLE = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
ROUNDS = os.path.join(ROOT, "data", "state_sample", "rounds.parquet")
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
B = 1000
SEED = 0
T, CT = 2, 3
TYPES = [(4, "glock"), (61, "usp_silencer"), (32, "hkp2000 (P2000)"), (1, "deagle"), (2, "elite (dual berettas)"),
         (30, "tec9"), (3, "fiveseven"), (36, "p250"), (63, "cz75a"), (64, "revolver (R8)")]


def chan_files(name):
    folder = glob.glob(os.path.join(TOME, name + ".*"))[0]
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


def main():
    cols = ["match_key", "round", "tick", "team_code", "money", "equipment_value_calc", "armor",
            "has_helmet", "inv_secondary"]
    lf = pl.scan_parquet(SAMPLE).select(cols)
    t0 = lf.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    fs_all = lf.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0")).collect()
    fs = fs_all.filter(pl.col("round").is_in([1, 13]))
    rd = pl.read_parquet(ROUNDS).select(["match_key", "round", "ct_win"]).to_dicts()
    ct_win = {(x["match_key"], x["round"]): x["ct_win"] for x in rd}

    # universe = all 150 sample matches (same as P1_pistol_sample.py), not only those with a pistol round
    matches = sorted(fs_all["match_key"].unique().to_list())
    M = len(matches)
    midx_of = {m: i for i, m in enumerate(matches)}
    rng = np.random.default_rng(SEED)
    Wb = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        Wb[b] = np.bincount(rng.integers(0, M, M), minlength=M)

    # side-level records
    sides = {}
    for x in fs.to_dicts():
        sides.setdefault((x["match_key"], x["round"], x["team_code"]), []).append(x)
    recs = []
    for (m, r, t), plist in sides.items():
        codes = [p["inv_secondary"] for p in plist]
        winner = CT if ct_win[(m, r)] == 1 else T
        recs.append(dict(midx=midx_of[m], round=r, team=t, win=int(t == winner), codes=Counter(codes), n=len(plist)))
    print("side-rounds:", len(recs), " matches:", M)

    print("\n== Within-team comparison: sides with >=1 player holding X vs sides with none (pooled R1 and R13) ==")
    print("(difference = rate(has X) - rate(no X), same team; bootstrap CI by match)")
    for t, tl in ((T, "T"), (CT, "CT")):
        base = [r for r in recs if r["team"] == t]
        print(f"-- {tl}: all sides n={len(base)} win rate={np.mean([r['win'] for r in base]):.4f}")
        for code, name in TYPES:
            has = [r for r in base if r["codes"].get(code, 0) >= 1]
            no = [r for r in base if r["codes"].get(code, 0) == 0]
            players = sum(r["codes"].get(code, 0) for r in base)
            if len(has) == 0 or len(no) == 0:
                print(f"   {name:<22} players={players:>4} sides_has={len(has):>4} sides_no={len(no):>4} (not estimable)")
                continue
            mh = np.array([r["midx"] for r in has])
            yh = np.array([r["win"] for r in has], dtype=float)
            mn = np.array([r["midx"] for r in no])
            yn = np.array([r["win"] for r in no], dtype=float)
            pt = yh.mean() - yn.mean()
            samp = np.empty(B)
            for b in range(B):
                wh = Wb[b][mh]
                wn = Wb[b][mn]
                if wh.sum() == 0 or wn.sum() == 0:
                    samp[b] = np.nan
                    continue
                samp[b] = (wh * yh).sum() / wh.sum() - (wn * yn).sum() / wn.sum()
            lo, hi = np.nanpercentile(samp, [2.5, 97.5])
            print(f"   {name:<22} players={players:>4} sides_has={len(has):>4} rate_has={yh.mean():.3f} "
                  f"sides_no={len(no):>4} rate_no={yn.mean():.3f} diff={pt:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ----------------------------------------------------------------
    print("\n== Sample-vs-tome check: pistol-round CT win rate ==")
    rw = pl.read_parquet(ROUNDS).select(["match_key", "round", "ct_win"])
    re_df = (pl.scan_parquet(chan_files("basic_round_end"))
             .select(["match_key", "round", "winner_team_code", "player_count"]).collect())
    for r in (1, 13):
        s = rw.filter(pl.col("round") == r)
        print(f"sample (rounds.parquet) round {r}: n={s.height} CT win rate={s['ct_win'].mean():.4f}")
        t_all = re_df.filter(pl.col("round") == r)
        cw_all = (t_all["winner_team_code"] == CT).mean()
        t10 = t_all.filter(pl.col("player_count") == 10)
        cw10 = (t10["winner_team_code"] == CT).mean()
        print(f"tome (round_end) round {r}: all n={t_all.height} CT rate={cw_all:.4f}; "
              f"player_count==10 n={t10.height} CT rate={cw10:.4f}")
    pc = re_df["player_count"].to_numpy()
    print("tome round_end player_count distribution (all rounds): ",
          sorted(Counter(pc.tolist()).items())[:12])
    print("\n== Done ==")


if __name__ == "__main__":
    main()
