"""D1-9 verification, part 3: independent per-round player count from player_spawn (read-only).

Purpose: rule out that player_info is undercounting, which would make pc=20 a correct count
for rounds that list 9 or 4 players. player_spawn is a separate channel (one row per player
spawn per round). Compared against round_end.player_count and player_info.
Also computes the exact size of the 'halve pc and keep pc/2 == 10' failure set (2024).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(200)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
S24 = ".2023-12-28,2024-02-25"
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S26 = ".2026-09-18,2026-10-08"
T_CODE, CT_CODE = 2, 3


def lf(base, suf, ch):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch} under {base}")
    return pl.scan_parquet(fs)


def hdr(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def run(name, base, suf):
    hdr(f"{name}: player_spawn vs round_end.player_count vs player_info")
    RE = lf(base, suf, "round_end").select("match_key", "round", "player_count").collect()
    RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
    PI = lf(base, suf, "player_info").select("match_key", "round", "player_id_fixed", "team_code").collect()
    PR = PI.group_by(["match_key", "round"]).agg(pl.col("player_id_fixed").n_unique().alias("n_pi"))
    SP = lf(base, suf, "player_spawn").select("match_key", "round", "player_id", "player_id_fixed",
                                              "player_team_code").collect()
    print(f"{name} player_spawn rows: {SP.height}; null player_id: {SP['player_id'].null_count()}; "
          f"team codes: {SP['player_team_code'].value_counts().sort('player_team_code').to_dicts()}")
    SR = SP.group_by(["match_key", "round"]).agg(
        pl.col("player_id").n_unique().alias("n_sp"),
        pl.col("player_id").filter(pl.col("player_team_code") == T_CODE).n_unique().alias("n_sp_t"),
        pl.col("player_id").filter(pl.col("player_team_code") == CT_CODE).n_unique().alias("n_sp_ct"),
        pl.len().alias("sp_rows"),
    )
    MSP = SP.group_by("match_key").agg(pl.col("player_id").n_unique().alias("m_sp"))
    J = (RE.join(PR, on=["match_key", "round"], how="left")
           .join(SR, on=["match_key", "round"], how="left")
           .join(MSP, on="match_key", how="left"))
    n_total = J.height
    print(f"{name} unique rounds: {n_total}; with spawn rows: {J['n_sp'].is_not_null().sum()}; "
          f"with player_info: {J['n_pi'].is_not_null().sum()}")

    print(f"{name} crosstab pc x n_sp (top 30):")
    print(J.group_by(["player_count", "n_sp"]).len().sort("len", descending=True).head(30))

    K = J.filter(pl.col("n_sp").is_not_null())
    n = K.height
    eq_sp = int((K["player_count"] == K["n_sp"]).sum())
    eq_2sp = int((K["player_count"] == 2 * K["n_sp"]).sum())
    print(f"{name} rounds with spawn rows: {n}; pc == n_sp: {eq_sp} ({100*eq_sp/n:.2f}%); "
          f"pc == 2 x n_sp: {eq_2sp} ({100*eq_2sp/n:.2f}%)")
    e5 = int(((K["n_sp_t"] == 5) & (K["n_sp_ct"] == 5)).sum())
    print(f"{name} rounds with exactly 5 T and 5 CT by spawn: {e5} ({100*e5/n:.2f}%)")
    print(f"{name} rounds with n_sp == 10: {int((K['n_sp'] == 10).sum())}")

    if name == "2024":
        p20 = K.filter(pl.col("player_count") == 20)
        print(f"2024 pc==20 rounds with spawn rows: {p20.height}; n_sp distribution:")
        print(p20.group_by("n_sp").len().sort("n_sp"))
        p20_not10 = p20.filter(pl.col("n_sp") != 10).height
        print(f"2024 pc==20 rounds whose spawn count is not 10: {p20_not10} "
              f"({100*p20_not10/p20.height:.2f}% of pc==20 rounds with spawn data)")
        print("2024 rounds with n_sp != 10 (any pc) and their pc values:")
        print(K.filter(pl.col("n_sp") != 10).group_by(["n_sp", "player_count"]).len()
              .sort("len", descending=True).head(25))
        print("2024 rounds with n_sp == 10 and their pc values:")
        print(K.filter(pl.col("n_sp") == 10).group_by("player_count").len().sort("len", descending=True).head(10))
        # wingman-like matches (m_sp == 4)
        mm = J.filter(pl.col("m_sp") == 4).select("match_key").unique()
        print(f"2024 matches with m_sp == 4 (whole-match distinct spawned players): {mm.height}")
        mmp = (J.filter(pl.col("m_sp") == 4).group_by("match_key")
               .agg(pl.col("player_count").mode().first().alias("pc_mode"),
                    pl.col("n_sp").max().alias("max_nsp")))
        print("  pc_mode distribution in m_sp==4 matches:", mmp["pc_mode"].value_counts().sort("count", descending=True).to_dicts())
        print("  max n_sp per match distribution:", mmp["max_nsp"].value_counts().sort("max_nsp").to_dicts())
        # m_sp distribution across all 2024 matches
        print("2024 m_sp distribution across matches (distinct spawned players per match):")
        print(J.group_by("match_key").agg(pl.col("m_sp").first()).group_by("m_sp").len().sort("m_sp").to_dicts())

        # Halving-normalization failure set: pc/2 == 10 keeps all pc==20 rounds
        all20 = J.filter(pl.col("player_count") == 20)
        not5v5_pi = all20.filter((pl.col("n_pi") != 10) | pl.col("n_pi").is_null()).height
        not5v5_sp = all20.filter((pl.col("n_sp") != 10) | pl.col("n_sp").is_null()).height
        print(f"2024 halving rule (pc/2 == 10): kept pc==20 rounds = {all20.height}; "
              f"of these, not 10 distinct players by player_info (or missing): {not5v5_pi}; "
              f"not 10 distinct spawned players (or missing): {not5v5_sp}")
        exact_5v5_pi = all20.filter((pl.col("n_pi") == 10)).height
        print(f"2024 pc==20 rounds with exactly 10 distinct player_info ids: {exact_5v5_pi}")
    else:
        w = K.filter(pl.col("player_count") == 4)
        print(f"2026 pc==4 rounds with spawn rows: {w.height}; n_sp distribution: "
              f"{w.group_by('n_sp').len().sort('n_sp').to_dicts()}")
        p10 = K.filter(pl.col("player_count") == 10)
        print(f"2026 pc==10 rounds with spawn rows: {p10.height}; n_sp distribution: "
              f"{p10.group_by('n_sp').len().sort('n_sp').to_dicts()}")
        print("2026 pc==10 rounds with n_sp != 10: ", p10.filter(pl.col("n_sp") != 10).height)


run("2024", T24, S24)
run("2026", T26, S26)
print("\nDONE")
