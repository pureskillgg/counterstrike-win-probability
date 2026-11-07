"""D1-10 verifier: row-level look at the 100 claim disagreements and their neighbours.
Reads the joined table written by verify_D1_10_recompute.py (scratchpad) and the 2026 tome
score_update / round_state rows for the affected matches only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


J = pl.read_parquet(SCRATCH / "verify_D1_10_joined_2026.parquet")
pc10 = J.filter(pl.col("player_count") == 10)
bad = pc10.filter(pl.col("dis_claim"))
print("disagreeing rows:", bad.height)
print("su_nteam > 1 anywhere (pc=10):", int((pc10["su_nteam"].fill_null(1) > 1).sum()),
      " su_n_inc != 1:", int((pc10["su_n_inc"].fill_null(1) != 1).sum()))
print("disagreeing rows with a bomb side:", int(bad["bomb_side"].is_not_null().sum()),
      " bomb agrees with winner there:", int((bad["bomb_side"] == bad["winner_team_code"]).sum()))
print("disagreeing rows: su side == winner:", int((bad["su_side"] == bad["winner_team_code"]).sum()), "of", bad.height)
print("disagreeing rows: claim side == other team (2 vs 3) :",
      int((bad["claim_side"] == (5 - bad["winner_team_code"])).sum()))
print("distinct matches with a disagreement:", bad["match_key"].n_unique())
print("disagreements per match (top 10):")
print(bad.group_by("match_key").len().sort("len", descending=True).head(10))

print("\n== the 100 rows (claim vs winner vs su vs bomb)")
print(bad.select(["match_key", "round", "winner_team_code", "claim_side", "su_side", "bomb_side",
                  "win_reason_code", "ev_first", "ev_last", "n_rs_rows"]).sort(["round", "match_key"]).head(40))

# Which matches: pull all round_state and score_update rows for these matches only
mk = bad["match_key"].unique().to_list()
print("\nmatches to pull:", len(mk))
rs_all = pl.scan_parquet(fl("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key").is_in(mk)).collect()
su_all = pl.scan_parquet(fl("score_update")).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).filter(pl.col("match_key").is_in(mk)).collect()
re_all = pl.scan_parquet(fl("round_end")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).filter(pl.col("match_key").is_in(mk)).collect()

# One example match: the first disagreeing match, rounds 25..30
ex = mk[0]
print("\n== example match:", ex)
print("round_end rows (rounds 24..34):")
print(re_all.filter((pl.col("match_key") == ex) & pl.col("round").is_between(24, 34)).sort("round"))
print("score_update rows (rounds 24..34):")
print(su_all.filter((pl.col("match_key") == ex) & pl.col("round").is_between(24, 34)).sort(["round", "tick"]))
print("round_state rows per round (rounds 26..28), event types and scores:")
rsx = rs_all.filter((pl.col("match_key") == ex) & pl.col("round").is_between(26, 28)).sort(["round", "tick"])
print(rsx.select(["round", "tick", "event_type", "t_score", "ct_score"]))
