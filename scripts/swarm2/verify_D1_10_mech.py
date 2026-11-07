"""D1-10 verifier: mechanism for the claim's round-27 / 33 / 39 / 45 disagreements.
Prints compact tuples (no wide tables). Reads the joined table written by
verify_D1_10_recompute.py and the 2026 round_state / score_update rows for affected matches.
"""
import glob
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


J = pl.read_parquet(SCRATCH / "verify_D1_10_joined_2026.parquet")
pc10 = J.filter(pl.col("player_count") == 10)
bad = pc10.filter(pl.col("dis_claim"))

# --- A. Off-by-one test: does claim_side match winner of r-1 or r+1 in the same match?
W = J.select(["match_key", "round", "winner_team_code"])
bw = bad.join(W.with_columns((pl.col("round") + 1).alias("round")).rename({"winner_team_code": "w_prev"}),
              on=["match_key", "round"], how="left")
bw = bw.join(W.with_columns((pl.col("round") - 1).alias("round")).rename({"winner_team_code": "w_next"}),
             on=["match_key", "round"], how="left")
print("A. off-by-one: claim_side == winner of round r+1 :", int((bw["claim_side"] == bw["w_prev"]).sum()), "of", bw.height)
print("   claim_side == winner of round r-1 :", int((bw["claim_side"] == bw["w_next"]).sum()), "of", bw.height)

# --- B. Which rounds have more than one +1 score_update team (ambiguous su side)?
SU = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
                for f in fl("score_update")])
inc = SU.filter((pl.col("new_score") - pl.col("old_score")) == 1)
amb = inc.group_by(["match_key", "round"]).agg(pl.col("team_code").n_unique().alias("nteam"),
                                               pl.len().alias("ninc"))
amb_pc = amb.filter(pl.col("nteam") > 1)
print("\nB. rounds with >1 distinct +1 team in score_update (all pc):", amb_pc.height)
print("   round numbers of those (counts):", sorted(Counter(amb_pc["round"].to_list()).items())[:30])
amb_keys = amb_pc.select(["match_key", "round"])
overlap = bad.join(amb_keys, on=["match_key", "round"], how="inner")
print("   overlap with the 100 claim disagreements:", overlap.height)
print("   ninc distribution among ambiguous rounds:", sorted(Counter(amb_pc["ninc"].to_list()).items()))

# --- C. Example match trajectory (rounds 25..29), compact
ex = bad.sort(["round", "match_key"]).row(0, named=True)["match_key"]
print("\nC. example match:", ex.split("/")[-1][:8])
re_ex = (pl.scan_parquet(fl("round_end")).select(["match_key", "round", "winner_team_code", "player_count"])
         .filter(pl.col("match_key") == ex).collect().sort("round"))
print("   round_end (round, winner, pc):", [(r, w, p) for r, w, p in
      re_ex.filter(pl.col("round").is_between(24, 30)).select(["round", "winner_team_code", "player_count"]).iter_rows()])
su_ex = SU.filter((pl.col("match_key") == ex) & pl.col("round").is_between(24, 30)).sort(["round", "tick"])
print("   score_update (round, tick, team, old->new):")
for r in su_ex.iter_rows(named=True):
    print("     ", r["round"], r["tick"], r["team_code"], f'{r["old_score"]}->{r["new_score"]}')
rs_all = pl.scan_parquet(fl("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key") == ex).collect()
rs_ex = rs_all.filter(pl.col("round").is_between(26, 28)).sort(["round", "tick"])
print("   round_state rows (round, tick, event, t_score, ct_score):")
for r in rs_ex.iter_rows(named=True):
    print("     ", r["round"], r["tick"], r["event_type"], r["t_score"], r["ct_score"])

# --- D. Claim disagreement mechanism: t/ct deltas for a sample of the 100 rows
print("\nD. sample of claim disagreement rows (match8, round, winner, claim, su_side, bomb, t_before->t_after, ct_before->ct_after, n_rs):")
RS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "t_score", "ct_score"]).collect()
                for f in fl("round_state")])
sc2 = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t_before"),
    pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
    pl.col("t_score").sort_by("tick").last().alias("t_after"),
    pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
sample = bad.sort(["round", "match_key"]).head(12).join(sc2, on=["match_key", "round"], how="left")
for r in sample.iter_rows(named=True):
    print("  ", r["match_key"].split("/")[-1][:8], r["round"], r["winner_team_code"], r["claim_side"], r["su_side"],
          r["bomb_side"], f'{r["t_before"]}->{r["t_after"]}', f'{r["ct_before"]}->{r["ct_after"]}', r["n_rs_rows"])

# --- E. Per-round summary of claim disagreements (ratio of rounds with su_nteam>1 to all rounds at same number)
print("\nE. per round-number: rounds, claim-side present, claim disagreements, su-ambiguous")
rn = pc10.group_by("round").agg(pl.len().alias("n"), pl.col("claim_side").is_not_null().sum().alias("n_claim"),
                                pl.col("dis_claim").sum().alias("n_dis"))
amb_rn = amb_pc.group_by("round").len().rename({"len": "n_amb"})
rn = rn.join(amb_rn, on="round", how="left").fill_null(0).sort("round")
print([r for r in rn.filter(pl.col("round").is_in([9, 13, 25, 26, 27, 28, 29, 33, 39, 45])).iter_rows()])
