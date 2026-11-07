# verify A1-2: read-only compare of raw E: parquet vs tome for match m_36b24642 (E: is only read, never written)
import os, glob
import polars as pl
BASE = r"E:\csds_final\csds\2026\10\01\m_36b246426c63e2e81c3165c3"
MK = "csds/2026/10/01/m_36b246426c63e2e81c3165c3/csds"
TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def raw(ch):
    p = os.path.join(BASE, ch)
    return pl.read_parquet(p)
def tome(ch):
    pages = sorted(glob.glob(os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    return pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == MK).collect()
rh = raw("header")
print("RAW header cols with starters/max_rounds:", [c for c in rh.columns if "starter" in c or c in ("max_rounds", "match_date", "key", "match_id")])
print(rh.select([c for c in rh.columns if "starter" in c or c in ("max_rounds", "match_date", "key")]).to_dicts())
th = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds"]).filter(pl.col("key") == MK)
print("TOME header:", th.to_dicts())
rr = raw("round_end").sort("round")
tr = tome("round_end").sort("round")
print("RAW round_end:", rr.select(["round", "tick", "winner_team_code", "win_reason_code"]).rows())
print("TOME round_end:", tr.select(["round", "tick", "winner_team_code", "win_reason_code"]).rows())
rs = raw("score_update").sort("tick"); ts = tome("score_update").sort("tick")
print("RAW score_update (round,tick,team_code,old,new):", rs.select(["round", "tick", "team_code", "old_score", "new_score"]).rows())
print("TOME score_update rows:", ts.height, "RAW rows:", rs.height)
rtc = raw("team_change"); ttc = tome("team_change")
print("RAW team_change cols:", rtc.columns[:12])
print("RAW team_change (round,tick,player_id,old,new):", rtc.select([c for c in ["round", "tick", "player_id", "old_team_code", "new_team_code"] if c in rtc.columns]).rows())
print("TOME team_change rows:", ttc.height)
rpi = raw("player_info").sort(["round", "player_id"]); tpi = tome("player_info").sort(["round", "player_id"])
print("RAW player_info rows", rpi.height, "TOME player_info rows", tpi.height)
print("RAW player_info (round,player_id,team_code) round1-2 & 9:", rpi.filter(pl.col("round").is_in([1, 2, 9])).select(["round", "player_id", "team_code"]).rows())
print("TOME player_info (round,player_id,team_code) round1-2 & 9:", tpi.filter(pl.col("round").is_in([1, 2, 9])).select(["round", "player_id", "team_code"]).rows())
rpd = raw("player_death").sort("tick"); print("RAW player_death rows", rpd.height, "TOME", tome("player_death").height)
