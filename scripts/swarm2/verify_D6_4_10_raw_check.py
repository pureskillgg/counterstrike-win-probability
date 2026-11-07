"""D6-4 verifier, step 10: read the RAW E: records for a few affected round-13 matches (read-only) and
compare them with the tome, to see which record is mis-written.

Raw layout (verified by listing): E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\02\\<DD>\\<match_id>\\<channel>
Each channel is a single Parquet file with no extension (magic PAR1). Only reads. Writes nothing under E:.
Per match, prints: raw round_end row for round 13 vs tome round_end row; raw bomb rows for round 13;
raw round_freeze_end scores for rounds 12-14; raw CT deaths before the defuse tick.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(12)
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def tome(name):
    return pl.scan_parquet(str([p for p in base.iterdir() if p.name.startswith(f"basic_{name}.")][0] / "dataframe_*"))

x = pl.read_parquet(str(SCR / "verify_D6_4_physical_2024_r13.parquet"))
pick = []
for day in ["12", "13", "15", "17", "20", "24"]:
    sub = x.filter(pl.col("aff") & (pl.col("day") == day) if "day" in x.columns else pl.col("aff") & (pl.col("win") == "d12-25"))
    sub = sub.filter(pl.col("match_key").str.contains(f"/2024/02/{day}/"))
    if sub.height:
        pick.append(sub.row(0, named=True))
# add one control (bomb == round_end) from day 12
ctl = x.filter((~pl.col("aff")) & pl.col("match_key").str.contains("/2024/02/12/"))
if ctl.height:
    pick.append(ctl.row(0, named=True))

re_tome = tome("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
bs_tome = tome("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
for p in pick:
    mk = p["match_key"]
    mid = mk.split("/")[3]
    day = mk.split("/")[2 + 1] if False else mk.split("/")[3 - 1]
    day = mk.split("/")[3]  # placeholder replaced below
    parts = mk.split("/")  # csds, 2024, 02, DD, match_id, csds
    day = parts[3]
    mid = parts[4]
    folder = RAW / day / mid
    kind = "AFFECTED" if p["aff"] else "CONTROL"
    print(f"\n=== {kind}  day {day}  match {mid}  bomb={p['bomb_side']} round_end(tome)={p['winner_team_code']} score={p['score_side']}  defuse_tick={p['defuse_tick']} plant_tick={p['plant_tick']}")
    if not folder.exists():
        print("  raw folder missing:", folder)
        continue
    try:
        rre = pl.read_parquet(str(folder / "round_end")).select([c for c in ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message"] if True])
    except Exception as ex:
        print("  raw round_end read error:", type(ex).__name__, str(ex)[:200])
        continue
    if p == pick[0]:
        print("  raw round_end columns:", rre.columns)
    r13_raw = rre.filter(pl.col("round") == 13)
    r13_tome = re_tome.filter((pl.col("match_key") == mk) & (pl.col("round") == 13))
    print("  RAW round_end r13:", r13_raw.select(["tick", "winner_team_code", "win_reason_code"]).rows())
    print("  TOME round_end r13:", r13_tome.select(["tick", "winner_team_code", "win_reason_code"]).rows())
    try:
        rbomb = pl.read_parquet(str(folder / "bomb_state"))
        rb13 = rbomb.filter(pl.col("round") == 13).select([c for c in ["tick", "event_type", "player_team_code"] if c in rbomb.columns])
        print("  RAW bomb_state r13:", rb13.sort("tick").rows())
    except Exception as ex:
        print("  raw bomb_state error:", type(ex).__name__, str(ex)[:200])
    print("  TOME bomb_state r13:", bs_tome.filter((pl.col("match_key") == mk) & (pl.col("round") == 13)).sort("tick").select(["tick", "event_type", "player_team_code"]).rows())
    try:
        rs = pl.read_parquet(str(folder / "round_state"))
        fe = rs.filter((pl.col("event_type") == "round_freeze_end") & pl.col("round").is_in([12, 13, 14])).select(["round", "tick", "t_score", "ct_score"]).sort("round")
        print("  RAW freeze_end scores r12-14:", fe.rows())
    except Exception as ex:
        print("  raw round_state error:", type(ex).__name__, str(ex)[:200])
