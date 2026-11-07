"""verify D5-5, step 4: compare the claim's example match in the RAW tail-recovered source (read-only on E:)
with the 2024 tome copy. Shows whether the round-13 round_end flip is already in the source.

Reads only: E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\02\\12\\05d51edb-...\\{round_end,bomb_state,round_start,header}
(parquet files, read in place, nothing written under E:).
Also cross-checks the same match's tome rows (round_end, bomb_state, round_state freeze_end) and prints the
raw round_end rows for a control match from the same day (0098ba6e, agree case) for comparison.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_rawE.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
MATCHES = {
    "claim_example_05d51edb": "05d51edb-04bc-420c-91b9-4bb95317ca19",
    "control_0098ba6e": "0098ba6e-b5e0-4c62-9daf-a79a423eb54c",
}


def raw_table(mid, channel):
    d = RAW / mid
    f = d / channel
    return pl.read_parquet(f)


print("raw round_end schema:", raw_table(MATCHES["claim_example_05d51edb"], "round_end").schema)
print("raw bomb_state schema:", raw_table(MATCHES["claim_example_05d51edb"], "bomb_state").schema)

for label, mid in MATCHES.items():
    key = f"csds/2024/02/12/{mid}/csds"
    print(f"\n==== {label}  ({key}) ====")
    re_raw = raw_table(mid, "round_end")
    cols = [c for c in ["round", "tick", "winner_team_code", "win_reason_code", "player_count"] if c in re_raw.columns]
    print("RAW round_end rows:", re_raw.height)
    print(re_raw.select(cols).sort(["round", "tick"]).filter(pl.col("round").is_between(11, 15)) if "round" in cols
          else re_raw.head(20))
    bs_raw = raw_table(mid, "bomb_state")
    bcols = [c for c in ["round", "tick", "event_type", "player_team_code"] if c in bs_raw.columns]
    print("RAW bomb_state rows (round 11-15):")
    print(bs_raw.select(bcols).filter(pl.col("round").is_between(11, 15)).sort(["round", "tick"]))

    # same match in the tome copy (round_end, bomb, freeze_end from the extracted intermediates)
    re_t = pl.read_parquet(SCR / "re_2024.parquet").filter(pl.col("match_key") == key)
    fe_t = pl.read_parquet(SCR / "fe_2024.parquet").filter(pl.col("match_key") == key)
    bs_t = pl.read_parquet(SCR / "bomb_2024.parquet").filter(pl.col("match_key") == key)
    print("TOME round_end (11-15):")
    print(re_t.filter(pl.col("round").is_between(11, 15)).sort(["round", "tick"])
          .select(["round", "tick", "winner_team_code", "win_reason_code", "player_count"]))
    print("TOME freeze_end (11-15):")
    print(fe_t.filter(pl.col("round").is_between(11, 15)).sort("round").select(["round", "tick", "t_score", "ct_score"]))
    print("TOME bomb rows (11-15):")
    print(bs_t.filter(pl.col("round").is_between(11, 15)).sort(["round", "tick"]))

    # raw vs tome round_end comparison for all labels in the match
    if "round" in cols:
        a = re_raw.select(["round", "tick", "winner_team_code"]).rename({"tick": "raw_tick", "winner_team_code": "raw_w"})
        b = re_t.select(["round", "tick", "winner_team_code"]).rename({"tick": "tome_tick", "winner_team_code": "tome_w"})
        j = a.join(b, on="round", how="full", coalesce=True).sort("round")
        print("raw vs tome round_end, all labels; rows where they differ:")
        print(j.filter((pl.col("raw_w") != pl.col("tome_w")) | (pl.col("raw_tick") != pl.col("tome_tick"))))
        print("labels in raw:", a.height, " labels in tome:", b.height)

print("\nRAW DONE")
