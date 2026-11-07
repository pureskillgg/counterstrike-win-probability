"""verify D7-5 schema probe: column lists, event types, duplicate checks. Prints only (no files written)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
for ch in ["round_state", "round_start", "round_end", "bomb_state", "bomb_action", "bomb_defuse"]:
    fs = files(BASE / f"basic_{ch}{SUF}")
    print(f"== basic_{ch}: files={len(fs)}")
    print("   schema:", dict(pl.read_parquet_schema(fs[0])))
    lf = pl.concat([pl.scan_parquet(f) for f in fs])
    n = lf.select(pl.len()).collect().item()
    print("   rows:", n)
    if "event_type" in dict(pl.read_parquet_schema(fs[0])):
        print("   event_type counts:", lf.group_by("event_type").len().sort("event_type").collect().rows())
    if "player_count" in dict(pl.read_parquet_schema(fs[0])):
        print("   player_count dist:", lf.group_by("player_count").len().sort("player_count").collect().rows())
H = pl.concat([pl.read_parquet(f) for f in files(BASE / "header")])
print("== header rows", H.height, "| cols with score/build:", [c for c in H.columns if "score" in c or "build" in c or "date" in c or "map" in c or "key" in c or "id" in c])
print("   key unique:", H["key"].n_unique(), "| match_id unique:", H["match_id"].n_unique())
print("   tick_rate:", H["tick_rate"].value_counts().rows())
print("   t_starters_score_final sample:", H.select(["t_starters_score_final", "ct_starters_score_final", "number_of_points"]).head(8).rows())
print("   match_type:", H["match_type"].value_counts().rows())
rs = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")]).filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score", "tick"]).collect()
print("== freeze_end rows", rs.height, "| unique (match,round):", rs.select(["match_key", "round"]).unique().height)
RE = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_end{SUF}")]).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]).collect()
print("== round_end rows", RE.height, "| unique (match,round):", RE.select(["match_key", "round"]).unique().height)
print("   winner_team_code dist:", RE["winner_team_code"].value_counts().sort("winner_team_code").rows())
print("   win_reason pairs:", RE.group_by(["win_reason_code", "win_reason_message"]).len().sort("win_reason_code").rows())
print("   round_end player_count dist top:", RE["player_count"].value_counts().sort("len", descending=True).head(6).rows())
