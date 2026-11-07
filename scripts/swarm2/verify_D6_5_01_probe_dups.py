"""D6-5 verifier probe: duplicates in round_end / round_state, win-reason vocabulary (2024 tome)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
def scan(ch):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
RE = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count", "legacy_code"])
print("round_end rows", RE.height, "distinct (match,round)", RE.select(["match_key", "round"]).unique().height)
d = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"),
                                            pl.col("tick").n_unique().alias("nt")).filter(pl.col("n") > 1)
print("duplicated (match,round) keys:", d.height, " of which conflicting winner:", int((d["nw"] > 1).sum()) if d.height else 0)
print(d.group_by(["n", "nw"]).len().sort(["n", "nw"]) if d.height else "none")
print("round_end round distribution of duplicates:", d.group_by("round").len().sort("round").to_dicts()[:30] if d.height else "none")
print("\nwin_reason vocabulary (code, message, winner_team_code counts):")
print(RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
print("\nplayer_count distribution in round_end:", RE.group_by("player_count").len().sort("player_count").to_dicts())
RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick", "t_score", "ct_score"])
print("\nround_freeze_end rows", RS.height, "distinct", RS.select(["match_key", "round"]).unique().height)
dd = RS.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("t_score").n_unique().alias("nt"), pl.col("ct_score").n_unique().alias("nc")).filter(pl.col("n") > 1)
print("freeze_end duplicated keys:", dd.height, " with conflicting scores:", int(((dd["nt"] > 1) | (dd["nc"] > 1)).sum()) if dd.height else 0)
print("event types in round_state:", scan("round_state").group_by("event_type").len().sort("len", descending=True).to_dicts())
