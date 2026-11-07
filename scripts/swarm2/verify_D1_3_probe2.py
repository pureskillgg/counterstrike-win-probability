"""Verifier probe 2: round_state event types, round_start coverage, header key, round_end duplicate. Read-only."""
import glob
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")

RE = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
d = RE.filter(pl.struct(["match_key", "round"]).is_duplicated())
print("round_end dup rows:\n", d)
RST = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"])
print("round_state rows", RST.height, "event_type counts:\n", RST.group_by("event_type").len().sort("len", descending=True))
print("phase counts:\n", RST.group_by("phase").len().sort("len", descending=True).head(8))
RSTA = rd("round_start", ["match_key", "round", "tick"])
print("round_start rows", RSTA.height, "unique", RSTA.select(["match_key","round"]).unique().height)
print("round_end unique", RE.select(["match_key","round"]).unique().height, "matches", RE["match_key"].n_unique())
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "match_date", "match_id"]) for f in hf], how="vertical_relaxed")
print("header rows", H.height, "unique key", H["key"].n_unique())
print("header sample:\n", H.head(3))
mk = set(RE["match_key"].unique().to_list())
hk = set(H["key"].to_list())
print("match_keys in round_end", len(mk), " in header", len(hk), " intersect", len(mk & hk))
print("map_name counts:\n", H.group_by("map_name").len().sort("len", descending=True).head(20))
print("match_date sample:", H["match_date"].head(5).to_list())
