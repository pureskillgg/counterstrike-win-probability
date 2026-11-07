"""verify_D7_8 part 5: is build 9961 separable by header fields or map, and do block-start mismatches hold on every map? (stdout only)."""
import io
import contextlib
import glob
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_header"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
XR, H, BASE = g["XR"], g["H"], g["BASE"]
HF = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))]).rename({"key": "match_key"})
cols = ["demo_version_name", "server_name", "client_name", "game_directory", "platform", "tick_rate", "tick_save_rate", "game", "rushb_version", "ppp_version", "fullpackets_version", "network_protocol", "demo_version_guid"]
for c in cols:
    ct = HF.group_by(["build_num", c]).len().sort("len", descending=True)
    print(f"{c}: distinct by build -> {ct.group_by('build_num').len().rows()} | top (build, value, n):", ct.head(4).rows())
print("\nmap_name x build (matches):", H.group_by(["build_num", "map_name"]).len().sort(["build_num", "map_name"]).rows())
RS = g["PI"].group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))
Qs = XR.filter(pl.col("winner_team_code").is_not_null() & pl.col("wside_p").is_not_null() & pl.col("is_start")).join(RS, on="match_key", how="left")
print("\nblock-start rows by build x map (n, mismatches):")
print(Qs.group_by(["build_num", "map_name"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm")).sort(["build_num", "map_name"]).rows())
print("\nblock-start rows by build x round (n, mismatches):")
print(Qs.group_by(["build_num", "round"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm")).sort(["build_num", "round"]).rows())
