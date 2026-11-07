"""verify_D7_4 probe 5: 2026 player_count convention and lag by build (all player counts); 2024 host prefix by build;
round-position lag profile in build 9961 (rounds 11-13, 23-25); E: source path existence for 2 match ids (read-only test).
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def channel(tome_csds, name):
    d = sorted(glob.glob(str(tome_csds / f"basic_{name}.*")))[0]
    return pl.scan_parquet(files(Path(d)))


def outcomes(tome_csds):
    H = pl.read_parquet(files(tome_csds / "header")[0], columns=["key", "build_num", "server_name"]).rename({"key": "match_key"})
    RE = channel(tome_csds, "round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
    RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
    BS = (channel(tome_csds, "bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
            .select(["match_key", "round", "tick", "event_type"]).collect())
    OUT = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"),
                                                  (pl.col("event_type") == "bomb_defused").any().alias("d"))
    O = (RE.join(OUT, on=["match_key", "round"], how="inner")
           .join(H, on="match_key", how="left")
           .with_columns(pl.when(pl.col("d")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                         (pl.col("out_tick") - pl.col("tick")).alias("lag")))
    return H, O.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))


T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
H26, O26 = outcomes(T26)
print("2026 header builds:", H26.group_by("build_num").len().sort("build_num").rows())
print("2026 round_end player_count dist:", channel(T26, "round_end").select(pl.col("player_count").value_counts()).collect().rows()[:1] if False else "")
print("2026 outcome rounds (all player counts):", O26.height, "| mm:", int(O26["mm"].sum()))
print("2026 outcome rows by player_count (n, mm):", O26.group_by("player_count").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("player_count").rows())
print("2026 lag top values:", O26.group_by("lag").len().sort("len", descending=True).head(8).rows())
print("2026 by build (n, lag0, lag-429, mm):", O26.group_by("build_num").agg(pl.len().alias("n"), (pl.col("lag") == 0).sum().alias("n0"),
      (pl.col("lag") == -429).sum().alias("n429"), pl.col("mm").sum().alias("mm")).sort("build_num").rows())

T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
H24, O24 = outcomes(T24)
H24 = H24.with_columns(pl.col("server_name").str.extract(r"^(Valve Counter-Strike 2|[^(]{0,30})", 1).alias("host_prefix"))
print("\n2024 host prefix x build:", H24.group_by(["build_num", "host_prefix"]).len().sort(["build_num", "host_prefix"]).rows())
O24 = O24.filter(pl.col("player_count") == 20)
p = O24.filter(pl.col("build_num") == 9961)
print("\nbuild 9961 (player_count==20) lag top values by round:")
for r in [11, 12, 13, 23, 24, 25]:
    s = p.filter(pl.col("round") == r)
    print("  round", r, "n", s.height, "top lags", s.group_by("lag").len().sort("len", descending=True).head(6).rows())

print("\nE: source path existence (read-only):")
for name, mid in [("9961 first match", "78a91cb9-d50e-459b-beb9-315d4d1cd570"), ("9842 post-switch match", "9cfb21ee-fd88-4b43-8898-931f2441cfe6")]:
    for label, pth in [("main", Path(r"E:\csds_final\csds\2024\02\12") / mid),
                       ("tail", Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12") / mid)]:
        print(f"  {name} [{label}] exists={pth.exists()}")
