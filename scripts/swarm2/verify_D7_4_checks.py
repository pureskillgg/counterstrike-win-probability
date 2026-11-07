"""verify_D7_4 checks (independent of D7_rounds_all.parquet): lag by bomb side, round-number profile, the single
post-switch build-9842 match, server field by build, and the 2026 tome lag by build.
ASSUMPTION: player_count == 20 for 2024 and for 2026 (same convention as the claim). ASSUMPTION: tick rate 64.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def build_outcomes(tome_root, suffix_glob):
    """Return header (H), per-round outcome rows (O) for one tome root."""
    base = tome_root / "tome" / "csds"
    H = (pl.read_parquet(files(base / "header")[0], columns=["key", "match_date", "build_num", "map_name", "server_name"])
           .rename({"key": "match_key"})
           .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt")))
    re_dir = sorted(glob.glob(str(base / f"basic_round_end{suffix_glob}")))[0]
    bs_dir = sorted(glob.glob(str(base / f"basic_bomb_state{suffix_glob}")))[0]
    RE = (pl.scan_parquet(sorted(glob.glob(str(Path(re_dir) / "dataframe_*"))))
            .select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
            .unique(["match_key", "round"], keep="first", maintain_order=True))
    BS = (pl.scan_parquet(sorted(glob.glob(str(Path(bs_dir) / "dataframe_*"))))
            .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
            .select(["match_key", "round", "tick", "event_type"]).collect())
    OUT = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"),
                                                  (pl.col("event_type") == "bomb_defused").any().alias("has_defuse"))
    O = (RE.join(OUT, on=["match_key", "round"], how="inner")
           .join(H.select(["match_key", "mdt", "build_num", "map_name", "server_name"]), on="match_key", how="left")
           .with_columns(pl.when(pl.col("has_defuse")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                         (pl.col("out_tick") - pl.col("tick")).alias("lag"))
           .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
           .filter(pl.col("player_count") == 20))
    return H, O


H24, O24 = build_outcomes(ROOT / "tomes" / "2024-02-g", ".2023-12-28,2024-02-25")
ONSET = O24.filter(pl.col("lag") == -429).sort("mdt").head(1).select("mdt").item()
print("2024 onset (first -429 row):", ONSET)
O24 = O24.with_columns(pl.when(pl.col("mdt") < ONSET).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
                       pl.when(pl.col("lag") == -429).then(pl.lit("-429")).when(pl.col("lag") == 0).then(pl.lit("0"))
                         .when(pl.col("lag") < 0).then(pl.lit("other_neg")).otherwise(pl.lit("positive")).alias("lagcat"))

print("\n== A1: 2024 post-onset, build 9961: lag category by bomb side (2=exploded, 3=defused) x mm")
P = O24.filter(pl.col("build_num") == 9961)
print(P.group_by(["bomb_side", "lagcat", "mm"]).len().sort(["bomb_side", "lagcat", "mm"]).rows())

print("\n== A2: 2024 build 9842 (pre-switch, all rounds): lag category by bomb side")
print(O24.filter(pl.col("build_num") == 9842).group_by(["bomb_side", "lagcat", "mm"]).len().sort(["bomb_side", "lagcat", "mm"]).rows())

print("\n== A3: round-number profile: outcome rounds (count, mismatches) by build, rounds 1..28 (player_count==20)")
for b in [9842, 9961]:
    sub = O24.filter(pl.col("build_num") == b)
    prof = sub.group_by("round").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), (pl.col("lag") == -429).sum().alias("n429")).sort("round")
    print(f"build {b}:", [(r, n, m, k) for r, n, m, k in prof.rows()])

print("\n== A4: round-13 post-onset rows: lag values for mismatch rows and clean rows, by bomb side")
r13 = O24.filter((pl.col("round") == 13) & (pl.col("period") == "post"))
print("r13 post rows:", r13.height, "mm:", int(r13["mm"].sum()))
print("mismatch rows lag (top):", r13.filter(pl.col("mm")).group_by("lag").len().sort("len", descending=True).head(10).rows())
print("clean rows lag (top):", r13.filter(~pl.col("mm")).group_by("lag").len().sort("len", descending=True).head(10).rows())
print("r13 post by lagcat x mm:", r13.group_by(["lagcat", "mm"]).len().sort(["lagcat", "mm"]).rows())

print("\n== A5: non-swap rounds (round not in {9,13,25,28}), post-onset: mismatch count by lagcat")
ns = O24.filter((pl.col("period") == "post") & ~pl.col("round").is_in([9, 13, 25, 28]))
print(ns.group_by(["lagcat", "mm"]).len().sort(["lagcat", "mm"]).rows())

print("\n== B1: first build-9961 match vs first -429 match (same match?)")
f9961 = H24.filter(pl.col("build_num") == 9961).sort("mdt").head(1)
f429 = O24.filter(pl.col("lag") == -429).sort("mdt").head(1).select(["match_key", "mdt"])
print("first 9961 header match:", f9961.select(["match_key", "mdt"]).rows())
print("first -429 outcome match:", f429.rows())
print("same match key:", f9961["match_key"].item() == f429["match_key"].item())

print("\n== B2: the single build-9842 match dated after onset")
inv = H24.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= ONSET))
print("header rows:", inv.select(["mdt", "map_name", "server_name"]).rows(), "| key:", inv["match_key"].item()[:60])
x = O24.filter(pl.col("match_key") == inv["match_key"].item()).sort("round")
print("its outcome rows:", x.select(["round", "bomb_side", "winner_team_code", "lag", "mm", "period"]).rows())

print("\n== C1: server_name by build (top 4 per build, 2024 header)")
for b in [9842, 9961]:
    s = H24.filter(pl.col("build_num") == b).group_by("server_name").len().sort("len", descending=True)
    print(f"build {b}: distinct servers {s.height}; top:", s.head(4).rows())
print("servers in both builds (count):", len(set(H24.filter(pl.col("build_num") == 9842)["server_name"].to_list()) &
                                                  set(H24.filter(pl.col("build_num") == 9961)["server_name"].to_list())))

print("\n== D1: 2026 tome lag by build (outcome rounds, player_count==20)")
H26, O26 = build_outcomes(ROOT / "tomes" / "2026-10-01_to_08", ".2026-09-18,2026-10-08")
print("2026 header rows:", H26.height, "| builds:", H26.group_by("build_num").len().sort("build_num").rows())
print("2026 outcome rounds (player_count==20):", O26.height, "| mismatches:", int(O26["mm"].sum()))
print("2026 lag top values:", O26.group_by("lag").len().sort("len", descending=True).head(8).rows())
print("2026 lag == -429 rows:", O26.filter(pl.col("lag") == -429).height)
print("2026 by build: lag==0 rows and -429 rows:",
      O26.group_by("build_num").agg((pl.col("lag") == 0).sum().alias("n_zero"), (pl.col("lag") == -429).sum().alias("n_429"), pl.len().alias("n")).sort("build_num").rows())
