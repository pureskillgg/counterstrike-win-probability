"""verify_D7_4 source check (read-only on E:): for every 2024 tome match, is its match_id folder present in the main 2024 copy
(E:\\csds_final\\csds\\2024\\02\\<DD>\\<match_id>) or in the tail-recovered copy
(E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\02\\<DD>\\<match_id>)? Cross-tab against build, the -429 lag
signature and mismatch (player_count==20 outcome rounds). Directory listings only; nothing is written.
"""
import glob, os, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def channel(name):
    d = sorted(glob.glob(str(T24 / f"basic_{name}.*")))[0]
    return pl.scan_parquet(files(Path(d)))


H = (pl.read_parquet(files(T24 / "header")[0], columns=["key", "build_num", "match_date", "server_name"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"),
                     pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/\d\d/([0-9a-f-]{36})/", 1).alias("mid"),
                     pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt")))
print("header rows:", H.height, "| mid parsed:", H["mid"].null_count() == 0, "| unique mids:", H["mid"].n_unique())

# directory listings: all day folders under the 2024/02 roots (both copies)
MAIN = Path(r"E:\csds_final\csds\2024\02")
TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
present = {}
for label, root in [("main", MAIN), ("tail", TAIL)]:
    found = set()
    days = []
    if root.exists():
        days = sorted(os.listdir(root))
    for dd in days:
        p = root / dd
        if p.is_dir():
            for mid in os.listdir(p):
                found.add((dd, mid))
    present[label] = found
    print(f"{label}: day folders={len(days)} | match folders listed={len(found)}")

rows = []
for mid, kday in H.select(["mid", "key_day"]).rows():
    dd = kday[-2:]
    rows.append({"mid": mid, "dd": dd, "in_main": (dd, mid) in present["main"], "in_tail": (dd, mid) in present["tail"],
                 "any_dd_main": any(m == mid for (_, m) in ()),})
S = pl.DataFrame(rows)
# also check whether the mid exists under any day folder (in case key day differs from folder day)
main_mids = {m for (_, m) in present["main"]}
tail_mids = {m for (_, m) in present["tail"]}
S = S.with_columns(pl.col("mid").is_in(list(main_mids)).alias("mid_in_main_any"),
                   pl.col("mid").is_in(list(tail_mids)).alias("mid_in_tail_any"))
S = S.with_columns(
    pl.when(pl.col("mid_in_main_any") & pl.col("mid_in_tail_any")).then(pl.lit("both"))
      .when(pl.col("mid_in_main_any")).then(pl.lit("main_only"))
      .when(pl.col("mid_in_tail_any")).then(pl.lit("tail_only"))
      .otherwise(pl.lit("neither")).alias("source"))
print("\nsource classification (any day folder), all 2024 tome matches:", S.group_by("source").len().sort("source").rows())
print("key-day folder match rate: in_main", int(S["in_main"].sum()), "| in_tail", int(S["in_tail"].sum()))

# attach build, mdt and -429 signature
RE = channel("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (channel("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
        .select(["match_key", "round", "tick", "event_type"]).collect())
OUT = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"), (pl.col("event_type") == "bomb_defused").any().alias("d"))
O = (RE.join(OUT, on=["match_key", "round"], how="inner").filter(pl.col("player_count") == 20)
       .with_columns(pl.when(pl.col("d")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                     (pl.col("out_tick") - pl.col("tick")).alias("lag"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
MM = (O.group_by("match_key").agg((pl.col("lag") == -429).any().alias("has429"), pl.col("mm").any().alias("has_mm"), pl.len().alias("n_out")))
X = H.select(["match_key", "mid", "build_num", "mdt"]).join(S.select(["mid", "source"]), on="mid", how="left").join(MM, on="match_key", how="inner")
print("\nmatches with outcome rows (player_count==20):", X.height)
print("build x source (matches):", X.group_by(["build_num", "source"]).len().sort(["build_num", "source"]).rows())
print("build x source x has429 (matches):", X.group_by(["build_num", "source", "has429"]).len().sort(["build_num", "source", "has429"]).rows())
print("source x has_mm (matches with >=1 mismatch):", X.group_by(["source", "has_mm"]).len().sort(["source", "has_mm"]).rows())
print("build x source x has_mm:", X.group_by(["build_num", "source", "has_mm"]).len().sort(["build_num", "source", "has_mm"]).rows())
print("\nbuild 9842 matches (count, tail_only):", X.filter((pl.col("build_num") == 9842) & (pl.col("source") == "tail_only")).height)
print("key-day folder counts in tail copy (21 expected from tome key days):", len(sorted(os.listdir(TAIL))) if TAIL.exists() else 0)
print("tail key-day folders:", sorted(os.listdir(TAIL)) if TAIL.exists() else [])
print("E:\\\\csds_final\\\\csds\\\\2024 exists:", Path(r"E:\csds_final\csds\2024").exists())
print("build 9961 matches in main copy (count):", X.filter((pl.col("build_num") == 9961) & (pl.col("source") == "main_only")).height)
print("pre-switch (mdt < 2024-02-07 01:01:18) matches by source:", X.filter(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18)).group_by("source").len().rows())
print("post-switch (mdt >= 2024-02-07 01:01:18) matches by source:", X.filter(pl.col("mdt") >= pl.datetime(2024, 2, 7, 1, 1, 18)).group_by("source").len().rows())
