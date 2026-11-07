"""A6 probe: is the 2024 round_end 'rebuilt' version a property of whole matches?

Per 2024 match (fingerprint-deduplicated set): count rounds coded 1/7 (event code), 8/9 (elimination code),
and other codes. Then, for matches with a bomb-ended round, check whether the round_end is logged at the
bomb event tick (offset 0) or ~429 ticks later. Finally cross with the round-13 / 25 side inversion
(W != score-derived side, from A6_threeway_2024.parquet).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_match_version_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def load(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*")))).select(cols).collect()


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).unique(
    ["match_key", "round"], keep="first").with_columns(
    pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
fp = (RE.sort(["match_key", "round"])
      .with_columns(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8),
                                   pl.col("winner_team_code").cast(pl.Utf8),
                                   pl.col("win_reason_code").cast(pl.Utf8)], separator=":").alias("part"))
      .group_by("match_key").agg(pl.col("part").alias("parts"))
      .with_columns(pl.col("parts").list.join("|").hash().alias("fph")))
grp = fp.group_by("fph").agg(pl.col("match_key").sort().alias("keys"), pl.len().alias("n"))
keep = set(grp["keys"].list.first().to_list())
REk = RE.filter(pl.col("match_key").is_in(list(keep)))

per = REk.group_by("match_key").agg(
    pl.col("day").first().alias("day"),
    pl.len().alias("rounds"),
    pl.col("win_reason_code").is_in([1, 7]).sum().alias("n_event_code"),
    pl.col("win_reason_code").is_in([8, 9]).sum().alias("n_elim_code"))
per = per.with_columns(
    (pl.col("n_event_code") / pl.col("rounds")).alias("share_event"),
    pl.when(pl.col("n_event_code") == 0).then(pl.lit("no_event_code"))
    .when(pl.col("n_event_code") == pl.col("n_event_code")).then(pl.lit("has_event_code")).alias("pattern"))
dd = per.filter(pl.col("day").cast(pl.Int32) >= 12)
print("== days 12-25 matches (deduped): count of rounds with event-code 1/7, bucketed")
print(dd.with_columns(pl.col("n_event_code").clip(0, 6).alias("bucket")).group_by("bucket").len().sort("bucket"))
print("days 01-07 matches (deduped): count of rounds with event-code 1/7, bucketed")
print(per.filter(pl.col("day").cast(pl.Int32) <= 7).with_columns(
    pl.col("n_event_code").clip(0, 6).alias("bucket")).group_by("bucket").len().sort("bucket"))

# bomb-ended rounds per match: offset of bomb event vs round_end
BS = load("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("btick"))
bj = REk.join(BS, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("btick") == pl.col("tick")).alias("off0"))
bm = bj.group_by("match_key").agg(pl.len().alias("n_bomb"), pl.col("off0").sum().alias("n_off0"))
bm = bm.with_columns((pl.col("n_off0") == pl.col("n_bomb")).alias("all_off0"))
bm2 = bm.join(per.select(["match_key", "day"]), on="match_key", how="left")
print("\n== matches with bomb-ended rounds: all bomb rounds logged at the event tick?")
print(bm2.with_columns(pl.when(pl.col("day").cast(pl.Int32) >= 12).then(pl.lit("d12_25")).otherwise(pl.lit("d01_07")).alias("grp"))
      .group_by(["grp", "all_off0"]).len().sort(["grp", "all_off0"]))

# cross with round 13 inversion
P2 = pl.read_parquet(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / "A6_threeway_2024.parquet").filter(
    pl.col("match_key").is_in(list(keep)) & (pl.col("round") == 13) & pl.col("S_ns").is_not_null()).with_columns(
    (pl.col("W") != pl.col("S_ns")).alias("inv13")).select(["match_key", "inv13"])
xm = P2.join(per, on="match_key", how="left").with_columns(
    pl.when(pl.col("n_event_code") == 0).then(pl.lit("pure_elim_codes")).otherwise(pl.lit("some_event_codes")).alias("version"))
print("\n== round-13 inversion by match version (deduped, round 13 scored)")
print(xm.group_by(["version", "inv13"]).len().sort(["version", "inv13"]))
print("\n== round-13 inversion by match version, days 12-25 only")
print(xm.filter(pl.col("day").cast(pl.Int32) >= 12).group_by(["version", "inv13"]).len().sort(["version", "inv13"]))
print("\n== share of event-coded rounds per match, days 12-25, quantiles")
print(dd.select(pl.col("share_event").quantile(0.1).alias("p10"), pl.col("share_event").median().alias("p50"),
                pl.col("share_event").quantile(0.9).alias("p90")))
