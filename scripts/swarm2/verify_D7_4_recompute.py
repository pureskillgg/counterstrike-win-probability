"""verify_D7_4 (independent recompute from raw 2024 tome channels; does NOT read D7_rounds_all.parquet).
Definitions re-implemented here (same as the claim's pipeline, written separately):
  outcome round = round with bomb_exploded or bomb_defused in basic_bomb_state; bomb_side = 3 if defused else 2 (exploded);
  mm = bomb_side != round_end.winner_team_code; lag = min(outcome tick) - round_end.tick (ticks).
ASSUMPTION: player_count == 20 (2024 full-match convention, as in the claim).
ASSUMPTION: header match_date is UTC as recorded.
ASSUMPTION: 'onset' is taken from data (first match with a -429 lag row), not assumed.
ASSUMPTION: tick rate 64 (header tick_rate is 64 for all 4,435 matches; checked in output).
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(files(BASE / f"basic_{ch}{SUF}"))


H = (pl.read_parquet(files(BASE / "header")[0], columns=["key", "match_date", "build_num", "map_name", "server_name", "tick_rate"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
                     pl.col("match_key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day")))
H = H.with_columns(pl.col("mdt").dt.strftime("%Y/%m/%d").alias("md_day"))
print("header rows:", H.height, "| build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("tick_rate values:", H["tick_rate"].unique().to_list())

RE_all = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
dupk = RE_all.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("\nround_end rows:", RE_all.height, "| duplicated (match,round) keys:", dupk.height)
if dupk.height:
    print("  duplicate detail:", RE_all.join(dupk.select(["match_key", "round"]), on=["match_key", "round"]).sort(["match_key", "round"]).rows())
RE = RE_all.unique(["match_key", "round"], keep="first", maintain_order=True)

BS = (scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
        .select(["match_key", "round", "tick", "event_type"]).collect())
OUT = BS.group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("out_tick"),
    (pl.col("event_type") == "bomb_defused").any().alias("has_defuse"),
    (pl.col("event_type") == "bomb_exploded").any().alias("has_explode"))
print("bomb outcome (match,round) keys:", OUT.height, "| with both defused and exploded:", OUT.filter(pl.col("has_defuse") & pl.col("has_explode")).height)

R = (RE.join(OUT, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "mdt", "md_day", "key_day", "build_num", "map_name", "server_name"]), on="match_key", how="left")
       .with_columns(
           pl.when(pl.col("has_defuse")).then(pl.lit(3)).when(pl.col("has_explode")).then(pl.lit(2)).otherwise(None).cast(pl.Int64).alias("bomb_side"),
           (pl.col("out_tick") - pl.col("tick")).alias("lag"),
           pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25")).otherwise(pl.lit("other")).alias("rbin")))
O = R.filter(pl.col("bomb_side").is_not_null()).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
O20 = O.filter(pl.col("player_count") == 20)
print("\n== outcome rounds: all player counts:", O.height, "mm:", int(O["mm"].sum()), "| player_count==20:", O20.height, "mm:", int(O20["mm"].sum()))
print("non-20 outcome rounds mismatches:", int(O.filter(pl.col("player_count") != 20)["mm"].sum()))
print("\n== mismatch by rbin (player_count==20):")
print(O20.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("rbin").rows())
print("mismatch round numbers (round, count), player_count==20:", O20.filter(pl.col("mm")).group_by("round").len().sort("round").rows())

# ---- onset from data
S429 = O20.filter(pl.col("lag") == -429).sort("mdt")
first429 = S429.head(1).select(["mdt", "match_key"]).rows()[0]
onset = first429[0]
print("\n== onset (from data): first -429 outcome row at", onset, "| match", first429[1][:40])
print("first mismatch (any round, player_count==20):", O20.filter(pl.col("mm")).sort("mdt").head(1).select(["mdt", "round", "build_num"]).rows())
print("first r13 mismatch:", O20.filter(pl.col("mm") & (pl.col("round") == 13)).sort("mdt").head(1).select(["mdt", "build_num"]).rows())
print("first build-9961 match (header):", H.filter(pl.col("build_num") == 9961).sort("mdt").head(1).select(["mdt"]).rows())
print("last build-9842 match (header):", H.filter(pl.col("build_num") == 9842).sort("mdt").tail(1).select(["mdt"]).rows())

O20 = O20.with_columns(pl.when(pl.col("mdt") < onset).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"))
C = O20.with_columns(
    pl.when(pl.col("lag") == -429).then(pl.lit("-429"))
      .when(pl.col("lag") == 0).then(pl.lit("0"))
      .when(pl.col("lag") < 0).then(pl.lit("other_neg"))
      .otherwise(pl.lit("positive")).alias("lagcat"))
print("\n== lag category by period x mm (player_count==20):")
print(C.group_by(["period", "mm", "lagcat"]).len().sort(["period", "mm", "lagcat"]).rows())
print("pre clean: lag==0:", C.filter((pl.col("period") == "pre") & ~pl.col("mm") & (pl.col("lag") == 0)).height,
      "of", C.filter((pl.col("period") == "pre") & ~pl.col("mm")).height,
      "| pre clean negative:", C.filter((pl.col("period") == "pre") & ~pl.col("mm") & (pl.col("lag") < 0)).height)
print("post clean -429:", C.filter((pl.col("period") == "post") & ~pl.col("mm") & (pl.col("lag") == -429)).height,
      "| post mismatch -429:", C.filter((pl.col("period") == "post") & pl.col("mm") & (pl.col("lag") == -429)).height,
      "| post mismatch total:", C.filter((pl.col("period") == "post") & pl.col("mm")).height)

# ---- match-level regime
M = (C.group_by("match_key").agg(pl.col("mdt").first(), pl.col("build_num").first(), pl.col("map_name").first(),
                                  pl.len().alias("n_out"),
                                  (pl.col("lag") == -429).sum().alias("n_m429"), (pl.col("lag") == 0).sum().alias("n_zero"),
                                  pl.col("mm").sum().alias("n_mm"))
       .with_columns(pl.when(pl.col("n_m429") > 0).then(pl.lit("neg429")).when(pl.col("n_zero") == pl.col("n_out")).then(pl.lit("allzero")).otherwise(pl.lit("other")).alias("regime"))
       .sort("mdt"))
print("\n== match-level regime (matches with >=1 player_count==20 outcome row):", M.height, M.group_by("regime").len().sort("regime").rows())
print("all-zero matches before onset:", M.filter((pl.col("regime") == "allzero") & (pl.col("mdt") < onset)).height,
      "| after onset:", M.filter((pl.col("regime") == "allzero") & (pl.col("mdt") >= onset)).height)
print("last all-zero match before onset:", M.filter((pl.col("regime") == "allzero") & (pl.col("mdt") < onset)).tail(1).select(["mdt", "build_num"]).rows())
print("build x regime (matches):", M.group_by(["build_num", "regime"]).len().sort(["build_num", "regime"]).rows())
print("build x has-429 (matches with outcome rows):",
      M.with_columns((pl.col("n_m429") > 0).alias("has429")).group_by(["build_num", "has429"]).len().sort(["build_num", "has429"]).rows())
print("build x period (matches):", M.with_columns(pl.when(pl.col("mdt") < onset).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("p"))
      .group_by(["build_num", "p"]).len().sort(["build_num", "p"]).rows())
inv = M.filter(((pl.col("build_num") == 9961) & (pl.col("mdt") < onset)) | ((pl.col("build_num") == 9842) & (pl.col("mdt") >= onset)))
print("\n== build/time inversions (9961 before onset OR 9842 after onset):", inv.height)
print(inv.select(["mdt", "build_num", "regime", "n_out", "n_m429", "n_zero", "n_mm", "map_name", "match_key"]).with_columns(pl.col("match_key").str.slice(0, 44)).rows())

# ---- within build 9961: is -429 necessary for mismatch?
print("\n== within build 9961 (player_count==20): mismatch by has-429 (match level) and rbin")
m9961 = M.filter(pl.col("build_num") == 9961).with_columns((pl.col("n_m429") > 0).alias("has429"))
C9961 = C.filter(pl.col("build_num") == 9961).join(m9961.select(["match_key", "has429"]), on="match_key", how="left")
print(C9961.group_by(["rbin", "has429", "mm"]).len().sort(["rbin", "has429", "mm"]).rows())
print("9961 matches without -429 (count, sample):", m9961.filter(~pl.col("has429")).height,
      m9961.filter(~pl.col("has429")).select(["mdt", "n_out", "n_mm", "regime", "map_name"]).sort("mdt").head(12).rows())
print("9961 matches without -429: lag values seen:",
      C9961.join(m9961.filter(~pl.col("has429")).select("match_key"), on="match_key").group_by("lag").len().sort("len", descending=True).head(8).rows())

print("\n== -429 by map (build 9961, matches):", m9961.group_by("map_name").agg(pl.len().alias("n"), pl.col("has429").sum().alias("n_429")).sort("map_name").rows())
print("== -429 by map (build 9842, matches):", M.filter(pl.col("build_num") == 9842).group_by("map_name").agg(pl.len().alias("n"), (pl.col("n_m429") > 0).sum().alias("n_429")).sort("map_name").rows())

# ---- day convention: key-path day vs match_date UTC day (claim's day table uses key path)
print("\n== day conventions for round-13 outcome rows (player_count==20)")
r13 = O20.filter(pl.col("rbin") == "r13").with_columns(pl.when(pl.col("mdt") < onset).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("p"))
print("key_day, r13 rows, mm (key_day, n, mm, period mix):")
print(r13.group_by("key_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), (pl.col("p") == "pre").sum().alias("n_pre_mdt")).sort("key_day").rows())
print("md_day, r13 rows, mm:")
print(r13.group_by("md_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("md_day").rows())
ks = r13.filter(pl.col("key_day").is_in(["2024/02/01", "2024/02/02", "2024/02/03", "2024/02/04", "2024/02/05", "2024/02/06", "2024/02/07"]))
print("claim-style 'key-path 02-01..07' r13: n", ks.height, "mm", int(ks["mm"].sum()))
kp = r13.filter(pl.col("key_day") >= "2024/02/12")
print("claim-style 'key-path >= 02-12' r13: n", kp.height, "mm", int(kp["mm"].sum()))
mp = r13.filter(pl.col("md_day") >= "2024/02/12")
print("match-date >= 02-12 r13: n", mp.height, "mm", int(mp["mm"].sum()))
print("md_day != key_day matches:", H.filter(pl.col("md_day") != pl.col("key_day")).height, "of", H.height)
