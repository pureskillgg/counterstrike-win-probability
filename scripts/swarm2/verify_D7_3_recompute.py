"""verify D7-3 (independent recompute). Read-only on tomes; writes nothing.

Round-winner truth sources used here, all taken from the raw 2024 tome, none from the swarm ledger:
  R1 reason  : round_end.win_reason_message mapped to a side (mapping learned on pre-onset matches, purity printed)
  R2 bomb    : bomb_state bomb_defused -> CT (3), bomb_exploded -> T (2)
Mismatch = winner_team_code (round_end) disagrees with the truth source.
The ledger-based w_ok from D7_rounds_with_ledger_dp.parquet is compared only as a third, claimed source.
All ASSUMPTIONS are printed as such in the log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
ONSET_BUILD = 9961
PRE_BUILD = 9842


def chan_dirs(base, ch):
    return sorted(p for p in base.glob(f"basic_{ch}.*") if p.is_dir())


def read_chan(base, ch, cols, where=None):
    parts = []
    for d in chan_dirs(base, ch):
        for f in sorted(glob.glob(str(d / "dataframe_*"))):
            lf = pl.scan_parquet(f).select(cols)
            if where is not None:
                lf = lf.filter(where)
            parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed")


def read_header(base):
    fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name"]) for f in fs],
                     how="vertical_relaxed")


def parse_mdt(col):
    return col.str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)


# ---------- header ----------
H = read_header(T24).rename({"key": "match_key"}).with_columns(
    parse_mdt(pl.col("match_date")).alias("mdt"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
)
print("=== 2024 header ===")
print("rows:", H.height, "| unique match_key:", H["match_key"].n_unique(), "| null mdt:", H["mdt"].null_count())
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("mdt range by build:")
print(H.group_by("build_num").agg(pl.col("mdt").min().alias("min"), pl.col("mdt").max().alias("max"), pl.len().alias("n")).sort("build_num"))
print("map_name counts:", H.group_by("map_name").len().sort("len", descending=True).rows())
print("build x map:")
print(H.group_by(["build_num", "map_name"]).len().sort(["build_num", "map_name"]).rows())
print("matches per folder day (day, n):", H.group_by("day").len().sort("day").rows())

# onset check (ASSUMPTION: onset = first build-9961 match by match_date)
b_on = H.filter(pl.col("build_num") == ONSET_BUILD)["mdt"].min()
b_pre_last = H.filter((pl.col("build_num") == PRE_BUILD) & (pl.col("mdt") < b_on))["mdt"].max()
post_9842 = H.filter((pl.col("build_num") == PRE_BUILD) & (pl.col("mdt") >= b_on))
print(f"\nfirst 9961 mdt: {b_on} | last 9842 before it: {b_pre_last}")
print(f"9842 matches AFTER first 9961 match: {post_9842.height}")
print(post_9842.select(["mdt", "day", "match_key", "map_name"]).sort("mdt").rows())

# ---------- round_end ----------
RE = read_chan(T24, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                  "win_reason_message", "player_count"])
print("\n=== round_end (2024) ===")
print("rows:", RE.height, "| unique (match_key, round):", RE.select(["match_key", "round"]).unique().height)
dup = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match_key, round) groups:", dup.height)
print("player_count counts:", RE.group_by("player_count").len().sort("player_count").rows())
print("winner_team_code counts:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("reason (code, message, winner) counts:")
print(RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len()
      .sort(["win_reason_code", "winner_team_code"]))

RE = RE.join(H.select(["match_key", "build_num", "mdt", "day", "map_name"]), on="match_key", how="left")
print("round_end rows with no header match:", RE.filter(pl.col("build_num").is_null()).height)

# ---------- reason -> side mapping, learned only on pre-onset (build 9842) matches ----------
pre = RE.filter((pl.col("build_num") == PRE_BUILD) & pl.col("winner_team_code").is_in([2, 3]))
mp = (pre.group_by(["win_reason_message", "winner_team_code"]).len()
      .sort(["win_reason_message", "len"], descending=[False, True]))
print("\nreason message x winner_team_code, pre-onset (build 9842) only:")
print(mp)
reason_side = (mp.group_by("win_reason_message").agg(pl.col("winner_team_code").first().alias("reason_side"),
                                                     pl.col("len").sum().alias("n_pre"),
                                                     pl.col("len").max().alias("n_top"))
               .with_columns((pl.col("n_top") / pl.col("n_pre")).alias("purity")))
print("reason -> side (ASSUMPTION: pre-onset build-9842 mapping is the truth for that reason):")
print(reason_side)

# ---------- bomb outcome per round ----------
BS = read_chan(T24, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
print("\nbomb_state event_type counts:", BS.group_by("event_type").len().sort("len", descending=True).rows())
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("bomb_tick")))
bo = bo.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
    pl.col("evs").list.len().alias("n_ev"),
)
print("rounds with a defuse/explode outcome:", bo.height, "| with >1 outcome event types:", bo.filter(pl.col("n_ev") > 1).height)

R = RE.join(reason_side.select(["win_reason_message", "reason_side"]), on="win_reason_message", how="left") \
      .join(bo.select(["match_key", "round", "bomb_side", "bomb_tick"]), on=["match_key", "round"], how="left") \
      .with_columns(
          (pl.col("winner_team_code") != pl.col("reason_side")).alias("mm_reason"),
          (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("mm_bomb"),
          (pl.col("bomb_side").is_not_null()).alias("has_bomb"),
          (pl.col("bomb_tick") - pl.col("tick")).alias("lag_bomb_minus_end"),
      )
R = R.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round") == 9).then(pl.lit("r9"))
    .when(pl.col("round") == 28).then(pl.lit("r28"))
    .when(pl.col("round") == 25).then(pl.lit("r25"))
    .otherwise(pl.lit("other")).alias("rbin"),
    (pl.col("mdt") >= b_on).alias("post_onset"),
)

print("\nreason-side vs bomb-side agreement where both exist (ASSUMPTION: both are truth sources):")
both = R.filter(pl.col("has_bomb") & pl.col("reason_side").is_not_null())
print("rows with both:", both.height, "| reason==bomb:", both.filter(pl.col("reason_side") == pl.col("bomb_side")).height)

# ---------- main tables ----------
print("\n=== mismatch by build x round bin (reason truth) ===")
print(R.filter(pl.col("reason_side").is_not_null()).group_by(["build_num", "rbin"])
      .agg(pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm"), pl.col("mm_reason").mean().alias("rate"))
      .sort(["build_num", "rbin"]))

print("\n=== mismatch by build x round bin (bomb truth, only rounds with a bomb outcome) ===")
print(R.filter(pl.col("has_bomb")).group_by(["build_num", "rbin"])
      .agg(pl.len().alias("n"), pl.col("mm_bomb").sum().alias("mm"), pl.col("mm_bomb").mean().alias("rate"))
      .sort(["build_num", "rbin"]))

print("\n=== round 13 by folder day x build (reason truth, ALL round-13 rows, no bomb needed) ===")
r13 = R.filter((pl.col("round") == 13) & pl.col("reason_side").is_not_null())
print(r13.group_by(["day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm"),
                                             pl.col("mm_reason").mean().alias("rate")).sort(["day", "build_num"]))

print("\n=== round 13 by build x has_bomb (reason truth) ===")
print(r13.group_by(["build_num", "has_bomb"]).agg(pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm"),
                                                  pl.col("mm_reason").mean().alias("rate")).sort(["build_num", "has_bomb"]))

print("\n=== non-round-13 rows, build x post_onset (reason truth), flips off the fas rounds ===")
oth = R.filter((pl.col("round") != 13) & pl.col("reason_side").is_not_null())
print(oth.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("mm_reason").sum().alias("mm"),
                                              pl.col("mm_reason").mean().alias("rate")).sort(["build_num", "rbin"]))

# ---------- match-level flip on round 13 ----------
m13 = (R.filter((pl.col("round") == 13) & pl.col("reason_side").is_not_null())
       .select(["match_key", "build_num", "day", "map_name", "mm_reason", "has_bomb", "lag_bomb_minus_end"]))
print("\nround-13 rows per match, distinct matches:", m13["match_key"].n_unique())
print("match-level flip (any round-13 row mismatched) by build:")
mm_match = m13.group_by(["match_key", "build_num"]).agg(pl.col("mm_reason").any().alias("flip"),
                                                       pl.col("mm_reason").sum().alias("k"),
                                                       pl.len().alias("n"))
print(mm_match.group_by(["build_num", "flip"]).len().sort(["build_num", "flip"]))
print("rows per match on round 13 (should be 1):", mm_match.group_by("n").len().rows())
print("flip by map x build (matches):")
print(m13.group_by(["build_num", "map_name"]).agg(pl.col("mm_reason").mean().alias("row_flip_rate"), pl.len().alias("n"))
      .sort(["build_num", "map_name"]))

# ---------- lag marker check ----------
print("\nlag (bomb_tick - round_end tick) distribution, rounds with bomb outcome, by build:")
lagd = R.filter(pl.col("has_bomb")).group_by(["build_num"]).agg(
    (pl.col("lag_bomb_minus_end") == -429).mean().alias("frac_lag_-429"),
    pl.col("lag_bomb_minus_end").median().alias("median_lag"), pl.len().alias("n"))
print(lagd.sort("build_num"))
mlag = (R.filter(pl.col("has_bomb")).group_by("match_key")
        .agg((pl.col("lag_bomb_minus_end") == -429).any().alias("has_lag429"))
        .join(H.select(["match_key", "build_num"]), on="match_key", how="left"))
print("matches with bomb outcome, build x has_lag429:", mlag.group_by(["build_num", "has_lag429"]).len()
      .sort(["build_num", "has_lag429"]).rows())
# the 71 build-9961 matches with no -429 lag: do they flip on round 13 (reason truth)?
no429 = mlag.filter((pl.col("build_num") == ONSET_BUILD) & (~pl.col("has_lag429")))["match_key"]
f71 = mm_match.filter(pl.col("match_key").is_in(no429))
print("\nbuild-9961 matches lacking lag -429 that have a round-13 row:", f71.height,
      "| flipped:", f71.filter(pl.col("flip")).height)
w429 = mm_match.filter((pl.col("build_num") == ONSET_BUILD) & (~pl.col("match_key").is_in(no429)))
print("build-9961 matches WITH lag -429 that have a round-13 row:", w429.height,
      "| flipped:", w429.filter(pl.col("flip")).height)

# save the per-round table to the session scratchpad (not the project), for the companion script
out = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_3_rounds.parquet")
R.select(["match_key", "round", "tick", "build_num", "day", "map_name", "mdt", "winner_team_code", "win_reason_message",
          "reason_side", "bomb_side", "bomb_tick", "has_bomb", "mm_reason", "mm_bomb", "lag_bomb_minus_end", "post_onset",
          "player_count"]) \
    .write_parquet(str(out))
print("\nwrote", out)
