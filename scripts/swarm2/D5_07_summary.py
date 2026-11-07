"""D5 step 7: round-end label 1 presence, flip rates by day and map, co-occurrence of flips
at the first post-swap rounds (13, 25, 28), and the single duplicate round_end pair.

Flip definition (chain-based, independent of bomb and of round_end): for label r in
{13, 25, 28}, freeze-end score step r -> r+1 gives a single-team +1 whose team differs
from round_end winner r. Only rounds with a clean +1 step are counted.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_07_summary.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))

# ---- label-1 round_end presence by day
r1 = R.filter(pl.col("round") == 1).select(["match_key", pl.col("in_end").alias("r1_end")])
MM = M.select(["match_key", "day", "map_name", "grp", "aff_r13"]).join(r1, on="match_key", how="left").with_columns(
    pl.col("r1_end").fill_null(False))
print("== round_end label 1 present, by day ==")
print(MM.group_by("day").agg(pl.len().alias("matches"), pl.col("r1_end").sum().alias("r1_end_present"),
                             pl.col("aff_r13").sum().alias("aff_r13")).sort("day"))

# ---- chain-based flip per (match, label) for labels 12..14, 24..29
base = R.select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner"])
rows = []
for lab in [12, 13, 24, 25, 27, 28]:
    cur = base.filter(pl.col("round") == lab).select(["match_key", pl.col("fe_t_score").alias("t"),
                                                     pl.col("fe_ct_score").alias("ct"), pl.col("winner").alias("w")])
    nxt = base.filter(pl.col("round") == lab + 1).select(["match_key", pl.col("fe_t_score").alias("tn"),
                                                         pl.col("fe_ct_score").alias("ctn")])
    j = cur.join(nxt, on="match_key", how="inner")
    j = j.with_columns(dt=pl.col("tn") - pl.col("t"), dct=pl.col("ctn") - pl.col("ct")).with_columns(
        ch=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
        .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)),
    ).filter(pl.col("w").is_not_null()).with_columns(lab=pl.lit(lab))
    rows.append(j.select(["match_key", "lab", "ch", "w"]))
F = pl.concat(rows).with_columns(
    usable=pl.col("ch") != 0,
    flipped=(pl.col("ch") != 0) & (pl.col("ch") != pl.col("w")))
F = F.join(MM.select(["match_key", "grp", "day", "map_name", "r1_end"]), on="match_key", how="left")

print("\n== chain-vs-round_end flip rate by label, group, r1_end (usable rounds only) ==")
t = (F.filter(pl.col("usable"))
     .group_by(["lab", "grp", "r1_end"]).agg(pl.len().alias("usable"), pl.col("flipped").sum().alias("flipped"))
     .with_columns((pl.col("flipped") / pl.col("usable")).round(3).alias("flip_rate"))
     .sort(["lab", "grp", "r1_end"]))
print(t)

# ---- flip rate for label 13 by day and by map (late, r1_end true)
l13 = F.filter((pl.col("lab") == 13) & pl.col("usable") & pl.col("grp").is_in(["ctrl_late", "aff_r13"]))
print("\n== label 13 flip rate by day (late groups, all r1_end) ==")
print(l13.group_by(["day", "r1_end"]).agg(pl.len().alias("usable"), pl.col("flipped").sum().alias("flipped"))
      .with_columns((pl.col("flipped") / pl.col("usable")).round(3).alias("rate")).sort(["day", "r1_end"]))
print("\n== label 13 flip rate by map (late groups, r1_end true) ==")
print(l13.filter(pl.col("r1_end")).group_by("map_name").agg(pl.len().alias("usable"), pl.col("flipped").sum().alias("flipped"))
      .with_columns((pl.col("flipped") / pl.col("usable")).round(3).alias("rate")).sort("usable", descending=True))
print("\n== label 13 flip rate by map (late groups, r1_end false) ==")
print(l13.filter(~pl.col("r1_end")).group_by("map_name").agg(pl.len().alias("usable"), pl.col("flipped").sum().alias("flipped"))
      .with_columns((pl.col("flipped") / pl.col("usable")).round(3).alias("rate")).sort("usable", descending=True))

# ---- co-occurrence: matches flipped at 13 vs at 25 / 28 (usable both)
ff = F.filter(pl.col("usable") & pl.col("lab").is_in([13, 25, 28]) & pl.col("grp").is_in(["ctrl_late", "aff_r13"])
              & pl.col("r1_end"))
wide = ff.pivot(on="lab", index="match_key", values="flipped", aggregate_function="first")
print("\n== late r1_end=true matches: flip at 13 vs 25 vs 28 (usable both) ==")
if {"13", "25", "28"}.issubset(set(wide.columns)) or {13, 25, 28}.issubset(set(wide.columns)):
    cols = {c: str(c) for c in wide.columns}
    wide = wide.rename(cols)
    print(wide.group_by(["13", "25", "28"]).agg(pl.len().alias("matches")).sort(["13", "25", "28"]))
else:
    print("pivot columns:", wide.columns)

# ---- duplicate round_end pair
print("\n== matches with duplicate round_end (match, round) pairs ==")
dup = R.filter(pl.col("n_end_rows") > 1).select(["match_key", "round", "n_end_rows", "end_tick", "winner", "reason_code"])
print(dup)
dm = dup.select("match_key").unique()["match_key"].to_list()
RE_raw = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "legacy_code"]).filter(
    pl.col("match_key").is_in(dm))
print(RE_raw.sort(["round", "tick"]))
rd = R.filter(pl.col("match_key").is_in(dm)).sort("round").select(
    ["round", "fe_tick", "fe_t_score", "fe_ct_score", "start_tick", "end_tick", "winner", "reason_code", "n_end_rows",
     "n_round_prestart", "n_round_officially_ended", "n_cs_pre_restart", "n_round_freeze_end"])
print(rd.filter(pl.col("round") <= 6))
print("labels present (end):", R.filter(pl.col("match_key").is_in(dm) & pl.col("in_end"))["round"].n_unique(),
      " max end label:", R.filter(pl.col("match_key").is_in(dm) & pl.col("in_end"))["round"].max())
print("header final sum:", M.filter(pl.col("match_key").is_in(dm))["final_sum"].to_list())

# ---- legacy_code and aff
print("\n== legacy_code != 0 by group (round rows with end) ==")
le = R.filter(pl.col("in_end")).join(M.select(["match_key", "grp"]), on="match_key", how="left")
print(le.group_by("grp").agg(pl.len().alias("rounds"), (pl.col("legacy_code") != 0).sum().alias("legacy_nonzero")).sort("grp"))
print("\n== aff round-13 reason codes (round_end) ==")
print(le.filter((pl.col("grp") == "aff_r13") & (pl.col("round") == 13)).group_by(["reason_code", "winner"]).agg(pl.len().alias("rounds")).sort(["reason_code", "winner"]))
