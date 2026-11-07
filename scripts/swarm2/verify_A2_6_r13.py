# verify_A2_6 part 2 (2024 round-13 bomb/side check, read-only).
# Reproduces scripts\swarm\e2_side_check_r13.py exactly (player_count==20, bomb_defused->3, bomb_exploded->2),
# then re-keys the same rows by header match_date and build_num, and tests whether
# build_num or folder day carries the round-13 mismatch.
# Reads only: 2024 tome basic_round_end, basic_bomb_state, header (one column set each).
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SUF = ".2023-12-28,2024-02-25"
H24 = os.path.join(TOME, "header", "dataframe_00000")
CUT = "2024-02-07"  # ASSUMPTION: the build cut found by verify_A2_6_header.py; used only to split rows


def files_of(ch):
    return sorted(glob.glob(os.path.join(TOME, f"basic_{ch}{SUF}", "dataframe_*")))


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(ch)], how="vertical_relaxed")


def show(title, df):
    print("-" * 100)
    print(title)
    print(df)


# ---- header: match_key, folder day, match_date (UTC date), build_num
hdr = pl.read_parquet(H24, columns=["key", "match_date", "build_num"]).with_columns(
    pl.col("key").alias("match_key"),
    pl.col("key").str.split("/").list.get(3).alias("fday"),
    pl.col("match_date").str.slice(0, 10).alias("mdate"),
)
odd = hdr.filter(~pl.col("match_date").str.contains(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"))
print("header rows with non-ISO match_date:", odd.height)
print(odd.select(["match_key", "match_date", "build_num", "fday"]))
print("header build_num x match_date<CUT:", hdr.with_columns((pl.col("mdate") < CUT).alias("before_cut"))
      .group_by(["build_num", "before_cut"]).len().sort(["build_num", "before_cut"]).rows())
print("header build exceptions to the CUT (9842 on/after CUT, or 9961 before CUT):")
print(hdr.filter(((pl.col("build_num") == 9842) & (pl.col("mdate") >= CUT)) |
                 ((pl.col("build_num") == 9961) & (pl.col("mdate") < CUT)))
      .select(["match_key", "mdate", "fday", "build_num"]))

# ---- round tables, same reads as E2
RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
print("round_end rows:", RE.height, " bomb_state rows:", BS.height)
dup_re = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end (match_key, round) groups with >1 row:", dup_re.height, " excess rows:", int((dup_re["len"] - 1).sum()) if dup_re.height else 0)

outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
planter = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))

j0 = (RE.filter(pl.col("player_count") == 20)
      .join(outcome, on=["match_key", "round"], how="inner")
      .join(planter, on=["match_key", "round"], how="left")
      .with_columns(
          pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
      )
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                    .otherwise(pl.lit("other")).alias("rbin")))
print(f"\nREPRODUCTION (E2 logic): rounds with outcome: {j0.height}, mismatches: {int(j0['mismatch'].sum())}")
e2_day = (j0.filter(pl.col("round") == 13)
          .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
          .group_by("day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
          .sort("day"))
show("E2 round-13 table by FOLDER day (should equal scripts\\swarm\\out\\e2_side_check_r13.log)", e2_day)
print("E2 sums: r13 n =", int(e2_day["n"].sum()), " mm =", int(e2_day["mm"].sum()),
      "  folder 01-07 n =", int(e2_day.filter(pl.col("day") <= "07")["n"].sum()),
      "  folder 12-25 n =", int(e2_day.filter(pl.col("day") >= "12")["n"].sum()))

# ---- re-key by header
j = j0.join(hdr.select(["match_key", "fday", "mdate", "build_num"]), on="match_key", how="left")
print("rows without header match:", int(j.filter(pl.col("build_num").is_null()).height))

r13 = j.filter(pl.col("round") == 13)
show("ROUND 13 by build_num (n, mismatches, rate)",
     r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                   pl.col("mismatch").mean().alias("rate")).sort("build_num"))
show("ROUND 13 by build_num x FOLDER day",
     r13.group_by(["build_num", "fday"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                             pl.col("mismatch").mean().alias("rate")).sort(["build_num", "fday"]))
show("ROUND 13 by build_num x MATCH_DATE (UTC day)",
     r13.group_by(["build_num", "mdate"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
     .sort(["build_num", "mdate"]))
show("ROUND 13 by cut on MATCH_DATE (before/on CUT vs after)",
     r13.with_columns((pl.col("mdate") >= CUT).alias("on_or_after_cut"))
     .group_by(["on_or_after_cut", "build_num"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                                     pl.col("mismatch").mean().alias("rate"))
     .sort(["on_or_after_cut", "build_num"]))

# the defect text says 'matches dated 2024-02-12..25'; check which key that set really is
fd_set = r13.filter((pl.col("fday") >= "12") & (pl.col("fday") <= "25"))
md_set = r13.filter((pl.col("mdate") >= "2024-02-12") & (pl.col("mdate") <= "2024-02-25"))
print(f"\nDEFECT-TEXT CHECK: folder 12..25 round-13 rows n={fd_set.height} mm={int(fd_set['mismatch'].sum())}")
print(f"                   match_date 02-12..02-25 round-13 rows n={md_set.height} mm={int(md_set['mismatch'].sum())}")
print(f"                   folder 01..07 round-13 rows n={r13.filter(pl.col('fday') <= '07').height}")

# within-build folder-day effect: is folder day informative once build is fixed?
for b in [9842, 9961]:
    sub = r13.filter(pl.col("build_num") == b)
    show(f"ROUND 13, build {b}, by FOLDER day",
         sub.group_by("fday").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("fday"))
    show(f"ROUND 13, build {b}, by MATCH_DATE day",
         sub.group_by("mdate").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("mdate"))

# all other rounds, by build, to see whether the defect is round-13 specific within each build
show("ROUNDS 13 vs 25 vs other, by build_num (n, mm)",
     j.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
     .sort(["build_num", "rbin"]))

# ---- neighbour-round check: does the bomb outcome of round 13 match the winner of round 12, 13 or 14?
RE_u = RE.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first", maintain_order=True)
r13b = r13.select(["match_key", "bomb_side", "build_num", "mismatch"])
for r in (12, 13, 14):
    wr = RE_u.filter(pl.col("round") == r).select(["match_key", pl.col("winner_team_code").alias(f"w{r}")])
    r13b = r13b.join(wr, on="match_key", how="left")
r13b = r13b.with_columns([(pl.col(f"w{r}") == pl.col("bomb_side")).alias(f"agree{r}") for r in (12, 13, 14)])
show("ROUND 13 bomb outcome == winner of round 12 / 13 / 14 (same match), by build_num",
     r13b.group_by("build_num").agg(pl.len().alias("n"),
                                    pl.col("agree12").sum().alias("eq_r12"),
                                    pl.col("agree13").sum().alias("eq_r13"),
                                    pl.col("agree14").sum().alias("eq_r14"),
                                    pl.col("mismatch").sum().alias("mm")).sort("build_num"))
print("DONE")
