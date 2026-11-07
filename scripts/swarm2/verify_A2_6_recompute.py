"""Independent verifier for finding A2-6.

Claim under test: the 2024 'day' is the folder day, not the match date
(header match_date day == folder day for 1,550 of 4,435 matches). The E2 by-day
round-13 table uses folder day, so its trend partly reflects build composition.
Proposed fix: rename to folder_day, use header match_date and build_num.

Re-derives every number from the raw tome with polars (column projection, one
parquet file at a time). Stdout is redirected to logs/swarm2/verify_A2_6.log by
the caller. Assumptions are printed with the word ASSUMPTION.
"""
import glob
import hashlib
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(250)

BASE = r"G:\pureskill\csds-cs2"
TOME24 = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
TOME26 = os.path.join(BASE, "tomes", "2026-10-01_to_08", "tome", "csds")
SUF24 = ".2023-12-28,2024-02-25"
F07 = ["01", "02", "03", "04", "05", "06", "07"]
F1225 = [f"{d:02d}" for d in range(12, 26)]

print("polars", pl.__version__)
print("ASSUMPTION: 'day' comes from key path element index 3 (csds/YYYY/MM/DD/<match_id>)")
print("ASSUMPTION: header match_date is UTC (trailing Z); folder date = 2024-02-<DD>")
print("ASSUMPTION: round-13 bomb mismatch logic = E2 logic (planter side not used for label)")


def read_cols(folder, cols):
    fs = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    if not fs:
        raise SystemExit(f"no parquet under {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def section(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def rate_line(label, d):
    n = d.height
    mm = int(d["mismatch"].sum()) if n else 0
    rate = (mm / n) if n else float("nan")
    print(f"  {label:<46} n={n:>5}  mismatch={mm:>4}  rate={rate:.4f}")


# ---------------------------------------------------------------- 1. header
section("1. HEADER (2024 tome): match_date day vs folder day")
hdr = read_cols(os.path.join(TOME24, "header"),
                ["key", "match_id", "match_date", "build_num", "map_name",
                 "t_starters_score_final", "ct_starters_score_final", "number_of_points"])
print("header rows", hdr.height, "distinct key", hdr["key"].n_unique(),
      "distinct match_id", hdr["match_id"].n_unique())
print("all keys match csds/2024/02/DD/<id> pattern:",
      int(hdr["key"].str.contains(r"^csds/2024/02/\d\d/").sum()), "of", hdr.height)
hdr = hdr.with_columns(
    pl.col("key").str.split("/").list.get(3).alias("folder_day"),
    pl.col("key").str.split("/").list.get(4).alias("key_mid"),
    pl.col("match_date").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("mdate"),
    pl.col("key").alias("match_key"),
)
hdr = hdr.with_columns(
    pl.col("mdate").dt.strftime("%d").alias("mday"),
    pl.col("mdate").dt.strftime("%Y-%m").alias("mmonth"),
    pl.concat_str([pl.lit("2024-02-"), pl.col("folder_day")]).str.to_date("%Y-%m-%d").alias("fdate"),
)
hdr = hdr.with_columns(
    (pl.col("mdate") - pl.col("fdate")).dt.total_days().alias("diff_days"),
    (pl.col("mday") == pl.col("folder_day")).alias("aligned"),
)
print("key folder id == match_id:", int((hdr["key_mid"] == hdr["match_id"]).sum()), "of", hdr.height)
print(f"CLAIM 1,550 of 4,435 | RECOMPUTED match_date day == folder day: "
      f"{int(hdr['aligned'].sum())} of {hdr.height} ({hdr['aligned'].mean():.4f})")
print("EXACT date equality (folder date == match date):", int((hdr["diff_days"] == 0).sum()),
      " | day-of-month equal but different month (inflates the claim's count):",
      int((hdr["aligned"] & (hdr["diff_days"] != 0)).sum()))
print("match_date month distribution (CLAIM 2023-12 3, 2024-01 1143, 2024-02 3289):",
      hdr.group_by("mmonth").len().sort("mmonth").rows())

section("1b. FOLDER DAY composition (CLAIM folder 01: 230 total, 70 aligned, 169 dated 2024-01; folder 12: 1004 total, 48 aligned)")
fd = (hdr.group_by("folder_day")
      .agg(pl.len().alias("n_matches"),
           pl.col("aligned").sum().alias("aligned"),
           (pl.col("mmonth") == "2023-12").sum().alias("dated_2023_12"),
           (pl.col("mmonth") == "2024-01").sum().alias("dated_2024_01"),
           (pl.col("mmonth") == "2024-02").sum().alias("dated_2024_02"),
           pl.col("build_num").filter(pl.col("build_num") == 9842).len().alias("build_9842"),
           pl.col("build_num").filter(pl.col("build_num") == 9961).len().alias("build_9961"))
      .sort("folder_day"))
print(fd)

section("1c. |match_date - folder date| in days (is it a timezone-scale shift or a batch index?)")
dd = hdr["diff_days"]
print("diff_days quantiles 0/5/25/50/75/95/100:",
      [int(dd.quantile(q, interpolation="nearest")) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)])
for label, mask in [("exactly 0 days", dd == 0),
                    ("|diff| == 1 day (timezone-scale)", dd.abs() == 1),
                    ("|diff| 2..7 days", dd.abs().is_between(2, 7)),
                    ("|diff| 8..31 days", dd.abs().is_between(8, 31)),
                    ("|diff| > 31 days", dd.abs() > 31)]:
    print(f"  {label:<40} {int(mask.sum())}")
print("matches dated outside Feb 2024 (cannot be a 1-day timezone shift):",
      int((hdr["mmonth"] != "2024-02").sum()))

# ---------------------------------------------------------------- 2. channels
section("2. ROUND-13 BOMB SIDE MISMATCH, E2 logic, re-keyed by header")
RE = read_cols(os.path.join(TOME24, f"basic_round_end{SUF24}"),
               ["match_key", "round", "winner_team_code", "player_count"])
BS = read_cols(os.path.join(TOME24, f"basic_bomb_state{SUF24}"),
               ["match_key", "round", "event_type", "player_team_code"])
print("round_end rows", RE.height, " distinct (match_key, round)",
      RE.select(["match_key", "round"]).unique().height)
print("bomb_state rows", BS.height)
hdr_keys = set(hdr["key"].to_list())
re_keys = set(RE["match_key"].unique().to_list())
print("round_end match_keys:", len(re_keys), " not present in header:", len(re_keys - hdr_keys))

outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
planter = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))

hdr_join = hdr.select(["match_key", "folder_day", "mdate", "mday", "mmonth", "aligned",
                       "build_num", "map_name", "diff_days"])
J = (RE.filter(pl.col("player_count") == 20)
     .join(outcome, on=["match_key", "round"], how="inner")
     .join(planter, on=["match_key", "round"], how="left")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
     .join(hdr_join, on="match_key", how="left")
     .with_columns(
         pl.when(pl.col("round") == 13).then(pl.lit("r13"))
         .when(pl.col("round") == 25).then(pl.lit("r25"))
         .otherwise(pl.lit("other")).alias("rbin")))
print("2024 round_end player_count distribution (top):", RE.group_by("player_count").len().sort("len", descending=True).head(4).rows())
print("rounds with bomb outcome (CLAIM/E2 15436):", J.height,
      " mismatches (CLAIM/E2 499):", int(J["mismatch"].sum()),
      " rows with no header match:", int(J["build_num"].is_null().sum()))
print("duplicate (match_key, round) rows after filter:", int(J.select(["match_key", "round"]).is_duplicated().sum()))
for b in ["r13", "r25", "other"]:
    rate_line(f"bin {b}", J.filter(pl.col("rbin") == b))
print("E2 values: r13 785/434, r25 49/24, other 14602/41")

fold_of = {k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 for k in hdr["key"].to_list()}
J = J.with_columns(pl.col("match_key").replace_strict(fold_of, default=0, return_dtype=pl.Int64).alias("fold"))
J = J.with_columns(pl.lit(1).alias("one"))
R13 = J.filter(pl.col("round") == 13)

section("2b. R13 by FOLDER DAY (reproduces E2 table; folder day is the claim's suspect key)")
print(R13.group_by("folder_day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                     pl.col("mismatch").mean().alias("rate")).sort("folder_day"))

section("2b2. BUILD vs CALENDAR CUTOVER (can build and match_date be separated in this tome?)")
b9842 = hdr.filter(pl.col("build_num") == 9842)
b9961 = hdr.filter(pl.col("build_num") == 9961)
print("build 9842 matches:", b9842.height, " min mdate", b9842["mdate"].min(), " max mdate", b9842["mdate"].max())
print("build 9961 matches:", b9961.height, " min mdate", b9961["mdate"].min(), " max mdate", b9961["mdate"].max())
import datetime as dtm
CUT = dtm.date(2024, 2, 7)
print("build 9842 rows dated on/after 2024-02-07:", int((b9842["mdate"] >= CUT).sum()))
print("build 9961 rows dated before 2024-02-07:", int((b9961["mdate"] < CUT).sum()))
print("match_date >= 2024-02-07 rule agrees with build_num==9961:",
      int(((hdr["mdate"] >= CUT) == (hdr["build_num"] == 9961)).sum()), "of", hdr.height)
print("per-day build table, header dates 2024-02-01..12 (all matches, and R13 rows with bomb outcome):")
days = hdr.filter(pl.col("mdate").is_between(dtm.date(2024, 2, 1), dtm.date(2024, 2, 12)))
per_day_all = days.group_by(["mdate", "build_num"]).len().rename({"len": "matches"})
per_day_r13 = (R13.filter(pl.col("mdate").is_between(dtm.date(2024, 2, 1), dtm.date(2024, 2, 12)))
               .group_by(["mdate", "build_num"]).agg(pl.len().alias("r13_rows"), pl.col("mismatch").sum().alias("r13_mm")))
print(per_day_all.join(per_day_r13, on=["mdate", "build_num"], how="full", coalesce=True).sort(["mdate", "build_num"]))
print("R13 rows with build 9842 dated 2024-02-07..12:",
      int(R13.filter((pl.col("build_num") == 9842) & (pl.col("mdate") >= CUT)).height),
      " mismatches:", int(R13.filter((pl.col("build_num") == 9842) & (pl.col("mdate") >= CUT))["mismatch"].sum()))
post_9842 = hdr.filter((pl.col("build_num") == 9842) & (pl.col("mdate") >= CUT))
pk = post_9842["match_key"].to_list()
pj = J.filter(pl.col("match_key").is_in(pk))
print("the 2 post-cutover build-9842 matches:", [(k[-36:], d) for k, d in zip(pk, post_9842["mdate"].to_list())],
      " bomb-outcome rounds (all rounds):", pj.height, " mismatches:", int(pj["mismatch"].sum()))
print("R13 rows with build 9961 dated 2024-02-07..25:",
      int(R13.filter((pl.col("build_num") == 9961) & (pl.col("mdate") >= CUT)).height),
      " mismatches:", int(R13.filter((pl.col("build_num") == 9961) & (pl.col("mdate") >= CUT))["mismatch"].sum()))

section("2c. R13 by HEADER match_date (full date)")
print(R13.group_by("mdate").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                pl.col("mismatch").mean().alias("rate"),
                                pl.col("folder_day").unique().sort().alias("folders"),
                                pl.col("build_num").unique().sort().alias("builds")).sort("mdate"))

section("2d. CLAIM WINDOWS scored two ways (folder-day window vs header match_date window)")
rate_line("folder 01..07 (E2 table; claim '0 of 180')", R13.filter(pl.col("folder_day").is_in(F07)))
rate_line("header date 2024-02-01..07", R13.filter(pl.col("mdate").is_between(pl.date(2024, 2, 1), pl.date(2024, 2, 7))))
rate_line("folder 12..25 (E2 table; claim '434 of 605')", R13.filter(pl.col("folder_day").is_in(F1225)))
rate_line("header date 2024-02-12..25", R13.filter(pl.col("mdate").is_between(pl.date(2024, 2, 12), pl.date(2024, 2, 25))))
rate_line("header date before 2024-02-01", R13.filter(pl.col("mdate") < pl.date(2024, 2, 1)))
rate_line("header date 2024-02-08..11", R13.filter(pl.col("mdate").is_between(pl.date(2024, 2, 8), pl.date(2024, 2, 11))))
print("cross-tab: header month (rows) x folder group (cols), R13 n / mismatches")
R13 = R13.with_columns(
    pl.when(pl.col("folder_day").is_in(F07)).then(pl.lit("fold01-07"))
    .when(pl.col("folder_day").is_in(F1225)).then(pl.lit("fold12-25"))
    .otherwise(pl.lit("fold_other")).alias("fgrp"))
print(R13.group_by(["mmonth", "fgrp"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort(["mmonth", "fgrp"]))

section("3. ALTERNATIVE: build_num (game version) vs folder day vs header date")
print("R13 by build_num:")
print(R13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                    pl.col("mismatch").mean().alias("rate")).sort("build_num"))
print("OTHER rounds by build_num (is the build effect general or r13-only?):")
OT = J.filter(pl.col("rbin") == "other")
print(OT.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                   pl.col("mismatch").mean().alias("rate")).sort("build_num"))
print("R13 by build_num x header month:")
print(R13.group_by(["build_num", "mmonth"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                               pl.col("mismatch").mean().alias("rate")).sort(["build_num", "mmonth"]))
print("R13 by build_num x folder_day:")
print(R13.group_by(["build_num", "folder_day"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                                   pl.col("mismatch").mean().alias("rate")).sort(["build_num", "folder_day"]))

section("3b. WITHIN-STRATUM tests separating build from date and from folder")
print("(a) folder 12 only: R13 by build_num")
for b in [9842, 9961]:
    rate_line(f"folder 12 build {b}", R13.filter((pl.col("folder_day") == "12") & (pl.col("build_num") == b)))
print("(b) header date 2024-02-12..25 only: R13 by build_num")
for b in [9842, 9961]:
    rate_line(f"header 02-12..25 build {b}",
              R13.filter(pl.col("mdate").is_between(pl.date(2024, 2, 12), pl.date(2024, 2, 25)) & (pl.col("build_num") == b)))
print("(c) header date in 2024-01 only: R13 by build_num")
for b in [9842, 9961]:
    rate_line(f"header 2024-01 build {b}", R13.filter((pl.col("mmonth") == "2024-01") & (pl.col("build_num") == b)))
print("(d) aligned (folder day == header day) vs not aligned, R13 overall and within build")
for al in [True, False]:
    rate_line(f"aligned={al}", R13.filter(pl.col("aligned") == al))
    for b in [9842, 9961]:
        rate_line(f"  aligned={al} build {b}", R13.filter((pl.col("aligned") == al) & (pl.col("build_num") == b)))


section("4. OUT-OF-SAMPLE explanatory power for R13 mismatch (2 folds by match, md5 parity)")
print("ASSUMPTION: smoothing a=1 Laplace toward training prior; Brier lower = better; threshold 0.5")


def cv(D, keys, a=1.0):
    D = D.with_columns(pl.col("mismatch").cast(pl.Float64).alias("y"))
    tot_brier, tot_correct, tot_n, tot_unseen = 0.0, 0, 0, 0
    for k in (0, 1):
        tr = D.filter(pl.col("fold") == k)
        te = D.filter(pl.col("fold") != k)
        prior = float(tr["y"].mean())
        cell = tr.group_by(keys).agg(pl.col("y").sum().alias("s"), pl.len().alias("cn"))
        p = (te.join(cell, on=keys, how="left")
             .with_columns(
                 ((pl.col("s").fill_null(0.0) + a * prior)
                  / (pl.col("cn").fill_null(0).cast(pl.Float64) + a)).alias("p"),
                 pl.col("cn").is_null().alias("unseen")))
        tot_brier += float(((p["p"] - p["y"]) ** 2).sum())
        tot_correct += int(((p["p"] >= 0.5).cast(pl.Float64) == p["y"]).sum())
        tot_n += p.height
        tot_unseen += int(p["unseen"].sum())
    return tot_brier / tot_n, tot_correct / tot_n, tot_unseen / tot_n


print(f"{'grouping':<34}{'cells':>7}{'CV Brier':>11}{'CV acc':>9}{'unseen%':>9}")
for keys in [["one"], ["aligned"], ["folder_day"], ["mdate"], ["mmonth"], ["map_name"],
             ["build_num"], ["build_num", "mmonth"], ["build_num", "folder_day"],
             ["aligned", "build_num"]]:
    cells = R13.group_by(keys).len().height
    b, acc, un = cv(R13, keys)
    print(f"{'+'.join(keys):<34}{cells:>7}{b:>11.4f}{acc:>9.4f}{un:>9.4f}")

section("5. DEDUPE SENSITIVITY (header signature groups; keep first key per group)")
sig = hdr.with_columns(pl.concat_str([pl.col("match_date"), pl.col("map_name"),
                                      pl.col("t_starters_score_final").cast(pl.Utf8),
                                      pl.col("ct_starters_score_final").cast(pl.Utf8),
                                      pl.col("number_of_points").cast(pl.Utf8)], separator="|").alias("sig"))
gsz = sig.group_by("sig").len()
dup_groups = int((gsz["len"] > 1).sum())
keep = (sig.sort("key").group_by("sig", maintain_order=True).agg(pl.col("match_key").first().alias("mk")))
kept_keys = set(keep["mk"].to_list())
print("signature groups with >1 member:", dup_groups, " excess copies:", int(gsz["len"].sum()) - gsz.height)
R13k = R13.filter(pl.col("match_key").is_in(list(kept_keys)))
print(f"R13 before dedupe n={R13.height} mm={int(R13['mismatch'].sum())}; after dedupe n={R13k.height} mm={int(R13k['mismatch'].sum())}")
print("R13 after dedupe, by build_num:")
print(R13k.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                     pl.col("mismatch").mean().alias("rate")).sort("build_num"))
print("R13 after dedupe, aligned vs not:")
print(R13k.group_by("aligned").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                   pl.col("mismatch").mean().alias("rate")).sort("aligned"))

section("6. 2026 COMPARATOR (same logic; claim: 0 mismatches in 7,427 rounds)")
h26 = read_cols(os.path.join(TOME26, "header"), ["key", "match_date", "build_num", "map_name"])
h26 = h26.with_columns(
    pl.col("key").str.split("/").list.get(3).alias("folder_day"),
    pl.col("match_date").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("mdate"),
    pl.col("key").alias("match_key"))
h26 = h26.with_columns(pl.col("mdate").dt.strftime("%d").alias("mday"),
                       pl.col("mdate").dt.strftime("%Y-%m").alias("mmonth"))
h26 = h26.with_columns((pl.col("mday") == pl.col("folder_day")).alias("aligned"))
print("2026 header rows", h26.height, " sample keys", h26["key"].head(2).to_list())
print("2026 build_num:", h26.group_by("build_num").len().rows())
print("2026 month:", h26.group_by("mmonth").len().sort("mmonth").rows())
print("2026 aligned:", int(h26["aligned"].sum()), "of", h26.height)
rb26 = sorted(glob.glob(os.path.join(TOME26, "basic_round_end*")))
bs26 = sorted(glob.glob(os.path.join(TOME26, "basic_bomb_state*")))
print("2026 round_end folders:", len(rb26), " bomb_state folders:", len(bs26))
RE26 = pl.concat([read_cols(f, ["match_key", "round", "winner_team_code", "player_count"]) for f in rb26],
                 how="vertical_relaxed")
BS26 = pl.concat([read_cols(f, ["match_key", "round", "event_type", "player_team_code"]) for f in bs26],
                 how="vertical_relaxed")
print("2026 round_end match_key sample:", RE26["match_key"].head(2).to_list(),
      " header key sample:", h26["key"].head(2).to_list())
RE26 = RE26.with_columns(pl.col("match_key").str.split("/").list.get(4).alias("mid"))
BS26 = BS26.with_columns(pl.col("match_key").str.split("/").list.get(4).alias("mid"))
h26 = h26.with_columns(pl.col("key").str.split("/").list.get(4).alias("mid"))
print("2026 distinct mids: round_end", RE26["mid"].n_unique(), " header", h26["mid"].n_unique(),
      " round_end mids not in header:", len(set(RE26["mid"].unique().to_list()) - set(h26["mid"].to_list())))
out26 = (BS26.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["mid", "round"]).agg(pl.col("event_type").unique().alias("evs")))
print("ASSUMPTION: 2026 full-roster rounds use player_count == 10 (E2's 20 selects 0 rows in 2026)")
print("2026 round_end player_count distribution (top):", RE26.group_by("player_count").len().sort("len", descending=True).head(4).rows())
J26 = (RE26.filter(pl.col("player_count") == 10)
       .join(out26, on=["mid", "round"], how="inner")
       .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
       .join(h26.select(["mid", "folder_day", "mdate", "mmonth", "aligned", "build_num"]),
             on="mid", how="left"))
anyJ = (RE26.join(out26, on=["mid", "round"], how="inner")
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
        .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
print("2026 rounds with bomb outcome, ANY player_count (claim denominator 7,427?):", anyJ.height,
      " mismatches:", int(anyJ["mismatch"].sum()))
print("2026 rounds with bomb outcome:", J26.height, " mismatches:", int(J26["mismatch"].sum()),
      " rows with no header match:", int(J26["build_num"].is_null().sum()))
print("2026 round-13 rows:", int((J26["round"] == 13).sum()),
      " mismatches:", int(J26.filter(pl.col("round") == 13)["mismatch"].sum()))
print("2026 by build_num (all rounds):")
print(J26.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("build_num"))

section("SUMMARY (numbers for the verdict)")
print("aligned (match_date day == folder day):", int(hdr["aligned"].sum()), "of", hdr.height)
print("header months:", hdr.group_by("mmonth").len().sort("mmonth").rows())
print("R13 rows total:", R13.height, " mismatches:", int(R13["mismatch"].sum()))
print("R13 by build:", R13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("build_num").rows())
print("R13 aligned:", R13.filter(pl.col("aligned")).height, int(R13.filter(pl.col("aligned"))["mismatch"].sum()),
      " not aligned:", R13.filter(~pl.col("aligned")).height, int(R13.filter(~pl.col("aligned"))["mismatch"].sum()))
print("DONE")
