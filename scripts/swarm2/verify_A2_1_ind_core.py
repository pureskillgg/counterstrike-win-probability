# verify_A2_1 independent recompute (verifier "ind"). Reads 2024 tome channels lazily with column projection.
# ASSUMPTION A: a round's score delta = change between consecutive round_freeze_end states of one match
#   (same rule as scripts\swarm2\A2_build_round.py): dT==1 -> T (2); else dCT==1 -> CT (3); else 0 (dropped).
# ASSUMPTION B: winner_team_code 2 = T, 3 = CT (scripts\swarm\e2_side_check_r13.py docstring).
# ASSUMPTION C: bomb_exploded -> T (2); bomb_defused -> CT (3) (same mapping as E2). Only single-outcome rounds used.
# ASSUMPTION D: header build_num is the game build of the whole match.
# ASSUMPTION E: round_end duplicates: variant A = first row in file order (A2 scripts, made deterministic),
#   variant B = player_count == 20 rows only (E2 filter), variant C = earliest-tick row.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
HDR = os.path.join(BASE, "header", "dataframe_00000")
TAIL = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
MAIN = r"E:\csds_final\csds\2024\02"
KS = [9, 13, 25, 27, 28]
KG = pl.when(pl.col("round").is_in(KS)).then(pl.col("round").cast(pl.Utf8)).otherwise(pl.lit("other"))


def files(ch):
    fs = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}{SUF}", "dataframe_*")))
    if not fs:
        raise SystemExit("no files for " + ch)
    return fs


def section(t):
    print("\n==== " + t, flush=True)


def totals(base):
    return (base.group_by("build_num")
            .agg(pl.len().alias("pairs"), pl.col("dis").sum().alias("dis"), pl.col("w_end").is_null().sum().alias("no_end"))
            .sort("build_num"))


def per_k(base, bn):
    return (base.filter(pl.col("build_num") == bn).group_by("round")
            .agg(pl.len().alias("n"), pl.col("dis").sum().alias("bad")).sort("round"))


def kc(tbl, k):
    r = tbl.filter(pl.col("round") == k).rows()
    return (r[0][1], r[0][2]) if r else (0, 0)


# ---------------------------------------------------------------- 1. header
section("1. header and build split")
hdr = (pl.read_parquet(HDR, columns=["key", "match_id", "match_date", "map_name", "build_num"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"),
                     pl.col("match_date").str.slice(0, 10).alias("mdate")))
print("header rows", hdr.height, "unique match_key", hdr["match_key"].n_unique(), flush=True)
print("build_num counts", hdr.group_by("build_num").len().sort("build_num").rows(), flush=True)
print("folder day x build (matches):", hdr.group_by(["day", "build_num"]).len().sort(["day", "build_num"]).rows(), flush=True)
hdr_s = hdr.select(["match_key", "build_num", "day", "map_name", "mdate", "match_id"])

# ---------------------------------------------------------------- 2. freeze_end and score deltas
section("2. round_freeze_end states and consecutive-round score deltas")
fe_raw = (pl.scan_parquet(files("round_state"))
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "tick", "t_score", "ct_score", "phase"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
print("freeze_end rows", fe_raw.height, flush=True)
dupfe = (fe_raw.group_by(["match_key", "round"])
         .agg(pl.len().alias("n"), pl.col("t_score").n_unique().alias("nt"), pl.col("ct_score").n_unique().alias("nc")))
print("freeze_end (match,round) groups", dupfe.height, "with n>1", dupfe.filter(pl.col("n") > 1).height,
      "with conflicting scores", dupfe.filter((pl.col("nt") > 1) | (pl.col("nc") > 1)).height, flush=True)
print("phase values", fe_raw.group_by("phase").len().sort("len", descending=True).rows(), flush=True)
fe = fe_raw.unique(["match_key", "round"], keep="first", maintain_order=True).sort(["match_key", "round"])
nx = fe.with_columns(pl.col("round").shift(-1).over("match_key").alias("r_next"),
                     pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                     pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"))
gap = nx.filter(pl.col("r_next").is_not_null() & (pl.col("r_next") != pl.col("round") + 1))
print("non-consecutive freeze_end steps", gap.height, "in matches", gap["match_key"].n_unique(), flush=True)
pr = (nx.filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns((pl.col("t_next") - pl.col("t_score")).alias("dT"), (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"))
      .with_columns(pl.when(pl.col("dT") == 1).then(pl.lit(2)).when(pl.col("dCT") == 1).then(pl.lit(3))
                    .otherwise(pl.lit(0)).alias("w_score")))
print("delta pattern top (dT, dCT, count):", pr.group_by(["dT", "dCT"]).len().sort("len", descending=True).head(10).rows(), flush=True)
print("pairs with nonzero w_score (all builds):", pr.filter(pl.col("w_score") != 0).height, flush=True)

# ---------------------------------------------------------------- 3. round_end winners
section("3. round_end winners (three de-duplication variants)")
re_raw = (pl.scan_parquet(files("round_end"))
          .select(["match_key", "round", "tick", "winner_team_code", "player_count"])
          .with_columns(pl.col("round").cast(pl.Int64))
          .collect())
print("round_end rows", re_raw.height, flush=True)
dre = re_raw.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"))
print("round_end (match,round) groups", dre.height, "n>1", dre.filter(pl.col("n") > 1).height,
      "conflicting winner", dre.filter(pl.col("nw") > 1).height, flush=True)
print("player_count dist top", re_raw.group_by("player_count").len().sort("len", descending=True).head(6).rows(), flush=True)
W = {
    "A_firstfile": re_raw.unique(["match_key", "round"], keep="first", maintain_order=True),
    "B_pc20": re_raw.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first", maintain_order=True),
    "C_earliest_tick": re_raw.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True),
}
W = {k: v.select(["match_key", "round", pl.col("winner_team_code").alias("w_end")]) for k, v in W.items()}
for k, v in W.items():
    print("variant", k, "(match,round) keys:", v.height, flush=True)


def attach(variant):
    return (pr.filter(pl.col("w_score") != 0)
            .join(W[variant], on=["match_key", "round"], how="left")
            .join(hdr_s, on="match_key", how="left")
            .with_columns((pl.col("w_end") != pl.col("w_score")).cast(pl.Int64).alias("dis")))


# ---------------------------------------------------------------- 4. recompute (variant A)
section("4. A2 recompute, variant A (first row in file order)")
bA = attach("A_firstfile")
print("nonzero pairs total", bA.height, " missing round_end", bA.filter(pl.col("w_end").is_null()).height,
      " missing header", bA.filter(pl.col("build_num").is_null()).height, flush=True)
print("totals by build (build, pairs, dis, no_end):", totals(bA).rows(), flush=True)
for bn in [9842, 9961]:
    t = per_k(bA, bn)
    print(f"build {bn} per-k (k, n, bad), all k:", [(r[0], r[1], r[2]) for r in t.rows()], flush=True)
mt = bA.group_by(["build_num", "match_key"]).agg(pl.col("dis").sum().alias("ndis"))
print("matches with >=1 disagreement (build, any, count):",
      mt.with_columns((pl.col("ndis") > 0).alias("any")).group_by(["build_num", "any"]).len().sort(["build_num", "any"]).rows(),
      flush=True)

# ---------------------------------------------------------------- 5. within-day controlled comparison
section("5. controlled comparison: folder day x build (variant A)")
mc = hdr.group_by(["day", "build_num"]).agg(pl.len().alias("matches"))
tp = bA.group_by(["day", "build_num"]).agg(pl.len().alias("pairs"), pl.col("dis").sum().alias("dis"))
t13 = (bA.filter(pl.col("round") == 13).group_by(["day", "build_num"])
       .agg(pl.len().alias("n13"), pl.col("dis").sum().alias("bad13")))
tday = mc.join(tp, on=["day", "build_num"], how="left").join(t13, on=["day", "build_num"], how="left").sort(["day", "build_num"])
print("columns: day, build_num, matches, pairs, dis, n13, bad13", flush=True)
for r in tday.rows():
    print("  ", r, flush=True)
s = (tday.filter(pl.col("day") >= "12").group_by("build_num")
     .agg(pl.col("matches").sum(), pl.col("pairs").sum(), pl.col("dis").sum(), pl.col("n13").sum(), pl.col("bad13").sum())
     .sort("build_num"))
print("days 12-25 by build (build, matches, pairs, dis, n13, bad13):", s.rows(), flush=True)

# ---------------------------------------------------------------- 6. bomb-outcome adjudication
section("6. bomb-outcome adjudication (uses bomb events, not score deltas)")
bo = (pl.scan_parquet(files("bomb_state"))
      .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"])
      .with_columns(pl.col("round").cast(pl.Int64))
      .collect())
print("bomb outcome rows", bo.height, bo.group_by("event_type").len().rows(), flush=True)
bg = bo.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
srows = []
conflict = 0
for mk, rd, evs in bg.rows():
    if len(evs) != 1:
        conflict += 1
        continue
    srows.append((mk, rd, 3 if evs[0] == "bomb_defused" else 2))
print("bomb (match,round) with outcome:", bg.height, " single-outcome:", len(srows), " conflicting:", conflict, flush=True)
bside = pl.DataFrame(srows, schema={"match_key": pl.Utf8, "round": pl.Int64, "bomb_side": pl.Int64}, orient="row")
be = (bside.join(W["A_firstfile"], on=["match_key", "round"], how="left")
      .filter(pl.col("w_end").is_not_null())
      .join(hdr_s, on="match_key", how="left")
      .with_columns((pl.col("bomb_side") != pl.col("w_end")).cast(pl.Int64).alias("mis"), KG.alias("kg")))
print("bomb-vs-round_end mismatch by build x kg (build, kg, n, mis):",
      be.group_by(["build_num", "kg"]).agg(pl.len().alias("n"), pl.col("mis").sum().alias("mis")).sort(["build_num", "kg"]).rows(),
      flush=True)
print("bomb-vs-round_end mismatch round 13 by build (build, n, mis):",
      be.filter(pl.col("round") == 13).group_by("build_num").agg(pl.len().alias("n"), pl.col("mis").sum().alias("mis")).sort("build_num").rows(),
      flush=True)
bA2 = bA.join(bside, on=["match_key", "round"], how="left")
bb = (bA2.filter(pl.col("bomb_side").is_not_null() & pl.col("w_end").is_not_null())
      .with_columns(KG.alias("kg"),
                    (pl.col("bomb_side") == pl.col("w_score")).cast(pl.Int64).alias("ag_score"),
                    (pl.col("bomb_side") == pl.col("w_end")).cast(pl.Int64).alias("ag_end")))
print("score-pairs with bomb outcome and round_end winner:", bb.height, flush=True)
bt = (bb.group_by(["build_num", "kg"])
      .agg(pl.len().alias("n"), pl.col("ag_score").sum().alias("agree_with_score_delta"), pl.col("ag_end").sum().alias("agree_with_round_end"))
      .sort(["build_num", "kg"]))
print("score-pair bomb agreement (build, kg, n, agree_score, agree_roundend):", bt.rows(), flush=True)
x13 = (bb.filter(pl.col("round") == 13).group_by(["build_num", "w_score", "w_end", "bomb_side"]).len()
       .sort(["build_num", "w_score", "w_end", "bomb_side"]))
print("k=13 crosstab (build, w_score, w_end, bomb_side, n):", x13.rows(), flush=True)

# ---------------------------------------------------------------- 7. round-index shift test
section("7. round-index shift test (score delta k vs round_end k-1, k, k+1)")
wprev = W["A_firstfile"].with_columns((pl.col("round") + 1).alias("round")).rename({"w_end": "w_prev"})
wnext = W["A_firstfile"].with_columns((pl.col("round") - 1).alias("round")).rename({"w_end": "w_next"})
bs = (bA.join(wprev, on=["match_key", "round"], how="left").join(wnext, on=["match_key", "round"], how="left")
      .filter(pl.col("w_prev").is_not_null() & pl.col("w_next").is_not_null() & pl.col("w_end").is_not_null())
      .with_columns(KG.alias("kg"),
                    (pl.col("w_end") == pl.col("w_score")).cast(pl.Int64).alias("ag_same"),
                    (pl.col("w_prev") == pl.col("w_score")).cast(pl.Int64).alias("ag_prev"),
                    (pl.col("w_next") == pl.col("w_score")).cast(pl.Int64).alias("ag_next")))
sh = (bs.group_by(["build_num", "kg"])
      .agg(pl.len().alias("n"), pl.col("ag_same").sum().alias("agree_same_k"),
           pl.col("ag_prev").sum().alias("agree_k_minus_1"), pl.col("ag_next").sum().alias("agree_k_plus_1"))
      .sort(["build_num", "kg"]))
print("shift test (build, kg, n, agree_k, agree_k-1, agree_k+1):", sh.rows(), flush=True)

# ---------------------------------------------------------------- 8. phase / halftime check
section("8. first 'second'-phase freeze_end round vs disagreement")
fsec = fe.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second"))
d13 = bA.filter(pl.col("round") == 13).group_by("match_key").agg(pl.col("dis").sum().alias("d13"))
d9 = bA.filter(pl.col("round") == 9).group_by("match_key").agg(pl.col("dis").sum().alias("d9"))
pm = (hdr_s.select(["match_key", "build_num"]).join(fsec, on="match_key", how="left")
      .join(d13, on="match_key", how="left").join(d9, on="match_key", how="left")
      .with_columns(pl.col("first_second").fill_null(-1)))
print("per match (build, first_second, matches, with d13>0, with d9>0):",
      pm.group_by(["build_num", "first_second"]).agg(pl.len().alias("matches"),
                                                     (pl.col("d13") > 0).sum().alias("with_d13"),
                                                     (pl.col("d9") > 0).sum().alias("with_d9"))
      .sort(["build_num", "first_second"]).rows(), flush=True)

# ---------------------------------------------------------------- 9. map within build
section("9. map within build (k=13, variant A)")
print("(build, map, n, bad):", bA.filter(pl.col("round") == 13).group_by(["build_num", "map_name"])
      .agg(pl.len().alias("n"), pl.col("dis").sum().alias("bad")).sort(["build_num", "map_name"]).rows(), flush=True)

# ---------------------------------------------------------------- 10. sensitivity to round_end de-duplication
section("10. winner-source sensitivity")
for name in ["B_pc20", "C_earliest_tick"]:
    bX = attach(name)
    print(name, "totals (build, pairs, dis, no_end):", totals(bX).rows(), flush=True)
    print(name, "k=13 (n,bad) 9961:", kc(per_k(bX, 9961), 13), " 9842:", kc(per_k(bX, 9842), 13), flush=True)
    print(name, "k=9  (n,bad) 9961:", kc(per_k(bX, 9961), 9), " 9842:", kc(per_k(bX, 9842), 9), flush=True)

# ---------------------------------------------------------------- 11. raw source membership (listing only)
section("11. raw source membership by folder day (directory listings, read-only)")
for day in [f"{d:02d}" for d in range(12, 26)]:
    ids = hdr.filter(pl.col("day") == day).select(["match_id", "build_num"]).rows()
    tail_ids, main_ids = set(), set()
    try:
        if os.path.isdir(os.path.join(TAIL, day)):
            tail_ids = set(os.listdir(os.path.join(TAIL, day)))
    except OSError as e:
        print("tail listdir error", day, repr(e), flush=True)
    try:
        if os.path.isdir(os.path.join(MAIN, day)):
            main_ids = set(os.listdir(os.path.join(MAIN, day)))
    except OSError as e:
        print("main listdir error", day, repr(e), flush=True)
    cnt = {}
    for mid, bn in ids:
        loc = "tail" if mid in tail_ids else ("main" if mid in main_ids else "neither")
        cnt[(bn, loc)] = cnt.get((bn, loc), 0) + 1
    print(day, "tail_entries", len(tail_ids), "main_entries", len(main_ids), sorted(cnt.items(), key=str), flush=True)

# ---------------------------------------------------------------- 12. summary
section("12. SUMMARY: claim vs independent recompute (variant A)")
tt = {r[0]: (r[1], r[2]) for r in totals(bA).rows()}
print("pairs/dis 9842: claim 32359/57 | recomputed", tt.get(9842), flush=True)
print("pairs/dis 9961: claim 42830/2642 | recomputed", tt.get(9961), flush=True)
print("total nonzero pairs: claim 75189 | recomputed", bA.height, flush=True)
K9961 = per_k(bA, 9961)
K9842 = per_k(bA, 9842)
print("9961 k=13 bad/n: claim 2051/2212 | recomputed", kc(K9961, 13), flush=True)
print("9961 k=9  bad/n: claim 292/2404  | recomputed", kc(K9961, 9), flush=True)
print("9961 k=25 bad/n: claim 124/124   | recomputed", kc(K9961, 25), flush=True)
print("9961 k=27 bad/n: claim 70/70     | recomputed", kc(K9961, 27), flush=True)
print("9961 k=28 bad/n: claim 105/105   | recomputed", kc(K9961, 28), flush=True)
print("9842 k=27 bad/n: claim 57/57     | recomputed", kc(K9842, 27), flush=True)
print("9842 k=13 bad/n: (not in claim)  | recomputed", kc(K9842, 13), flush=True)
print("9842 k=9  bad/n: (not in claim)  | recomputed", kc(K9842, 9), flush=True)
print("9842 k=25 bad/n: (not in claim)  | recomputed", kc(K9842, 25), flush=True)
print("9842 k=28 bad/n: (not in claim)  | recomputed", kc(K9842, 28), flush=True)
mm = mt.with_columns((pl.col("ndis") > 0).alias("any")).group_by(["build_num", "any"]).len()
print("matches with >=1 dis, build 9961: claim 2343 of 2476 | recomputed", mm.filter((pl.col("build_num") == 9961) & pl.col("any")).rows(), flush=True)
print("matches with >=1 dis, build 9842: claim 57 of 1954 | recomputed", mm.filter((pl.col("build_num") == 9842) & pl.col("any")).rows(), flush=True)
print("DONE", flush=True)
