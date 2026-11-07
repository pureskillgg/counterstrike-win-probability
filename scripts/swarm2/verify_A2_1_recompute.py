# verify_A2_1 recompute: independent re-derivation of the A2-1 build split and round-level disagreement.
# Part 1 reproduces the claim's method (score delta vs round_end winner_team_code).
# Part 2 adds score-free oracles (round_end win_reason_message, bomb_state explode/defuse events).
# Part 3 checks header alternatives to build_num (server, protocol, game dir, demo guid, ppp version).
# Memory: schema-projected lazy scans; only small per-round tables are collected (~1e5 rows).
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(90)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
HDR = os.path.join(BASE, "header", "dataframe_00000")


def chan(ch):
    fs = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}{SUF}", "dataframe_*")))
    assert len(fs) == 1, (ch, fs)
    return pl.scan_parquet(fs[0])


def sec(t):
    print("\n=== " + t, flush=True)


# ---------- header ----------
HCOLS = ["key", "match_id", "match_date", "map_name", "build_num", "server_name", "network_protocol",
         "game_directory", "demo_version_guid", "ppp_version", "tick_rate",
         "t_starters_score_final", "ct_starters_score_final", "number_of_points"]
hdr = pl.scan_parquet(HDR).select(HCOLS).collect().rename({"key": "match_key"})
hdr = hdr.with_columns(
    pl.col("match_key").str.split("/").list.get(3).alias("fday"),
    pl.col("match_date").str.slice(0, 10).alias("mdate"),
    pl.col("server_name").str.extract(r"Server \(srcds\d+-(\w+)", 1).alias("srv_region"),
)
hdr = hdr.with_columns(
    pl.when(pl.col("fday") == "01").then(pl.lit("A_01"))
    .when(pl.col("fday") <= "07").then(pl.lit("B_02_07"))
    .otherwise(pl.lit("C_12_25")).alias("fgrp")
)
sec("HEADER")
print("rows", hdr.height, "| unique match_key", hdr["match_key"].n_unique(),
      "| unique match_id", hdr["match_id"].n_unique(), flush=True)
print("build_num counts:", hdr.group_by("build_num").len().sort("build_num").rows(), flush=True)
print("folder-group x build:", hdr.group_by(["fgrp", "build_num"]).len().sort(["fgrp", "build_num"]).rows(), flush=True)
print("match_date range by build:")
print(hdr.group_by("build_num").agg(pl.col("mdate").min().alias("min_mdate"),
                                    pl.col("mdate").max().alias("max_mdate"),
                                    pl.len().alias("n")).sort("build_num"), flush=True)
print("match_date month x build:", hdr.group_by(["build_num", pl.col("mdate").str.slice(0, 7)]).len()
      .sort(["build_num", "mdate"]).rows(), flush=True)
print("folder day == match_date day (overall):",
      int((hdr["fday"] == hdr["mdate"].str.slice(8, 2)).sum()), "of", hdr.height, flush=True)

# ---------- round_end ----------
re_all = chan("round_end").select(["match_key", "round", "tick", "winner_team_code",
                                   "win_reason_code", "win_reason_message", "player_count"]).collect()
sec("ROUND_END")
print("rows", re_all.height, "| distinct (match,round)", re_all.select(["match_key", "round"]).unique().height, flush=True)
print("player_count dist:", re_all.group_by("player_count").len().sort("player_count").rows(), flush=True)
print("winner_team_code dist:", re_all.group_by("winner_team_code").len().sort("winner_team_code").rows(), flush=True)
print("reason vocabulary (code, message, winner_team_code, n):")
for r in re_all.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len() \
        .sort(["win_reason_code", "winner_team_code"]).iter_rows():
    print("  ", r, flush=True)
dupg = re_all.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                                   pl.col("winner_team_code").n_unique().alias("nw"),
                                                   pl.col("tick").n_unique().alias("nt"))
print("duplicate (match,round) groups:", dupg.filter(pl.col("n") > 1).height,
      "| with differing winner:", dupg.filter(pl.col("nw") > 1).height,
      "| with differing tick:", dupg.filter(pl.col("nt") > 1).height, flush=True)

re_u = (re_all.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
# game-side reason -> side. T-side reasons: Terrorists_Win, Target_Bombed. CT-side: CTs_Win, Bomb_Defused, Target_Saved.
re_u = re_u.with_columns(
    pl.when(pl.col("win_reason_message").str.contains("Terrorists_Win|Target_Bombed")).then(2)
    .when(pl.col("win_reason_message").str.contains("CTs_Win|Bomb_Defused|Target_Saved")).then(3)
    .otherwise(None).alias("reason_side")
)
print("reason_side null count (unmapped messages):", re_u.filter(pl.col("reason_side").is_null()).height, flush=True)
print("unmapped messages:", re_u.filter(pl.col("reason_side").is_null()).group_by(
    ["win_reason_code", "win_reason_message", "winner_team_code"]).len().rows(), flush=True)

# ---------- round_state freeze_end ----------
ev = chan("round_state").select("event_type").group_by("event_type").len().collect()
sec("ROUND_STATE event vocabulary")
print(ev.sort("len", descending=True).rows(), flush=True)
fe_all = (chan("round_state")
          .select(["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                   "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
          .filter(pl.col("event_type") == "round_freeze_end")
          .collect())
sec("FREEZE_END")
print("rows", fe_all.height, "| distinct (match,round)", fe_all.select(["match_key", "round"]).unique().height, flush=True)
print("phase dist:", fe_all.group_by("phase").len().sort("phase").rows(), flush=True)
print("is_warmup dist:", fe_all.group_by("is_warmup").len().rows(), flush=True)
print("t_score != t_score_raw rows:", fe_all.filter(pl.col("t_score") != pl.col("t_score_raw")).height,
      "| ct_score != ct_score_raw rows:", fe_all.filter(pl.col("ct_score") != pl.col("ct_score_raw")).height, flush=True)
fdup = fe_all.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("t_score").n_unique().alias("nt"))
print("freeze_end duplicate (match,round) groups:", fdup.filter(pl.col("n") > 1).height,
      "| with differing t_score:", fdup.filter(pl.col("nt") > 1).height, flush=True)

fe = fe_all.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
nx = (fe.sort(["match_key", "round"])
      .with_columns([
          pl.col("round").shift(-1).over("match_key").alias("r_next"),
          pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
          pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
          pl.col("t_score_raw").shift(-1).over("match_key").alias("tr_next"),
          pl.col("ct_score_raw").shift(-1).over("match_key").alias("ctr_next"),
      ])
      .filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns([
          (pl.col("t_next") - pl.col("t_score")).alias("dT"),
          (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"),
          (pl.col("tr_next") - pl.col("t_score_raw")).alias("dTr"),
          (pl.col("ctr_next") - pl.col("ct_score_raw")).alias("dCTr"),
      ])
      .with_columns([
          # claim's rule: dT==1 -> T (2); elif dCT==1 -> CT (3); else 0 (no pair)
          pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"),
          pl.when(pl.col("dTr") == 1).then(2).when(pl.col("dCTr") == 1).then(3).otherwise(0).alias("w_raw"),
      ])
      .select(["match_key", "round", "w_score", "w_raw", "dT", "dCT", "dTr", "dCTr", "phase", "is_warmup"]))

# ---------- bomb_state (score-free oracle) ----------
bo = (chan("bomb_state")
      .select(["match_key", "round", "tick", "event_type", "player_team_code"])
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .collect())
sec("BOMB_STATE")
print("rows", bo.height, "| event vocab:", bo.group_by("event_type").len().rows(), flush=True)
bsum = bo.group_by(["match_key", "round"]).agg(
    pl.col("event_type").eq("bomb_exploded").any().alias("exp"),
    pl.col("event_type").eq("bomb_defused").any().alias("dfs"),
    pl.col("event_type").eq("bomb_planted").any().alias("planted"),
    pl.col("player_team_code").filter(pl.col("event_type") == "bomb_planted").first().alias("planter_side"),
)
bsum = bsum.with_columns(
    pl.when(pl.col("exp") & ~pl.col("dfs")).then(2)
    .when(pl.col("dfs") & ~pl.col("exp")).then(3)
    .otherwise(None).alias("bomb_side")
)
print("bomb rounds:", bsum.height, "| both exp&defused:", bsum.filter(pl.col("exp") & pl.col("dfs")).height,
      "| bomb_side null:", bsum.filter(pl.col("bomb_side").is_null()).height, flush=True)

# ---------- joined per-round table (all rounds) ----------
R = (re_u.select(["match_key", "round", "winner_team_code", "reason_side", "player_count", "win_reason_code"])
     .join(bsum.select(["match_key", "round", "bomb_side", "planter_side", "planted"]), on=["match_key", "round"], how="left")
     .join(hdr.select(["match_key", "build_num", "fgrp", "map_name", "srv_region", "network_protocol",
                       "game_directory", "demo_version_guid", "ppp_version", "server_name"]),
           on="match_key", how="left"))
R = R.with_columns(
    (pl.col("reason_side") != pl.col("winner_team_code")).alias("win_vs_reason_bad"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("win_vs_bomb_bad"),
    (pl.col("bomb_side") != pl.col("reason_side")).alias("bomb_vs_reason_bad"),
)
R = R.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round") == 9).then(pl.lit("r09"))
    .when(pl.col("round") == 12).then(pl.lit("r12"))
    .when(pl.col("round") == 25).then(pl.lit("r25"))
    .when(pl.col("round") == 27).then(pl.lit("r27"))
    .when(pl.col("round") == 28).then(pl.lit("r28"))
    .otherwise(pl.lit("rOther")).alias("rbin")
)
R.write_parquet(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A2_1_round_table.parquet")

sec("ORACLE AGREEMENT on ALL rounds (score-free): winner_team_code vs reason vs bomb, by build and round")
Rr = R.filter(pl.col("reason_side").is_not_null())
print("rounds with reason_side:", Rr.height, flush=True)
print("win_vs_reason mismatch by build x fgrp:",
      Rr.group_by(["build_num", "fgrp"]).agg(pl.len().alias("n"), pl.col("win_vs_reason_bad").sum().alias("bad"))
      .sort(["build_num", "fgrp"]).rows(), flush=True)
Rb = R.filter(pl.col("bomb_side").is_not_null())
print("win_vs_bomb mismatch by build x fgrp:",
      Rb.group_by(["build_num", "fgrp"]).agg(pl.len().alias("n"), pl.col("win_vs_bomb_bad").sum().alias("bad"))
      .sort(["build_num", "fgrp"]).rows(), flush=True)
print("bomb_vs_reason mismatch (both present) by build x fgrp:",
      R.filter(pl.col("bomb_side").is_not_null() & pl.col("reason_side").is_not_null())
      .group_by(["build_num", "fgrp"]).agg(pl.len().alias("n"), pl.col("bomb_vs_reason_bad").sum().alias("bad"))
      .sort(["build_num", "fgrp"]).rows(), flush=True)

sec("WINNER vs REASON mismatches by build x round (all rounds, count of bad/n)")
t = (Rr.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("win_vs_reason_bad").sum().alias("bad"))
     .sort(["build_num", "round"]))
for bn in [9842, 9961]:
    rows = [(r[1], r[2], r[3]) for r in t.filter(pl.col("build_num") == bn).rows() if r[1] <= 30]
    print(f"build {bn} (k,n,bad):", rows, flush=True)

sec("WINNER vs BOMB mismatches by build x round (all rounds, bad/n)")
t2 = (Rb.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("win_vs_bomb_bad").sum().alias("bad"))
      .sort(["build_num", "round"]))
for bn in [9842, 9961]:
    rows = [(r[1], r[2], r[3]) for r in t2.filter(pl.col("build_num") == bn).rows() if r[1] <= 30]
    print(f"build {bn} (k,n,bad):", rows, flush=True)

# ---------- Part 1: claim method (score delta vs round_end) ----------
P = (nx.filter(pl.col("w_score") != 0)
     .join(R.select(["match_key", "round", "winner_team_code", "build_num", "fgrp", "map_name",
                     "srv_region", "network_protocol", "game_directory", "demo_version_guid",
                     "ppp_version", "reason_side", "bomb_side", "planter_side"]),
           on=["match_key", "round"], how="left"))
P = P.with_columns((pl.col("w_score") != pl.col("winner_team_code")).alias("bad_score_vs_end"))
sec("PART1 claim method: score delta (w_score!=0) vs round_end winner, by build (disagreeing rounds, pairs)")
g = P.group_by("build_num").agg(pl.col("bad_score_vs_end").sum().alias("bad"), pl.len().alias("pairs")).sort("build_num")
print(g, flush=True)
mm = P.group_by(["match_key", "build_num"]).agg(pl.col("bad_score_vs_end").sum().alias("nbad"), pl.len().alias("np"))
print("matches with >=1 disagreement by build:",
      mm.with_columns((pl.col("nbad") > 0).alias("any")).group_by(["build_num", "any"]).len()
      .sort(["build_num", "any"]).rows(), flush=True)
print("by build x k (k, n, bad) for k in 1..30:")
for bn in [9842, 9961]:
    sub = P.filter(pl.col("build_num") == bn).group_by("round").agg(pl.len().alias("n"),
                                                                     pl.col("bad_score_vs_end").sum().alias("bad")).sort("round")
    print(f"  build {bn}:", [(r[0], r[1], r[2]) for r in sub.rows() if r[0] <= 30], flush=True)

# ---------- Part 1b: 3-way agreement at each k for score delta, round_end winner, reason, bomb ----------
sec("PART1b three-way agreement on pairs where all oracles present (score, end, reason, bomb)")
P3 = P.filter(pl.col("reason_side").is_not_null() & pl.col("bomb_side").is_not_null())
print("pairs with reason & bomb present:", P3.height, flush=True)
for bn in [9842, 9961]:
    for k in [9, 12, 13, 25, 27, 28]:
        s = P3.filter((pl.col("build_num") == bn) & (pl.col("round") == k))
        if s.height == 0:
            print(f"  build {bn} k={k}: no rows", flush=True)
            continue
        c = s.select([
            pl.len().alias("n"),
            (pl.col("w_score") == pl.col("winner_team_code")).sum().alias("score_eq_end"),
            (pl.col("w_score") == pl.col("reason_side")).sum().alias("score_eq_reason"),
            (pl.col("w_score") == pl.col("bomb_side")).sum().alias("score_eq_bomb"),
            (pl.col("winner_team_code") == pl.col("reason_side")).sum().alias("end_eq_reason"),
            (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("end_eq_bomb"),
            (pl.col("reason_side") == pl.col("bomb_side")).sum().alias("reason_eq_bomb"),
        ]).rows()[0]
        print(f"  build {bn} k={k}: n={c[0]} score==end {c[1]} | score==reason {c[2]} | score==bomb {c[3]} | "
              f"end==reason {c[4]} | end==bomb {c[5]} | reason==bomb {c[6]}", flush=True)

sec("PART1c raw score fields (t_score_raw/ct_score_raw) vs round_end winner, by build x k (k=9,12,13,25,27,28)")
PR = nx.filter(pl.col("w_raw") != 0).join(R.select(["match_key", "round", "winner_team_code", "build_num"]),
                                          on=["match_key", "round"], how="left")
PR = PR.with_columns((pl.col("w_raw") != pl.col("winner_team_code")).alias("bad_raw_vs_end"))
for bn in [9842, 9961]:
    sub = PR.filter(pl.col("build_num") == bn).group_by("round").agg(pl.len().alias("n"),
                                                                      pl.col("bad_raw_vs_end").sum().alias("bad")).sort("round")
    print(f"  build {bn} raw (k,n,bad) k<=30:", [(r[0], r[1], r[2]) for r in sub.rows() if r[0] <= 30], flush=True)

sec("PART1d score field vs raw field equality by build (rows where t_score != t_score_raw)")
print(fe_all.join(hdr.select(["match_key", "build_num"]), on="match_key", how="left")
      .with_columns((pl.col("t_score") != pl.col("t_score_raw")).alias("neq"))
      .group_by(["build_num", "neq"]).len().sort(["build_num", "neq"]).rows(), flush=True)

print("DONE recompute", flush=True)
