# verify_A2_1 checks: alternative explanations for the build-9961 round-boundary disagreement.
#  (1) tick-window attribution: do round_end and bomb events fall inside their labelled round window?
#  (2) label-shift test: is winner(k) a copy of winner(k-1) or winner(k+1) at the mismatch rounds?
#  (3) halftime / format (MR8 vs MR12) and overtime structure per match, by build.
#  (4) same-calendar-date control (match_date 2024-02-07..12 exists in both builds).
#  (5) map and header-field alternatives (server region, protocol, game dir, demo guid, ppp version).
#  (6) signature duplicates and their build agreement.
# Inputs: round table written by verify_A2_1_recompute.py (scratchpad) plus fresh small scans.
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(70)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
HDR = os.path.join(BASE, "header", "dataframe_00000")
SP = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"


def chan(ch):
    fs = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}{SUF}", "dataframe_*")))
    assert len(fs) == 1, (ch, fs)
    return pl.scan_parquet(fs[0])


def sec(t):
    print("\n=== " + t, flush=True)


HCOLS = ["key", "match_id", "match_date", "map_name", "build_num", "server_name", "network_protocol",
         "game_directory", "demo_version_guid", "ppp_version", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points"]
hdr = pl.scan_parquet(HDR).select(HCOLS).collect().rename({"key": "match_key"})
hdr = hdr.with_columns(pl.col("match_date").str.slice(0, 10).alias("mdate"),
                       pl.col("server_name").str.extract(r"Server \(srcds\d+-(\w+)", 1).alias("srv_region"),
                       pl.col("server_name").str.extract(r"us_(\w+) Server", 1).alias("srv_us_region"))

R = pl.read_parquet(os.path.join(SP, "verify_A2_1_round_table.parquet"))

# ---------- ticks for windows ----------
fe_t = (chan("round_state").select(["match_key", "round", "tick", "event_type"])
        .filter(pl.col("event_type") == "round_freeze_end").collect()
        .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
        .select(["match_key", "round", "tick"]).rename({"tick": "tick_fe"}))
re_t = (chan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
        .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
        .rename({"tick": "tick_re"}))
bo = (chan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"])
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())

# window of round k: [fe_k.tick, fe_{k+1}.tick)
win = (fe_t.sort(["match_key", "round"])
       .with_columns(pl.col("tick_fe").shift(-1).over("match_key").alias("tick_fe_next"),
                     pl.col("round").shift(-1).over("match_key").alias("r_next"))
       .filter(pl.col("r_next") == pl.col("round") + 1))

# (1) round_end tick inside own window?
A = re_t.join(win, on=["match_key", "round"], how="inner").join(hdr.select(["match_key", "build_num"]), on="match_key")
A = A.with_columns(((pl.col("tick_re") >= pl.col("tick_fe")) & (pl.col("tick_re") < pl.col("tick_fe_next"))).alias("in_own"))
sec("(1a) round_end tick inside own round window [fe_k, fe_k+1): by build, count and violations")
print(A.group_by("build_num").agg(pl.len().alias("n"), (~pl.col("in_own")).sum().alias("outside")).sort("build_num").rows(), flush=True)
print("violations (build, k) for 9961 with k in {8,9,12,13,24..28}:",
      A.filter((pl.col("build_num") == 9961) & (~pl.col("in_own")) & pl.col("round").is_in([8, 9, 12, 13, 24, 25, 26, 27, 28]))
      .group_by("round").len().sort("round").rows(), flush=True)

# bomb events inside their labelled window
B = bo.join(win, on=["match_key", "round"], how="inner").join(hdr.select(["match_key", "build_num"]), on="match_key")
B = B.with_columns(((pl.col("tick") >= pl.col("tick_fe")) & (pl.col("tick") < pl.col("tick_fe_next"))).alias("in_own"))
sec("(1b) bomb events tick inside own round window, by build (n, outside)")
print(B.group_by("build_num").agg(pl.len().alias("n"), (~pl.col("in_own")).sum().alias("outside")).sort("build_num").rows(), flush=True)
print("bomb events outside own window by build x round (9961, k in 9,13,25,28):",
      B.filter((~pl.col("in_own")) & pl.col("round").is_in([9, 13, 25, 28])).group_by(["build_num", "round"])
      .len().sort(["build_num", "round"]).rows(), flush=True)

# (2) label-shift test for winner_team_code at k=13 (9961 vs 9842)
W = re_t.select(["match_key", "round", "winner_team_code", "win_reason_code"])
sec("(2) label-shift test: winner(k) vs winner(k-1) and winner(k+1) at k=13 and k=9, by build")
for k in [9, 13]:
    for nb, lab in [(-1, "k-1"), (1, "k+1")]:
        cur = W.filter(pl.col("round") == k).select(["match_key", "winner_team_code"]).rename({"winner_team_code": "wk"})
        oth = W.filter(pl.col("round") == k + nb).select(["match_key", "winner_team_code"]).rename({"winner_team_code": "wo"})
        j = cur.join(oth, on="match_key", how="inner").join(hdr.select(["match_key", "build_num"]), on="match_key")
        print(f"  k={k} vs {lab}:", j.group_by("build_num").agg(pl.len().alias("n"),
                                                                (pl.col("wk") == pl.col("wo")).mean().alias("p_equal"))
              .sort("build_num").rows(), flush=True)

# (2b) reason code of mismatched 9961 round-13 rows (bomb present): what does the round_end say?
Rm = R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null())
sec("(2b) 9961 round 13 with bomb outcome: crosstab bomb_side vs round_end winner and win_reason_code")
print(Rm.group_by(["bomb_side", "winner_team_code", "win_reason_code"]).len().sort(["bomb_side", "winner_team_code"]).rows(), flush=True)
sec("(2c) 9961 round 13 with bomb: planter side crosstab (should be 2=T when bomb planted)")
print(Rm.group_by(["planter_side", "bomb_side", "winner_team_code"]).len().sort(["planter_side", "bomb_side"]).rows(), flush=True)

# (3) halftime and overtime structure per match
fe_phase = (chan("round_state").select(["match_key", "round", "tick", "event_type", "phase"])
            .filter(pl.col("event_type") == "round_freeze_end").collect()
            .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fs = (fe_phase.filter(pl.col("phase") == "second").group_by("match_key")
      .agg(pl.col("round").min().alias("first_second")))
mx = re_t.group_by("match_key").agg(pl.col("round").max().alias("last_round"))
M = (hdr.select(["match_key", "build_num", "mdate", "map_name", "srv_region", "network_protocol",
                 "game_directory", "demo_version_guid", "ppp_version"])
     .join(fs, on="match_key", how="left").join(mx, on="match_key", how="left"))
M = M.with_columns(pl.when(pl.col("first_second") == 9).then(pl.lit("MR8_second_at9"))
                   .when(pl.col("first_second") == 13).then(pl.lit("MR12_second_at13"))
                   .when(pl.col("first_second").is_null()).then(pl.lit("no_second_phase"))
                   .otherwise(pl.lit("other")).alias("fmt"))
sec("(3a) match format by build (first round with phase 'second')")
print(M.group_by(["build_num", "fmt", "first_second"]).len().sort(["build_num", "fmt", "first_second"]).rows(), flush=True)
print("OT (last_round>24) by build:", M.with_columns((pl.col("last_round") > 24).alias("ot"))
      .group_by(["build_num", "ot"]).len().sort(["build_num", "ot"]).rows(), flush=True)

# per-round defect by format: score-vs-end, bomb-vs-end
P = R.select(["match_key", "round", "winner_team_code", "bomb_side", "reason_side"]).join(
    M.select(["match_key", "build_num", "fmt", "first_second", "last_round"]), on="match_key", how="left")
# score-delta winner recomputed here for the format split (same rule as the claim)
fe_nx = (fe_phase.sort(["match_key", "round"]).select(["match_key", "round"])) # placeholder for shape
# rebuild score deltas with the claim's rule
fes = (chan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
       .filter(pl.col("event_type") == "round_freeze_end").collect()
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
nx = (fes.sort(["match_key", "round"])
      .with_columns(pl.col("round").shift(-1).over("match_key").alias("r_next"),
                    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
                    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"))
      .filter(pl.col("r_next") == pl.col("round") + 1)
      .with_columns(((pl.col("t_next") - pl.col("t_score")).alias("dT")),
                    ((pl.col("ct_next") - pl.col("ct_score")).alias("dCT")))
      .with_columns(pl.when(pl.col("dT") == 1).then(2).when(pl.col("dCT") == 1).then(3).otherwise(0).alias("w_score"))
      .select(["match_key", "round", "dT", "dCT", "w_score"]))
P = P.join(nx, on=["match_key", "round"], how="left")
P = P.with_columns((pl.col("w_score") != 0).alias("has_score"),
                   (pl.col("w_score") != pl.col("winner_team_code")).alias("score_bad"),
                   (pl.col("bomb_side") != pl.col("winner_team_code")).alias("bomb_bad"))
P = P.with_columns(pl.col("build_num").cast(pl.Int64))
sec("(3b) per build x format x round k in {8,9,12,13,24..28}: score-vs-end bad/n | bomb-vs-end bad/n (bomb rows)")
for bn in [9842, 9961]:
    for fmt in ["MR8_second_at9", "MR12_second_at13", "no_second_phase", "other"]:
        sub = P.filter((pl.col("build_num") == bn) & (pl.col("fmt") == fmt))
        if sub.height == 0:
            continue
        rows = []
        for k in [8, 9, 12, 13, 24, 25, 26, 27, 28, 29]:
            s = sub.filter(pl.col("round") == k)
            sc = s.filter(pl.col("has_score"))
            bm = s.filter(pl.col("bomb_side").is_not_null())
            rows.append((k, f"{int(sc['score_bad'].sum())}/{sc.height}", f"{int(bm['bomb_bad'].sum())}/{bm.height}"))
        print(f"  build {bn} {fmt} (k, score_bad/n, bomb_bad/n):", rows, flush=True)

# (3c) 9961 matches with NO disagreement: what are they?
sec("(3c) 9961 matches with zero score-vs-end disagreement: format and OT breakdown")
mdis = (P.filter(pl.col("has_score")).group_by(["match_key", "build_num", "fmt"])
        .agg(pl.col("score_bad").sum().alias("nbad"), pl.len().alias("npairs")))
zero = mdis.filter((pl.col("build_num") == 9961) & (pl.col("nbad") == 0))
print("9961 zero-disagreement matches:", zero.height, "| by fmt:", zero.group_by("fmt").len().rows(), flush=True)
nz = mdis.filter((pl.col("build_num") == 9961) & (pl.col("nbad") > 0))
print("9961 disagreeing matches by fmt:", nz.group_by("fmt").len().rows(), flush=True)
print("9961 zero-disagreement matches' last_round:", M.join(zero.select("match_key"), on="match_key")
      .group_by("last_round").len().sort("last_round").rows()[:15], flush=True)

# (4) same calendar-date control: 2024-02-07..2024-02-12 has both builds
sec("(4) same match_date window control (2024-02-07..2024-02-12): builds present and defect at k=13 (MR12)")
ctrl = M.filter((pl.col("mdate") >= "2024-02-07") & (pl.col("mdate") <= "2024-02-12"))
print("matches in window by build:", ctrl.group_by("build_num").len().sort("build_num").rows(), flush=True)
c13 = P.filter((pl.col("round") == 13) & pl.col("has_score")).join(
    hdr.select(["match_key", "mdate"]), on="match_key", how="left").filter(
    (pl.col("mdate") >= "2024-02-07") & (pl.col("mdate") <= "2024-02-12"))
print("k=13 score-vs-end by build in window (bad/n):", c13.group_by("build_num").agg(
    pl.col("score_bad").sum().alias("bad"), pl.len().alias("n")).sort("build_num").rows(), flush=True)

# (5) map and header alternatives: defect at k=13 among MR12 matches, by each field and build
sec("(5) k=13 score-vs-end defect by map and by header field, within build (MR12 matches only)")
K13 = P.filter((pl.col("round") == 13) & pl.col("has_score") & (pl.col("fmt") == "MR12_second_at13"))
K13 = K13.join(hdr.select(["match_key", "map_name", "srv_region", "network_protocol", "game_directory",
                           "demo_version_guid", "ppp_version"]), on="match_key", how="left")
for fld in ["map_name", "srv_region", "network_protocol", "game_directory", "demo_version_guid", "ppp_version"]:
    g = (K13.group_by(["build_num", fld]).agg(pl.col("score_bad").sum().alias("bad"), pl.len().alias("n"))
         .sort(["build_num", fld]))
    print(f"  field {fld}: rows (build, value, bad, n) count={g.height}", flush=True)
    for r in g.rows()[:40]:
        print("     ", r, flush=True)
print("build x network_protocol x game_directory (collinearity with build):",
      hdr.group_by(["build_num", "network_protocol", "game_directory"]).len().sort(["build_num"]).rows(), flush=True)
print("build x srv_region counts:", hdr.group_by(["build_num", "srv_region"]).len().sort(["build_num", "srv_region"]).rows(),
      flush=True)
print("build x ppp_version counts:", hdr.group_by(["build_num", "ppp_version"]).len().sort(["build_num", "ppp_version"]).rows(),
      flush=True)
print("build x demo_version_guid counts:", hdr.group_by(["build_num", "demo_version_guid"]).len().sort(["build_num", "demo_version_guid"]).rows(),
      flush=True)

# (6) signature duplicates and build agreement
sec("(6) signature duplicates (same date|map|scores|points) and build agreement")
hs = hdr.with_columns(pl.concat_str([pl.col("mdate"), pl.col("map_name"),
                                     pl.col("t_starters_score_final").cast(pl.Utf8),
                                     pl.col("ct_starters_score_final").cast(pl.Utf8),
                                     pl.col("number_of_points").cast(pl.Utf8)], separator="|").alias("sig"))
dg = hs.group_by("sig").agg(pl.len().alias("n"), pl.col("build_num").n_unique().alias("nb"),
                            pl.col("build_num").alias("builds"))
dgd = dg.filter(pl.col("n") > 1)
print("signature groups with n>1:", dgd.height, "| groups spanning two builds:", dgd.filter(pl.col("nb") > 1).height,
      "| total matches in dup groups:", int(dgd["n"].sum()), flush=True)
print("dup groups by build set:", dgd.group_by("nb").len().rows(), flush=True)

sec("(7) sample k=27 and k=13 rows (9961) for direct inspection of score deltas vs end and bomb")
for k, bn in [(27, 9842), (27, 9961), (13, 9961)]:
    s = (P.filter((pl.col("round") == k) & (pl.col("build_num") == bn) & pl.col("has_score"))
         .select(["match_key", "round", "dT", "dCT", "w_score", "winner_team_code", "bomb_side", "reason_side"]).head(6))
    print(f"  build {bn} k={k} sample:", s.rows(), flush=True)

sec("(8) 9842 and 9961 round-27 disagreements: OT overtime structure (last_round, fmt)")
k27 = P.filter((pl.col("round") == 27) & pl.col("has_score")).group_by(["build_num", "score_bad"]).len().sort(["build_num", "score_bad"])
print(k27.rows(), flush=True)
print("DONE checks", flush=True)
