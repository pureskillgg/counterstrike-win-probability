"""verify D4-5, step 15: overtime side-swap convention for the 2024 MR12 overtime matches (pattern '13,25,28' group).

Complete 2024 MR12 matches (halftime label 11), days 12-25, labels from 1, with max_label > 24 (overtime).
For each candidate set of swap labels after regulation (convention A = bakeoff: 12, then 27, 33, ...;
B = add 24; C = add 24 and 30; D = add 24 and 27 only), compute header agreement (X total == hT) for:
  RE  : round_end winners
  fix : claim fix on rounds 9, 13, 25, 28 (bomb side, else score side, else round_end)
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, T, CT  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
DISP = [9, 13, 25, 28]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
bs = scan(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).collect()
bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .sort("tick").group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bev")))
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_label"))
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
Dp = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))
M = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"),
                                  pl.col("round").max().alias("max_label")).join(H, on="match_key").join(
    half, on="match_key", how="left")
M = M.with_columns(
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("day"))
M = M.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25) &
             (pl.col("half_label") == 11) & (pl.col("max_label") > 24))
keep = M["match_key"].to_list()
print("MR12 overtime complete matches:", len(keep))
R = re_.filter(pl.col("match_key").is_in(keep)).join(Dp.select(["match_key", "round", "ss"]),
                                                       on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left")
R = R.with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside")).sort(
    ["match_key", "round"])
conv = {
    "A_bakeoff(12,27,33)": lambda r: r == 12 or (r > 24 and (r - 27) % 6 == 0),
    "B_add24(12,24,27,33)": lambda r: r in (12, 24) or (r > 24 and (r - 27) % 6 == 0),
    "C_add24_30(12,24,27,30)": lambda r: r in (12, 24, 30) or (r > 24 and (r - 27) % 6 == 0),
    "D_add24_only_once(12,24,27)": lambda r: r in (12, 24, 27),
    "E_none_in_reg_except12(12)": lambda r: r == 12,
}
R = R.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .otherwise(pl.col("winner_team_code")).alias("w_bomb"))
R = R.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix_any"))
st = rs.filter(pl.col("match_key").is_in(keep)).sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last"))
rows = R.select(["match_key", "round", "winner_team_code", "w_fix_any"]).rows()
from collections import defaultdict  # noqa: E402
by = defaultdict(list)
for mk, r, w, wf in rows:
    by[mk].append((r, w, wf))
hT = dict(M.select(["match_key", "hT"]).rows())
for name, f in conv.items():
    ok_re = ok_fix = 0
    for mk, seq in by.items():
        s = 0
        xre = xfix = 0
        for r, w, wf in seq:
            on_T = (s % 2 == 0)
            if (w == T and on_T) or (w == CT and not on_T):
                xre += 1
            if (wf == T and on_T) or (wf == CT and not on_T):
                xfix += 1
            if f(r):
                s += 1
        ok_re += int(xre == hT[mk])
        ok_fix += int(xfix == hT[mk])
    print(f"{name:32s} RE matches header: {ok_re:3d} of {len(by)} | fix matches header: {ok_fix:3d} of {len(by)}")
