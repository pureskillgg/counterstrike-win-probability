"""D5 step 5: is the round-13 disagreement a round-number shift, or a side/score-field artifact?

Per match, using only round_state freeze_end scores, round_end winner, and bomb_state outcomes:
  sw12      halftime score-field status across the 12 -> 13 boundary:
              'side_swap'  t_score(13) == ct_score(12) + [w12==3] and ct_score(13) == t_score(12) + [w12==2]
              'no_swap'    t_score(13) == t_score(12) + [w12==2] and ct_score(13) == ct_score(12) + [w12==3]
              'other'      neither
  chain13   winner of round 13 from the freeze-end score step 13 -> 14 (2 = T, 3 = CT, 0 = not a +1 step)
  rw13      round_end winner_team_code for label 13
  bomb13    bomb outcome side for label 13 (3 if bomb_defused, 2 if bomb_exploded, null if none)
Cross-tab by group (aff_r13 / ctrl_late / early) and sw12.
Also repeats the same cross-tab for labels 25 and 28 (first rounds after OT side swaps).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_05_swap.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
B = pl.read_parquet(SCR / "bomb.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))

# bomb side per (match, label)
BSIDE = (B.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
         .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                       .otherwise(pl.lit(2)))
         .select(["match_key", "round", "bomb_side"]))

base = R.select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner", "in_fe", "in_end"])


def at(label):
    return base.filter(pl.col("round") == label).select(
        ["match_key", pl.col("fe_t_score").alias(f"t{label}"), pl.col("fe_ct_score").alias(f"ct{label}"),
         pl.col("winner").alias(f"w{label}")])


T = at(12).join(at(13), on="match_key", how="full", coalesce=True).join(at(14), on="match_key", how="full", coalesce=True)
T = T.join(M.select(["match_key", "grp"]), on="match_key", how="left")
T = T.with_columns(
    sw12=pl.when(
        (pl.col("t13") == pl.col("ct12") + pl.when(pl.col("w12") == 3).then(1).otherwise(0))
        & (pl.col("ct13") == pl.col("t12") + pl.when(pl.col("w12") == 2).then(1).otherwise(0))
    ).then(pl.lit("side_swap"))
    .when(
        (pl.col("t13") == pl.col("t12") + pl.when(pl.col("w12") == 2).then(1).otherwise(0))
        & (pl.col("ct13") == pl.col("ct12") + pl.when(pl.col("w12") == 3).then(1).otherwise(0))
    ).then(pl.lit("no_swap")).otherwise(pl.lit("other")),
    dt=pl.col("t14") - pl.col("t13"),
    dct=pl.col("ct14") - pl.col("ct13"),
)
T = T.with_columns(chain13=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                   .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
T = T.join(BSIDE.filter(pl.col("round") == 13).select(["match_key", "bomb_side"]), on="match_key", how="left")

print("== label 13: sw12 status (halftime score fields) by group ==")
print(T.group_by(["grp", "sw12"]).agg(pl.len().alias("matches")).sort(["grp", "sw12"]))

X = T.filter(pl.col("w13").is_not_null()).with_columns(
    rw_eq_chain=pl.when(pl.col("chain13") == 0).then(pl.lit("chain_zero")).when(pl.col("chain13") == pl.col("w13"))
    .then(pl.lit("agree")).otherwise(pl.lit("disagree")),
    bomb_status=pl.when(pl.col("bomb_side").is_null()).then(pl.lit("no_bomb_r13"))
    .when(pl.col("bomb_side") == pl.col("w13")).then(pl.lit("bomb=rw"))
    .otherwise(pl.lit("bomb!=rw")),
    chain_vs_bomb=pl.when(pl.col("bomb_side").is_null()).then(pl.lit("no_bomb_r13"))
    .when(pl.col("chain13") == 0).then(pl.lit("chain_zero"))
    .when(pl.col("chain13") == pl.col("bomb_side")).then(pl.lit("chain=bomb")).otherwise(pl.lit("chain!=bomb")),
)
print("\n== label 13: chain vs round_end winner vs bomb, by group x sw12 ==")
print(X.group_by(["grp", "sw12", "rw_eq_chain", "bomb_status", "chain_vs_bomb"]).agg(pl.len().alias("matches"))
      .sort(["grp", "sw12", "rw_eq_chain", "bomb_status", "chain_vs_bomb"]))

print("\n== label 13 'disagree' rows: sw12 status and whether a bomb outcome exists ==")
dis = X.filter(pl.col("rw_eq_chain") == "disagree")
print(dis.group_by(["grp", "sw12", "bomb_status"]).agg(pl.len().alias("matches")).sort(["grp", "sw12", "bomb_status"]))

# label-level flip rate across labels (first rounds after side swaps: 13, 25, 28)
print("\n== per-label chain vs round_end agreement for labels 24..29, by group and sw status of prior boundary ==")
rows = []
for lab in [12, 13, 14, 24, 25, 26, 27, 28, 29]:
    cur = base.filter(pl.col("round") == lab).select(["match_key", pl.col("fe_t_score").alias("t"),
                                                    pl.col("fe_ct_score").alias("ct"), pl.col("winner").alias("w")])
    nxt = base.filter(pl.col("round") == lab + 1).select(["match_key", pl.col("fe_t_score").alias("tn"),
                                                        pl.col("fe_ct_score").alias("ctn")])
    j = cur.join(nxt, on="match_key", how="inner").join(M.select(["match_key", "grp"]), on="match_key", how="left")
    j = j.with_columns(dt=pl.col("tn") - pl.col("t"), dct=pl.col("ctn") - pl.col("ct")).with_columns(
        ch=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
        .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
    j = j.filter(pl.col("w").is_not_null())
    g = j.group_by("grp").agg(pl.len().alias("pairs"),
                              (pl.col("ch") == pl.col("w")).sum().alias("agree"),
                              (pl.col("ch") == 0).sum().alias("chain_zero")).with_columns(lab=pl.lit(lab))
    rows.append(g)
print(pl.concat(rows).with_columns((pl.col("agree") / pl.col("pairs")).round(3).alias("agree_rate"))
      .select(["lab", "grp", "pairs", "agree", "chain_zero", "agree_rate"]).sort(["lab", "grp"]))

# sw status for OT boundary (24 -> 25 and 27 -> 28)
print("\n== OT boundaries: sw status for 24->25 and 27->28 (score-field side mapping) ==")
for lab in [24, 27]:
    prev = base.filter(pl.col("round") == lab).select(["match_key", pl.col("fe_t_score").alias("t0"),
                                                      pl.col("fe_ct_score").alias("ct0"), pl.col("winner").alias("w0")])
    nxt = base.filter(pl.col("round") == lab + 1).select(["match_key", pl.col("fe_t_score").alias("t1"),
                                                         pl.col("fe_ct_score").alias("ct1")])
    k = prev.join(nxt, on="match_key", how="inner").join(M.select(["match_key", "grp"]), on="match_key", how="left")
    k = k.with_columns(
        sw=pl.when((pl.col("t1") == pl.col("ct0") + pl.when(pl.col("w0") == 3).then(1).otherwise(0))
                   & (pl.col("ct1") == pl.col("t0") + pl.when(pl.col("w0") == 2).then(1).otherwise(0))).then(pl.lit("side_swap"))
        .when((pl.col("t1") == pl.col("t0") + pl.when(pl.col("w0") == 2).then(1).otherwise(0))
              & (pl.col("ct1") == pl.col("ct0") + pl.when(pl.col("w0") == 3).then(1).otherwise(0))).then(pl.lit("no_swap"))
        .otherwise(pl.lit("other")))
    print(f"boundary {lab}->{lab + 1}:")
    print(k.group_by(["grp", "sw"]).agg(pl.len().alias("matches")).sort(["grp", "sw"]))

# Late-period ctrl matches: do label-13 disagreements coincide with round_end label-1 presence?
print("\n== label 1 round_end presence vs label-13 chain agreement (late/early groups) ==")
r1 = base.filter(pl.col("round") == 1).select(["match_key", pl.col("in_end").alias("r1_end")])
Y = X.join(r1, on="match_key", how="left")
print(Y.group_by(["grp", "r1_end", "rw_eq_chain"]).agg(pl.len().alias("matches")).sort(["grp", "r1_end", "rw_eq_chain"]))
