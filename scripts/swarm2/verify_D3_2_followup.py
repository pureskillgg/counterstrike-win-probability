"""D3-2 verifier follow-up: is round_end the odd source on NON-bomb rounds too? Elimination and placement tests.

Reads only the G: 2024 tome (column-projected). Writes nothing except stdout (redirected to the log by the caller).
ASSUMPTIONS (marked): 5v5 rosters (a side with 5 distinct deaths was eliminated); round_state 'round_end'
event carries the post-round score; player_team_code on player_death is the victim's side.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_followup.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
DAY_RE = r"2024[/\\]02[/\\](\d\d)"
ROSTER = 5  # ASSUMPTION: 5v5


def g_read(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def side_from_delta(dt, dct):
    return (pl.when((pl.col(dct) == 1) & (pl.col(dt) == 0)).then(pl.lit(CT))
            .when((pl.col(dt) == 1) & (pl.col(dct) == 0)).then(pl.lit(T))
            .otherwise(pl.lit(None, dtype=pl.Int64)))


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).filter(
    pl.col("player_count") == 20)
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
PD = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
print(f"rows RE20={RE.height} BS={BS.height} RS={RS.height} PD={PD.height}")

day = lambda c: pl.col(c).str.extract(DAY_RE, 1).cast(pl.Int64)

# bomb outcome per round
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("ev")))
has_bomb = outc.select(["match_key", "round"]).with_columns(pl.lit(True).alias("has_bomb"))

# engine score change r -> r+1 (freeze_end based, claim logic)
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tb"),
                                            pl.col("ct_score").first().alias("cb"),
                                            pl.col("tick").first().alias("fe_tick")))
fe_nx = fe.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("tb").alias("ta"),
                   pl.col("cb").alias("ca"), pl.col("fe_tick").alias("fe_tick_next")])
sc = fe.join(fe_nx, on=["match_key", "round"], how="left").with_columns(
    (pl.col("ta") - pl.col("tb")).alias("dt"), (pl.col("ca") - pl.col("cb")).alias("dc"))
sc = sc.with_columns(side_from_delta("dt", "dc").alias("score_side"))

# round_state round_end event: post-round score (third engine source)
rse = (RS.filter(pl.col("event_type") == "round_end").sort("tick")
       .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("te"),
                                             pl.col("ct_score").first().alias("ce"),
                                             pl.col("tick").first().alias("rse_tick")))
rse = rse.join(fe.select(["match_key", "round", "tb", "cb"]), on=["match_key", "round"], how="left").with_columns(
    (pl.col("te") - pl.col("tb")).alias("dte"), (pl.col("ce") - pl.col("cb")).alias("dce"))
rse = rse.with_columns(side_from_delta("dte", "dce").alias("rse_side"))

# elimination truth from player deaths (ASSUMPTION: 5v5)
kills = (PD.filter(pl.col("player_team_code").is_in([T, CT]))
         .group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nd")))
dT = kills.filter(pl.col("player_team_code") == T).select(["match_key", "round", pl.col("nd").alias("dT")])
dC = kills.filter(pl.col("player_team_code") == CT).select(["match_key", "round", pl.col("nd").alias("dC")])
elim = dT.join(dC, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("dT").fill_null(0), pl.col("dC").fill_null(0))
elim = elim.with_columns(
    pl.when((pl.col("dT") == ROSTER) & (pl.col("dC") < ROSTER)).then(pl.lit(CT))
    .when((pl.col("dC") == ROSTER) & (pl.col("dT") < ROSTER)).then(pl.lit(T))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))

# base: every RE20 round with its flags
base = (RE.join(has_bomb, on=["match_key", "round"], how="left")
        .join(outc, on=["match_key", "round"], how="left")
        .join(sc.select(["match_key", "round", "score_side", "fe_tick", "fe_tick_next"]),
              on=["match_key", "round"], how="left")
        .join(rse.select(["match_key", "round", "rse_side", "rse_tick"]), on=["match_key", "round"], how="left")
        .join(elim.select(["match_key", "round", "elim_side", "dT", "dC"]), on=["match_key", "round"], how="left")
        .with_columns(day("match_key").alias("day"),
                      pl.col("has_bomb").fill_null(False).alias("bomb"),
                      (pl.col("ev") == "bomb_exploded").alias("exploded")))
base = base.with_columns(
    pl.when(pl.col("bomb") & pl.col("exploded")).then(pl.lit(T))
    .when(pl.col("bomb")).then(pl.lit(CT)).otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
base = base.with_columns(
    pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("period"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rb"))

# ---------------------------------------------------------------- A. validate elimination proxy on non-bomb rounds
print("\n[A] validate elimination proxy on NON-bomb rounds (elim_side vs engine score_side vs round_end winner):")
nb = base.filter(~pl.col("bomb") & pl.col("elim_side").is_not_null())
for per in ["d01-07", "d12-25"]:
    for rb in ["r13", "other"]:
        s = nb.filter((pl.col("period") == per) & (pl.col("rb") == rb))
        if s.height == 0:
            continue
        ve = s.filter(pl.col("score_side").is_not_null())
        a_sc = int((ve["elim_side"] == ve["score_side"]).sum())
        a_re = int((s["elim_side"] == s["winner_team_code"]).sum())
        print(f"    {per} {rb}: n_elim={s.height}; elim==score {a_sc}/{ve.height}; elim==round_end {a_re}/{s.height}")

# ---------------------------------------------------------------- B. the same three-way test on round 13, days 12-25
print("\n[B] round 13 days 12-25 (all rounds, bomb and non-bomb), agreement of round_end with each engine/elim source:")
r = base.filter((pl.col("rb") == "r13") & (pl.col("period") == "d12-25"))
for lab, flt in [("bomb rounds", pl.col("bomb")), ("non-bomb rounds", ~pl.col("bomb"))]:
    s = r.filter(flt)
    e = s.filter(pl.col("score_side").is_not_null())
    el = s.filter(pl.col("elim_side").is_not_null())
    print(f"    {lab}: n={s.height}; score defined {e.height}: round_end==score {int((e['winner_team_code'] == e['score_side']).sum())}; "
          f"elim defined {el.height}: round_end==elim {int((el['winner_team_code'] == el['elim_side']).sum())}, "
          f"score==elim {int((el['score_side'] == el['elim_side']).sum())}")
    print(f"      round_state round_end-event side defined {s.filter(pl.col('rse_side').is_not_null()).height}: "
          f"round_end==rse_side {int((s.filter(pl.col('rse_side').is_not_null())['winner_team_code'] == s.filter(pl.col('rse_side').is_not_null())['rse_side']).sum())}, "
          f"score==rse_side {int((s.filter(pl.col('rse_side').is_not_null() & pl.col('score_side').is_not_null()).filter(pl.col('score_side') == pl.col('rse_side')).height))}")

# per-round round_end disagreement against elim (non-bomb), days 12-25, to see which rounds are affected
print("\n[C] non-bomb rounds, days 12-25: round_end vs elim_side disagreement by round (round: n, disagree):")
c = (base.filter((pl.col("period") == "d12-25") & ~pl.col("bomb") & pl.col("elim_side").is_not_null())
     .with_columns((pl.col("winner_team_code") != pl.col("elim_side")).alias("dis"))
     .group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("round"))
print([(x, n, d) for x, n, d in c.rows() if d > 0])

# ---------------------------------------------------------------- D. per-match clustering on round 13 (score-defined rows)
print("\n[D] per-match pattern on round 13, days 12-25 (score defined): matches by (n, disagree-with-score):")
m = (base.filter((pl.col("rb") == "r13") & (pl.col("period") == "d12-25") & pl.col("score_side").is_not_null())
     .with_columns((pl.col("winner_team_code") != pl.col("score_side")).alias("dis"))
     .group_by("match_key").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")))
pat = m.with_columns(
    pl.when(pl.col("dis") == 0).then(pl.lit("none"))
    .when(pl.col("dis") == pl.col("n")).then(pl.lit("all"))
    .otherwise(pl.lit("mixed")).alias("pattern"))
print("    matches:", pat.height)
print(pat.group_by("pattern").len().sort("pattern"))
print("    per-match disagreement fraction histogram (dis/n bins):")
print(pat.with_columns((pl.col("dis") / pl.col("n")).round(1).alias("frac")).group_by("frac").len().sort("frac"))

# ---------------------------------------------------------------- E. placement test: round_end tick between own freeze_end and next freeze_end
print("\n[E] placement: round_end tick inside (freeze_end(r), freeze_end(r+1)) by round, days 12-25 (round: n, inside):")
pl_ = base.filter((pl.col("period") == "d12-25") & pl.col("fe_tick").is_not_null() & pl.col("fe_tick_next").is_not_null())
pl_ = pl_.with_columns(((pl.col("tick") > pl.col("fe_tick")) & (pl.col("tick") < pl.col("fe_tick_next"))).alias("inside"))
e_rows = pl_.group_by("round").agg(pl.len().alias("n"), pl.col("inside").sum().alias("inside")).sort("round")
print([(x, n, i) for x, n, i in e_rows.rows() if x in (9, 12, 13, 14, 24, 25, 26, 27, 28, 29)])
print("    overall inside fraction:", f"{int(pl_['inside'].sum())}/{pl_.height}")

# ---------------------------------------------------------------- F. round_end tick relative to bomb outcome, by flag (seconds at 64 ticks/s)
print("\n[F] round_end tick minus bomb outcome tick (ticks), exploded rounds in d12-25 with bomb_state outcome tick:")
out_t = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"),
                                               pl.col("event_type").first().alias("ev2")))
fx = (base.filter(pl.col("bomb") & (pl.col("period") == "d12-25"))
      .join(out_t, on=["match_key", "round"], how="left")
      .with_columns((pl.col("tick") - pl.col("out_tick")).alias("d_ticks"),
                    (pl.col("winner_team_code") != pl.col("bomb_side")).alias("mm")))
for ev in ["bomb_exploded", "bomb_defused"]:
    s = fx.filter(pl.col("ev2") == ev)
    for flag in [True, False]:
        q = s.filter(pl.col("mm") == flag)
        if q.height:
            print(f"    {ev} mm={flag}: n={q.height}; d_ticks median={q['d_ticks'].median()}; "
                  f"top values={q.group_by('d_ticks').len().sort('len', descending=True).head(4).rows()}")

# ---------------------------------------------------------------- G. E: raw tail copy, a few round-13 matches (read only)
print("\n[G] E: tail copy spot check (round_end winner, round 13 day 12 matches):")
TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12")
r13_12 = (base.filter((pl.col("rb") == "r13") & (pl.col("day") == 12) & pl.col("bomb"))
          .select(["match_key", "winner_team_code", "bomb_side"]).head(3).rows())
for mk, win_g, bside in r13_12:
    uuid = mk.split("/")[3]
    folder = TAIL / uuid
    print(f"    match {uuid}: exists={folder.exists()}")
    if not folder.exists():
        continue
    subs = sorted(os.listdir(folder))
    print(f"      children: {subs[:12]}")
    cand = [s for s in subs if "round_end" in s]
    if not cand:
        for s in subs:
            p = folder / s
            if p.is_dir():
                inner = [x for x in os.listdir(p) if "round_end" in x]
                if inner:
                    cand = [os.path.join(s, x) for x in inner]
                    break
    print(f"      round_end candidates: {cand[:3]}")
    if cand:
        base_p = folder / cand[0]
        fs = sorted(glob.glob(str(base_p / "dataframe_*")))
        if fs:
            raw = pl.concat([pl.read_parquet(f, columns=["round", "winner_team_code", "tick"]) for f in fs])
            raw13 = raw.filter(pl.col("round") == 13)
            print(f"      raw round 13 rows: {raw13.rows()}  | tome winner={win_g}, bomb_side={bside}")
        else:
            print("      no dataframe_* files under candidate")

print("\nDONE")
