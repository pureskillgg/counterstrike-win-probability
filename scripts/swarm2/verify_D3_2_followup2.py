"""D3-2 verifier follow-up 2: per-round source table, round 27 / round 9 score anomalies, planter side-code check,
cross-round defect co-occurrence, and the E: tail copy spot check (fixed path index).

Reads only the G: 2024 tome (column-projected) and, for the spot check, a few parquet files on E: (read only).
ASSUMPTIONS (marked): 5v5 rosters; a side with 5 distinct deaths was eliminated (valid only in non-bomb rounds,
since a planted bomb keeps the round alive after the planting side is wiped).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_followup2.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(260)

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
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
PD = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])

day = lambda c: pl.col(c).str.extract(DAY_RE, 1).cast(pl.Int64)

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("ev")))
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tb"),
                                            pl.col("ct_score").first().alias("cb")))
fe_nx = fe.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("tb").alias("ta"),
                   pl.col("cb").alias("ca")])
sc = fe.join(fe_nx, on=["match_key", "round"], how="left").with_columns(
    (pl.col("ta") - pl.col("tb")).alias("dt"), (pl.col("ca") - pl.col("cb")).alias("dc"))
sc = sc.with_columns(side_from_delta("dt", "dc").alias("score_side"))

kills = (PD.filter(pl.col("player_team_code").is_in([T, CT]))
         .group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nd")))
dT = kills.filter(pl.col("player_team_code") == T).select(["match_key", "round", pl.col("nd").alias("dT")])
dC = kills.filter(pl.col("player_team_code") == CT).select(["match_key", "round", pl.col("nd").alias("dC")])
elim = dT.join(dC, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("dT").fill_null(0), pl.col("dC").fill_null(0)).with_columns(
    pl.when((pl.col("dT") == ROSTER) & (pl.col("dC") < ROSTER)).then(pl.lit(CT))
    .when((pl.col("dC") == ROSTER) & (pl.col("dT") < ROSTER)).then(pl.lit(T))
    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))

base = (RE.join(outc, on=["match_key", "round"], how="left")
        .join(sc.select(["match_key", "round", "dt", "dc", "score_side"]), on=["match_key", "round"], how="left")
        .join(elim.select(["match_key", "round", "elim_side"]), on=["match_key", "round"], how="left")
        .with_columns(day("match_key").alias("day"), pl.col("ev").is_not_null().alias("bomb"),
                      (pl.col("ev") == "bomb_exploded").alias("exploded")))
base = base.with_columns(
    pl.when(pl.col("bomb") & pl.col("exploded")).then(pl.lit(T))
    .when(pl.col("bomb")).then(pl.lit(CT)).otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
    (pl.col("winner_team_code") != pl.col("score_side")).alias("re_ne_sc"),
    (pl.col("winner_team_code") != pl.col("elim_side")).alias("re_ne_el"),
    (pl.col("score_side") != pl.col("elim_side")).alias("sc_ne_el"))

# ---------------------------------------------------------------- 1. per-round table, days 12-25
d = base.filter(pl.col("day") >= 12)
print("[1] per-round source table, days 12-25 (player_count==20):")
print("    round | rows | bomb | score_def | RE!=score | elim_def(nonbomb) | RE!=elim(nonbomb) | score!=elim(nonbomb)")
tab = d.group_by("round").agg(
    pl.len().alias("rows"),
    pl.col("bomb").sum().alias("bomb"),
    pl.col("score_side").is_not_null().sum().alias("sc_def"),
    pl.col("re_ne_sc").sum().alias("re_ne_sc"),
    (pl.col("elim_side").is_not_null() & ~pl.col("bomb")).sum().alias("el_def_nb"),
    (pl.col("re_ne_el") & ~pl.col("bomb")).sum().alias("re_ne_el_nb"),
    (pl.col("sc_ne_el") & ~pl.col("bomb") & pl.col("score_side").is_not_null() & pl.col("elim_side").is_not_null())
    .sum().alias("sc_ne_el_nb")).sort("round")
for row in tab.iter_rows(named=True):
    if row["rows"] >= 1:
        print(f"    {row['round']:>5} | {row['rows']:>4} | {row['bomb']:>4} | {row['sc_def']:>9} | {row['re_ne_sc']:>9} | "
              f"{row['el_def_nb']:>17} | {row['re_ne_el_nb']:>17} | {row['sc_ne_el_nb']:>17}")

# ---------------------------------------------------------------- 2. round 27 and round 9 score anomalies
for rnd in [9, 27]:
    s = d.filter(pl.col("round") == rnd)
    print(f"\n[2] round {rnd}, days 12-25: score (dt,dc) pair counts (top 8):")
    print(s.group_by(["dt", "dc"]).len().sort("len", descending=True).head(8).rows())
    sd = s.filter(pl.col("score_side").is_not_null())
    print(f"    score defined {sd.height}/{s.height}; bomb rows {int(s['bomb'].sum())}; "
          f"elim defined nonbomb {int((s['elim_side'].is_not_null() & ~s['bomb']).sum())}")
    print("    among score-defined rows: RE==score, RE==elim(nonbomb), score==elim(nonbomb):")
    nb = sd.filter(~pl.col("bomb") & pl.col("elim_side").is_not_null())
    print(f"      RE==score {int((~sd['re_ne_sc']).sum())}/{sd.height}; "
          f"RE==elim {int((~nb['re_ne_el']).sum())}/{nb.height}; score==elim {int((~nb['sc_ne_el']).sum())}/{nb.height}")
    bb = sd.filter(pl.col("bomb"))
    print(f"      bomb rows with score: n={bb.height}; RE!=score {int(bb['re_ne_sc'].sum())}; "
          f"score!=bomb_side {int((bb['score_side'] != bb['bomb_side']).sum())}")

# ---------------------------------------------------------------- 3. planter side-code consistency (PD vs BS codes)
print("\n[3] planter side-code check: BS planter team code vs PD victim team code for the same player and round:")
planters = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
            .group_by(["match_key", "round"]).agg(pl.col("player_id").first().alias("pid"),
                                                  pl.col("player_team_code").first().alias("bs_side")))
pdp = PD.select(["match_key", "round", pl.col("player_id").alias("pid"),
                 pl.col("player_team_code").alias("pd_side")]).unique(["match_key", "round", "pid"])
pj = planters.join(pdp, on=["match_key", "round", "pid"], how="inner").with_columns(day("match_key").alias("day"))
print(f"    planters who also died in the same round: {pj.height} of {planters.height}")
print("    (bs_side, pd_side) counts, days 12-25 vs 1-7:")
print(pj.with_columns(pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("p"))
      .group_by(["p", "bs_side", "pd_side"]).len().sort(["p", "bs_side", "pd_side"]).rows())
pl13 = pj.filter(pl.col("round") == 13)
print("    round 13 only, (bs_side, pd_side):", pl13.group_by(["bs_side", "pd_side"]).len().rows())

# ---------------------------------------------------------------- 4. co-occurrence of round_end defects across rounds within a match
print("\n[4] per-match co-occurrence (days 12-25), rounds 9/13/25/28, flag = RE != score (score defined):")
flags = (d.filter(pl.col("score_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 28]))
         .select(["match_key", "round", "re_ne_sc"])
         .pivot(values="re_ne_sc", index="match_key", on="round", aggregate_function="first"))
print("    columns:", flags.columns)
cols = [c for c in ["9", "13", "25", "28"] if c in flags.columns]
combo = flags.group_by(cols).len().sort("len", descending=True)
print(combo)

# ---------------------------------------------------------------- 5. E: tail copy spot check (correct uuid index)
print("\n[5] E: tail copy spot check, round 13 rows, first 4 day-12 bomb matches and first 2 non-bomb matches:")
TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12")
picks = (base.filter((pl.col("round") == 13) & (pl.col("day") == 12) & pl.col("bomb"))
         .select(["match_key", "winner_team_code", "bomb_side", "score_side"]).head(4).rows()
         + base.filter((pl.col("round") == 13) & (pl.col("day") == 12) & ~pl.col("bomb"))
         .select(["match_key", "winner_team_code", "bomb_side", "score_side"]).head(2).rows())
for mk, win, bside, sside in picks:
    uuid = mk.split("/")[4]
    folder = TAIL / uuid
    print(f"    {uuid}: folder exists={folder.exists()} tome r13 winner={win} bomb_side={bside} score_side={sside}")
    if not folder.exists():
        continue
    subs = sorted(os.listdir(folder))
    print(f"      children: {subs[:10]}")
    rel = None
    for s in subs:
        p = folder / s
        if p.is_dir() and "round_end" in s:
            rel = p
            break
    if rel is None:
        for s in subs:
            p = folder / s
            if p.is_dir():
                for t in os.listdir(p):
                    if "round_end" in t:
                        rel = p / t
                        break
            if rel is not None:
                break
    if rel is None:
        print("      no round_end folder found")
        continue
    fs = sorted(glob.glob(str(rel / "dataframe_*")))
    if not fs:
        print(f"      no dataframe_* under {rel}")
        continue
    raw = pl.concat([pl.read_parquet(f, columns=["round", "winner_team_code", "player_count"]) for f in fs])
    r13 = raw.filter((pl.col("round") == 13) & (pl.col("player_count") == 20))
    print(f"      raw r13 (player_count 20) rows={r13.height}: winners={r13['winner_team_code'].to_list()}")

print("\nDONE")
