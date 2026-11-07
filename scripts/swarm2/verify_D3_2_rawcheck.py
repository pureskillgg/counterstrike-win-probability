"""D3-2 verifier: (1) E: raw vs G: tome copy check on sampled 2024-02-12..25 matches (read only);
(2) round-9 defect rows: round duration and death counts (is round_end's claimed wipe supported?).
ASSUMPTIONS (marked): tick rate 64/s (measured in verify_D3_2_recompute.py); 5v5 rosters.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_rawcheck.py
"""
import glob
import random
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
RAWROOT = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
DAY_RE = r"2024[/\\]02[/\\](\d\d)"
TICKS_PER_S = 64.0  # ASSUMPTION (measured slope 64.000 in recompute script)
ROSTER = 5


def g_read(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def side_from_delta(dt, dct):
    return (pl.when((pl.col(dct) == 1) & (pl.col(dt) == 0)).then(pl.lit(CT))
            .when((pl.col(dt) == 1) & (pl.col(dct) == 0)).then(pl.lit(T))
            .otherwise(pl.lit(None, dtype=pl.Int64)))


RE_all = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
PD = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
day = lambda c: pl.col(c).str.extract(DAY_RE, 1).cast(pl.Int64)

RE = RE_all.filter(pl.col("player_count") == 20)
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tb"),
                                            pl.col("ct_score").first().alias("cb"),
                                            pl.col("tick").first().alias("fe_tick")))
fe_nx = fe.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("tb").alias("ta"),
                   pl.col("cb").alias("ca")])
sc = fe.join(fe_nx, on=["match_key", "round"], how="left").with_columns(
    (pl.col("ta") - pl.col("tb")).alias("dt"), (pl.col("ca") - pl.col("cb")).alias("dc"))
sc = sc.with_columns(side_from_delta("dt", "dc").alias("score_side"))
kills = (PD.filter(pl.col("player_team_code").is_in([T, CT]))
         .group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nd")))
dT = kills.filter(pl.col("player_team_code") == T).select(["match_key", "round", pl.col("nd").alias("dT")])
dC = kills.filter(pl.col("player_team_code") == CT).select(["match_key", "round", pl.col("nd").alias("dC")])
dd = dT.join(dC, on=["match_key", "round"], how="full", coalesce=True).with_columns(
    pl.col("dT").fill_null(0), pl.col("dC").fill_null(0))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("ev")))

base = (RE.join(sc.select(["match_key", "round", "fe_tick", "score_side"]), on=["match_key", "round"], how="left")
        .join(dd, on=["match_key", "round"], how="left")
        .join(outc, on=["match_key", "round"], how="left")
        .with_columns(day("match_key").alias("day"), pl.col("ev").is_not_null().alias("bomb"),
                      (pl.col("winner_team_code") != pl.col("score_side")).alias("re_ne_sc")))

# ---------------------------------------------------------------- 1. raw E: vs tome G: copy
print("[1] E: raw vs G: tome copy, sampled 2024-02-12..25 matches (round_end and bomb_state rows).")
d_ok = base.filter(pl.col("day") >= 12)
r13_def = d_ok.filter((pl.col("round") == 13) & pl.col("re_ne_sc")).select("match_key").unique().to_series().to_list()
r13_ok = d_ok.filter((pl.col("round") == 13) & ~pl.col("re_ne_sc") & pl.col("score_side").is_not_null()) \
    .select("match_key").unique().to_series().to_list()
random.seed(20261013)
sample = random.sample(r13_def, min(8, len(r13_def))) + random.sample(r13_ok, min(4, len(r13_ok)))
tot_rows = tot_eq = 0
for mk in sample:
    uuid = mk.split("/")[4]
    rdir = RAWROOT / f"{'%02d' % int(mk.split('/')[3])}" / uuid
    tome_re = RE_all.filter(pl.col("match_key") == mk).sort(["round", "tick"])
    if not (rdir / "round_end").exists():
        print(f"    {uuid}: raw round_end missing at {rdir}")
        continue
    raw_re = pl.read_parquet(rdir / "round_end", columns=["round", "tick", "winner_team_code", "win_reason_code",
                                                          "player_count"]).sort(["round", "tick"])
    tome_cmp = tome_re.select(["round", "tick", "winner_team_code", "win_reason_code", "player_count"]).sort(["round", "tick"])
    same = (raw_re.height == tome_cmp.height) and raw_re.equals(tome_cmp)
    n_diff = 0
    if raw_re.height == tome_cmp.height:
        n_diff = int((raw_re["winner_team_code"] != tome_cmp["winner_team_code"]).sum())
    tot_rows += tome_cmp.height
    tot_eq += int(same) * tome_cmp.height
    r13raw = raw_re.filter((pl.col("round") == 13) & (pl.col("player_count") == 20))["winner_team_code"].to_list()
    r13tome = tome_cmp.filter((pl.col("round") == 13) & (pl.col("player_count") == 20))["winner_team_code"].to_list()
    print(f"    {uuid} (day {mk.split('/')[3]}): raw rows={raw_re.height} tome rows={tome_cmp.height} "
          f"identical={same} winner_diffs={n_diff}; r13 raw={r13raw} tome={r13tome} "
          f"sample_kind={'defect' if mk in r13_def else 'ok'}")
print(f"    total compared rows {tot_rows}; identical-match rows {tot_eq}")

# ---------------------------------------------------------------- 2. round 9 defect rows: duration and deaths
print("\n[2] round 9, days 12-25, score-defined rows: duration (s, ASSUMPTION 64 ticks/s) and deaths by group.")
r9 = base.filter((pl.col("round") == 9) & (pl.col("day") >= 12) & pl.col("score_side").is_not_null()).with_columns(
    ((pl.col("tick") - pl.col("fe_tick")) / TICKS_PER_S).alias("dur_s"),
    pl.col("dT").fill_null(0).alias("dT0"), pl.col("dC").fill_null(0).alias("dC0"))
r9 = r9.with_columns(pl.when(pl.col("re_ne_sc")).then(pl.lit("RE!=score")).otherwise(pl.lit("RE==score")).alias("grp"))
print(r9.group_by("grp").agg(pl.len().alias("n"), pl.col("bomb").sum().alias("bomb"),
                             pl.col("dur_s").median().round(1).alias("dur_median_s"),
                             pl.col("dur_s").min().round(1).alias("dur_min_s"),
                             pl.col("dur_s").max().round(1).alias("dur_max_s"),
                             (pl.col("dT0") + pl.col("dC0")).median().alias("deaths_median"),
                             pl.col("dT0").max().alias("max_dT"), pl.col("dC0").max().alias("max_dC")).sort("grp").rows())
nb = r9.filter(~pl.col("bomb") & pl.col("re_ne_sc"))
print("    RE!=score non-bomb rows: RE winner x RE reason x score side (counts):")
print(nb.group_by(["winner_team_code", "win_reason_code", "score_side"]).len().sort("len", descending=True).rows())
print("    RE!=score non-bomb rows: max deaths on one side (should be >=5 if a wipe occurred):",
      nb.select(pl.max_horizontal("dT0", "dC0").max()).item(), "| rows with a side at >=5 deaths:",
      int(((nb["dT0"] >= ROSTER) | (nb["dC0"] >= ROSTER)).sum()), "of", nb.height)
print("    RE!=score non-bomb rows where RE reason implies a wipe of the other side but that side has <5 deaths:")
impl_ct_wipe = nb.filter(pl.col("win_reason_code") == 9)  # Terrorists_Win -> CT wiped
impl_t_wipe = nb.filter(pl.col("win_reason_code") == 8)   # CTs_Win -> T wiped
print("      Terrorists_Win rows:", impl_ct_wipe.height, "with dC>=5:", int((impl_ct_wipe["dC0"] >= ROSTER).sum()))
print("      CTs_Win rows:", impl_t_wipe.height, "with dT>=5:", int((impl_t_wipe["dT0"] >= ROSTER).sum()))
print("\nDONE")
