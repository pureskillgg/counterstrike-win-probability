"""verify_D9_2_b: regime (calendar time) check and an elimination-based check of which field is wrong.

Part 1: is any 9842 match dated after the first 9961 match (pre/post switch regime)? By regime, build,
        and round class (13, 9, 25, 28, other), for pc20 bomb-outcome rounds.
Part 2: elimination consistency on NO-PLANT rounds (no bomb_planted, no bomb outcome), which do not use
        bomb events. A T-side round_end winner needs every CT player dead (CT_all_dead). If the recorded
        winner is flipped in build 9961 round 13, the T-winner rows would show a low CT_all_dead rate there.
        ASSUMPTION: n_T / n_CT are distinct player_info players per side in the round (team_code 2 / 3);
        death victims are distinct (match_key, round, player_id), side = player_death.player_team_code.
Part 3: bomb-outcome round 13 pc20 rows: bomb event tick inside [round_start tick, round_end tick]
        (ASSUMPTION: round_start = min tick of round_start rows; round_end = round_end tick), and the
        death pattern of mismatches vs matches.
Memory: projected columns and lazy scans.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    miss = [c for c in cols if c not in have]
    print(f"[load] {name}: cols={[c for c in cols if c in have]} missing={miss}")
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "match_date", "build_num", "ppp_version"]).with_columns(
    pl.col("key").alias("match_key"),
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("folder_day"),
)
T9961 = H.filter(pl.col("build_num") == 9961)["match_date"].min()
print("switch = earliest 9961 match_date:", T9961)
H = H.with_columns((pl.col("match_date") >= T9961).alias("post_switch"))

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).group_by(
    ["match_key", "round"]).agg(pl.col("tick").min().alias("re_tick"),
                                pl.col("winner_team_code").first().alias("winner"),
                                pl.col("player_count").first().alias("pc"))

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type"])
OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).sort("tick").group_by(
    ["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
    pl.col("tick").min().alias("out_tick"))
PLANT = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.len().alias("n_plant"), pl.col("tick").min().alias("plant_tick"))

RS = scan("round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("rs_tick"))

DE_raw = scan("player_death", ["match_key", "round", "player_id", "player_team_code", "tick"])
DE = DE_raw.sort("tick").unique(subset=["match_key", "round", "player_id"], keep="first")
DEAD = DE.group_by(["match_key", "round"]).agg(
    (pl.col("player_team_code") == 2).sum().alias("T_dead"),
    (pl.col("player_team_code") == 3).sum().alias("CT_dead"))

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
SIDES = PIu.group_by(["match_key", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("n_T"),
    (pl.col("team_code") == 3).sum().alias("n_CT"))

R = (RE.join(OUT, on=["match_key", "round"], how="left")
       .join(PLANT, on=["match_key", "round"], how="left")
       .join(RS, on=["match_key", "round"], how="left")
       .join(DEAD, on=["match_key", "round"], how="left")
       .join(SIDES, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num", "post_switch", "ppp_version", "folder_day"]),
             on="match_key", how="left")
       .with_columns(pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
                     pl.col("n_plant").fill_null(0), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0), pl.col("n_T").fill_null(0),
                     pl.col("n_CT").fill_null(0))
       .with_columns((pl.col("defused") | pl.col("exploded")).alias("has_out"),
                     (pl.col("pc") == 20).alias("pc20"),
                     (pl.col("build_num") == 9961).alias("is9961"))
       .with_columns(
           pl.when(pl.col("has_out")).then((pl.when(pl.col("defused")).then(3).otherwise(2) != pl.col("winner")).cast(pl.Int64))
             .otherwise(None).alias("mm"),
       )
       .with_columns(
           ((pl.col("n_CT") > 0) & (pl.col("CT_dead") >= pl.col("n_CT"))).alias("CT_all_dead"),
           ((pl.col("n_T") > 0) & (pl.col("T_dead") >= pl.col("n_T"))).alias("T_all_dead"),
           pl.when(pl.col("has_out")).then(
               (pl.col("rs_tick") <= pl.col("out_tick")) & (pl.col("out_tick") <= pl.col("re_tick")))
             .otherwise(None).alias("bomb_in_window"),
       ))

print("\n== PART 1. regime x build x round class, pc20 bomb-outcome rounds")
P = R.filter(pl.col("has_out") & pl.col("pc20"))
P = P.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                   .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28_OT"))
                   .when(pl.col("round") == 9).then(pl.lit("r9"))
                   .otherwise(pl.lit("other")).alias("rclass"))
print(P.group_by(["post_switch", "build_num", "rclass"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
        .sort(["post_switch", "build_num", "rclass"]))
print("9842 post-switch matches (match_date >= switch):",
      H.filter((pl.col("build_num") == 9842) & pl.col("post_switch")).select(["match_key", "match_date", "folder_day", "ppp_version"]).rows())
print("9842 pc20 outcome rounds post-switch, all rounds:",
      P.filter((pl.col("build_num") == 9842) & pl.col("post_switch")).height)
print("9842 any-pc outcome rounds post-switch:",
      R.filter(pl.col("has_out") & (pl.col("build_num") == 9842) & pl.col("post_switch")).height)
print("9842 any-pc rounds (no outcome needed) post-switch:",
      R.filter((pl.col("build_num") == 9842) & pl.col("post_switch")).height)
print("9842 rounds post-switch by pc:",
      R.filter((pl.col("build_num") == 9842) & pl.col("post_switch")).group_by("pc").len().sort("pc").rows())

print("\n== PART 2. no-plant rounds (no bomb_planted, no outcome): elimination consistency")
NP = R.filter((pl.col("n_plant") == 0) & ~pl.col("has_out") & pl.col("pc20"))
print("no-plant pc20 rounds total:", NP.height)
print("  by build:", NP.group_by("build_num").agg(pl.len().alias("n")).sort("build_num").rows())
NP13 = NP.filter(pl.col("round") == 13)
print("\n  round 13 no-plant pc20 by build x recorded winner (T=2, CT=3):")
print(NP13.group_by(["build_num", "winner"]).agg(
    pl.len().alias("n"),
    pl.col("CT_all_dead").sum().alias("CT_all_dead"),
    pl.col("T_all_dead").sum().alias("T_all_dead"),
    pl.col("n_CT").mean().alias("mean_nCT"),
    pl.col("n_T").mean().alias("mean_nT")).sort(["build_num", "winner"]))
print("  note: T-winner consistency = CT_all_dead; CT-winner consistency = T_all_dead (timeouts make CT-winner ambiguous)")
NPx = NP.filter(pl.col("round") != 13)
print("\n  non-13 no-plant pc20 by build x recorded winner (baseline):")
print(NPx.group_by(["build_num", "winner"]).agg(
    pl.len().alias("n"),
    pl.col("CT_all_dead").sum().alias("CT_all_dead"),
    pl.col("T_all_dead").sum().alias("T_all_dead")).sort(["build_num", "winner"]))
NP5 = NP13.filter((pl.col("n_T") == 5) & (pl.col("n_CT") == 5))
print("\n  round 13 no-plant 5v5 only:")
print(NP5.group_by(["build_num", "winner"]).agg(
    pl.len().alias("n"),
    pl.col("CT_all_dead").sum().alias("CT_all_dead"),
    pl.col("T_all_dead").sum().alias("T_all_dead")).sort(["build_num", "winner"]))

print("\n== PART 3. round 13 pc20 bomb-outcome rows: mismatch vs death pattern and bomb tick window")
B13 = P.filter(pl.col("round") == 13)
print("  by build x mm: (n, T_all_dead, CT_all_dead)")
print(B13.group_by(["build_num", "mm"]).agg(pl.len().alias("n"),
                                              pl.col("T_all_dead").sum().alias("T_all_dead"),
                                              pl.col("CT_all_dead").sum().alias("CT_all_dead")).sort(["build_num", "mm"]))
print("  bomb event tick inside round window (rs_tick..re_tick) by build x mm:")
print(B13.group_by(["build_num", "mm", "bomb_in_window"]).agg(pl.len().alias("n")).sort(["build_num", "mm", "bomb_in_window"]))
print("  outcome kind x recorded winner x mm (r13, 9961):")
print(B13.filter(pl.col("is9961")).with_columns(
    pl.when(pl.col("defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("kind"))
     .group_by(["kind", "winner", "mm"]).agg(pl.len().alias("n")).sort(["kind", "winner", "mm"]))
print("  9961 r13 mismatch rows: winner text messages are in the round_end table (not re-read here)")
print("\nDONE part b")
