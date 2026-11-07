"""verify D5-6, step 2: build an independent per-(match, round) table for the 2024 tome.

Reads the 2024 tome on G: directly (not the D5 scratch parquet), with column projection and
event-type filters applied before collect. Writes parquet only to the session scratchpad.
Definitions follow the claim where it defines them (D5_02 / D5_06) and are marked ASSUMPTION
where the claim is silent.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
SCR.mkdir(parents=True, exist_ok=True)
K = ["match_key", "round"]


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(ch)
    return pl.scan_parquet(fs).select(cols)


print("polars", pl.__version__)

# ---------- header ----------
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = (pl.scan_parquet(hf).select(["key", "match_date", "map_name"]).collect()
     .rename({"key": "match_key"})
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))
print("header rows:", H.height, " unique keys:", H["match_key"].n_unique())

# ---------- round_end (raw, with duplicates) ----------
RE_raw = (scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
          .collect().with_row_index("ridx"))
print("\nround_end rows:", RE_raw.height, " unique match_key:", RE_raw["match_key"].n_unique())
dup = RE_raw.group_by(K).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"),
                             pl.col("tick").n_unique().alias("nt"))
print("(match,round) pairs in round_end:", dup.height)
print("rows per pair distribution:")
print(dup.group_by("n").len().sort("n"))
print("pairs with >1 distinct winner_team_code:", dup.filter(pl.col("nw") > 1).height,
      " pairs with >1 distinct tick:", dup.filter(pl.col("nt") > 1).height)

RE_s = RE_raw.sort(["match_key", "round", "tick", "ridx"])
RE_first = RE_s.unique(subset=K, keep="first", maintain_order=True).select(
    K + [pl.col("tick").alias("re_tick"), pl.col("winner_team_code").alias("rw"),
         pl.col("win_reason_code").alias("rw_reason"), pl.col("player_count").alias("rw_pc")])
RE_last = RE_s.unique(subset=K, keep="last", maintain_order=True).select(
    K + [pl.col("tick").alias("re_tick_last"), pl.col("winner_team_code").alias("rw_last")])
RE_dup_meta = dup.select(K + [pl.col("n").alias("re_nrows"), pl.col("nw").alias("re_nwin")])

# ---------- round_start ----------
RST = (scan("round_start", ["match_key", "round", "tick"]).collect()
       .group_by(K).agg(pl.col("tick").min().alias("start_tick"), pl.len().alias("n_start"))
       .sort(K)
       .with_columns(nxt_start=pl.col("start_tick").shift(-1).over("match_key"),
                     nxt_round=pl.col("round").shift(-1).over("match_key")))
RST = RST.with_columns(nxt_start=pl.when(pl.col("nxt_round") == pl.col("round") + 1)
                       .then(pl.col("nxt_start")).otherwise(None)).drop("nxt_round")
print("\nround_start pairs:", RST.height)

# ---------- round_state: freeze end, round_end, bomb events ----------
RS_EV = ["round_freeze_end", "round_end", "bomb_planted", "bomb_defused", "bomb_exploded", "round_start"]
RS = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(RS_EV)).collect())
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(subset=K, keep="first", maintain_order=True)
      .select(K + [pl.col("tick").alias("fe_tick"), pl.col("t_score").alias("fe_t"), pl.col("ct_score").alias("fe_ct")]))
RS_END = (RS.filter(pl.col("event_type") == "round_end").group_by(K)
          .agg(pl.col("tick").min().alias("rs_end_tick"), pl.len().alias("rs_end_n")))
RS_PL = (RS.filter(pl.col("event_type") == "bomb_planted").group_by(K)
         .agg(pl.len().alias("rs_planted")))
print("round_state freeze_end pairs:", FE.height, " round_state round_end pairs:", RS_END.height,
      " rs bomb_planted pairs:", RS_PL.height)

# ---------- bomb_state (bomb events) ----------
BS = (scan("bomb_state", ["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())
BS_P = BS.filter(pl.col("event_type") == "bomb_planted").group_by(K).agg(pl.len().alias("bs_planted"))
BS_D = BS.filter(pl.col("event_type") == "bomb_defused").group_by(K).agg(pl.len().alias("bs_defused"))
BS_X = BS.filter(pl.col("event_type") == "bomb_exploded").group_by(K).agg(pl.len().alias("bs_exploded"))
print("bomb_state planted pairs:", BS_P.height, " defused pairs:", BS_D.height, " exploded pairs:", BS_X.height)

# ---------- player_death ----------
DE = (scan("player_death", ["match_key", "round", "tick", "player_team_code"]).collect()
      .with_row_index("didx"))
print("\nplayer_death rows:", DE.height, " null victim side:", DE.filter(pl.col("player_team_code").is_null()).height)
V = DE.filter(pl.col("player_team_code").is_in([2, 3]))
vmax = V.group_by(K).agg(pl.col("tick").max().alias("last_tick"), pl.len().alias("n_victims"))
V_at = V.join(vmax, on=K).filter(pl.col("tick") == pl.col("last_tick"))
last_tbl = V_at.group_by(K).agg(
    pl.col("player_team_code").n_unique().alias("n_side_at_max"),
    pl.col("player_team_code").min().alias("last_side_min"),
    pl.col("player_team_code").max().alias("last_side_max"),
    pl.len().alias("n_at_max_tick"))
last_tbl = last_tbl.with_columns(
    last_victim=pl.when(pl.col("n_side_at_max") == 1).then(pl.col("last_side_min")).otherwise(None))
# first victim side (tick order), for the alternative 'first death' truth
first_tbl = (V.sort(["match_key", "round", "tick", "didx"]).unique(subset=K, keep="first", maintain_order=True)
             .select(K + [pl.col("player_team_code").alias("first_victim")]))
DEATH = vmax.join(last_tbl, on=K, how="left").join(first_tbl, on=K, how="left")
DEATH = DEATH.with_columns(
    implied=pl.when(pl.col("last_victim") == 2).then(pl.lit(3)).when(pl.col("last_victim") == 3).then(pl.lit(2))
    .otherwise(None),
    implied_first=pl.when(pl.col("first_victim") == 2).then(pl.lit(3)).when(pl.col("first_victim") == 3).then(pl.lit(2))
    .otherwise(None))
print("death pairs with >=1 victim side:", DEATH.height, " tie at max tick (ambiguous last victim):",
      DEATH.filter(pl.col("last_victim").is_null()).height)

# death-window check: a death belongs to its labelled round if start_tick <= tick < next round start
DW = (DE.filter(pl.col("player_team_code").is_in([2, 3])).select(K + ["tick"])
      .join(RST.select(K + ["start_tick", "nxt_start"]), on=K, how="left"))
DW = DW.with_columns(
    in_win=(pl.col("tick") >= pl.col("start_tick")) & (pl.col("nxt_start").is_null() | (pl.col("tick") < pl.col("nxt_start"))),
    before_start=pl.col("tick") < pl.col("start_tick"))
DWS = DW.group_by(K).agg(pl.len().alias("dw_n"), pl.col("in_win").sum().alias("dw_in"),
                         pl.col("before_start").sum().alias("dw_before"))

# ---------- assemble per-(match, round) table ----------
keys_all = pl.concat([RST.select(K), RE_first.select(K), FE.select(K), RS_END.select(K),
                      DEATH.select(K), BS_P.select(K), BS_D.select(K), BS_X.select(K)]).unique()
T = (keys_all.join(RST, on=K, how="left").join(RE_first, on=K, how="left").join(RE_last, on=K, how="left")
     .join(RE_dup_meta, on=K, how="left").join(FE, on=K, how="left").join(RS_END, on=K, how="left")
     .join(RS_PL, on=K, how="left").join(BS_P, on=K, how="left").join(BS_D, on=K, how="left")
     .join(BS_X, on=K, how="left").join(DEATH, on=K, how="left").join(DWS, on=K, how="left"))
T = T.with_columns(
    rs_planted=pl.col("rs_planted").fill_null(0), bs_planted=pl.col("bs_planted").fill_null(0),
    bs_defused=pl.col("bs_defused").fill_null(0), bs_exploded=pl.col("bs_exploded").fill_null(0),
    dw_n=pl.col("dw_n").fill_null(0), dw_in=pl.col("dw_in").fill_null(0), dw_before=pl.col("dw_before").fill_null(0),
    re_present=pl.col("rw").is_not_null())
# next-round freeze-end scores for the score-step (chain) truth, same match and round+1 only
T = T.sort(K).with_columns(
    nr=pl.col("round").shift(-1).over("match_key"),
    fe_t_n=pl.col("fe_t").shift(-1).over("match_key"),
    fe_ct_n=pl.col("fe_ct").shift(-1).over("match_key"))
T = T.with_columns(
    dt=pl.when(pl.col("nr") == pl.col("round") + 1).then(pl.col("fe_t_n") - pl.col("fe_t")).otherwise(None),
    dct=pl.when(pl.col("nr") == pl.col("round") + 1).then(pl.col("fe_ct_n") - pl.col("fe_ct")).otherwise(None))
T = T.with_columns(
    chain=pl.when(pl.col("dt").is_null() | pl.col("dct").is_null()).then(None)
    .when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(0)))

# ---------- bomb outcome side (defused -> 3, exploded -> 2), as in D5_02 ----------
T = T.with_columns(
    bomb_side=pl.when(pl.col("bs_defused") > 0).then(pl.lit(3))
    .when(pl.col("bs_exploded") > 0).then(pl.lit(2)).otherwise(None))

# ---------- match-level flags ----------
r13 = T.filter((pl.col("round") == 13) & pl.col("rw").is_not_null() & pl.col("bomb_side").is_not_null())
aff_pc20 = r13.filter(pl.col("rw_pc") == 20).filter(pl.col("bomb_side") != pl.col("rw")).select("match_key").unique()
aff_any = r13.filter(pl.col("bomb_side") != pl.col("rw")).select("match_key").unique()
r1e = T.filter((pl.col("round") == 1) & pl.col("re_present")).select("match_key").unique()
M = (H.join(r1e.with_columns(r1_end=pl.lit(True)), on="match_key", how="left")
     .with_columns(r1_end=pl.col("r1_end").fill_null(False),
                   aff_pc20=pl.col("match_key").is_in(aff_pc20["match_key"].to_list()),
                   aff_any=pl.col("match_key").is_in(aff_any["match_key"].to_list())))
print("\nmatches in header:", M.height, " r1_end true:", int(M["r1_end"].sum()),
      " aff (D5 def, player_count==20 filter):", int(M["aff_pc20"].sum()),
      " aff (no pc filter):", int(M["aff_any"].sum()))

T.write_parquet(SCR / "v_rounds.parquet")
M.write_parquet(SCR / "v_matches.parquet")
DE_win = DW.select(K + ["tick", "in_win", "before_start"])
print("\nrows written:", T.height, "pairs;", M.height, "matches. scratch dir:", SCR)
print("deaths with side in {2,3}:", DW.height, " in own window:", int(DW["in_win"].sum()),
      " before own round start:", int(DW["before_start"].sum()))
