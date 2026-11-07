r"""Verifier for D4-5, step F: full round-number grid, independent evidence, 2024 (and 2026 control).

Evidence per (match, round):
  bomb side   : from bomb_state outcome rows (bomb_defused -> CT, bomb_exploded -> T). Game rule, no team codes.
  wipe side   : no-bomb rounds, exactly one side with >= TEAM_SIZE distinct dead player_ids inside the live window
                (freeze_end .. round_end). Wiped side loses. ASSUMPTION TEAM_SIZE = 5.
  score side  : freeze_end(r) -> freeze_end(r+1), unambiguous. Only when no swap is expected after r
                (2024 uses mr=24 per the D4 claim; 2026 uses header max_rounds).

Grid output, by era x day group x round number (all rounds 1..30, including swap rounds):
  bomb_n / bomb_end_ne   : bomb rounds, round_end winner != bomb side
  wipe_n / wipe_end_ne   : no-bomb rounds with a unique wipe, round_end winner != wipe-implied winner
  score_n / score_end_ne : valid unambiguous score side, score != round_end  (the D4 check)
  score_n / score_wipe_ne: same rows, score != wipe-implied winner (where a wipe exists)

Headline recomputation with the claim's own definitions (valid = no swap after r; unambiguous score side):
  2024 d12-25 round 13: valid rounds, disagreements with round_end, by day 12 and 17
  2024 d12-25 rounds 9, 25, 28 and all other rounds
  2024 d01-07 total valid and disagreements
  2026 all rounds (control)

Reads projected columns only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_f_grid.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(260)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3
TEAM_SIZE = 5  # ASSUMPTION


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def run(era, base):
    print(f"\n######################## ERA {era} ########################")
    if era == "2026":
        H = read_dir(base / "header", ["key", "max_rounds"])
        mr_map = {k: int(v) for k, v in zip(H["key"].to_list(), H["max_rounds"].to_list())}
        print("2026 max_rounds distribution:")
        print(H.group_by("max_rounds").len().sort("max_rounds"))
    else:
        mr_map = None
        print("2024: no max_rounds in header; mr = 24 for all matches (ASSUMPTION, as in the D4 claim)")

    def read(ch, cols):
        return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)

    fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
          .unique(["match_key", "round"], keep="first").rename({"tick": "live"}))
    re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code"])
           .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename({"tick": "end"}))
    base_rows = fe.join(re_, on=["match_key", "round"], how="inner").with_columns(
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
        pl.when(pl.col("day") == "12").then(pl.lit("d12"))
        .when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
        .when(pl.col("day").is_in(["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"])).then(pl.lit("d13-25"))
        .otherwise(pl.lit("d-other")).alias("grp"))

    # score side
    nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
        {"t_score": "t2", "ct_score": "c2"})
    sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
        (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
    if mr_map is None:
        sd = sd.with_columns(pl.lit(24).alias("mr"))
    else:
        sd = sd.with_columns(pl.col("match_key").replace_strict(mr_map, default=24, return_dtype=pl.Int64).alias("mr"))
    sd = sd.with_columns(pl.struct(["round", "mr"]).map_elements(
        lambda s: not swap_expected(s["round"], s["mr"]), return_dtype=pl.Boolean).alias("valid"))
    base_rows = base_rows.join(sd.select(["match_key", "round", "score_side", "valid"]), on=["match_key", "round"], how="left")

    # bomb side
    bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
    bo = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("bevs")).with_columns(
        pl.when(pl.col("bevs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
    base_rows = base_rows.join(bo.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")

    # wipe side (distinct dead players per side in the live window)
    de = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(
        pl.col("player_team_code").is_in([T, CT]))
    DW = de.join(base_rows.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
        (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
    vic = DW.group_by(["match_key", "round", "player_team_code"]).agg(pl.col("player_id").n_unique().alias("nv")).pivot(
        on="player_team_code", index=["match_key", "round"], values="nv")
    vic = vic.rename({"2": "nT", "3": "nCT"}) if "2" in vic.columns else vic
    base_rows = base_rows.join(vic, on=["match_key", "round"], how="left").with_columns(
        pl.col("nT").fill_null(0) if "nT" in vic.columns else pl.lit(0).alias("nT"),
        pl.col("nCT").fill_null(0) if "nCT" in vic.columns else pl.lit(0).alias("nCT"))
    base_rows = base_rows.with_columns(
        pl.when((pl.col("nT") >= TEAM_SIZE) & (pl.col("nCT") < TEAM_SIZE)).then(pl.lit(CT))
        .when((pl.col("nCT") >= TEAM_SIZE) & (pl.col("nT") < TEAM_SIZE)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("wipe_winner"))

    # grid
    G = base_rows.with_columns(
        pl.when(pl.col("bside").is_not_null()).then(pl.lit("bomb")).otherwise(pl.lit("nobomb")).alias("ev"))
    bomb_g = G.filter(pl.col("bside").is_not_null()).group_by(["grp", "round"]).agg(
        pl.len().alias("bomb_n"), (pl.col("winner_team_code") != pl.col("bside")).sum().alias("bomb_end_ne"))
    wipe_g = G.filter(pl.col("bside").is_null() & (pl.col("wipe_winner") != 0)).group_by(["grp", "round"]).agg(
        pl.len().alias("wipe_n"), (pl.col("winner_team_code") != pl.col("wipe_winner")).sum().alias("wipe_end_ne"))
    sc_g = G.filter(pl.col("valid") & (pl.col("score_side") != 0)).group_by(["grp", "round"]).agg(
        pl.len().alias("score_n"), (pl.col("score_side") != pl.col("winner_team_code")).sum().alias("score_end_ne"),
        (pl.col("score_side") != pl.col("bside")).filter(pl.col("bside").is_not_null()).sum().alias("score_bomb_ne"),
        (pl.col("bside").is_not_null()).sum().alias("score_bomb_rows"),
        (pl.col("score_side") != pl.col("wipe_winner")).filter((pl.col("wipe_winner") != 0) & pl.col("bside").is_null()).sum().alias("score_wipe_ne"),
        ((pl.col("wipe_winner") != 0) & pl.col("bside").is_null()).sum().alias("score_wipe_rows"))
    grid = (bomb_g.join(wipe_g, on=["grp", "round"], how="full", coalesce=True)
            .join(sc_g, on=["grp", "round"], how="full", coalesce=True).sort(["grp", "round"]).fill_null(0))
    print("\nGRID (all rounds 1-30; swap rounds marked with * where valid=False for 2024/2026 mr rule):")
    sw = {r: (r == 12 or (r > 24 and (r - 27) % 6 == 0)) for r in range(1, 40)}
    grid = grid.filter(pl.col("round") <= 30).with_columns(
        pl.when(pl.col("round").map_elements(lambda r: sw.get(r, False), return_dtype=pl.Boolean)).then(pl.lit("*"))
        .otherwise(pl.lit("")).alias("swap_mr24"))
    print(grid.select(["grp", "round", "swap_mr24", "bomb_n", "bomb_end_ne", "wipe_n", "wipe_end_ne",
                       "score_n", "score_end_ne", "score_bomb_rows", "score_bomb_ne", "score_wipe_rows", "score_wipe_ne"]))

    # headline with the claim's definitions
    V = base_rows.filter(pl.col("valid") & (pl.col("score_side") != 0))
    print("\nHEADLINE (claim definitions): valid & unambiguous score side vs round_end, round 13 by day group:")
    r13 = V.filter(pl.col("round") == 13).with_columns(
        (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
    print(r13.group_by("grp").agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"),
                                  pl.col("dis").mean().alias("rate")).sort("grp"))
    r13d = r13.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("dd"))
    print("round 13 by day (dd):")
    print(r13d.group_by("dd").agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree")).sort("dd"))
    print("\nrounds 9, 25, 28 by group:")
    print(V.filter(pl.col("round").is_in([9, 25, 28])).with_columns(
        (pl.col("score_side") != pl.col("winner_team_code")).alias("dis")).group_by(["grp", "round"]).agg(
        pl.len().alias("n"), pl.col("dis").sum().alias("disagree")).sort(["grp", "round"]))
    other = V.filter(~pl.col("round").is_in([9, 13, 25, 28])).with_columns(
        (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
    print("\nall other rounds (valid, unambiguous), by group: n and disagreements")
    print(other.group_by("grp").agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree")).sort("grp"))
    tot = V.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("dis")).group_by("grp").agg(
        pl.len().alias("valid_unamb_rounds"), pl.col("dis").sum().alias("disagree_all_rounds")).sort("grp")
    print("\nall rounds valid & unambiguous, by group:")
    print(tot)


for era, base in TOMES.items():
    run(era, base)
