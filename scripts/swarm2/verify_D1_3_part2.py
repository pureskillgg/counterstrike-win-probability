"""verify_D1_3 part 2: build-number alternative, neighbour-index shift, validated within-round score
side, and an elimination-based winner (player_death roster) that uses neither round_end nor bomb events.

Read-only on the 2024 tome. Output is redirected to logs\\swarm2\\verify_D1_3.log by the caller.

ASSUMPTIONS (marked in the log):
  B1 build era = header.build_num (9842 vs 9961). Matches with 'unknown' platform are kept.
  B2 neighbour window: prev round_end.tick < bomb outcome tick <= this round_end.tick (no tolerance).
  B3 within-round score side: t/ct score at round_prestart (pre) vs round_officially_ended (post)
     of the same round (fallback: first/last row). Validated against bomb side on bomb rounds.
  B4 elimination winner: if all CT players (roster from player_info) died before round_end -> T (2);
     if all T players died and no bomb_planted in the round -> CT (3). Roster = player_info of the
     same round, death tick <= round_end tick. Uses no round_end and no bomb outcome.
  B5 only player_count == 20 rounds are scored.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
CLAIM_ROUNDS = [9, 13, 25, 28]


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def rbin_expr():
    return (pl.when(pl.col("round").is_in(CLAIM_ROUNDS)).then(pl.col("round").cast(pl.Utf8))
            .when(pl.col("round").is_between(14, 24)).then(pl.lit("14-24"))
            .otherwise(pl.lit("other")).alias("rbin"))


# ------------------------------------------------------------------ load
RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True))
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
RST = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"])
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD = (pl.concat([pl.read_parquet(f, columns=["key", "map_name", "match_date", "build_num", "demo_version_name", "platform"])
                 for f in hf], how="vertical_relaxed")
      .rename({"key": "match_key"}))
MATCH = HD.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("fday"))
print(f"loaded: round_end={RE.height} bomb_state={BS.height} round_state={RST.height} "
      f"player_info={PI.height} player_death={PD.height} header={MATCH.height}")
print("build_num counts:", MATCH.group_by("build_num").len().sort("build_num").to_dicts())

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("out_tick")))
planted = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))

RE_s = RE.sort(["match_key", "round"]).with_columns(
    pl.col("tick").shift(1).over("match_key").alias("prev_end"),
    pl.col("round").shift(1).over("match_key").alias("prev_round"))
RO_all = (RE_s.join(outc, on=["match_key", "round"], how="inner")
          .join(planted, on=["match_key", "round"], how="left")
          .with_columns(
              pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
              pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("otype"))
          .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
          .join(MATCH.select(["match_key", "fday", "match_date", "build_num", "map_name"]), on="match_key", how="left")
          .with_columns(rbin_expr()))
RO = RO_all.filter(pl.col("player_count") == 20)
print(f"bomb rounds p20={RO.height} mismatches={int(RO['mismatch'].sum())}")

# ------------------------------------------------------------------ B2 neighbour window
hdr("B2. OUTCOME TICK INSIDE (previous round_end, this round_end] ?")
w = RO_all.with_columns(
    ((pl.col("out_tick") > pl.col("prev_end")) & (pl.col("out_tick") <= pl.col("tick"))).alias("in_nb_window"),
    (pl.col("tick") - pl.col("out_tick")).alias("gap"))
print(w.group_by(["rbin", "mismatch", "in_nb_window"]).len().sort(["rbin", "mismatch", "in_nb_window"]))
print("gap == 0 vs gap == 429 vs other, by mismatch (all p20 bomb rounds):")
print(RO.with_columns((pl.col("tick") - pl.col("out_tick")).alias("gap")).with_columns(
    pl.when(pl.col("gap") == 0).then(pl.lit("0"))
      .when(pl.col("gap") == 429).then(pl.lit("429"))
      .otherwise(pl.lit("other")).alias("gcat")).group_by(["mismatch", "gcat"]).len().sort(["mismatch", "gcat"]))

# ------------------------------------------------------------------ B1 build number
hdr("B1. BUILD NUMBER: is the defect a build-era property?")
RO_b = RO.with_columns(pl.col("build_num").cast(pl.Utf8))
print("bomb rounds p20 by build x round bin, mismatch / n:")
print(RO_b.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
      .sort(["build_num", "rbin"]))
print("\nmatches by build with any mismatch (p20 bomb rounds):")
mb = (RO_b.group_by(["match_key", "build_num"]).agg(pl.col("mismatch").any().alias("any_mm"),
                                                    pl.col("mismatch").sum().alias("n_mm"),
                                                    pl.len().alias("n_bomb"))
      .group_by(["build_num", "any_mm"]).len().sort(["build_num", "any_mm"]))
print(mb)
print("\nr13 bomb rounds by build and match_date:")
r13b = RO_b.filter(pl.col("round") == 13)
print(r13b.group_by(["build_num", "match_date"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
      .with_columns(pl.col("match_date").str.slice(0, 10)).sort(["build_num", "match_date"]).filter(pl.col("match_date") >= "2024-02-01"))

# ------------------------------------------------------------------ B3 within-round score side
hdr("B3. WITHIN-ROUND SCORE SIDE (pre = round_prestart, post = round_officially_ended), validated vs bomb")
ev_rows = RST.filter(pl.col("event_type").is_in(["round_prestart", "round_officially_ended"]))
pre = (ev_rows.filter(pl.col("event_type") == "round_prestart")
       .sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tpre"),
                                                          pl.col("ct_score").first().alias("ctpre")))
post = (ev_rows.filter(pl.col("event_type") == "round_officially_ended")
        .sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tpost"),
                                                           pl.col("ct_score").last().alias("ctpost")))
fallback = RST.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("tf"), pl.col("ct_score").first().alias("ctf"),
    pl.col("t_score").last().alias("tl"), pl.col("ct_score").last().alias("ctl"))
SW = (fallback.join(pre, on=["match_key", "round"], how="left").join(post, on=["match_key", "round"], how="left")
      .with_columns(pl.coalesce(["tpre", "tf"]).alias("T0"), pl.coalesce(["ctpre", "ctf"]).alias("CT0"),
                    pl.coalesce(["tpost", "tl"]).alias("T1"), pl.coalesce(["ctpost", "ctl"]).alias("CT1"))
      .with_columns(((pl.col("T1") - pl.col("T0")) == 1).alias("dT1"), ((pl.col("CT1") - pl.col("CT0")) == 1).alias("dC1"))
      .with_columns(pl.when(pl.col("dT1") & ~pl.col("dC1")).then(pl.lit(2))
                      .when(pl.col("dC1") & ~pl.col("dT1")).then(pl.lit(3))
                      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("side_within")))
print("within-round side non-null share:", round(SW.filter(pl.col("side_within").is_not_null()).height / SW.height, 4),
      "of", SW.height, "rounds")
SWn = SW.select(["match_key", "round", "side_within", "T0", "CT0", "T1", "CT1"])
# cross-round variant for comparison
CR = (fallback.select(["match_key", "round", "tf", "ctf"]).with_columns((pl.col("round") - 1).alias("round"))
      .rename({"tf": "tnext", "ctf": "ctnext"}))
CRj = (fallback.select(["match_key", "round", "tf", "ctf"]).join(CR, on=["match_key", "round"], how="inner")
       .with_columns(((pl.col("tnext") - pl.col("tf")) == 1).alias("a"), ((pl.col("ctnext") - pl.col("ctf")) == 1).alias("b"))
       .with_columns(pl.when(pl.col("a") & ~pl.col("b")).then(pl.lit(2))
                       .when(pl.col("b") & ~pl.col("a")).then(pl.lit(3))
                       .otherwise(pl.lit(None, dtype=pl.Int64)).alias("side_cross"))
       .select(["match_key", "round", "side_cross"]))
VAL = RO.join(SWn, on=["match_key", "round"], how="left").join(CRj, on=["match_key", "round"], how="left")
print("validation on p20 bomb rounds (side == bomb side):")
print(VAL.group_by("rbin").agg(
    pl.len().alias("n"),
    (pl.col("side_within") == pl.col("bomb_side")).sum().alias("within_eq_bomb"),
    pl.col("side_within").is_null().sum().alias("within_null"),
    (pl.col("side_cross") == pl.col("bomb_side")).sum().alias("cross_eq_bomb"),
    pl.col("side_cross").is_null().sum().alias("cross_null")).sort("rbin"))

# ------------------------------------------------------------------ B4 elimination winner from roster
hdr("B4. ELIMINATION WINNER FROM PLAYER DEATHS (no round_end, no bomb outcome)")
pid = pl.Int64
roster = PI.with_columns(pl.col("player_id_fixed").cast(pid, strict=False).alias("pid")).filter(pl.col("pid").is_not_null()) \
    .select(["match_key", "round", "pid", "team_code"]).unique(["match_key", "round", "pid"])
deaths = PD.with_columns(pl.col("player_id_fixed").cast(pid, strict=False).alias("pid")).filter(pl.col("pid").is_not_null()) \
    .select(["match_key", "round", "pid", "tick"]).sort("tick").unique(["match_key", "round", "pid"], keep="first")
rd_ = roster.join(deaths, on=["match_key", "round", "pid"], how="left").join(
    RE.select(["match_key", "round", pl.col("tick").alias("end_tick")]), on=["match_key", "round"], how="inner")
rd_ = rd_.with_columns(((pl.col("tick").is_not_null()) & (pl.col("tick") <= pl.col("end_tick"))).alias("dead_by_end"))
alive = (rd_.group_by(["match_key", "round"]).agg(
    ((pl.col("team_code") == 2) & ~pl.col("dead_by_end")).sum().alias("T_alive"),
    ((pl.col("team_code") == 3) & ~pl.col("dead_by_end")).sum().alias("CT_alive"),
    (pl.col("team_code") == 2).sum().alias("nT"),
    (pl.col("team_code") == 3).sum().alias("nCT")))
alive = alive.join(MATCH.select(["match_key", "build_num", "map_name"]), on="match_key", how="left")
planted_any = planted.select(["match_key", "round"]).with_columns(pl.lit(True).alias("planted"))
alive = alive.join(planted_any, on=["match_key", "round"], how="left").with_columns(pl.col("planted").fill_null(False))
alive = alive.with_columns(
    pl.when((pl.col("nCT") == 5) & (pl.col("nT") == 5) & (pl.col("CT_alive") == 0) & (pl.col("T_alive") > 0)).then(pl.lit(2))
      .when((pl.col("nCT") == 5) & (pl.col("nT") == 5) & (pl.col("T_alive") == 0) & (pl.col("CT_alive") > 0) & (~pl.col("planted"))).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))
print("roster rounds with 5v5 and a derivable elimination winner:", alive.filter(pl.col("elim_side").is_not_null()).height)
E = (RO_all.select(["match_key", "round", "player_count", "winner_team_code", "bomb_side", "mismatch", "rbin", "build_num", "otype"])
     .join(alive.select(["match_key", "round", "elim_side", "T_alive", "CT_alive", "nT", "nCT", "planted"]), on=["match_key", "round"], how="left")
     .join(SWn.select(["match_key", "round", "side_within"]), on=["match_key", "round"], how="left")
     .filter(pl.col("player_count") == 20))
print("bomb rounds p20 with elim side:", E.filter(pl.col("elim_side").is_not_null()).height)
print(E.filter(pl.col("elim_side").is_not_null()).group_by(["build_num", "rbin", "elim_side", "winner_team_code"]).len()
      .sort(["build_num", "rbin", "elim_side", "winner_team_code"]))

# all rounds: winner vs elimination winner (non-bomb rounds included)
A = (alive.filter((pl.col("elim_side").is_not_null()) & (pl.col("nCT") == 5) & (pl.col("nT") == 5))
     .join(RE.select(["match_key", "round", "winner_team_code", "player_count"]), on=["match_key", "round"], how="inner")
     .filter(pl.col("player_count") == 20)
     .join(SWn.select(["match_key", "round", "side_within"]), on=["match_key", "round"], how="left")
     .with_columns(rbin_expr(), (pl.col("elim_side") != pl.col("winner_team_code")).alias("elim_ne_winner"),
                   (pl.col("side_within") == pl.col("elim_side")).alias("within_eq_elim")))
print("\nAll rounds (bomb or not) with a derivable elimination winner, p20, by build x rbin:")
print(A.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"), pl.col("elim_ne_winner").sum().alias("elim_ne_winner"),
    pl.col("elim_ne_winner").mean().round(3).alias("rate")).sort(["build_num", "rbin"]))
print("\nelimination winner vs within-round score side (both independent of round_end):")
print(A.filter(pl.col("side_within").is_not_null()).group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"), (pl.col("within_eq_elim")).sum().alias("within_eq_elim"),
    (~pl.col("within_eq_elim")).sum().alias("within_ne_elim")).sort(["build_num", "rbin"]))

# ------------------------------------------------------------------ 3. neighbour shift test on bomb rounds
hdr("C. NEIGHBOUR SHIFT: is winner_team_code of round r equal to the bomb side of r-1 or r+1?")
BSd = RO_all.select(["match_key", "round", "bomb_side"])
nxt = BSd.with_columns((pl.col("round") - 1).alias("round")).rename({"bomb_side": "bomb_prev"})
prv = BSd.with_columns((pl.col("round") + 1).alias("round")).rename({"bomb_side": "bomb_next"})
S2 = (RO_all.select(["match_key", "round", "winner_team_code", "bomb_side", "mismatch", "rbin", "build_num", "player_count"])
      .join(nxt, on=["match_key", "round"], how="left").join(prv, on=["match_key", "round"], how="left"))
S2 = S2.filter((pl.col("player_count") == 20) & pl.col("mismatch"))
print("mismatch rounds p20 (n=%d): winner==bomb_prev rate, winner==bomb_next rate (nulls excluded)" % S2.height)
print(S2.group_by(["rbin"]).agg(
    pl.len().alias("n_mm"),
    (pl.col("winner_team_code") == pl.col("bomb_prev")).sum().alias("eq_prev"),
    pl.col("bomb_prev").is_not_null().sum().alias("prev_avail"),
    (pl.col("winner_team_code") == pl.col("bomb_next")).sum().alias("eq_next"),
    pl.col("bomb_next").is_not_null().sum().alias("next_avail")).sort("rbin"))

print("\nDONE part2")
