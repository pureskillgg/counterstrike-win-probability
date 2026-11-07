"""D1-10 verifier: test the mechanism behind the claim's 100 disagreements, and apply the same
checks to the 2024 tome at the claimed rounds 9, 13, 25, 28.

Parts:
  P1 (2026): swap-safe score side = team_code of the EARLIEST +1 score_update row in the round.
             Compare with round_end.winner_team_code over all rounds and pc=10 rounds.
  P2 (2026): swap signature in round_state: last tick (t,ct) equals round_end tick (ct,t) swapped.
             Show where it occurs and whether it coincides with the claim's 100 disagreements.
  P3 (2024): claim method (round_state first vs last tick) and the swap signature, against the
             bomb outcome (bomb_exploded -> T, bomb_defused -> CT) and winner_team_code.

ASSUMPTIONS:
  A1 team codes 2 = T, 3 = CT.
  A2 the round_end row in round_state (event_type == 'round_end') carries the score before any
     post-round side swap (checked empirically in P2: exactly one such row per round is assumed).
  A3 score_update tick for the winning +1 equals the round_end tick (checked, not assumed).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
CLAIM_ROUNDS = [9, 13, 25, 27, 28, 33, 39, 45]


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def side_table(root, label):
    RE = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
                     for f in fl(root, "round_end")]))
    RS = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
                     for f in fl(root, "round_state")]))
    BS = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]).collect()
                     for f in fl(root, "bomb_state")]))
    print(f"[{label}] round_end={RE.height} round_state={RS.height} bomb_state={BS.height}; "
          f"distinct matches={RE['match_key'].n_unique()}")
    return RE, RS, BS


def claim_scores(RS):
    s = RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a"),
    )
    s = s.with_columns(
        pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
          .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
    return s


def swap_signature(RS, s):
    re_rows = (RS.filter(pl.col("event_type") == "round_end")
               .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t_re"),
                                                     pl.col("ct_score").first().alias("ct_re"),
                                                     pl.len().alias("n_re_rows")))
    j = s.join(re_rows, on=["match_key", "round"], how="left")
    j = j.with_columns(
        ((pl.col("t_a") == pl.col("ct_re")) & (pl.col("ct_a") == pl.col("t_re")) & (pl.col("t_re") != pl.col("ct_re")))
        .alias("swap_after_re"),
        ((pl.col("t_a") == pl.col("t_re")) & (pl.col("ct_a") == pl.col("ct_re"))).alias("no_change_after_re"))
    return j


# ======================= P1 + P2: 2026 =======================
RE26, RS26, BS26 = side_table(T26, "2026")
SU26 = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
                   for f in fl(T26, "score_update")]))
RE26 = RE26.unique(["match_key", "round"])
print("2026 round_end distinct (match, round):", RE26.height)

inc = SU26.filter((pl.col("new_score") - pl.col("old_score")) == 1)
inc_min = inc.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("su_tmin"))
inc_at_min = inc.join(inc_min, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("su_tmin"))
su_first = inc_at_min.group_by(["match_key", "round"]).agg(
    pl.col("team_code").n_unique().alias("su_nteam_at_min"),
    pl.col("team_code").min().alias("su_first"),
    pl.col("su_tmin").first().alias("su_tmin"),
    pl.len().alias("su_n_at_min"))
su_any = inc.group_by(["match_key", "round"]).agg(pl.len().alias("su_ninc_all"),
                                                   pl.col("team_code").n_unique().alias("su_nteam_all"))

s26 = claim_scores(RS26)
s26 = swap_signature(RS26, s26)
J26 = (RE26.join(su_first, on=["match_key", "round"], how="left")
       .join(su_any, on=["match_key", "round"], how="left")
       .join(s26.select(["match_key", "round", "claim_side", "swap_after_re", "no_change_after_re"]),
             on=["match_key", "round"], how="left"))
J26 = J26.with_columns(
    (pl.col("su_first") != pl.col("winner_team_code")).alias("dis_su_first"),
    (pl.col("claim_side") != pl.col("winner_team_code")).alias("dis_claim"),
    (pl.col("su_tmin") == pl.col("tick")).alias("su_tick_eq_re_tick"))

print("\n== P1 (2026): earliest +1 score_update row vs winner_team_code")
print("  rounds (all pc):", J26.height, " with su_first:", J26.filter(pl.col("su_first").is_not_null()).height)
print("  disagreements (all pc):", int(J26["dis_su_first"].fill_null(False).sum()))
p10 = J26.filter(pl.col("player_count") == 10)
print("  pc=10 rounds:", p10.height, " with su_first:", p10.filter(pl.col("su_first").is_not_null()).height,
      " disagreements:", int(p10["dis_su_first"].fill_null(False).sum()))
print("  ties at earliest tick (su_nteam_at_min > 1):", int((J26["su_nteam_at_min"].fill_null(1) > 1).sum()))
print("  earliest +1 tick equals round_end tick:", int(J26["su_tick_eq_re_tick"].fill_null(False).sum()), "of",
      int(J26["su_tmin"].is_not_null().sum()))
print("  per-round disagreements (su_first) at claimed rounds, pc=10:")
for r in CLAIM_ROUNDS:
    sub = p10.filter(pl.col("round") == r)
    print(f"    round {r:>2}: rounds={sub.height:>5} su_first_present={sub.filter(pl.col('su_first').is_not_null()).height:>5} "
          f"su_dis={int(sub['dis_su_first'].fill_null(False).sum()):>3}")

print("\n== P2 (2026): swap signature (last round_state tick = round_end values swapped)")
p10_claim = p10.filter(pl.col("claim_side").is_not_null())
print("  pc=10 rounds with claim side:", p10_claim.height,
      " claim disagreements:", int(p10_claim["dis_claim"].sum()),
      " of which swap_after_re:", int((p10_claim["dis_claim"] & p10_claim["swap_after_re"].fill_null(False)).sum()))
print("  claim disagreement rows where claim_side == other team:", int((p10_claim.filter(pl.col("dis_claim"))["claim_side"] == 5 - p10_claim.filter(pl.col("dis_claim"))["winner_team_code"]).sum()))
sw = p10.group_by("round").agg(pl.len().alias("n"), pl.col("swap_after_re").fill_null(False).sum().alias("n_swap"),
                              pl.col("no_change_after_re").fill_null(False).sum().alias("n_nochange")).sort("round")
print("  swap_after_re by round number (rounds where n_swap > 0):",
      [(r["round"], r["n_swap"], r["n"]) for r in sw.iter_rows(named=True) if r["n_swap"] > 0][:20])
print("  rounds with claim side but no swap and disagreement (should be 0):",
      int((p10_claim["dis_claim"] & ~p10_claim["swap_after_re"].fill_null(False)).sum()))

# ======================= P3: 2024 =======================
RE24, RS24, BS24 = side_table(T24, "2024")
RE24 = RE24.unique(["match_key", "round"])
s24 = claim_scores(RS24)
s24 = swap_signature(RS24, s24)
bomb24 = (BS24.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .group_by(["match_key", "round"]).agg(
              (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
              (pl.col("event_type") == "bomb_defused").any().alias("b_def")))
bomb24 = bomb24.with_columns(
    pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2))
      .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
J24 = (RE24.join(s24.select(["match_key", "round", "claim_side", "swap_after_re", "no_change_after_re"]),
                 on=["match_key", "round"], how="left")
       .join(bomb24.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left"))
J24 = J24.with_columns(
    (pl.col("claim_side") != pl.col("winner_team_code")).alias("dis_claim"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("dis_bomb"))
print("\n== P3 (2024): claim method vs bomb outcome vs winner, claimed rounds")
print("  rounds (all pc):", J24.height, " claim side:", J24.filter(pl.col("claim_side").is_not_null()).height,
      " bomb side:", J24.filter(pl.col("bomb_side").is_not_null()).height)
print("  pc=10 subset rounds:", J24.filter(pl.col("player_count") == 10).height)
print(f"  {'round':>5} {'n_all':>6} {'claim_side':>10} {'claim_dis':>9} {'swap':>5} {'bomb_side':>9} {'bomb_dis':>8} {'claim_dis&bomb_present':>22}")
for r in [9, 13, 25, 27, 28, 33, 39, 45]:
    sub = J24.filter(pl.col("round") == r)
    cs = sub.filter(pl.col("claim_side").is_not_null())
    bs = sub.filter(pl.col("bomb_side").is_not_null())
    cd = int(cs["dis_claim"].sum()) if cs.height else 0
    bd = int(bs["dis_bomb"].sum()) if bs.height else 0
    sw_n = int(sub["swap_after_re"].fill_null(False).sum())
    both = sub.filter(pl.col("claim_side").is_not_null() & pl.col("bomb_side").is_not_null())
    cd_b = int((both["dis_claim"] & both["dis_bomb"]).sum())
    print(f"  {r:>5} {sub.height:>6} {cs.height:>10} {cd:>9} {sw_n:>5} {bs.height:>9} {bd:>8} {cd_b:>12} of {both.height}")

print("\n  2024 claim disagreements (all rounds) by round number, top 15:")
d24 = J24.filter(pl.col("dis_claim"))
print("   ", sorted([(r, n) for r, n in d24.group_by("round").len().rows()], key=lambda x: -x[1])[:15])
print("  2024 claim disagreements where bomb also disagrees (bomb present):",
      int((d24["dis_bomb"].fill_null(False) & d24["bomb_side"].is_not_null()).sum()),
      " where bomb agrees with winner:",
      int(((~d24["dis_bomb"].fill_null(True)) & d24["bomb_side"].is_not_null()).sum()),
      " where no bomb side:", int(d24["bomb_side"].is_null().sum()), " of", d24.height)
print("  2024 swap signature by round number, top 15:",
      sorted([(r, n) for r, n in J24.filter(pl.col("swap_after_re").fill_null(False)).group_by("round").len().rows()],
             key=lambda x: -x[1])[:15])
print("DONE")
