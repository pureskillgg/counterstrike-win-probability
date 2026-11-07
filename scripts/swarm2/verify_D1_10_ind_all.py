"""verify_D1_10_ind_all.py -- consolidated independent check of claim D1-10 (2026 score-vs-winner cross-check).

Read-only. Prints to stdout; the runner appends the output to logs\\swarm2\\verify_D1_10.log.
Consolidates the checks run by this verifier (reproducible from this single file):
  Part 1  2026 headline: reproduce the claim (claim method = round_state first vs last tick, pc10).
  Part 2  2026 independent references: score_update single +1 rule, earliest +1 rule, bomb_state, win_reason.
  Part 3  2026 mechanism: straight / swapped / neither classification of round_state against winner.
  Part 4  2026 raw E: vs tome for four claim-flagged matches (rounds 27/33/39/45).
  Part 5  2024 control: the same classifications and bomb check on the 2024 tome (no score_update channel there).

ASSUMPTIONS (labelled):
  A1 team codes 2 = T, 3 = CT.
  A2 claim rule: side T if t_last - t_first == 1, CT if ct_last - ct_first == 1, else null.
  A3 single-+1 score_update rule: exactly one +1 row in the round; its team_code is the side.
  A4 bomb rule: bomb_exploded only -> T, bomb_defused only -> CT.
  A5 day = match_key path day; 2024 windows 02-01..07 and 02-12..25 as in the claim.
  A6 earliest-+1 rule: the earliest +1 score_update row by tick is the scoring event; later swap-reassignment rows are ignored.
  A7 straight/swapped frames: straight = state columns stay with the side; swapped = columns follow teams across a side swap.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(18)
pl.Config.set_fmt_str_lengths(30)
pl.Config.set_tbl_width_chars(220)

TOMES = {
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
}
RAW = Path(r"E:\csds_final")  # read-only
FLAG_ROUNDS = [27, 33, 39, 45]
CLAIM_ROUNDS = [9, 12, 13, 25, 27, 28, 33, 39, 45]
INT = pl.Int64


def load(root, ch, cols, optional=False):
    p = sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))
    if not p:
        if optional:
            print(f"  [{root.parts[-3]}] channel {ch}: ABSENT (treated as no information)")
            return None
        raise SystemExit(f"missing channel {ch} under {root}")
    df = pl.scan_parquet(p).select(cols).collect()
    exprs = [pl.col("match_key").cast(pl.String)]
    for c in cols:
        if c in ("round", "tick", "t_score", "ct_score", "team_code", "old_score", "new_score",
                 "winner_team_code", "win_reason_code", "player_count"):
            exprs.append(pl.col(c).cast(INT))
    return df.with_columns(exprs)


def empty_su():
    return pl.DataFrame(schema={"match_key": pl.String, "round": INT, "tick": INT, "team_code": INT,
                                "old_score": INT, "new_score": INT})


def build(tag):
    root = TOMES[tag]
    print(f"\n==================== TOME {tag}: {root} ====================")
    RE = load(root, "round_end", ["match_key", "round", "winner_team_code", "player_count", "win_reason_code"]).unique(["match_key", "round"])
    print(f"  round_end rows={RE.height} matches={RE['match_key'].n_unique()}")
    RS = load(root, "round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
    g = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
            pl.col("t_score").sort_by("tick").first().alias("t_b"),
            pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
            pl.col("t_score").sort_by("tick").last().alias("t_a"),
            pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
    SU = load(root, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"], optional=True)
    if SU is None:
        SU = empty_su()
    inc = SU.filter((pl.col("new_score") - pl.col("old_score")) == 1).sort(["match_key", "round", "tick"])
    su_agg = inc.group_by(["match_key", "round"], maintain_order=True).agg(
        pl.len().alias("su_n"),
        pl.col("team_code").first().alias("first_side"),
        pl.col("team_code").min().alias("su_min_tc"))
    su_agg = su_agg.with_columns(
        pl.when(pl.col("su_n") == 1).then(pl.col("su_min_tc")).otherwise(pl.lit(None, dtype=INT)).alias("su_side"))
    BS = load(root, "bomb_state", ["match_key", "round", "event_type"], optional=True)
    if BS is None:
        BS = pl.DataFrame(schema={"match_key": pl.String, "round": INT, "event_type": pl.String})
    bomb = BS.group_by(["match_key", "round"]).agg(
        (pl.col("event_type") == "bomb_exploded").any().alias("b_exp"),
        (pl.col("event_type") == "bomb_defused").any().alias("b_def"))
    bomb = bomb.with_columns(
        pl.when(pl.col("b_exp") & ~pl.col("b_def")).then(pl.lit(2, dtype=INT))
          .when(pl.col("b_def") & ~pl.col("b_exp")).then(pl.lit(3, dtype=INT))
          .otherwise(pl.lit(None, dtype=INT)).alias("bomb_side"))

    J = RE.join(g, on=["match_key", "round"], how="left")
    J = J.join(su_agg.select(["match_key", "round", "su_n", "first_side", "su_side"]), on=["match_key", "round"], how="left")
    J = J.join(bomb.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
    J = J.with_columns(
        pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2, dtype=INT))
          .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3, dtype=INT))
          .otherwise(pl.lit(None, dtype=INT)).alias("claim_side"),
        (pl.col("winner_team_code") == 2).cast(INT).alias("tw"),
        (pl.col("winner_team_code") == 3).cast(INT).alias("cw"),
        (pl.col("t_b") == pl.col("ct_b")).alias("tied"))
    J = J.with_columns(
        ((pl.col("t_a") == pl.col("t_b") + pl.col("tw")) & (pl.col("ct_a") == pl.col("ct_b") + pl.col("cw"))).alias("straight"),
        ((pl.col("t_a") == pl.col("ct_b") + pl.col("cw")) & (pl.col("ct_a") == pl.col("t_b") + pl.col("tw"))).alias("swapped"))
    J = J.with_columns(
        pl.when(pl.col("straight") & pl.col("swapped")).then(pl.lit("both"))
          .when(pl.col("straight")).then(pl.lit("straight"))
          .when(pl.col("swapped")).then(pl.lit("swapped"))
          .otherwise(pl.lit("neither")).alias("cat"),
        (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_dis"),
        (pl.col("su_side").is_not_null() & (pl.col("su_side") != pl.col("winner_team_code"))).alias("su_dis"),
        (pl.col("first_side").is_not_null() & (pl.col("first_side") != pl.col("winner_team_code"))).alias("first_dis"),
        (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_dis"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).cast(INT).alias("day"))
    return J, RE


def per_round_table(J, rounds, title):
    print(f"\n  {title}")
    print(f"    {'rnd':>3} {'n':>5} {'tied':>5} {'straight':>8} {'swapped':>7} {'both':>5} {'neither':>7} | "
          f"{'claim':>5} {'c_dis':>5} | {'su':>5} {'su_dis':>6} | {'first_dis':>9} | {'bomb':>5} {'b_dis':>5}")
    for r in rounds:
        s = J.filter(pl.col("round") == r)
        if s.height == 0:
            continue
        print(f"    {r:>3} {s.height:>5} {int(s['tied'].sum()):>5} {int((s['cat'] == 'straight').sum()):>8} "
              f"{int((s['cat'] == 'swapped').sum()):>7} {int((s['cat'] == 'both').sum()):>5} {int((s['cat'] == 'neither').sum()):>7} | "
              f"{int(s['claim_side'].is_not_null().sum()):>5} {int(s['claim_dis'].sum()):>5} | "
              f"{int(s['su_side'].is_not_null().sum()):>5} {int(s['su_dis'].sum()):>6} | {int(s['first_dis'].sum()):>9} | "
              f"{int(s['bomb_side'].is_not_null().sum()):>5} {int(s['bomb_dis'].sum()):>5}")


# ---------------- Part 1 and 2 and 3: 2026 ----------------
J26, RE26 = build("2026")
print("\n[P1] claim headline (claim method, recomputed)")
for label, sub in [("player_count == 10", J26.filter(pl.col("player_count") == 10)), ("all player_count", J26)]:
    ws = sub.filter(pl.col("claim_side").is_not_null())
    print(f"  {label}: round_end={sub.height} claim-side rounds={ws.height} "
          f"claim-null={sub.height - ws.height} claim disagreements={int(sub.filter(pl.col('claim_dis')).height)}")
P10 = J26.filter(pl.col("player_count") == 10)
per_round_table(P10, sorted(set(CLAIM_ROUNDS) | set(range(1, 49))) if False else CLAIM_ROUNDS, "pc10 per-round (claim rounds)")
null_by_round = (P10.filter(pl.col("claim_side").is_null()).group_by("round").len().sort("len", descending=True).head(6))
print("  pc10 claim-null rounds by round (top 6):", null_by_round.rows())
print("  pc10 round 12: round_end rows", P10.filter(pl.col("round") == 12).height,
      " claim side present", P10.filter((pl.col("round") == 12) & pl.col("claim_side").is_not_null()).height,
      " single-+1 score_update side present", P10.filter((pl.col("round") == 12) & pl.col("su_side").is_not_null()).height)

print("\n[P2] independent references, 2026")
print("  win_reason codes mapping to >1 winner:",
      RE26.group_by("win_reason_code").agg(pl.col("winner_team_code").n_unique().alias("k")).filter(pl.col("k") > 1).height)
print("  pc10 single-+1 score_update side vs winner: rounds with side", P10.filter(pl.col("su_side").is_not_null()).height,
      " disagreements", int(P10.filter(pl.col("su_dis")).height))
print("  all-pc earliest-+1 (A6) vs winner: rounds with a +1 row", J26.filter(pl.col("first_side").is_not_null()).height,
      " disagreements", int(J26.filter(pl.col("first_dis")).height))
print("  pc10 bomb side vs winner: rounds with side", P10.filter(pl.col("bomb_side").is_not_null()).height,
      " disagreements", int(P10.filter(pl.col("bomb_dis")).height))
print("  all-pc bomb side vs winner: rounds with side", J26.filter(pl.col("bomb_side").is_not_null()).height,
      " disagreements", int(J26.filter(pl.col("bomb_dis")).height))
flag26 = J26.filter(pl.col("claim_dis"))
print(f"  claim-flagged rows (all pc): {flag26.height}; earliest-+1 side present {flag26.filter(pl.col('first_side').is_not_null()).height},"
      f" earliest-+1 disagreements {flag26.filter(pl.col('first_dis')).height};"
      f" bomb side present {flag26.filter(pl.col('bomb_side').is_not_null()).height},"
      f" bomb agrees with winner {flag26.filter(pl.col('bomb_side').is_not_null() & (pl.col('bomb_side') == pl.col('winner_team_code'))).height}")

print("\n[P3] mechanism, 2026 pc10")
print("  category counts:", P10.group_by("cat").len().sort("cat").rows())
cd = P10.filter(pl.col("claim_dis"))
print(f"  claim disagreements pc10: {cd.height}; categories {cd.group_by('cat').len().rows()};"
      f" tied_before {cd.group_by('tied').len().rows()}; rounds {cd.group_by('round').len().sort('round').rows()}")
print(f"  rounds tied before: {P10.filter(pl.col('tied')).height}; of which swapped {P10.filter(pl.col('tied') & (pl.col('cat') == 'swapped')).height}")
print("  rounds 25-30 (pc10) classification:")
per_round_table(P10, [25, 26, 27, 28, 29, 30], "pc10 rounds 25-30")
print("  denominators (pc10): round 27 round_end=", P10.filter(pl.col("round") == 27).height,
      " claim-kept=", P10.filter((pl.col("round") == 27) & pl.col("claim_side").is_not_null()).height,
      " single-+1 kept=", P10.filter((pl.col("round") == 27) & pl.col("su_side").is_not_null()).height,
      " both kept=", P10.filter((pl.col("round") == 27) & pl.col("claim_side").is_not_null() & pl.col("su_side").is_not_null()).height)

# ---------------- Part 4: raw E: vs tome ----------------
print("\n[P4] raw E: vs tome, one claim-flagged 2026 match per flagged round (pc10)")
root26 = TOMES["2026"]
for r in FLAG_ROUNDS:
    sub = P10.filter(pl.col("claim_dis") & (pl.col("round") == r)).sort("match_key")
    if sub.height == 0:
        print(f"  round {r}: no flagged rows")
        continue
    mk = sub["match_key"][0]
    parts = mk.split("/")  # csds / YYYY / MM / DD / guid / csds
    raw_dir = RAW.joinpath(*parts[:5])
    print(f"  round {r}: {mk} raw dir exists={raw_dir.exists()}")
    for ch, cols in [("score_update", ["round", "tick", "team_code", "old_score", "new_score"]),
                     ("round_end", ["round", "tick", "winner_team_code", "win_reason_code"])]:
        raw = pl.read_parquet(raw_dir / ch).select(cols).with_columns(pl.all().cast(INT))
        tome = load(root26, ch, ["match_key"] + cols).filter(pl.col("match_key") == mk).select(cols)
        same = raw.sort(["round", "tick"]).equals(tome.sort(["round", "tick"]))
        print(f"    {ch}: raw rows={raw.height} tome rows={tome.height} identical={same}")
    rr = raw.filter(pl.col("round") == r)
    print(f"    raw round_end round {r}: winner_team_code={rr['winner_team_code'].to_list()} win_reason_code={rr['win_reason_code'].to_list()}")
    raw_su = pl.read_parquet(raw_dir / "score_update").filter((pl.col("round") == r) & ((pl.col("new_score") - pl.col("old_score")) == 1)).sort("tick")
    print(f"    raw earliest +1 team_code in round {r}: {raw_su['team_code'][0] if raw_su.height else None}")

# ---------------- Part 5: 2024 control ----------------
J24, RE24 = build("2024")
print("\n[P5] 2024 control (no score_update channel in this tome; bomb and claim method only)")
for label, sub in [("player_count == 20", J24.filter(pl.col("player_count") == 20)), ("all player_count", J24)]:
    print(f"  {label}: round_end={sub.height}; category counts {sub.group_by('cat').len().sort('cat').rows()}")
    print(f"    claim disagreements={int(sub.filter(pl.col('claim_dis')).height)} by category "
          f"{sub.filter(pl.col('claim_dis')).group_by('cat').len().sort('cat').rows()}")
    print(f"    bomb-labelled={int(sub.filter(pl.col('bomb_side').is_not_null()).height)} bomb disagreements={int(sub.filter(pl.col('bomb_dis')).height)}"
          f" by round {sub.filter(pl.col('bomb_dis')).group_by('round').len().sort('round').rows()}")
per_round_table(J24.filter(pl.col("player_count") == 20), [9, 12, 13, 25, 27, 28], "2024 pc20 per-round")
per_round_table(J24, [9, 12, 13, 25, 27, 28], "2024 all-pc per-round")
print("\n  2024 bomb windows (pc20 and all), grouped as in the claim:")
J24b = J24.filter(pl.col("bomb_side").is_not_null()).with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([9, 25, 28])).then(pl.lit("r9_25_28")).otherwise(pl.lit("other")).alias("grp"),
    pl.when(pl.col("day").is_between(12, 25)).then(pl.lit("02-12..25"))
      .when(pl.col("day").is_between(1, 7)).then(pl.lit("02-01..07")).otherwise(pl.lit("other days")).alias("window"))
for label, sub in [("pc20", J24b.filter(pl.col("player_count") == 20)), ("all", J24b)]:
    t = sub.group_by(["window", "grp"]).agg(pl.len().alias("n"), pl.col("bomb_dis").sum().alias("mismatch")).sort(["window", "grp"])
    print(f"  {label}:", [(r["window"], r["grp"], r["n"], r["mismatch"]) for r in t.iter_rows(named=True)
                          if r["window"] in ("02-01..07", "02-12..25")])
    pr = sub.filter(pl.col("bomb_dis")).group_by("round").len().sort("round").rows()
    print(f"    mismatch rounds {label}: {pr}")
print("DONE")
