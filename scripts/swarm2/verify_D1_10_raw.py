r"""D1-10 verifier: raw E: per-match check of the tome labels for a few named matches.

Picks matches from the 2026 and 2024 tomes, then reads the RAW per-match parquet files on E:
(read-only) and compares them with the tome rows for the same match:
  - raw round_end winner vs tome round_end winner
  - raw score side (earliest +1 score_update) vs raw winner
  - raw claim-rule side (round_state first vs last tick) vs raw winner
  - raw bomb side (bomb_state exploded/defused) vs raw winner
Prints one block per match; only rounds of interest or rounds with a disagreement are printed.

ASSUMPTIONS:
  A1 raw match folder = E:\csds_final\csds\<YYYY>\<MM>\<DD>\<match_id> (2026) or the tail copy
     E:\csds_final\csds\csds_tail_recovered_20261011\csds\<YYYY>\<MM>\<DD>\<match_id> (2024), chosen by existence.
  A2 team codes 2 = T, 3 = CT.
  A3 the raw channel files are single parquet files named after the channel (as observed for 2026).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
E_ROOT = Path(r"E:\csds_final\csds")
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
ROUNDS_OF_INTEREST = {9, 13, 25, 27, 28, 33, 39, 45}


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def tome_load(root, ch, cols):
    return pl.concat([pl.scan_parquet(f).select(cols).collect() for f in fl(root, ch)])


def raw_dir_for(match_key):
    # match_key looks like csds/2026/10/08/<id>/csds
    parts = match_key.split("/")
    yyyy, mm, dd, mid = parts[1], parts[2], parts[3], parts[4]
    cands = [E_ROOT / yyyy / mm / dd / mid, E_TAIL / yyyy / mm / dd / mid]
    for c in cands:
        if c.exists():
            return c
    return None


def read_raw(d, ch, cols):
    p = d / ch
    if not p.exists():
        return None
    return pl.read_parquet(p, columns=cols)


def analyse(match_key, tome_root, label):
    d = raw_dir_for(match_key)
    print(f"\n==== {label} match {match_key.split('/')[4][:8]} ({match_key.split('/')[1]}/{match_key.split('/')[2]}/{match_key.split('/')[3]})")
    if d is None:
        print("   raw folder not found on E: for this key")
        return
    print("   raw folder:", d)
    raw_re = read_raw(d, "round_end", ["round", "tick", "winner_team_code"])
    raw_su = read_raw(d, "score_update", ["round", "tick", "team_code", "old_score", "new_score"])
    raw_rs = read_raw(d, "round_state", ["round", "tick", "event_type", "t_score", "ct_score"])
    raw_bs = read_raw(d, "bomb_state", ["round", "tick", "event_type"])
    if raw_re is None or raw_rs is None:
        print("   raw round_end or round_state missing")
        return

    # tome rows for the same match
    t_re = tome_load(tome_root, "round_end", ["match_key", "round", "winner_team_code"]).filter(pl.col("match_key") == match_key)
    t_su = None
    if (tome_root / "basic_score_update.2026-09-18,2026-10-08").exists() or fl(tome_root, "score_update"):
        t_su = tome_load(tome_root, "score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"]).filter(pl.col("match_key") == match_key)
    t_bs = tome_load(tome_root, "bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("match_key") == match_key)

    # raw derivations
    rw = raw_re.unique(["round"]).select(["round", "winner_team_code"]).rename({"winner_team_code": "raw_winner"})
    if raw_su is not None:
        inc = raw_su.filter((pl.col("new_score") - pl.col("old_score")) == 1)
        mn = inc.group_by("round").agg(pl.col("tick").min().alias("tmin"))
        first = inc.join(mn, on="round").filter(pl.col("tick") == pl.col("tmin")).group_by("round").agg(
            pl.col("team_code").min().alias("raw_su_first"))
    else:
        first = pl.DataFrame({"round": [], "raw_su_first": []}, schema={"round": pl.Int64, "raw_su_first": pl.Int64})
    sc = raw_rs.sort("tick").group_by("round").agg(
        pl.col("t_score").first().alias("t_b"), pl.col("ct_score").first().alias("ct_b"),
        pl.col("t_score").last().alias("t_a"), pl.col("ct_score").last().alias("ct_a"))
    sc = sc.with_columns(
        pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
          .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("raw_claim_side"))
    if raw_bs is not None:
        bo = raw_bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).group_by("round").agg(
            (pl.col("event_type") == "bomb_exploded").any().alias("e"),
            (pl.col("event_type") == "bomb_defused").any().alias("f"))
        bo = bo.with_columns(
            pl.when(pl.col("e") & ~pl.col("f")).then(pl.lit(2))
              .when(pl.col("f") & ~pl.col("e")).then(pl.lit(3))
              .otherwise(pl.lit(None, dtype=pl.Int64)).alias("raw_bomb_side")).select(["round", "raw_bomb_side"])
    else:
        bo = pl.DataFrame({"round": [], "raw_bomb_side": []}, schema={"round": pl.Int64, "raw_bomb_side": pl.Int64})

    R = (rw.join(first, on="round", how="left").join(sc.select(["round", "raw_claim_side"]), on="round", how="left")
         .join(bo, on="round", how="left")
         .join(t_re.select(["round", "winner_team_code"]).rename({"winner_team_code": "tome_winner"}), on="round", how="left")
         .sort("round"))
    R = R.with_columns(
        (pl.col("raw_su_first") != pl.col("raw_winner")).alias("raw_su_dis"),
        (pl.col("raw_claim_side") != pl.col("raw_winner")).alias("raw_claim_dis"),
        (pl.col("raw_bomb_side") != pl.col("raw_winner")).alias("raw_bomb_dis"),
        (pl.col("tome_winner") != pl.col("raw_winner")).alias("tome_vs_raw_winner_dis"),
    )
    n_round = R.height
    print(f"   rounds: raw {n_round}, tome {t_re.height}; tome winner != raw winner: {int(R['tome_vs_raw_winner_dis'].fill_null(False).sum())}")
    print(f"   raw su_first != raw winner: {int(R['raw_su_dis'].fill_null(False).sum())}  "
          f"raw claim-rule != raw winner: {int(R['raw_claim_dis'].fill_null(False).sum())}  "
          f"raw bomb != raw winner: {int(R['raw_bomb_dis'].fill_null(False).sum())} (bomb present {int(R['raw_bomb_side'].is_not_null().sum())})")
    if t_su is not None:
        t_inc = t_su.filter((pl.col("new_score") - pl.col("old_score")) == 1)
        t_first = None
        if t_inc.height:
            t_mn = t_inc.group_by("round").agg(pl.col("tick").min().alias("tt"))
            t_first = (t_inc.join(t_mn, on="round").filter(pl.col("tick") == pl.col("tt"))
                       .group_by("round").agg(pl.col("team_code").min().alias("tome_su_first")))
        if t_first is not None:
            R = R.join(t_first.select(["round", "tome_su_first"]), on="round", how="left")
            print("   tome su_first != raw su_first (rounds):", int((R["tome_su_first"] != R["raw_su_first"]).fill_null(False).sum()))
    shown = R.filter(pl.col("round").is_in(list(ROUNDS_OF_INTEREST)) | pl.col("raw_claim_dis") | pl.col("raw_bomb_dis") | pl.col("raw_su_dis") | pl.col("tome_vs_raw_winner_dis"))
    print("   round  raw_winner  tome_winner  raw_su_first  raw_claim_side  raw_bomb_side  (flags)")
    for r in shown.iter_rows(named=True):
        flags = []
        if r["tome_vs_raw_winner_dis"]: flags.append("TOME!=RAW")
        if r["raw_su_dis"]: flags.append("su!=win")
        if r["raw_claim_dis"]: flags.append("claim!=win")
        if r["raw_bomb_dis"]: flags.append("bomb!=win")
        print(f"   {r['round']:>5}  {r['raw_winner']!s:>10}  {r['tome_winner']!s:>11}  {r['raw_su_first']!s:>12}  "
              f"{r['raw_claim_side']!s:>14}  {r['raw_bomb_side']!s:>13}  {' '.join(flags)}")


# -------- choose matches from the tomes (the rules are the same as verify_D1_10_recompute.py) --------
RE26 = tome_load(T26, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
RS26 = tome_load(T26, "round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
sc26 = (RS26.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t_b"), pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
    pl.col("t_score").sort_by("tick").last().alias("t_a"), pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
sc26 = sc26.with_columns(pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
                           .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
                           .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
J26 = RE26.join(sc26.select(["match_key", "round", "claim_side"]), on=["match_key", "round"], how="left")
dis26 = J26.filter((pl.col("player_count") == 10) & pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code")))
pick26 = []
for rnd in [27, 33, 39]:
    x = dis26.filter(pl.col("round") == rnd).sort("match_key")
    if x.height:
        pick26.append((x["match_key"][0], f"2026 claim-dis round {rnd}"))
ctrl = J26.filter((pl.col("player_count") == 10) & (pl.col("round") == 13) & (pl.col("claim_side") == pl.col("winner_team_code"))).sort("match_key")
if ctrl.height:
    pick26.append((ctrl["match_key"][0], "2026 control round 13 (claim agrees)"))

RS24 = tome_load(T24, "round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
RE24 = tome_load(T24, "round_end", ["match_key", "round", "winner_team_code"])
BS24 = tome_load(T24, "bomb_state", ["match_key", "round", "event_type"])
sc24 = (RS24.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t_b"), pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
    pl.col("t_score").sort_by("tick").last().alias("t_a"), pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
sc24 = sc24.with_columns(pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
                           .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
                           .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
bo24 = (BS24.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
        .group_by(["match_key", "round"]).agg((pl.col("event_type") == "bomb_exploded").any().alias("e"),
                                              (pl.col("event_type") == "bomb_defused").any().alias("f")))
bo24 = bo24.with_columns(pl.when(pl.col("e") & ~pl.col("f")).then(pl.lit(2))
                           .when(pl.col("f") & ~pl.col("e")).then(pl.lit(3))
                           .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
J24 = (RE24.join(sc24.select(["match_key", "round", "claim_side"]), on=["match_key", "round"], how="left")
       .join(bo24.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left"))
pick24 = []
x = J24.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).sort("match_key")
if x.height:
    pick24.append((x["match_key"][0], "2024 bomb-dis round 13"))
x = J24.filter((pl.col("round") == 9) & (pl.col("claim_side") != pl.col("winner_team_code")) & pl.col("bomb_side").is_not_null()
               & (pl.col("bomb_side") != pl.col("winner_team_code"))).sort("match_key")
if x.height:
    pick24.append((x["match_key"][0], "2024 claim-dis and bomb-dis round 9"))
x = J24.filter((pl.col("round") == 25) & (pl.col("claim_side") != pl.col("winner_team_code"))).sort("match_key")
if x.height:
    pick24.append((x["match_key"][0], "2024 claim-dis round 25"))

print("PICKED:")
for k, lab in pick26 + pick24:
    print("  ", lab, "->", k)

for k, lab in pick26:
    analyse(k, T26, lab)
for k, lab in pick24:
    analyse(k, T24, lab)
print("\nDONE")
