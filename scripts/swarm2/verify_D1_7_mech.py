"""verify D1-7 (swarm2 verifier): test the counter-exchange mechanism directly.

ASSUMPTIONS:
  A1 side codes 2 = T, 3 = CT. A2 bomb_exploded => T won, bomb_defused => CT won.
  A3 claim side = first/last round_state row diff, as in D1_followup2.py.
  A4 exchange = consecutive round_state rows where (t,ct) becomes (old_ct, old_t) with old_t != old_ct.
  A8 (2026 only) the earliest +1 score_update event of a round is that round's winning increment.
  A9 pre-swap side = diff between first round_state row and the last row BEFORE the exchange row.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(18)
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def scan(base, name, suf=""):
    return pl.scan_parquet(sorted(glob.glob(str(base / f"basic_{name}{suf}*" / "dataframe_*"))))


def sec(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


def presw(base, suf, year):
    rs = (scan(base, "round_state", suf).select(["match_key", "round", "tick", "t_score", "ct_score"]).collect()
          .sort(["match_key", "round", "tick"]))
    g = ["match_key", "round"]
    rs = rs.with_columns(
        pl.col("t_score").shift(1).over(g).alias("pt"), pl.col("ct_score").shift(1).over(g).alias("pc"),
        pl.col("t_score").first().over(g).alias("t0"), pl.col("ct_score").first().over(g).alias("c0"))
    sw = rs.filter((pl.col("pt").is_not_null()) & (pl.col("pt") != pl.col("pc"))
                   & (pl.col("t_score") == pl.col("pc")) & (pl.col("ct_score") == pl.col("pt")))
    # A9: pre-swap side from the row before the first exchange in the round
    first_sw = sw.sort(["match_key", "round", "tick"]).group_by(g).agg(
        pl.col("pt").first().alias("pre_t"), pl.col("pc").first().alias("pre_c"),
        pl.col("t0").first().alias("t0"), pl.col("c0").first().alias("c0"), pl.len().alias("n_swap"))
    first_sw = first_sw.with_columns(
        pl.when((pl.col("pre_t") - pl.col("t0")) == 1).then(pl.lit(2))
          .when((pl.col("pre_c") - pl.col("c0")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("pre_side"))
    first_sw = first_sw.with_columns(pl.lit(year).alias("year"))
    return first_sw.select(["match_key", "round", "n_swap", "pre_side", "year"])


P24 = presw(T24, SUF24, 2024)
P26 = presw(T26, "", 2026)
J24 = pl.read_parquet(OUT / "verify_D1_7_rounds2024.parquet")
J26 = pl.read_parquet(OUT / "verify_D1_7_rounds2026.parquet")
M24 = J24.join(P24.select(["match_key", "round", "pre_side"]), on=["match_key", "round"], how="left")
M26 = J26.join(P26.select(["match_key", "round", "pre_side"]), on=["match_key", "round"], how="left")
M24.write_parquet(OUT / "verify_D1_7_mech_rounds2024.parquet")
M26.write_parquet(OUT / "verify_D1_7_mech_rounds2026.parquet")

sec("M1 2026: pre-swap side vs winner_team_code, rounds with an in-window exchange (all pc)")
x = M26.filter(pl.col("n_swap") > 0).with_columns(
    pl.when(pl.col("pre_side").is_null()).then(pl.lit("null")).when(pl.col("pre_side") == pl.col("winner_team_code")).then(pl.lit("agree")).otherwise(pl.lit("disagree")).alias("pre_vs_w"))
print("exchange rows by round and pre_side vs winner:")
print(x.group_by(["round", "pre_vs_w"]).len().sort(["round", "pre_vs_w"]))
print("2026 pre-swap side: rows with exchange whose pre_side == winner:", x.filter(pl.col("pre_vs_w") == "agree").height,
      " disagree:", x.filter(pl.col("pre_vs_w") == "disagree").height, " null:", x.filter(pl.col("pre_vs_w") == "null").height)

sec("M2 2026: claim-side vs pre-swap side on exchange rows (pc10), and bomb evidence on exchange rows")
xb = M26.filter((pl.col("n_swap") > 0) & pl.col("bomb_side").is_not_null())
print("2026 exchange rows with bomb: n =", xb.height, " bomb==winner:", int((xb["bomb_side"] == xb["winner_team_code"]).sum()),
      " bomb==pre_side:", int((xb["bomb_side"] == xb["pre_side"]).sum()), " bomb==claim_side (non-null):",
      int(xb.filter(pl.col("claim_side").is_not_null()).pipe(lambda d: (d["bomb_side"] == d["claim_side"]).sum())))
print(xb.group_by("round").len().sort("round"))

sec("M3 2026 score_update: earliest +1 event side vs winner_team_code, all rounds")
RE = scan(T26, "round_end").select(["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"]).collect()
su = scan(T26, "score_update").select(["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
su = su.with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta"))
p1 = su.filter(pl.col("delta") == 1).sort(["match_key", "round", "tick"])
first_p1 = p1.group_by(["match_key", "round"]).agg(pl.col("team_code").first().alias("first_p1"),
                                                   pl.len().alias("n_p1"),
                                                   pl.col("team_code").last().alias("last_p1"))
R = RE.join(first_p1, on=["match_key", "round"], how="left")
R = R.with_columns((pl.col("first_p1") == pl.col("winner_team_code")).alias("first_eq_w"))
print("rounds:", R.height, " first+1 == winner:", int(R["first_eq_w"].sum()), " first+1 != winner:",
      int((R["first_eq_w"] == False).sum()), " no +1 at all:", int(R["first_p1"].is_null().sum()))
print("first+1 != winner by round:", R.filter(pl.col("first_eq_w") == False).group_by("round").len().sort("round").to_dicts())
two = R.filter(pl.col("n_p1") == 2)
print("two-+1 rounds:", two.height, " first+1 == winner:", int(two["first_eq_w"].sum()))
print("two-+1 rounds by round:", two.group_by("round").len().sort("round").to_dicts())
print("two-+1 rounds, pc10 (claim population), first+1==winner:", int(two.filter(pl.col("player_count") == 10)["first_eq_w"].sum()),
      "of", two.filter(pl.col("player_count") == 10).height)
print("round 27 pc10 2026, all rounds with +1 events, first+1==winner:",
      R.filter((pl.col("round") == 27) & (pl.col("player_count") == 10)).group_by("first_eq_w").len().to_dicts())

sec("M4 2024 pre-swap side and claim side by round, exchange rounds (all pc)")
x4 = M24.filter(pl.col("n_swap") > 0)
print("2024 exchange rows by round:", x4.group_by("round").len().sort("round").to_dicts())
print("2024 exchange rows: claim_side non-null and claim_mm, by round:")
print(x4.filter(pl.col("claim_side").is_not_null()).group_by(["round", "claim_mm"]).len().sort(["round", "claim_mm"]))
print("2024 exchange rows: bomb rows and bomb vs pre_side vs winner, by round:")
b4 = x4.filter(pl.col("bomb_side").is_not_null())
print(b4.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("bomb_side") == pl.col("pre_side")).sum().alias("bomb_eq_pre"),
    (pl.col("bomb_side") == pl.col("winner_team_code")).sum().alias("bomb_eq_winner"),
    (pl.col("bomb_side") == pl.col("claim_side")).sum().alias("bomb_eq_claim")).sort("round"))

sec("M5 2024 round 27 (all pc): where is the claim value non-null, and is n_swap 1 there")
r27 = M24.filter(pl.col("round") == 27)
print(r27.group_by(["n_swap", "claim_side"]).len().sort(["n_swap", "claim_side"]))
print("2024 round 27 claim_mm by n_swap:", r27.filter(pl.col("claim_side").is_not_null()).group_by("n_swap").agg(pl.len().alias("n"), pl.col("claim_mm").sum().alias("mm")).to_dicts())

sec("M6 2024 d12-25 winner_team_code vs bomb, and claim vs bomb, rounds 9/13/25/27/28 (all pc) -- summary")
M24d = M24.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
M24d = M24d.with_columns(pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
sumt = (M24d.filter(pl.col("bomb_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 27, 28]))
        .group_by(["dg", "round"]).agg(
            pl.len().alias("bomb_rows"),
            (pl.col("bomb_side") != pl.col("winner_team_code")).sum().alias("bomb_ne_winner"),
            pl.col("claim_side").is_not_null().sum().alias("with_claim"),
            ((pl.col("claim_side").is_not_null()) & (pl.col("bomb_side") != pl.col("claim_side"))).sum().alias("bomb_ne_claim"),
            (pl.col("n_swap") > 0).sum().alias("n_exchange_in_window")).sort(["dg", "round"]))
print(sumt)
