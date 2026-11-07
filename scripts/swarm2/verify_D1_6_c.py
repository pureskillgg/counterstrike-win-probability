"""verify D1-6 (swarm2 verifier), part C: match-level and timing tests for the 2024 round-9 mismatch rows.
  C1 header test: regulation-only 2024 matches (no OT, round count == header final sum). Team that STARTED on T:
     tally_W = rounds 1-12 won by side T + rounds 13-n won by side CT, using round_end winner_team_code;
     tally_L = same with the freeze-to-freeze ledger (ledger_side, falling back to winner where the ledger is null).
     Compare each tally with header t_starters_score_final. Groups: round-9 mismatch matches with no other mismatch in
     rounds 1-24 (test) and matches with no mismatch at all (control). The header is produced by the game, not by the
     winner field, so an agreement with tally_L and not tally_W is an independent signal.
  C2 timing: round_end tick inside [freeze tick of round 9, freeze tick of round 10] for mismatch vs agreeing rows.
  C3 neighbour: round-9 (winner, reason) equal to the (winner, reason) of round 8 or round 10 in the same match.
  C4 overlap: round-9 mismatch matches vs round-13 ledger mismatch (d12-25, pc20).
ASSUMPTIONS:
  C-A1 2024 matches are MR12 with halftime swap after round 12 and no swap before round 12 (bake-off swap rule, r == max_rounds // 2).
  C-A2 header max_rounds is 24 for the regulation-only test (checked, not assumed, via the round count rule).
  C-A3 only matches with round count == header t+ct final sum and <= 24 rounds are used for C1 (no overtime).
  C-A4 ledger = freeze-end score change between round r and r+1 (no swap between them for r = 8, 9, 13 in MR12).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(50)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]
REASON_SIDE = {1: 2, 7: 3, 8: 3, 9: 2, 11: 3, 12: 3, 13: 2, 17: 3, 18: 2}


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------- header
hf = sorted(glob.glob(str(T24 / "header" / "dataframe_*")))
hs = pl.read_parquet_schema(hf[0])
print("header schema (2024):", {k: str(v) for k, v in hs.items() if "score" in k or "round" in k or k == "match_key"})
H = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"]) for f in hf],
              how="vertical_relaxed")
print("header rows:", H.height, " distinct key:", H["key"].n_unique())
H = H.select([pl.col("key").alias("match_key"), pl.col("t_starters_score_final").alias("S_T"),
              pl.col("ct_starters_score_final").alias("S_CT")]).unique("match_key")
_rk = set(rd(f24("round_end"), ["match_key"])["match_key"].unique().to_list())
print("header keys that appear in round_end:", sum(1 for k in H["match_key"].to_list() if k in _rk), "of", H.height,
      " round_end matches:", len(_rk))

# ----------------------------------------------------------------------------- round table
RE = (rd(f24("round_end"), ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "end_tick"}))
RS = rd(f24("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"),
                                            pl.col("ct_score").first().alias("c0"),
                                            pl.col("tick").min().alias("ftick")))
FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1"), pl.col("ftick").alias("nftick")])
FZall = FZ.join(FZn, on=["match_key", "round"], how="left").with_columns(
    pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
      .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side"))
SC = RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("tb"), pl.col("ct_score").sort_by("tick").first().alias("cb"),
    pl.col("t_score").sort_by("tick").last().alias("ta"), pl.col("ct_score").sort_by("tick").last().alias("ca"))
SC = SC.with_columns(
    pl.when((pl.col("ta") - pl.col("tb")) == 1).then(pl.lit(2))
      .when((pl.col("ca") - pl.col("cb")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")).select(["match_key", "round", "score_side"])
reason = None
for code, side in REASON_SIDE.items():
    cond = pl.col("win_reason_code") == code
    reason = pl.when(cond).then(pl.lit(side)) if reason is None else reason.when(cond).then(pl.lit(side))
reason = reason.otherwise(pl.lit(None, dtype=pl.Int64)).alias("reason_side")
R = (RE.join(FZall, on=["match_key", "round"], how="left").join(SC, on=["match_key", "round"], how="left")
     .with_columns(reason))
R = R.filter(pl.col("player_count") == 20)
R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
R = R.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
R = R.with_columns(
    ((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code"))).alias("mmL"),
    ((pl.col("score_side").is_not_null()) & (pl.col("score_side") != pl.col("winner_team_code"))).alias("mmC"),
    pl.coalesce(["ledger_side", "winner_team_code"]).alias("L"))

# ----------------------------------------------------------------------------- C1 header test
hdr("C1. header test on regulation-only 2024 matches (pc==20): header final vs winner tally vs ledger tally")
X = R.filter(pl.col("round") <= 24).with_columns(
    pl.when(pl.col("round") <= 12).then(pl.col("winner_team_code") == 2).otherwise(pl.col("winner_team_code") == 3)
      .cast(pl.Int32).alias("sW"),
    pl.when(pl.col("round") <= 12).then(pl.col("L") == 2).otherwise(pl.col("L") == 3).cast(pl.Int32).alias("sL"))
M = X.group_by("match_key").agg(
    pl.len().alias("nr"), pl.col("round").max().alias("maxr"), pl.col("dg").first().alias("dg"),
    pl.col("sW").sum().alias("tW_T"), pl.col("sL").sum().alias("tL_T"),
    pl.col("mmL").sum().alias("n_mmL"),
    (pl.col("round") == 9).any().alias("has9"),
    (pl.col("mmL") & (pl.col("round") == 9)).any().alias("mm9"))
M = M.join(H, on="match_key", how="inner")
M = M.filter((pl.col("maxr") == pl.col("nr")) & (pl.col("nr") <= 24) & ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr"))
             & (pl.max_horizontal(["S_T", "S_CT"]) >= 13))
print("regulation-only matches with complete rounds 1..n and header sum == n:", M.height)
print("  by dg:", M.group_by("dg").len().sort("dg").to_dicts())
M = M.with_columns(
    (pl.col("n_mmL") - pl.col("mm9").cast(pl.Int64)).alias("n_other"),
    (pl.col("tW_T") == pl.col("S_T")).alias("W_ok"), (pl.col("tL_T") == pl.col("S_T")).alias("L_ok"))
test = M.filter(pl.col("mm9") & (pl.col("n_other") == 0) & (pl.col("dg") == "d12-25"))
ctrl = M.filter((pl.col("n_mmL") == 0) & (pl.col("dg") == "d12-25"))
ctrl_d1 = M.filter((pl.col("n_mmL") == 0) & (pl.col("dg") == "d01-07"))
for name, g in [("TEST: round-9 mismatch, no other mismatch in 1-24, d12-25", test),
                ("CONTROL: no mismatch in 1-24, d12-25", ctrl),
                ("CONTROL: no mismatch in 1-24, d01-07", ctrl_d1)]:
    nW = int(g["W_ok"].sum())
    nL = int(g["L_ok"].sum())
    print(f"{name}: matches={g.height}  header==winner_tally={nW}  header==ledger_tally={nL}")
print("TEST group detail (first 8):")
print(test.select(["match_key", "nr", "S_T", "S_CT", "tW_T", "tL_T", "W_ok", "L_ok"]).head(8))

# ----------------------------------------------------------------------------- C2 timing
hdr("C2. round-9 round_end tick inside the freeze window [freeze9, freeze10] (d12-25, pc20)")
r9 = R.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25") & pl.col("score_side").is_not_null()).with_columns(
    ((pl.col("end_tick") >= pl.col("ftick")) & (pl.col("end_tick") <= pl.col("nftick"))).alias("in_win"),
    (pl.col("end_tick") > pl.col("nftick")).alias("after_next_freeze"),
    (pl.col("end_tick") < pl.col("ftick")).alias("before_freeze"))
print(r9.with_columns(pl.col("mmC").alias("mismatch")).group_by("mismatch").agg(
    pl.len().alias("rows"), pl.col("in_win").sum().alias("in_window"),
    pl.col("after_next_freeze").sum().alias("after_next_freeze"), pl.col("before_freeze").sum().alias("before_freeze")))

# ----------------------------------------------------------------------------- C3 neighbour
hdr("C3. round-9 (winner, reason) equal to a neighbouring round's (winner, reason) in the same match (d12-25)")
nb = R.select(["match_key", "round", "winner_team_code", "win_reason_code"])
prev = nb.with_columns((pl.col("round") + 1).alias("round")).rename(
    {"winner_team_code": "wp", "win_reason_code": "rp"})
nxt = nb.with_columns((pl.col("round") - 1).alias("round")).rename({"winner_team_code": "wn", "win_reason_code": "rn"})
c3 = (r9.join(prev.select(["match_key", "round", "wp", "rp"]), on=["match_key", "round"], how="left")
      .join(nxt.select(["match_key", "round", "wn", "rn"]), on=["match_key", "round"], how="left")
      .with_columns(((pl.col("wp") == pl.col("winner_team_code")) & (pl.col("rp") == pl.col("win_reason_code"))).alias("eqP"),
                    ((pl.col("wn") == pl.col("winner_team_code")) & (pl.col("rn") == pl.col("win_reason_code"))).alias("eqN")))
print(c3.with_columns(pl.col("mmC").alias("mismatch")).group_by("mismatch").agg(
    pl.len().alias("rows"), pl.col("eqP").sum().alias("same_as_prev"), pl.col("eqN").sum().alias("same_as_next")))

# ----------------------------------------------------------------------------- C4 overlap
hdr("C4. overlap of round-9 mismatch with round-13 ledger mismatch (d12-25, pc20, matches with both rows)")
d = R.filter(pl.col("dg") == "d12-25")
m9 = d.filter((pl.col("round") == 9) & pl.col("ledger_side").is_not_null()).select(["match_key", pl.col("mmL").alias("mm9")])
m13 = d.filter((pl.col("round") == 13) & pl.col("ledger_side").is_not_null()).select(["match_key", pl.col("mmL").alias("mm13")])
j = m9.join(m13, on="match_key", how="inner")
print(j.group_by(["mm9", "mm13"]).len().sort(["mm9", "mm13"]))
print("P(mm13 | mm9) and P(mm13 | not mm9):")
print(j.group_by("mm9").agg(pl.len().alias("n"), pl.col("mm13").mean().alias("rate_mm13")).sort("mm9"))
per = d.filter(pl.col("ledger_side").is_not_null()).group_by("match_key").agg(pl.col("mmL").sum().alias("n_mm"))
print("ledger mismatch rounds per match (d12-25, all rounds):", per.group_by("n_mm").len().sort("n_mm").to_dicts())
