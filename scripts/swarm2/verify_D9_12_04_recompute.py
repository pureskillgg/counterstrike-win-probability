"""verify D9-12 step 4: recompute the claim's core numbers from the raw basic tomes (2024 and 2026), then
test which round label of round_start is physically correct using two label-independent anchors:

  A. Score anchor. t_score + ct_score in round_state counts completed rounds (ASSUMPTION: cumulative across
     halves, side-swapped values still sum to completed rounds; checked in section 4). Physical round k starts
     with sum = k - 1 and ends with sum = k. Expected sum at round_freeze_end(label r) = r - 1 if the label is right.
  B. Freeze-delay anchor. round_freeze_end minus its own round_start is a constant (the freeze time), so the
     pairing that gives one constant delay is the physically right one. Compare fe(r)-rs(r) and fe(r)-rs(r-1).

Labels: rs = basic round_start (min tick per match, round); re = basic round_end (min tick, winner_team_code);
fe = round_state event_type round_freeze_end (min tick). ASSUMPTION: 64 ticks per second (step 2 showed
round_end tick == 64 * second exactly in both tomes).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    lf = pl.scan_parquet(fs)
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


def quant(s):
    return (s.quantile(0.01), s.quantile(0.5), s.quantile(0.99))


def asof_score(df, col, sc, outname):
    left = df.select(["match_key", "round", col]).sort(col).with_columns(pl.col(col).alias("_t"))
    right = sc.rename({"tick": "_t"})
    out = left.join_asof(right, on="_t", by="match_key", strategy="backward")
    return out.rename({"ssum": outname})


for year, (base, suf) in TOMES.items():
    print(f"\n################ {year}")
    rs_raw = scan(base, suf, "round_start", ["match_key", "round", "tick"])
    re_raw = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code"])
    st = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe_raw = st.filter(pl.col("event_type") == "round_freeze_end")

    for name, df in [("round_start", rs_raw), ("round_end", re_raw), ("round_freeze_end", fe_raw)]:
        g = df.group_by(["match_key", "round"]).len()
        print(f"{name}: rows={df.height} (match,round) keys={g.height} "
              f"rows-per-key dist={g.group_by('len').agg(pl.len().alias('n_keys')).sort('len').rows()}")

    rs = rs_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs"))
    re = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("re"),
                                                     pl.col("winner_team_code").first().alias("winner"))
    fe = fe_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe"))

    # ---- 1. claim recompute: own round, previous round, next round ----
    re_next = re.select(["match_key", "round", "re"]).with_columns((pl.col("round") - 1).alias("round")).rename({"re": "re_next"})
    re_prev = re.select(["match_key", "round", "re"]).with_columns((pl.col("round") + 1).alias("round")).rename({"re": "re_prev"})
    A = rs.join(re.select(["match_key", "round", "re"]), on=["match_key", "round"], how="inner")
    A = A.join(re_next, on=["match_key", "round"], how="left")
    print(f"\n[1] rounds with round_start AND own round_end: {A.height}")
    print(f"    rs > own end: {A.filter(pl.col('rs') > pl.col('re')).height}   "
          f"rs < own end: {A.filter(pl.col('rs') < pl.col('re')).height}")
    wn = A.filter(pl.col("re_next").is_not_null())
    print(f"    with next round_end (n={wn.height}): rs in (own end, next end]: "
          f"{wn.filter((pl.col('rs') > pl.col('re')) & (pl.col('rs') <= pl.col('re_next'))).height}")
    B = fe.join(re.select(["match_key", "round", "re"]), on=["match_key", "round"], how="inner")
    B = B.join(re_prev, on=["match_key", "round"], how="left")
    print(f"    rounds with freeze_end AND own round_end: {B.height}; fe < own end: "
          f"{B.filter(pl.col('fe') < pl.col('re')).height}; fe > own end: {B.filter(pl.col('fe') > pl.col('re')).height}")
    bp = B.filter(pl.col("re_prev").is_not_null())
    print(f"    with previous round_end (n={bp.height}): fe in (prev end, own end]: "
          f"{bp.filter((pl.col('fe') > pl.col('re_prev')) & (pl.col('fe') <= pl.col('re'))).height}")
    print(f"    freeze_end rounds with NO previous round_end row: {B.height - bp.height}")

    # ---- 2. freeze-delay anchor ----
    F = fe.join(rs, on=["match_key", "round"], how="inner")
    Fp = fe.join(rs.with_columns((pl.col("round") + 1).alias("round")).rename({"rs": "rs_prev"}),
                 on=["match_key", "round"], how="inner")
    print(f"\n[2] freeze delay fe(r) - rs(label r): n={F.height}")
    print("    top values (delay ticks, count):",
          (F["fe"] - F["rs"]).value_counts().sort("count", descending=True).head(6).rows())
    print("    quantiles p1/p50/p99:", quant(F["fe"] - F["rs"]))
    print(f"    fe(r) - rs(label r-1): n={Fp.height}")
    print("    top values (delay ticks, count):",
          (Fp["fe"] - Fp["rs_prev"]).value_counts().sort("count", descending=True).head(6).rows())
    print("    quantiles p1/p50/p99:", quant(Fp["fe"] - Fp["rs_prev"]))

    # ---- 3. score anchor ----
    sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
          .select(["match_key", "tick", "t_score", "ct_score"])
          .with_columns((pl.col("t_score") + pl.col("ct_score")).alias("ssum"))
          .select(["match_key", "tick", "ssum"]).sort("tick"))
    s_rs = asof_score(rs.rename({"rs": "rs_t"}), "rs_t", sc, "sum_rs")
    s_fe = asof_score(fe.rename({"fe": "fe_t"}), "fe_t", sc, "sum_fe")
    s_re = asof_score(re.rename({"re": "re_t"}), "re_t", sc, "sum_re")
    print("\n[3] score anchor: sum of t_score+ct_score at the event tick (backward asof)")
    d_rs = s_rs.with_columns((pl.col("sum_rs") - (pl.col("round") - 1)).alias("dev"))
    print("    round_start: sum - (label-1) counts [0 => label r is physical round r; 1 => physical round r+1]:",
          d_rs.group_by("dev").len().sort("dev").rows())
    d_fe = s_fe.with_columns((pl.col("sum_fe") - (pl.col("round") - 1)).alias("dev"))
    print("    round_freeze_end: sum - (label-1) counts:",
          d_fe.group_by("dev").len().sort("dev").rows())
    d_re = s_re.with_columns((pl.col("sum_re") - pl.col("round")).alias("dev"))
    print("    round_end: sum - label counts [0 => label r ends physical round r]:",
          d_re.group_by("dev").len().sort("dev").rows())

    # ---- 4. round_end winner vs score change across the end tick ----
    pre = asof_score(re.with_columns((pl.col("re") - 1).alias("re_m1")).rename({"re_m1": "re_m1_t"}),
                     "re_m1_t", sc, "sum_pre")
    W = re.join(pre.select(["match_key", "round", "sum_pre"]), on=["match_key", "round"])
    W = W.join(s_re.select(["match_key", "round", "sum_re"]), on=["match_key", "round"])
    W = W.with_columns((pl.col("sum_re") - pl.col("sum_pre")).alias("dsum"))
    print("\n[4] round_end: score-sum change across end tick (1 expected) and winner counts:")
    print("    dsum counts:", W.group_by("dsum").len().sort("dsum").rows())
    print("    winner counts (2=T, 3=CT):", W.group_by("winner").len().sort("winner").rows())
