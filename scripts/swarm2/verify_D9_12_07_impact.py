"""verify D9-12 step 7: sizing the impact of the round_start label offset and sanity checks.

  (0) round_end player_count distribution per tome (the bake-off rule requires 10; 2024 may use other values).
  (a) 2024 membership flip: label-keyed rule 'round_start row exists for (match, label)'. Under the offset,
      label 1 is physically round 2 and physical round 1 has no round_start row. Count physical round-1
      candidates (round_end label 1) with the modal player_count that the label rule keeps.
  (b) Freeze-delay tails: share of pairs with delay in [600, 2000] ticks (ASSUMPTION: 10 s to 31 s at 64 tps).
  (c) round_end winner vs side score change, for rounds with the modal player_count. ASSUMPTION: t_score tracks the
      T side and ct_score the CT side in the same row, so the winning side's score rises by 1 across the end tick.
  (d) 2024 round_start offset by build_num (header joined on the match uuid in match_key).

Read-only. Writes nothing except stdout (captured by the caller).
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


def asof_side(df, col, sc):
    left = df.select(["match_key", "round", col]).sort(col).with_columns(pl.col(col).alias("_t"))
    return left.join_asof(sc.rename({"tick": "_t"}), on="_t", by="match_key", strategy="backward")


for year, (base, suf) in TOMES.items():
    print(f"\n################ {year}")
    rs = scan(base, suf, "round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("tick").min().alias("rs"))
    re_raw = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    print("(0) round_end player_count value counts:", re_raw.group_by("player_count").len().sort("len", descending=True).rows()[:6])
    re = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("re"),
                                                     pl.col("winner_team_code").first().alias("winner"),
                                                     pl.col("player_count").first().alias("pc"))
    modal_pc = re_raw.group_by("player_count").len().sort("len", descending=True)["player_count"][0]
    st = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = st.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
        pl.col("tick").min().alias("fe"))

    # (a) membership flip sizing (modal player_count)
    has_rs = re.join(rs.select(["match_key", "round"]).with_columns(pl.lit(1).alias("has_rs")),
                     on=["match_key", "round"], how="left")
    print(f"[a] modal player_count = {modal_pc}")
    print(f"    round_end rows: {re.height}; with NO round_start label: {has_rs.filter(pl.col('has_rs').is_null()).height}")
    r1 = re.filter(pl.col("round") == 1)
    print(f"    physical round-1 candidates (round_end label 1): {r1.height}; with modal pc: "
          f"{r1.filter(pl.col('pc') == modal_pc).height}; round_start label 1 present for them (label rule keeps): "
          f"{r1.join(rs.select(['match_key', 'round']), on=['match_key', 'round']).height}")
    lastr = re.group_by("match_key").agg(pl.col("round").max().alias("rmax"))
    lastm = re.join(lastr, on="match_key").filter(pl.col("round") == pl.col("rmax"))
    print(f"    last round_end label per match: {lastm.height}; with round_start label present: "
          f"{lastm.join(rs.select(['match_key', 'round']), on=['match_key', 'round']).height}; with modal pc: "
          f"{lastm.filter(pl.col('pc') == modal_pc).height}")

    # (b) freeze-delay tails
    pairs_same = fe.join(rs, on=["match_key", "round"]).with_columns((pl.col("fe") - pl.col("rs")).alias("d"))
    pairs_prev = fe.join(rs.with_columns((pl.col("round") + 1).alias("round")).rename({"rs": "rs_prev"}),
                         on=["match_key", "round"]).with_columns((pl.col("fe") - pl.col("rs_prev")).alias("d"))
    for nm, p in [("label r (same)", pairs_same), ("label r-1 (shifted)", pairs_prev)]:
        n = p.height
        inr = p.filter((pl.col("d") >= 600) & (pl.col("d") <= 2000)).height
        neg = p.filter(pl.col("d") < 0).height
        print(f"[b] fe(r) vs rs {nm}: n={n}; delay in [600,2000]: {inr} ({inr / max(n, 1):.4f}); negative: {neg}")

    # (c) winner vs side score change
    sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
          .select(["match_key", "tick", "t_score", "ct_score"]).sort("tick"))
    post = asof_side(re.select(["match_key", "round", "re"]), "re", sc).rename(
        {"t_score": "t_post", "ct_score": "ct_post"}).select(["match_key", "round", "t_post", "ct_post"])
    pre = asof_side(re.select(["match_key", "round", "re"]).with_columns((pl.col("re") - 1).alias("re_m1")),
                    "re_m1", sc).rename({"t_score": "t_pre", "ct_score": "ct_pre"}).select(
        ["match_key", "round", "t_pre", "ct_pre"])
    W = re.join(post, on=["match_key", "round"]).join(pre, on=["match_key", "round"])
    W = W.with_columns((pl.col("t_post") - pl.col("t_pre")).alias("dt"),
                       (pl.col("ct_post") - pl.col("ct_pre")).alias("dct"))
    ok = (((pl.col("winner") == 2) & (pl.col("dt") == 1) & (pl.col("dct") == 0)) |
          ((pl.col("winner") == 3) & (pl.col("dct") == 1) & (pl.col("dt") == 0)))
    for lab, sub in [("all rounds", W), (f"pc=={modal_pc}", W.filter(pl.col("pc") == modal_pc))]:
        n = sub.height
        k = sub.filter(ok).height
        print(f"[c] {lab}: rounds={n}; winner side score rises by 1 (dt/dct): {k} ({k / max(n, 1):.4f})")
    bad = W.filter(~ok)
    print("    rounds failing the check, by label (top 8):", bad.group_by("round").len().sort("len", descending=True).head(8).rows())
    print("    failing rounds by winner code:", bad.group_by("winner").len().sort("winner").rows())
    print("    failing rounds by (dt, dct, winner) (top 6):", bad.group_by(["dt", "dct", "winner"]).len().sort("len", descending=True).head(6).rows())

    # (d) build_num by match uuid (2024 only)
    if year == "2024":
        hdr_fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
        H = pl.scan_parquet(hdr_fs).select(["match_id", "build_num"]).unique(subset=["match_id"]).collect()
        rsk = rs.with_columns(pl.col("match_key").str.split("/").list.get(4).alias("match_id")).join(
            H, on="match_id", how="left")  # match_key = csds/YYYY/MM/DD/<uuid>/csds
        print("[d] round_start rows by build_num (uuid join):", rsk.group_by("build_num").len().sort("build_num").rows())
        print("    rows with no header match:", rsk.filter(pl.col("build_num").is_null()).height)
