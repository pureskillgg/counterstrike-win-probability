"""verify_D2_6 probe 3: match-level classification of the R-chain vs S_delta round-13 flags, and 2026 days 01-02.

Prints only. Uses verify_D2_6_recompute.build / chains (read-only reuse).

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_probe3.py
"""
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_6_recompute as m  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    tab, _ = m.build("2024")
    res = m.chains(tab, "S")
    nxt = tab.select(["match_key", (pl.col("round") - 1).alias("round"),
                      pl.col("t0").alias("t1"), pl.col("ct0").alias("ct1"), pl.col("sx").alias("sx_next")])
    t = tab.join(nxt, on=["match_key", "round"], how="left").with_columns(
        (pl.col("ct1") - pl.col("ct0")).alias("dCT"), (pl.col("t1") - pl.col("t0")).alias("dT"))
    t = t.with_columns(
        pl.when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
        .when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("S_delta"))
    r13 = t.filter(pl.col("round") == 13).select(["match_key", "R", "S", "S_delta", "sx", "t0", "ct0", "t1", "ct1", "sx_next"])
    flag13 = set(r13.filter(pl.col("S_delta").is_not_null() & (pl.col("R") != pl.col("S_delta")))
                 .select("match_key").to_series().to_list())
    cset = res["cset"]
    mR = res["mR"]
    print(f"2024 contiguous matches: {len(cset)}; R-bad (rs vs R-chain): {len(mR)}; "
          f"R != S_delta at 13 (any match): {len(flag13)}; of which contiguous: {len(flag13 & cset)}")
    print(f"R-bad and flagged at 13: {len(mR & flag13)}; R-bad not flagged: {len(mR - flag13)}; "
          f"flagged not R-bad: {len(flag13 - mR)}")

    # R-bad but not flagged at 13 by S_delta
    rb = r13.filter(pl.col("match_key").is_in(list(mR - flag13)))
    print("  R-bad, not flagged at 13: sample of round-13 state (match tail, R, S, S_delta, sx13, sx14, t0_13, ct0_13, t_14, ct_14):")
    for row in rb.head(10).iter_rows():
        print("    ", row[0][-12:], row[1:])
    # classify: how many have S_delta undefined at 13 (no sx match or missing freeze-end 14)
    undef = rb.filter(pl.col("S_delta").is_null()).height if "S_delta" in rb.columns else None
    print(f"  of these, S_delta undefined at 13: {rb.filter(pl.col('S_delta').is_null()).height} of {rb.height}; "
          f"S defined: {rb.filter(pl.col('S').is_not_null()).height}")
    # flagged at 13 but not R-bad: why
    fl = r13.filter(pl.col("match_key").is_in(list(flag13 - mR)))
    print("  flagged at 13 but not R-bad: reasons:")
    rs14_missing = 0
    sx_break = 0
    for row in fl.iter_rows(named=True):
        mk = row["match_key"]
        r14 = t.filter((pl.col("match_key") == mk) & (pl.col("round") == 14)).select(["t0", "ct0", "sx"]).rows()
        if not r14 or r14[0][0] is None or r14[0][1] is None:
            rs14_missing += 1
        else:
            sx_break += 1
    print(f"    rs(14) missing: {rs14_missing}; rs(14) present but match not R-bad (chain cut earlier or no compare): {sx_break}")

    # 2026 days 01-02 check (HANDOFF caution 2: team-attached score fields on days 01-02)
    tab26, _ = m.build("2026")
    res26 = m.chains(tab26, "S")
    days = Counter()
    bad_days = Counter()
    cs = res26["cset"]
    comp = tab26.filter(pl.col("match_key").is_in(list(cs)))
    dd = comp.with_columns(pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2}/\d{2})/", 1).alias("day"))
    for row in dd.select(["match_key", "day"]).unique().iter_rows():
        days[row[1]] += 1
    print("2026 contiguous matches by day (first 6 days):", sorted(days.items())[:6])
    d12 = [k for k, v in days.items() if k.endswith("/10/01") or k.endswith("/10/02")]
    print("2026 contiguous matches on 10/01 and 10/02:", sum(days[k] for k in d12))
    bad_mk = res26["mrsS"] | res26["mR"]
    print(f"2026 rs-vs-S-chain disagreements: rounds {res26['c']['rs_bad_S_rounds']} in {len(res26['mrsS'])} matches; "
          f"R-chain disagreements {res26['c']['rs_bad_R_rounds']}")
    print("2026 rs-vs-S disagreement matches on 10/01-10/02:",
          sum(1 for mk in res26["mrsS"] if ("/2026/10/01/" in mk or "/2026/10/02/" in mk)),
          "(0 expected if HANDOFF team-attached claim holds nowhere on those days)")
    print("2026 sample keys:", list(cs)[:2])


if __name__ == "__main__":
    main()
