"""verify_D2_3 final: sx-free truth with validated score deltas, match-level pattern, header final-score test,
map/build split, and evaluation of the proposed match-level fix.

Reads base_2024.parquet / base_2026.parquet written by verify_D2_3_recompute.py (scratchpad).
Header table (match_date, build_num, map_name, starter final scores) read directly from the tome (read-only).

Validated sx-free truth (T_v):
  s_score_v   side-attached score delta FE(r)->FE(r+1), used ONLY when sx(r) == sx(r+1)
              (no team-side change across the boundary). Naive use across a swap boundary is invalid
              (verify_D2_3_channels: 100% mismatch with recorded and bomb truth at round 27 in both years).
  s_bomb_w    bomb outcome in the round window (sx-free)
  s_elim_free victim-side elimination rule (sx-free)
  T_v = coalesce(s_score_v, s_bomb_w, s_elim_free)

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_final.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
KEY = ["match_key", "round"]


def header(label):
    fs = sorted(glob.glob(str(TOME[label] / "header" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name",
                                                  "t_starters_score_final", "ct_starters_score_final"])
                      for f in fs], how="vertical_relaxed").rename({"key": "match_key"}).unique("match_key", keep="first")


def prep(label):
    b = pl.read_parquet(SCRATCH / f"base_{label}.parquet")
    b = b.with_columns(
        pl.when(pl.col("sx") == pl.col("sx_nx")).then(pl.col("s_score_free")).otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("s_score_v"))
    b = b.with_columns(
        pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"]).alias("T_v"),
        pl.coalesce(["s_score_v", "s_bomb_w"]).alias("T_v_nelim"))
    b = b.with_columns(
        (pl.col("R") != pl.col("T_v")).alias("flip_v"),
        (pl.col("R") != pl.col("T_v_nelim")).alias("flip_v_nelim"))
    return b


def match_pattern(b, col, label):
    pat = (b.filter(pl.col("round").is_in([13, 25, 28]) & pl.col(col).is_not_null())
           .select(["match_key", "round", col]).pivot(on="round", index="match_key", values=col))
    cols = set(pat.columns)
    out = {}
    if {"13", "25", "28"}.issubset(cols):
        full = pat.drop_nulls(["13", "25", "28"])
        out["all3_n"] = full.height
        out["all3_pattern"] = full.group_by(["13", "25", "28"]).len().sort(["13", "25", "28"])
    if {"13", "25"}.issubset(cols):
        p2 = pat.drop_nulls(["13", "25"])
        out["13_25_n"] = p2.height
        out["13_25_pattern"] = p2.group_by(["13", "25"]).len().sort(["13", "25"])
    if {"13", "28"}.issubset(cols):
        p3 = pat.drop_nulls(["13", "28"])
        out["13_28_n"] = p3.height
        out["13_28_pattern"] = p3.group_by(["13", "28"]).len().sort(["13", "28"])
    out["n13"] = pat.drop_nulls(["13"]).height if "13" in cols else 0
    return out


def main():
    res = {}
    for label in ["2024", "2026"]:
        b = prep(label)
        res[label] = b
        print(f"\n################ {label} ################")
        print("rows:", b.height, " T_v known:", b.filter(pl.col("T_v").is_not_null()).height,
              " T_v from score_v:", b.filter(pl.col("s_score_v").is_not_null()).height,
              " T_v from bomb_w only:", b.filter(pl.col("s_score_v").is_null() & pl.col("s_bomb_w").is_not_null()).height,
              " T_v from elim only:", b.filter(pl.col("s_score_v").is_null() & pl.col("s_bomb_w").is_null()
                                               & pl.col("s_elim_free").is_not_null()).height)
        k = b.filter(pl.col("T_v").is_not_null())
        print("\n-- flip rate by round group, validated sx-free truth T_v --")
        print(k.group_by("grp").agg(pl.len().alias("n"), pl.col("flip_v").sum().alias("R_ne_T"),
                                    pl.col("flip_v").mean().round(4).alias("rate")).sort("grp"))
        print("-- same, without the elimination channel (T_v_nelim) --")
        kn = b.filter(pl.col("T_v_nelim").is_not_null())
        print(kn.group_by("grp").agg(pl.len().alias("n"), pl.col("flip_v_nelim").sum().alias("R_ne_T"),
                                     pl.col("flip_v_nelim").mean().round(4).alias("rate")).sort("grp"))
        print("-- flipped rows outside rounds 13/25/28 (T_v) by round --")
        out_rows = k.filter(pl.col("flip_v") & ~pl.col("round").is_in([13, 25, 28]))
        print("count:", out_rows.height)
        print(out_rows.group_by("round").len().sort("round"))

    b24 = res["2024"]
    print("\n################ 2024 match-level pattern ################")
    for col, lab in [("flip_v", "T_v (validated sx-free)"), ("flip_v_nelim", "score+bomb only")]:
        m = match_pattern(b24, col, "2024")
        print(f"\n-- {lab}: matches with truth at 13 = {m['n13']}")
        if "all3_pattern" in m:
            print(f"   matches with truth at 13, 25, 28: {m['all3_n']}")
            print(m["all3_pattern"])
        if "13_25_pattern" in m:
            print(f"   matches with truth at 13 and 25: {m['13_25_n']}")
            print(m["13_25_pattern"])
        if "13_28_pattern" in m:
            print(f"   matches with truth at 13 and 28: {m['13_28_n']}")
            print(m["13_28_pattern"])

    print("\n################ 2024 round-13 flips by day (T_v) ################")
    r13 = b24.filter((pl.col("round") == 13) & pl.col("T_v").is_not_null())
    print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("flip_v").sum().alias("flipped"),
                                  pl.col("flip_v").mean().round(3).alias("rate")).sort("day"))

    # ---- header: map, build, date; starters' final scores (team-attached)
    hdr = header("2024")
    print("\n################ 2024 header join ################")
    hj = b24.select(["match_key"]).unique().join(hdr, on="match_key", how="left")
    print("matches in base:", b24["match_key"].n_unique(), " matched header:", hj.filter(pl.col("map_name").is_not_null()).height)
    print("header rows:", hdr.height)

    # per-match flags
    per = (b24.group_by("match_key").agg(
        pl.col("flip_v").filter(pl.col("round") == 13).first().alias("f13"),
        pl.col("T_v").filter(pl.col("round") == 13).first().alias("t13"),
        pl.col("flip_v").filter(pl.col("round") == 25).first().alias("f25"),
        pl.col("T_v").filter(pl.col("round") == 25).first().alias("t25"),
        pl.col("flip_v").filter(pl.col("round") == 28).first().alias("f28"),
        pl.col("T_v").filter(pl.col("round") == 28).first().alias("t28"),
        pl.col("flip_v").sum().alias("n_flip_all"),
        pl.len().alias("n_rounds")).join(hdr, on="match_key", how="left"))

    print("\n################ 2024 map and build split of round-13 flags (T_v) ################")
    m13 = per.filter(pl.col("t13").is_not_null())
    print("matches with truth at 13:", m13.height, " flagged (flip at 13):", m13.filter(pl.col("f13")).height)
    print(m13.group_by("map_name").agg(pl.len().alias("matches"), pl.col("f13").sum().alias("flagged"),
                                       pl.col("f13").mean().round(3).alias("rate")).sort("matches", descending=True).head(20))
    print(m13.group_by("build_num").agg(pl.len().alias("matches"), pl.col("f13").sum().alias("flagged"),
                                        pl.col("f13").mean().round(3).alias("rate")).sort("build_num").head(30))
    print("match_date vs flag at 13 (by date string):")
    print(m13.group_by("match_date").agg(pl.len().alias("matches"), pl.col("f13").sum().alias("flagged"))
          .sort("match_date").head(40))

    # ---- header final-score test (team-attached starters' final score)
    print("\n################ 2024 header final-score test (match-level, complete matches) ################")
    rc = b24.group_by("match_key").agg(pl.len().alias("N"), pl.col("round").min().alias("rmin"),
                                       pl.col("round").max().alias("rmax"),
                                       pl.col("sx").is_not_null().all().alias("sx_all"),
                                       pl.col("T_v").is_not_null().all().alias("T_all"),
                                       pl.col("sx").filter(pl.col("round") == 1).first().alias("sx1"))
    comp = rc.filter((pl.col("rmin") == 1) & (pl.col("rmax") == pl.col("N")) & pl.col("sx_all") & pl.col("T_all")
                     & pl.col("sx1").is_not_null())
    print("complete matches (rounds 1..N, sx and T_v known for every round, sx at round 1):", comp.height)
    x = b24.join(comp.select("match_key"), on="match_key", how="inner").group_by("match_key").agg(
        ((pl.col("R") == pl.col("sx")).cast(pl.Int64)).sum().alias("xw_R"),
        ((pl.col("T_v") == pl.col("sx")).cast(pl.Int64)).sum().alias("xw_T"),
        ((pl.col("T_sx") == pl.col("sx")).cast(pl.Int64)).sum().alias("xw_S"),
        pl.col("flip_v").sum().alias("n_flip"),
        pl.len().alias("N"))
    x = x.join(hdr, on="match_key", how="inner")
    x = x.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("N")).alias("sum_ok"))
    print("header total == N rounds:", x.filter(pl.col("sum_ok")).height, "of", x.height)
    x = x.filter(pl.col("sum_ok"))
    print("\nA. header T-starters score vs team-X wins (X = T side at round 1):")
    print("   matches with xw_R == X_final:", x.filter(pl.col("xw_R") == pl.col("t_starters_score_final")).height,
          "  xw_T == X_final:", x.filter(pl.col("xw_T") == pl.col("t_starters_score_final")).height,
          "  xw_S (claim) == X_final:", x.filter(pl.col("xw_S") == pl.col("t_starters_score_final")).height)
    x = x.with_columns(
        pl.when(pl.col("n_flip") > 0).then(pl.lit("some_flip")).otherwise(pl.lit("no_flip")).alias("cohort"))
    print(x.group_by("cohort").agg(
        pl.len().alias("n"),
        (pl.col("xw_R") == pl.col("t_starters_score_final")).sum().alias("R_matches_header"),
        (pl.col("xw_T") == pl.col("t_starters_score_final")).sum().alias("T_v_matches_header"),
        (pl.col("xw_S") == pl.col("t_starters_score_final")).sum().alias("T_sx_matches_header")))
    print("\nB. same test, orientation swapped (X = team on CT at round 1):")
    print("   matches with xw_R == CT-starters final:", x.filter(pl.col("xw_R") == pl.col("ct_starters_score_final")).height)

    # ---- evaluation of the proposed fix (flag the match on round-13 disagreement; relabel 13/25/28)
    print("\n################ 2024 evaluation of the proposed match-level fix (T_v) ################")
    flagged = per.filter(pl.col("f13") == True)  # noqa: E712
    print("flagged matches (flip at 13):", flagged.height)
    print("  of which with truth at 25:", flagged.filter(pl.col("t25").is_not_null()).height,
          " with truth at 28:", flagged.filter(pl.col("t28").is_not_null()).height)
    fl_ot = flagged.filter(pl.col("t25").is_not_null())
    print("  flagged OT matches (truth at 25) flip pattern at 25 / 28:")
    print(fl_ot.group_by(["f25", "f28"]).len().sort(["f25", "f28"]))
    unf = per.filter((pl.col("t13").is_not_null()) & (pl.col("f13") == False))  # noqa: E712
    print("unflagged matches with truth at 13:", unf.height)
    print("  unflagged with a flip at 25 or 28 (missed by rule):",
          unf.filter((pl.col("f25") == True) | (pl.col("f28") == True)).height)  # noqa: E712
    print("  unflagged with truth at 25 flipped:", unf.filter(pl.col("f25") == True).height)  # noqa: E712
    no13 = per.filter(pl.col("t13").is_null())
    print("matches with NO truth at 13 (rule cannot fire):", no13.height,
          " of which flips at 25 or 28:", no13.filter((pl.col("f25") == True) | (pl.col("f28") == True)).height)  # noqa: E712
    print("total rows the fix would relabel (flip at 13/25/28 in flagged matches):",
          b24.join(flagged.select("match_key"), on="match_key", how="inner")
          .filter(pl.col("round").is_in([13, 25, 28]) & (pl.col("flip_v") == True)).height)  # noqa: E712
    print("total 2024 truth-bearing rows flipped outside 13/25/28 (T_v):",
          b24.filter(pl.col("flip_v") & ~pl.col("round").is_in([13, 25, 28]) & pl.col("T_v").is_not_null()).height)

    # ---- streak test (sx-free, supporting only): does the truth at the flipped round agree with neighbours
    # more often than the recorded winner does?
    print("\n################ 2024 neighbour-agreement test at 13, 25, 28 (supporting evidence only) ################")
    nxt = b24.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("R").alias("R_next")])
    tt = b24.join(nxt, on=KEY, how="inner").filter(pl.col("T_v").is_not_null() & pl.col("R_next").is_not_null())
    for r in [13, 25, 28]:
        s = tt.filter(pl.col("round") == r)
        fl = s.filter(pl.col("flip_v"))
        ok = s.filter(~pl.col("flip_v"))
        print(f"round {r}: flipped rows n={fl.height}: R(r+1)==R(r) {fl.filter(pl.col('R_next') == pl.col('R')).height},"
              f" R(r+1)==T(r) {fl.filter(pl.col('R_next') == pl.col('T_v')).height}"
              f" | unflipped n={ok.height}: R(r+1)==R(r) {ok.filter(pl.col('R_next') == pl.col('R')).height}")

    # ---- claim check: the 448 claim-T rows that the validated truth misses
    print("\n################ 2024 claim-T rows not covered by my T_sx (448 expected) ################")
    claim = pl.read_parquet(ROOT / "logs" / "swarm2" / "D2_truth_2024.parquet").select(
        KEY + [pl.col("s_score").alias("c_score"), pl.col("s_bomb").alias("c_bomb"), pl.col("s_elim").alias("c_elim"),
               pl.col("T").alias("c_T")])
    mine = b24.select(KEY + ["s_score_sx", "s_bomb_w", "s_elim_sx", "T_sx"])
    miss = claim.join(mine, on=KEY, how="inner").filter(pl.col("c_T").is_not_null() & pl.col("T_sx").is_null())
    print("rows:", miss.height)
    print(miss.select([pl.col("c_score").is_not_null().alias("claim_has_score"),
                       pl.col("c_bomb").is_not_null().alias("claim_has_bomb"),
                       pl.col("c_elim").is_not_null().alias("claim_has_elim"),
                       pl.col("s_bomb_w").is_not_null().alias("mine_has_bomb"),
                       pl.col("s_elim_sx").is_not_null().alias("mine_has_elim")]).group_by(
        ["claim_has_score", "claim_has_bomb", "claim_has_elim", "mine_has_bomb", "mine_has_elim"]).len().sort("len", descending=True))
    print("rows by round among the 448:")
    print(miss.group_by("round").len().sort("len", descending=True).head(10))


if __name__ == "__main__":
    main()
