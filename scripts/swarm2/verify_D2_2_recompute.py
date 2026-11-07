"""verify_D2_2_recompute.py: independent recomputation of claim D2-2 (2024 tome; 2026 as control).

Truth T is built from side-attached sources that do NOT read winner_team_code (R) and do NOT need the
team-X construct (sx) for the truth itself:
  s_score  side whose round_state score rose between freeze_end(r) and freeze_end(r+1)
  s_bomb   first bomb_exploded (-> 2 = T) or bomb_defused (-> 3 = CT) in [freeze_end(r), round_end(r) + 40 ticks]
  s_elim   side whose five players are all dead (victim side = player_info.team_code of that round)
  T        = coalesce(s_score, s_bomb, s_elim)
flagged  = R != T, where R = round_end.winner_team_code (first row per match and round).

Then the cited readings are rebuilt exactly as the claim defines them, to test the algebra:
  X        = players on code 2 in the first valid 5v5 round of the match
  sx       = side of X in round r (majority of X members), sx_prev = side of X in round r-1
  X_won    = (sx == T)                       (uses T, as in D2_final.py)
  pred_static = 2 if X_won else 3            (claim's "static identity")
  pred_lag    = sx_prev if X_won else 5 - sx_prev   (claim's "lag-1 team code")

ASSUMPTION: round window = [round_freeze_end tick, round_end tick + 40 ticks]; first row per key by tick.
ASSUMPTION: elimination = all five players of one side dead in the round (player_death, first per player).
ASSUMPTION: planter = first bomb_planted event in the window.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_2_recompute.py
Memory: lazy scans with column projection; one year at a time; no writes except stdout.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}


def files(base, ch):
    fd = sorted(glob.glob(str(base / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def lz(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols)


def build(year):
    base = TOMES[year]
    RE = (lz(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
          .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True).collect()
          .with_columns(pl.col("winner_team_code").cast(pl.Int64).alias("R")))

    FE = (lz(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end")
          .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True)
          .select(["match_key", "round", pl.col("tick").alias("t_fe"),
                   pl.col("t_score").cast(pl.Int64), pl.col("ct_score").cast(pl.Int64)]).collect())

    # team X and sx (side of X in each round). Needed to track scores across swaps: t_score/ct_score
    # follow the side, so after a swap the field of a team changes (ASSUMPTION, checked below vs s_bomb).
    PI = (lz(base, "player_info", ["match_key", "round", "player_id", "team_code"]).collect()
          .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
          .with_columns(pl.col("team_code").cast(pl.Int64)))
    PC = PI.group_by(["match_key", "round"]).agg(
        pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
    VAL = PC.filter((pl.col("np") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5))
    F = VAL.group_by("match_key").agg(pl.col("round").min().alias("F"))
    XS = (PI.join(F, on="match_key").filter((pl.col("round") == pl.col("F")) & (pl.col("team_code") == 2))
          .select(["match_key", "player_id"]).unique().with_columns(pl.lit(1).alias("isX")))
    PX = PI.join(XS, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(0))
    SXC = PX.group_by(["match_key", "round"]).agg(
        ((pl.col("isX") == 1) & (pl.col("team_code") == 2)).sum().alias("xt"),
        ((pl.col("isX") == 1) & (pl.col("team_code") == 3)).sum().alias("xc"))
    SXC = (SXC.with_columns(pl.when(pl.col("xt") > pl.col("xc")).then(pl.lit(2))
                            .when(pl.col("xc") > pl.col("xt")).then(pl.lit(3))
                            .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
           .select(["match_key", "round", "sx"]))

    # team-tracked score winner: X score = t_score if sx == 2 else ct_score (same for Y), at freeze_end r and r+1
    F0 = FE.join(SXC, on=["match_key", "round"], how="left")
    F1 = FE.select(["match_key", (pl.col("round") - 1).alias("round"),
                    pl.col("t_score").alias("t1"), pl.col("ct_score").alias("c1")])
    S1 = SXC.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_nx")])
    SC = (F0.join(F1, on=["match_key", "round"], how="left").join(S1, on=["match_key", "round"], how="left")
          .with_columns(
              pl.when(pl.col("sx") == 2).then(pl.col("t_score")).otherwise(pl.col("ct_score")).alias("x0"),
              pl.when(pl.col("sx") == 3).then(pl.col("t_score")).otherwise(pl.col("ct_score")).alias("y0"),
              pl.when(pl.col("sx_nx") == 2).then(pl.col("t1")).otherwise(pl.col("c1")).alias("x1"),
              pl.when(pl.col("sx_nx") == 3).then(pl.col("t1")).otherwise(pl.col("c1")).alias("y1"))
          .with_columns((pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
          .with_columns(pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
                        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
                        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score"))
          .select(["match_key", "round", "s_score"]))
    # pure side-field increment (no swap tracking), kept only as a diagnostic
    F1p = FE.select(["match_key", (pl.col("round") - 1).alias("round"),
                     pl.col("t_score").alias("tp1"), pl.col("ct_score").alias("cp1")])
    DIAG = (FE.join(F1p, on=["match_key", "round"], how="left")
            .with_columns((pl.col("tp1") - pl.col("t_score")).alias("dTp"), (pl.col("cp1") - pl.col("ct_score")).alias("dCp"))
            .with_columns(pl.when((pl.col("dTp") == 1) & (pl.col("dCp") == 0)).then(pl.lit(2))
                          .when((pl.col("dCp") == 1) & (pl.col("dTp") == 0)).then(pl.lit(3))
                          .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score_untracked"))
            .select(["match_key", "round", "s_score_untracked"]))

    BS = (lz(base, "bomb_state", ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"])
          .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"])).collect())
    win = (BS.join(FE.select(["match_key", "round", "t_fe"]), on=["match_key", "round"], how="inner")
           .join(RE.select(["match_key", "round", pl.col("tick").alias("t_end")]), on=["match_key", "round"], how="inner")
           .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + 40))
           .sort("tick"))
    res = (win.filter(pl.col("event_type") != "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bo_first"),
                                                 pl.col("event_type").n_unique().alias("bo_types"))
           .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2))
                         .otherwise(pl.lit(3)).cast(pl.Int64).alias("s_bomb")))
    plant = (win.filter(pl.col("event_type") == "bomb_planted")
             .group_by(["match_key", "round"]).agg(pl.col("player_id").first().alias("planter_id"),
                                                   pl.col("player_team_code").first().cast(pl.Int64).alias("planter_code")))

    DT = (lz(base, "player_death", ["match_key", "round", "player_id"]).collect()
          .unique(["match_key", "round", "player_id"], keep="first"))
    dv = DT.join(PI.select(["match_key", "round", "player_id", "team_code"]),
                 on=["match_key", "round", "player_id"], how="left")
    EL = (dv.group_by(["match_key", "round"]).agg(
        (pl.col("team_code") == 2).sum().alias("t_dead"), (pl.col("team_code") == 3).sum().alias("ct_dead"))
          .with_columns(pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
                        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
                        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim"))
          .select(["match_key", "round", "s_elim"]))

    B = (RE.select(["match_key", "round", "R", "player_count"])
         .join(SC, on=["match_key", "round"], how="left")
         .join(DIAG, on=["match_key", "round"], how="left")
         .join(res.select(["match_key", "round", "s_bomb", "bo_types"]), on=["match_key", "round"], how="left")
         .join(plant, on=["match_key", "round"], how="left")
         .join(EL, on=["match_key", "round"], how="left")
         .with_columns(pl.coalesce(["s_score", "s_bomb", "s_elim"]).cast(pl.Int64).alias("T"),
                       pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day")))

    # sx / X were built above (team X = code-2 players of the first valid 5v5 round)
    SXP = SXC.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sx").alias("sx_prev")])
    B = (B.join(SXC, on=["match_key", "round"], how="left")
         .join(SXP, on=["match_key", "round"], how="left")
         .join(PI.select(["match_key", "round", "player_id", pl.col("team_code").alias("planter_pi_code")])
               .rename({"player_id": "planter_id"}), on=["match_key", "round", "planter_id"], how="left"))
    B = B.with_columns(
        (pl.col("R") != pl.col("T")).alias("flag"),
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
        .when(pl.col("round") == 28).then(pl.lit("r28")).otherwise(pl.lit("other")).alias("bin"),
    )
    B = B.with_columns(
        (pl.col("sx") == pl.col("T")).alias("X_won"),
        (pl.col("sx") != pl.col("sx_prev")).alias("swap"),
    )
    B = B.with_columns(
        pl.when(pl.col("X_won")).then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int64).alias("pred_static"),
        pl.when(pl.col("X_won")).then(pl.col("sx_prev")).otherwise(5 - pl.col("sx_prev")).cast(pl.Int64).alias("pred_lag"),
    )
    return B, F


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def analyse(year):
    B, F = build(year)
    section(f"{year}: coverage and source agreement (T built without R, sx, or X)")
    print("match-round rows (round_end):", B.height, " matches:", B["match_key"].n_unique(),
          " matches with valid 5v5 round F:", F.height)
    for c in ["s_score", "s_bomb", "s_elim", "T", "sx", "sx_prev"]:
        print(f"  non-null {c}: {B.filter(pl.col(c).is_not_null()).height}")
    print("player_count distribution (round_end):", B.group_by("player_count").len().sort("player_count").rows())
    print("\nSource agreement (rows where both known; all player counts):")
    for a, c in [("s_score", "s_bomb"), ("s_score_untracked", "s_bomb"), ("s_score", "s_elim"), ("s_bomb", "s_elim"),
                 ("s_score", "R"), ("s_bomb", "R"), ("s_elim", "R")]:
        k = B.filter(pl.col(a).is_not_null() & pl.col(c).is_not_null())
        eq = k.filter(pl.col(a) == pl.col(c)).height
        print(f"  {a} vs {c}: n={k.height} equal={eq} differ={k.height - eq}")
    scope = 20 if year == "2024" else 10
    print(f"\nPRIMARY SCOPE: player_count == {scope} (2024 scope = 20 as in D2_truth.py; 2026 tome rounds have 10 players). Coverage on this scope:")
    B = B.filter(pl.col("player_count") == scope)
    for c in ["s_score", "s_bomb", "s_elim", "T", "sx", "sx_prev"]:
        print(f"  non-null {c} (20-player rounds): {B.filter(pl.col(c).is_not_null()).height} of {B.height}")

    k = B.filter(pl.col("T").is_not_null())
    section(f"{year}: flagged = R != T (T from side-attached sources)")
    print("rows with T:", k.height, " flagged:", k.filter(pl.col("flag")).height,
          f" rate={k.filter(pl.col('flag')).height / max(k.height,1):.4f}")
    print("flagged by round (round, n):",
          k.filter(pl.col("flag")).group_by("round").len().sort("round").rows())
    print(k.group_by("bin").agg(pl.len().alias("n_T"), pl.col("flag").sum().alias("flagged"),
                                pl.col("flag").mean().round(4).alias("rate")).sort("bin"))
    print("flagged by player_count:", k.filter(pl.col("flag")).group_by("player_count").len().rows())
    kk = k.filter(pl.col("sx").is_not_null() & pl.col("sx_prev").is_not_null())
    print(f"flagged with sx and sx_prev known: {kk.filter(pl.col('flag')).height}")
    print("flagged with sx and sx_prev, by round bin:")
    print(kk.filter(pl.col("flag")).group_by("bin").len().sort("bin"))

    section(f"{year}: physical check, R vs bomb outcome (no sx, no X)")
    kb = B.filter(pl.col("s_bomb").is_not_null())
    print("rounds with bomb outcome:", kb.height, " R != s_bomb:", kb.filter(pl.col("R") != pl.col("s_bomb")).height)
    print(kb.group_by("bin").agg(pl.len().alias("n"), (pl.col("R") != pl.col("s_bomb")).sum().alias("R!=bomb"),
                                 (pl.col("s_score") != pl.col("s_bomb")).sum().alias("score!=bomb (both known)")).sort("bin"))
    kr = kb.filter(pl.col("bin") == "r13")
    print("round 13 bomb rows by day (n, R!=bomb):")
    print(kr.group_by("day").agg(pl.len().alias("n"), (pl.col("R") != pl.col("s_bomb")).sum().alias("R!=bomb")).sort("day").rows())
    kp = B.filter(pl.col("planter_code").is_not_null())
    print("planted rounds, planter bomb-event code (should be 2 = T) by bin:")
    print(kp.group_by(["bin", "planter_code"]).len().sort(["bin", "planter_code"]))
    kpp = kp.filter(pl.col("planter_pi_code").is_not_null())
    print("planter bomb-event code == planter's player_info code in same round:",
          kpp.filter(pl.col("planter_code") == pl.col("planter_pi_code")).height, "of", kpp.height)

    section(f"{year}: claim algebra. Flagged rows vs readings pred_static (identity) and pred_lag (lag-1)")
    kx = kk
    print("flagged rows: by bin and sx -> n, R==pred_static, R==pred_lag, swap")
    print(kx.filter(pl.col("flag")).group_by(["bin", "sx"]).agg(
        pl.len().alias("n"), (pl.col("R") == pl.col("pred_static")).sum().alias("R==static"),
        (pl.col("R") == pl.col("pred_lag")).sum().alias("R==lag1"),
        pl.col("swap").sum().alias("swap")).sort(["bin", "sx"]))
    print("unflagged rows: by bin and sx -> n, R==pred_static, R==pred_lag, swap")
    print(kx.filter(~pl.col("flag")).group_by(["bin", "sx"]).agg(
        pl.len().alias("n"), (pl.col("R") == pl.col("pred_static")).sum().alias("R==static"),
        (pl.col("R") == pl.col("pred_lag")).sum().alias("R==lag1"),
        pl.col("swap").sum().alias("swap")).sort(["bin", "sx"]))
    sw = kx.filter(pl.col("swap") & (pl.col("round").is_in([13, 25, 28])))
    print("swap rows at rounds 13/25/28: sx distribution and flag rate by sx")
    print(sw.group_by(["round", "sx"]).agg(pl.len().alias("n"), pl.col("flag").sum().alias("flagged"),
                                           pl.col("flag").mean().round(3).alias("rate")).sort(["round", "sx"]))
    print("identity-model test: flagged rows with sx == 2 (identity predicts R == T there):",
          kx.filter(pl.col("flag") & (pl.col("sx") == 2)).group_by("round").len().sort("round").rows())
    # algebra: pred_static = 2 if X won else 3, X won = (sx == T). So R == pred_static should equal
    # (flag and sx == 3) or (not flag and sx == 2) on every row. Count rows where this identity fails.
    alg = kx.with_columns(
        ((pl.col("R") == pl.col("pred_static")) ==
         ((pl.col("flag") & (pl.col("sx") == 3)) | (~pl.col("flag") & (pl.col("sx") == 2)))).alias("holds"))
    print(f"algebra check (R==pred_static iff flag&sx==3, or ~flag&sx==2): holds on {alg.filter(pl.col('holds')).height} of {alg.height} rows")

    section(f"{year}: match-level flag pattern at rounds 13, 25, 28 (matches with T at those rounds)")
    M = (B.filter(pl.col("round").is_in([13, 25, 28]) & pl.col("T").is_not_null())
         .group_by("match_key").agg(
             pl.col("flag").filter(pl.col("round") == 13).first().alias("f13"),
             pl.col("flag").filter(pl.col("round") == 25).first().alias("f25"),
             pl.col("flag").filter(pl.col("round") == 28).first().alias("f28")))
    print("matches: (f13, f25, f28) counts; null = that round absent")
    print(M.group_by(["f13", "f25", "f28"]).len().sort(["f13", "f25", "f28"]))
    print("round-13 rows with T by day: (day, n rows, flagged):")
    d13 = B.filter((pl.col("round") == 13) & pl.col("T").is_not_null())
    print(d13.group_by("day").agg(pl.len().alias("n"), pl.col("flag").sum().alias("flagged")).sort("day").rows())

    section(f"{year}: non-swap-round flags (rounds other than 13, 25, 28) and header confounds")
    oth = B.filter(pl.col("flag") & ~pl.col("round").is_in([13, 25, 28]))
    print("other flagged rows:", oth.height, " distinct matches:", oth["match_key"].n_unique(),
          " of which in matches flagged at round 13:",
          oth.join(B.filter((pl.col("round") == 13) & pl.col("flag")).select("match_key"), on="match_key").height)
    print("other flagged rows by round (round, n):", oth.group_by("round").len().sort("round").rows())
    hd_files = sorted(glob.glob(str(TOMES[year] / "header" / "dataframe_*")))
    HD = pl.scan_parquet(hd_files).select(
        ["key", "map_name", "build_num", "server_name", "tick_rate", "match_date"]).unique("key").collect()
    m13 = (B.filter((pl.col("round") == 13) & pl.col("T").is_not_null())
           .select(["match_key", "flag"]).rename({"match_key": "key"}))
    j = m13.join(HD, on="key", how="left")
    print("round-13 matches joined to header:", j.height, " header hits:", j.filter(pl.col("map_name").is_not_null()).height)
    for col in ["map_name", "build_num", "server_name", "tick_rate"]:
        g = (j.group_by(col).agg(pl.len().alias("n_matches"), pl.col("flag").sum().alias("flagged_matches"),
                                 pl.col("flag").mean().round(3).alias("rate")).sort("n_matches", descending=True))
        print(f"flag rate by {col} (top 12):")
        print(g.head(12))
    jd = j.with_columns(pl.col("match_date").cast(pl.Utf8))
    print("flag rate by header match_date (day count):")
    print(jd.group_by("match_date").agg(pl.len().alias("n_matches"), pl.col("flag").sum().alias("flagged")).sort("match_date").rows())
    # build vs match day (match_key day) and flags at each swap round, per build
    HB = HD.select(["key", "build_num"]).rename({"key": "match_key"})
    Bb = B.join(HB, on="match_key", how="left")
    print("build_num by match day (day, build, matches with a round_end row):")
    print(Bb.select(["match_key", "day", "build_num"]).unique().group_by(["day", "build_num"]).len().sort(["day", "build_num"]).rows())
    print("flag rate by build_num and round (round, build, rows with T, flagged):")
    rr = (Bb.filter(pl.col("T").is_not_null() & pl.col("round").is_in([2, 3, 5, 9, 12, 13, 14, 20, 24, 25, 27, 28, 29]))
          .group_by(["round", "build_num"]).agg(pl.len().alias("n"), pl.col("flag").sum().alias("flagged"))
          .sort(["round", "build_num"]))
    print(rr)
    ex = (Bb.filter((pl.col("round") == 13) & pl.col("T").is_not_null() & (pl.col("build_num") == 9961) & ~pl.col("flag"))
          .select(["match_key", "day", "sx", "sx_prev", "R", "T", "s_score", "s_bomb", "s_elim"]))
    print("build-9961 round-13 matches with T but NOT flagged (exceptions):", ex.height)
    print(ex.head(15))
    print("physical test by build: bomb-known rounds, R != bomb outcome (side truth independent of sx/T):")
    pb = (Bb.filter(pl.col("s_bomb").is_not_null())
          .group_by(["build_num", "bin"]).agg(pl.len().alias("n_bomb_rounds"),
                                              (pl.col("R") != pl.col("s_bomb")).sum().alias("R_ne_bomb"))
          .sort(["build_num", "bin"]))
    print(pb)
    print("round 13 with T, by day AND build (day, build, rows, flagged): tests per-match build vs per-day date")
    print((Bb.filter((pl.col("round") == 13) & pl.col("T").is_not_null())
           .group_by(["day", "build_num"]).agg(pl.len().alias("rows"), pl.col("flag").sum().alias("flagged"))
           .sort(["day", "build_num"])).rows())
    print("map x build, round-13 rows with T (flagged / rows):")
    mb = (Bb.filter((pl.col("round") == 13) & pl.col("T").is_not_null())
          .join(HD.select(["key", "map_name"]).rename({"key": "match_key"}), on="match_key", how="left")
          .group_by(["map_name", "build_num"]).agg(pl.len().alias("rows"), pl.col("flag").sum().alias("flagged"))
          .sort(["map_name", "build_num"]))
    print(mb)


def main():
    for year in sys.argv[1:] or ["2024", "2026"]:
        analyse(year)


if __name__ == "__main__":
    main()
