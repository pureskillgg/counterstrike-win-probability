"""verify D5-5, step 2: recompute the round-13 claim from the extracted raw tables and test alternatives.

Input: intermediates written by verify_D5_5_extract.py (scratchpad). Nothing is written to the project.

Definitions (ASSUMPTIONS, all marked):
  chain(L)   freeze-end step fe(L) -> fe(L+1): +1 for T only -> 2, +1 for CT only -> 3, else 0.
             ASSUMPTION: score fields are per current side (verified by the swap pattern in the claim example).
             Labels 12, 24, 27 are excluded because their step crosses a side swap.
  rw(L)      round_end winner_team_code of the first round_end row for (match, L) (ASSUMPTION: first by tick;
             one duplicate pair exists in 2024, reported below).
  bomb(L)    bomb outcome for (match, L): defused -> side 3, exploded -> side 2; planted_only = planted with no
             defuse/explode row. Winner side is inferred from the event type alone (ASSUMPTION: game rules).
  pc20       player_count == 20 (same clean-round filter as scripts/swarm/e2_side_check_r13.py).
  late/early day groups as in the claim: late = 2024-02-12..25, early = 2024-02-01..07.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_check.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(70)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
BOUNDARY = [12, 24, 27]
EXAMPLE = "csds/2024/02/12/m_161472e7ab8d0a97e6250fef/csds"


def build(year):
    hd = pl.read_parquet(SCR / f"hdr_{year}.parquet").with_columns(
        day=pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1))
    re = pl.read_parquet(SCR / f"re_{year}.parquet")
    fe = pl.read_parquet(SCR / f"fe_{year}.parquet")
    bs = pl.read_parquet(SCR / f"bomb_{year}.parquet")

    RE1 = (re.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("tick").first().alias("end_tick"),
        pl.col("winner_team_code").first().alias("rw"),
        pl.col("win_reason_code").first().alias("reason"),
        pl.col("player_count").first().alias("pc"),
        pl.col("tick").n_unique().alias("n_end_ticks"),
        pl.len().alias("n_end")))
    FE1 = (fe.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("tick").first().alias("fe_tick"),
        pl.col("t_score").first().alias("t"),
        pl.col("ct_score").first().alias("ct"),
        pl.len().alias("n_fe")))
    BX = (bs.sort(["match_key", "round", "tick"]).group_by(["match_key", "round", "event_type"]).agg(
        pl.col("tick").first().alias("tick"), pl.len().alias("n_ev")))
    BP = BX.pivot(on="event_type", index=["match_key", "round"], values="tick", aggregate_function="first")
    for c in ["bomb_planted", "bomb_defused", "bomb_exploded"]:
        if c not in BP.columns:
            BP = BP.with_columns(pl.lit(None, dtype=pl.Int64).alias(c))
    BP = BP.select(["match_key", "round", "bomb_planted", "bomb_defused", "bomb_exploded"]).rename(
        {"bomb_planted": "plant_tick", "bomb_defused": "defuse_tick", "bomb_exploded": "explode_tick"})
    BP = BP.with_columns(
        status=pl.when(pl.col("defuse_tick").is_not_null() & pl.col("explode_tick").is_not_null())
        .then(pl.lit("both"))
        .when(pl.col("defuse_tick").is_not_null()).then(pl.lit("defused"))
        .when(pl.col("explode_tick").is_not_null()).then(pl.lit("exploded"))
        .when(pl.col("plant_tick").is_not_null()).then(pl.lit("planted_only"))
        .otherwise(pl.lit("none")),
        bside=pl.when(pl.col("defuse_tick").is_not_null()).then(pl.lit(3))
        .when(pl.col("explode_tick").is_not_null()).then(pl.lit(2)).otherwise(pl.lit(None, dtype=pl.Int64)),
        out_tick=pl.when(pl.col("defuse_tick").is_not_null()).then(pl.col("defuse_tick"))
        .otherwise(pl.col("explode_tick")),
    )

    cur = FE1.select(["match_key", "round", "t", "ct", "fe_tick", "n_fe"])
    nxt = FE1.select(["match_key", (pl.col("round") - 1).alias("round"),
                      pl.col("t").alias("t_n"), pl.col("ct").alias("ct_n")])
    J = (cur.join(nxt, on=["match_key", "round"], how="left")
         .join(RE1, on=["match_key", "round"], how="left")
         .join(BP, on=["match_key", "round"], how="left")
         .join(hd.select(["match_key", "day", "t_starters_score_final", "ct_starters_score_final"]),
               on="match_key", how="left"))
    if year == "2026":
        grp = pl.lit("y2026")
    else:
        grp = (pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late"))
               .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
    J = J.with_columns(
        year=pl.lit(year),
        grp=grp,
        dt=pl.col("t_n") - pl.col("t"),
        dct=pl.col("ct_n") - pl.col("ct"),
    ).with_columns(
        chain=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
        .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)),
    ).with_columns(
        cmp=pl.when(pl.col("rw").is_null() | pl.col("t_n").is_null()).then(pl.lit("no_data"))
        .when(pl.col("chain") == 0).then(pl.lit("chain0"))
        .when(pl.col("chain") == pl.col("rw")).then(pl.lit("agree"))
        .otherwise(pl.lit("disagree")),
        lc=pl.when(pl.col("round") == 13).then(pl.lit("L13"))
        .when(pl.col("round") == 25).then(pl.lit("L25"))
        .when(pl.col("round") == 28).then(pl.lit("L28"))
        .otherwise(pl.lit("other")),
        bmm=pl.when(pl.col("bside").is_null() | pl.col("rw").is_null())
        .then(pl.lit(None, dtype=pl.Boolean)).otherwise(pl.col("bside") != pl.col("rw")),
    )
    return J, RE1, hd


def show(df, title, n=60):
    print(f"\n-- {title}")
    print(df.head(n))


J24, RE24, HD24 = build("2024")
J26, RE26, HD26 = build("2026")

print("== data sanity ==")
print("2024 round_end (match,round) with >1 row:")
print(RE24.filter(pl.col("n_end") > 1))
print("2024 round_end rows with >1 distinct tick:", RE24.filter(pl.col("n_end_ticks") > 1).height)
print("2024 freeze_end (match,round) with >1 row:")
print(J24.filter(pl.col("n_fe") > 1).select(["match_key", "round", "n_fe"]))
print("2024 match count:", HD24.height, " matches with any round_end:", RE24["match_key"].n_unique())
print("2026 match count:", HD26.height, " matches with any round_end:", RE26["match_key"].n_unique())

# ---------------- S1: reproduce the e2 counts ----------------
print("\n== S1 reproduce e2: bomb outcome vs round_end winner (2024, pc20, rounds with defuse/explode) ==")
x = J24.filter((pl.col("pc") == 20) & pl.col("bside").is_not_null() & pl.col("rw").is_not_null())
print("rounds with outcome (all labels):", x.height, " mismatches:", int(x["bmm"].sum()))
print(x.group_by(["lc", "grp"]).agg(pl.len().alias("n"), pl.col("bmm").sum().alias("mm"),
                                    pl.col("bmm").mean().round(4).alias("rate")).sort(["lc", "grp"]))
print("bomb status counts among pc20 rounds with outcome (should be exactly one of defused/exploded):")
print(x.group_by("status").len())

# ---------------- S2: the claimed affected set ----------------
print("\n== S2 claimed affected set: matches with a round-13 bomb/round_end mismatch ==")
aff_keys = x.filter((pl.col("round") == 13) & pl.col("bmm"))["match_key"].unique().to_list()
print("affected matches (aff_r13):", len(aff_keys))
A13 = J24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("match_key").is_in(aff_keys))
print("rows at L13 for affected matches:", A13.height)
print("chain status:", A13.group_by("chain").len().sort("chain").to_dicts())
print("cmp status (chain vs rw):", A13.group_by("cmp").len().to_dicts())
A13c = A13.filter(pl.col("chain") != 0)
print("nonzero chain:", A13c.height,
      " chain == bomb side:", int((A13c["chain"] == A13c["bside"]).sum()),
      " chain == rw:", int((A13c["chain"] == A13c["rw"]).sum()))
print("split (chain, bomb, rw) -> count:")
print(A13c.group_by(["chain", "status", "rw", "reason"]).len().sort(["chain", "status", "rw", "reason"]))
print("bomb tick inside [fe13, end13]:", int(((A13["out_tick"] >= A13["fe_tick"]) &
                                              (A13["out_tick"] <= A13["end_tick"])).sum()), "of", A13.height)
print("offset end_tick - out_tick (affected L13):")
print(A13.with_columns((pl.col("end_tick") - pl.col("out_tick")).alias("off"))["off"].describe())
print("offset value counts (top 10):")
print(A13.with_columns((pl.col("end_tick") - pl.col("out_tick")).alias("off")).group_by("off").len().sort("len", descending=True).head(10))

# ---------------- S3: unselected control at L13 ----------------
print("\n== S3 control: L13 rounds with outcome in NON-affected matches (pc20, late+early) ==")
C13 = J24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("bside").is_not_null()
                 & pl.col("rw").is_not_null() & ~pl.col("match_key").is_in(aff_keys))
print("rows:", C13.height, " rw == bomb:", int((~C13["bmm"]).sum()))
print(C13.group_by(["grp", "status", "cmp"]).len().sort(["grp", "status", "cmp"]))
print("nonzero chain & chain == bomb (control):",
      int(((C13["chain"] != 0) & (C13["chain"] == C13["bside"])).sum()),
      " nonzero chain & chain != bomb:", int(((C13["chain"] != 0) & (C13["chain"] != C13["bside"])).sum()))

# ---------------- S4: L13 chain-vs-round_end over ALL matches ----------------
print("\n== S4 L13: chain vs round_end winner over ALL pc20 matches, split by bomb status ==")
L13 = J24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("rw").is_not_null() & pl.col("t_n").is_not_null())
print(L13.group_by(["grp", "status", "cmp"]).len().sort(["grp", "status", "cmp"]))
print("late L13 rounds total:", L13.filter(pl.col("grp") == "late").height,
      " late disagree:", L13.filter((pl.col("grp") == "late") & (pl.col("cmp") == "disagree")).height)
print("early L13 rounds total:", L13.filter(pl.col("grp") == "early").height,
      " early disagree:", L13.filter((pl.col("grp") == "early") & (pl.col("cmp") == "disagree")).height)

# ---------------- S5: planted-only rounds (bomb planted, no defuse/explode row) ----------------
print("\n== S5 planted_only rounds (pc20): round_end winner and reason, by label class and group ==")
PO = J24.filter((pl.col("status") == "planted_only") & (pl.col("pc") == 20) & pl.col("rw").is_not_null())
print(PO.group_by(["grp", "lc", "rw", "reason"]).len().sort(["grp", "lc", "rw", "reason"]))
print("planted_only with rw == 3 (CT win) and no defuse row -- impossible if plant implies T alive and CT must defuse:")
print(PO.group_by(["grp", "lc"]).agg(pl.len().alias("n"), (pl.col("rw") == 3).sum().alias("rw3"),
                                    (pl.col("rw") == 2).sum().alias("rw2")).sort(["grp", "lc"]))

# ---------------- S6: chain vs round_end agreement by label ----------------
print("\n== S6 per-label disagreement rate of chain vs round_end (labels 12, 24, 27 excluded: swap boundaries) ==")
Lab = J24.filter(~pl.col("round").is_in(BOUNDARY) & (pl.col("pc") == 20) & (pl.col("chain") != 0)
                 & pl.col("rw").is_not_null())
r6 = (Lab.group_by(["grp", "round"]).agg(pl.len().alias("n_chain"),
                                         (pl.col("cmp") == "disagree").sum().alias("disagree"))
      .with_columns((pl.col("disagree") / pl.col("n_chain")).round(3).alias("rate")).sort(["grp", "round"]))
print(r6.filter(pl.col("grp").is_in(["late", "early"])))

# ---------------- S7: per-day disagreement at L13 (late) ----------------
print("\n== S7 L13 per-day chain-vs-round_end disagreement (late and early, pc20, nonzero chain) ==")
d7 = (L13.filter(pl.col("chain") != 0).group_by("day").agg(pl.len().alias("n"),
                                                          (pl.col("cmp") == "disagree").sum().alias("disagree"),
                                                          pl.col("bmm").sum().alias("bomb_mm"),
                                                          (pl.col("bside").is_not_null()).sum().alias("n_outcome"))
      .with_columns((pl.col("disagree") / pl.col("n")).round(3).alias("rate")).sort("day"))
print(d7)

# ---------------- S8: tick offset between bomb outcome and round_end ----------------
print("\n== S8 tick offset end_tick - out_tick (pc20, outcome rounds) by class ==")
O = J24.filter((pl.col("pc") == 20) & pl.col("bside").is_not_null() & pl.col("rw").is_not_null()).with_columns(
    off=pl.col("end_tick") - pl.col("out_tick"),
    cls=pl.when(pl.col("round") == 13).then(pl.lit("L13"))
    .when(pl.col("match_key").is_in(aff_keys)).then(pl.lit("aff_other_labels"))
    .otherwise(pl.lit("other")))
O = O.with_columns(is_aff=pl.col("match_key").is_in(aff_keys))
print(O.group_by(["grp", "is_aff", "lc", "bmm"]).agg(
    pl.len().alias("n"),
    pl.col("off").median().alias("median_off"),
    pl.col("off").min().alias("min_off"),
    pl.col("off").max().alias("max_off"),
    (pl.col("off") == 429).sum().alias("n_off429"),
    (pl.col("off") == 0).sum().alias("n_off0"),
).sort(["grp", "is_aff", "lc", "bmm"]))
print("offset 429 histogram among L13 affected rows:")
print(O.filter((pl.col("round") == 13) & pl.col("is_aff")).group_by("off").len().sort("off").head(20))

# ---------------- S9: header final score as an independent reference ----------------
print("\n== S9 header final score test (independent of round_end and freeze_end) ==")


def header_test(year, RE1, J, hd):
    c13 = J.filter(pl.col("round") == 13).select(["match_key", pl.col("chain").alias("chain13")])
    R = RE1.join(c13, on="match_key", how="left").with_columns(
        w_fix=pl.when((pl.col("round") == 13) & (pl.col("chain13") != 0)).then(pl.col("chain13"))
        .otherwise(pl.col("rw")))
    R = R.with_columns(
        a_on_t=(pl.col("round") <= 12) | ((pl.col("round") >= 25) & ((((pl.col("round") - 25) // 3) % 2) == 0)))
    R = R.with_columns(
        a_win_rw=((pl.col("a_on_t") & (pl.col("rw") == 2)) | (~pl.col("a_on_t") & (pl.col("rw") == 3))).cast(pl.Int32),
        a_win_fix=((pl.col("a_on_t") & (pl.col("w_fix") == 2)) | (~pl.col("a_on_t") & (pl.col("w_fix") == 3))).cast(pl.Int32),
    )
    M = R.group_by("match_key").agg(
        pl.len().alias("n_rounds"),
        pl.col("a_win_rw").sum().alias("A_rw"),
        pl.col("a_win_fix").sum().alias("A_fix"),
        pl.col("rw").is_null().sum().alias("n_null_rw"),
        pl.col("round").max().alias("max_round"),
    ).join(hd.select(["match_key", "day", "t_starters_score_final", "ct_starters_score_final"]),
           on="match_key", how="left")
    M = M.with_columns(
        total=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"),
        complete=(pl.col("n_rounds") == pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")),
        aff=pl.col("match_key").is_in(aff_keys),
        A_eq_t_rw=pl.col("A_rw") == pl.col("t_starters_score_final"),
        A_eq_c_rw=pl.col("A_rw") == pl.col("ct_starters_score_final"),
        A_eq_t_fix=pl.col("A_fix") == pl.col("t_starters_score_final"),
        A_eq_c_fix=pl.col("A_fix") == pl.col("ct_starters_score_final"),
    )
    print(f"\n{year}: matches {M.height}, complete (rounds == header final sum): {int(M['complete'].sum())}")
    print(M.group_by(["complete"]).agg(pl.len().alias("n")).sort("complete"))
    C = M.filter(pl.col("complete") & ~pl.col("aff"))
    print("non-affected complete matches:", C.height,
          " A_rw == t_final:", int(C["A_eq_t_rw"].sum()), " A_rw == ct_final:", int(C["A_eq_c_rw"].sum()))
    Af = M.filter(pl.col("complete") & pl.col("aff"))
    print("affected complete matches:", Af.height,
          " A_rw == t_final:", int(Af["A_eq_t_rw"].sum()), " A_fix == t_final:", int(Af["A_eq_t_fix"].sum()),
          " A_rw == ct_final:", int(Af["A_eq_c_rw"].sum()), " A_fix == ct_final:", int(Af["A_eq_c_fix"].sum()))
    print("affected complete: A_rw matches (t or c):",
          int((Af["A_eq_t_rw"] | Af["A_eq_c_rw"]).sum()), " A_fix matches (t or c):",
          int((Af["A_eq_t_fix"] | Af["A_eq_c_fix"]).sum()))
    print("affected complete, t-final orientation: A_rw==t:", int(Af["A_eq_t_rw"].sum()),
          " A_fix==t:", int(Af["A_eq_t_fix"].sum()), " of", Af.height)
    print("grp-level non-affected complete (A_rw == t_final by group):")
    print(C.group_by("day").agg(pl.len().alias("n"), pl.col("A_eq_t_rw").sum().alias("eq_t"),
                                pl.col("A_eq_c_rw").sum().alias("eq_c")).sort("day"))
    return M


M24 = header_test("2024", RE24, J24, HD24)
M26 = header_test("2026", RE26, J26, HD26)

# ---------------- S10: 2026 control ----------------
print("\n== S10 2026 control (same tests, all pc20 matches) ==")
print("2026 outcome rounds pc20:", J26.filter((pl.col("pc") == 20) & pl.col("bside").is_not_null()).height,
      " bomb/round_end mismatches:", int(J26.filter((pl.col("pc") == 20) & pl.col("bmm").fill_null(False)).height))
print("2026 L13 cmp (chain vs rw):")
print(J26.filter((pl.col("round") == 13) & (pl.col("pc") == 20)).group_by(["cmp"]).len().sort("cmp"))
r10 = (J26.filter(~pl.col("round").is_in(BOUNDARY) & (pl.col("pc") == 20) & (pl.col("chain") != 0)
                  & pl.col("rw").is_not_null())
       .group_by("round").agg(pl.len().alias("n_chain"), (pl.col("cmp") == "disagree").sum().alias("disagree"))
       .with_columns((pl.col("disagree") / pl.col("n_chain")).round(3).alias("rate")).sort("round"))
print(r10)

# ---------------- S11: examples ----------------
print("\n== S11 worked examples ==")


def show_match(J, key, labels=range(11, 16)):
    sub = J.filter((pl.col("match_key") == key) & pl.col("round").is_in(list(labels))).sort("round")
    print(f"\n{key}")
    print(sub.select(["round", "t", "ct", "t_n", "ct_n", "chain", "rw", "reason", "pc", "fe_tick", "end_tick",
                      "status", "plant_tick", "out_tick", "cmp"]))


for key in [EXAMPLE,
            "csds/2024/02/12/m_fd60894d07788430d780ef75/csds",
            "csds/2024/02/12/m_a6f7a940a3cb6238e1ffc830/csds",
            "csds/2024/02/12/m_e43036da6af5ea4d2411b7a4/csds"]:
    show_match(J24, key)

print("\nDONE")
