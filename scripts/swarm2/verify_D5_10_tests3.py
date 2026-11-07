"""Verifier tests 3 for D5-10 (read only; caches in scratchpad only).

V1  Same boundary test on the 2026 bake-off tome (tomes\\2026-10-01_to_08, suffix .2026-09-18,2026-10-08):
    does 2026 show the round-13 flip signature, any regime-A marker, or any flip at any boundary?
    Tie-aware: flip and rec are counted only for non-tie pre-round states.
V2  2024 header-clean matches (header total == round count), regime A: which convention reproduces the header for the
    early-diff matches (first diff at 9 or 10) and for the swap8 group. Conventions: swap sets and inverted rounds.

ASSUMPTION A10: header total == round count is the 'clean' filter (no truncated matches); truncation is inferred from
                header total > last round label and is not otherwise verified.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_tests3.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
T, CT = 2, 3
ROOT = Path(r"G:\pureskill\csds-cs2")


def scan(folder):
    return pl.scan_parquet(sorted(glob.glob(str(folder / "dataframe_*"))))


# ---------------------------------------------------------------- V1 2026 tome
print("== V1 2026 bake-off tome: boundary test (tie-aware) ==")
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
S26 = ".2026-09-18,2026-10-08"
rs26 = scan(B26 / f"basic_round_state{S26}").select(["match_key", "round", "event_type", "t_score", "ct_score"]).collect()
rs26 = rs26.with_columns(pl.col("round").cast(pl.Int64))
ev26 = rs26["event_type"].value_counts().sort("count", descending=True)
print("2026 round_state event types with 'round_announce' or 'cs_win' or 'round_start':",
      [r for r in ev26.rows() if any(k in r[0] for k in ["announce_match_start", "cs_win_panel_round", "round_start"])])
fe26 = (rs26.filter(pl.col("event_type") == "round_freeze_end")
        .unique(subset=["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"])
        .rename({"t_score": "t0", "ct_score": "c0"}))
re26 = scan(B26 / f"basic_round_end{S26}").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
re26 = re26.with_columns(pl.col("round").cast(pl.Int64), pl.col("winner_team_code").cast(pl.Int64)).unique(
    subset=["match_key", "round"], keep="first")
print("2026 freeze-end rows:", fe26.height, "; round_end rows:", re26.height,
      "; round_end labels == 1 present for matches:", re26.filter(pl.col("round") == 1)["match_key"].n_unique(),
      "of", re26["match_key"].n_unique())
RIGHT = fe26.select(["match_key", pl.col("round") - 1, pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
P26 = (fe26.join(RIGHT, on=["match_key", "round"], how="inner")
       .join(re26.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
       .filter(pl.col("winner_team_code").is_in([T, CT])))
P26 = P26.with_columns(tie0=(pl.col("t0") == pl.col("c0")))
wT = (pl.col("winner_team_code") == T).cast(pl.Int64)
wC = (pl.col("winner_team_code") == CT).cast(pl.Int64)
P26 = P26.with_columns(
    H0=(pl.col("t1") == pl.col("t0") + wT) & (pl.col("c1") == pl.col("c0") + wC),
    H1=(pl.col("t1") == pl.col("c0") + wC) & (pl.col("c1") == pl.col("t0") + wT),
    H0f=(pl.col("t1") == pl.col("t0") + wC) & (pl.col("c1") == pl.col("c0") + wT),
    H1f=(pl.col("t1") == pl.col("c0") + wT) & (pl.col("c1") == pl.col("t0") + wC))
P26 = P26.with_columns(cls=pl.when(pl.col("H0") & ~pl.col("H1") & ~pl.col("H0f") & ~pl.col("H1f")).then(pl.lit("rec_H0_only"))
                       .when(pl.col("H1") & ~pl.col("H0") & ~pl.col("H0f") & ~pl.col("H1f")).then(pl.lit("rec_H1_only"))
                       .when(pl.col("H0") | pl.col("H1")).then(pl.lit("rec_fits_ambiguous"))
                       .when(pl.col("H0f") | pl.col("H1f")).then(pl.lit("flip_fits"))
                       .otherwise(pl.lit("neither")))
print("2026 boundary pairs:", P26.height)
print("2026 classes, non-tie pre-state:")
print(P26.filter(~pl.col("tie0")).group_by("cls").len().sort("cls"))
print("2026 flip_fits (non-tie) by boundary label, first 30 labels:")
print(P26.filter(~pl.col("tie0") & (pl.col("cls") == "flip_fits") & (pl.col("round") <= 30))
      .group_by("round").len().sort("round").rows())
print("2026 rec_H1_only (non-tie) by boundary label, first 30 labels:")
print(P26.filter(~pl.col("tie0") & (pl.col("cls") == "rec_H1_only") & (pl.col("round") <= 30))
      .group_by("round").len().sort("round").rows())
print("2026 neither (all states):", P26.filter(pl.col("cls") == "neither").height)

# ---------------------------------------------------------------- V2 2024 header-clean subset, conventions
print("\n== V2 2024 header-clean matches (header total == round count), regulation N <= 24 ==")
PAIR = pl.read_parquet(SCR / "pairs.parquet").with_columns(tie0=(pl.col("t0") == pl.col("c0")))
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
hdr = scan(BASE / "header").select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})
MR = MR.join(hdr.select(["match_key"]), on="match_key", how="left")
swap8 = PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").unique().with_columns(swap8=pl.lit(True))
MR = MR.join(swap8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))
NM = RE.group_by("match_key").agg(pl.col("round").max().alias("N"), pl.len().alias("cnt"))
MRX = MR.join(NM, on="match_key", how="left").with_columns(
    clean=(pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("N")),
    grp=pl.when(pl.col("regime") == "B").then(pl.lit("B"))
    .when(pl.col("swap8")).then(pl.lit("A_swap8")).otherwise(pl.lit("A_noswap8")))
print("clean (header == N) by group:", MRX.group_by(["grp", "clean"]).len().sort(["grp", "clean"]).rows())

# early-diff matches (claim population, first diff at 9 or 10): from the D5 replication (noswap8, recorded winners)
Q = (RE.join(FE, on=["match_key", "round"], how="left").join(MRX.select(["match_key", "grp"]), on="match_key", how="inner")
     .filter(pl.col("grp") == "A_noswap8"))
FE2 = FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2"), pl.col("c0").alias("c2")])
Q = Q.join(FE2, on="match_key", how="left").sort(["match_key", "round"])
onT = (pl.col("round") <= 12) | pl.col("round").is_between(25, 27)
Q = Q.with_columns(a_fe=pl.when(onT).then(pl.col("t0")).otherwise(pl.col("c0")),
                   a_won=pl.when(pl.col("round") == 1).then(pl.when(onT).then(pl.col("t2")).otherwise(pl.col("c2")))
                   .otherwise(pl.when(onT & (pl.col("winner_team_code") == T)).then(1)
                              .when((~onT) & (pl.col("winner_team_code") == CT)).then(1).otherwise(0)))
Q = Q.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
Q = Q.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
ck = Q.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2) & pl.col("winner_team_code").is_not_null())
early_m = ck.filter(pl.col("diff") != 0).group_by("match_key").agg(pl.col("round").min().alias("first_r")).filter(pl.col("first_r") <= 12)
print("early-diff matches (first diff <= 12):", early_m.height)
early_clean = early_m.join(MRX.filter(pl.col("clean")).select("match_key"), on="match_key", how="inner")
print("  of which header-clean:", early_clean.height)
e_pair = PAIR.filter((pl.col("round") == 9) & pl.col("match_key").is_in(early_m["match_key"].to_list()))
print("  boundary 9 pre-state tie status among early-diff matches:", e_pair.group_by("tie0").len().sort("tie0").rows())

swap_sets = {"swap{12}": [12], "swap{8}": [8], "swap{8,12}": [8, 12], "swap{}": []}
inv_sets = {"inv{}": [], "inv{9}": [9], "inv{13}": [13], "inv{9,13}": [9, 13]}


def team_a_total(df, swaps, inv):
    nsw = pl.lit(0)
    for s in swaps:
        nsw = nsw + (pl.col("round") > s).cast(pl.Int64)
    onTx = (nsw % 2 == 0)
    w = pl.col("winner_team_code")
    if inv:
        w = pl.when(pl.col("round").is_in(inv) & (w == T)).then(pl.lit(CT)).when(
            pl.col("round").is_in(inv) & (w == CT)).then(pl.lit(T)).otherwise(w)
    aw = pl.when(onTx & (w == T)).then(1).when((~onTx) & (w == CT)).then(1).otherwise(0)
    return df.with_columns(aw=aw).group_by("match_key").agg(pl.col("aw").sum().alias("A"),
                                                              pl.col("t_starters_score_final").first().alias("hT"))


def grid(mkeys, title):
    R = (RE.filter(pl.col("match_key").is_in(mkeys))
         .join(MRX.select(["match_key", "t_starters_score_final"]), on="match_key", how="left")
         .select(["match_key", "round", "winner_team_code", "t_starters_score_final"]))
    print(f"\n{title}: matches {len(mkeys)}")
    out = []
    for sn, ss in swap_sets.items():
        for iname, ii in inv_sets.items():
            tot = team_a_total(R, ss, ii)
            out.append((sn, iname, int((tot["A"] == tot["hT"]).sum()), tot.height))
    out.sort(key=lambda x: -x[2])
    for row in out[:6]:
        print(f"  {row[0]:<12} {row[1]:<10} header match {row[2]} of {row[3]}")


clean_all = MRX.filter(pl.col("clean") & (pl.col("grp") == "A_noswap8")).select("match_key")["match_key"].to_list()
grid(clean_all, "regime A noswap8, header-clean (all)")
grid(early_clean["match_key"].to_list(), "regime A noswap8 early-diff matches, header-clean")
clean_swap8 = MRX.filter(pl.col("clean") & (pl.col("grp") == "A_swap8")).select("match_key")["match_key"].to_list()
grid(clean_swap8, "regime A swap8, header-clean")
clean_B = MRX.filter(pl.col("clean") & (pl.col("grp") == "B")).select("match_key")["match_key"].to_list()
grid(clean_B, "regime B, header-clean (round 1 missing in round_end; grid uses rounds present)")
