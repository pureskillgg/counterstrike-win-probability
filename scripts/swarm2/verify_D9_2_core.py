"""verify_D9_2 core: independent recomputation of the build_num 9961 round-13 claim (2024-02 tome, read-only).

Reads header, round_end, bomb_state only (column projection, lazy scans). Writes nothing outside the log.
Sections:
  1. reconcile population with scripts\\swarm\\out\\e2_side_check_r13.log (15,436 / 499; r13 785 / 434)
  2. round-13 pc20 by build_num: 2x2, chi-square (uncorrected), log10 p
  3. round-13 pc20 by path day and by header match_date (UTC) day, per build
  4. within-stratum permutation (2,000 perms, day strata) and Mantel-Haenszel (Haldane) for both stratifications
  5. bomb-derived mismatch by round index and build (does the defect stay at 9/13/25/28?)
  6. final-score test (header t/ct_starters_score_final vs round_end winner codes, regulation matches only)
     this uses neither bomb events nor roster codes
ASSUMPTIONS are printed inline with the word ASSUMPTION.
"""
import glob
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2, chi2_contingency

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
RNG = np.random.default_rng(20260614)


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    miss = [c for c in cols if c not in have]
    if miss:
        print(f"  MISSING in {name}: {miss}")
    return lf.select([c for c in cols if c in have]).collect()


def log10p_chi2(stat):
    return float(chi2.logsf(stat, 1) / math.log(10))


# ---------------------------------------------------------------- load
print("== load")
H = scan("header", ["key", "match_id", "match_date", "build_num", "game_directory", "rushb_version",
                    "map_name", "tick_rate", "match_type", "t_starters_score_final", "ct_starters_score_final"])
H = H.with_columns(
    pl.col("key").alias("match_key"),
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day_path"),
    pl.col("match_date").str.slice(0, 10).alias("date_utc"),
    pl.col("match_date").str.slice(8, 2).alias("day_utc"),
    pl.col("match_date").str.slice(11, 2).alias("hour_utc"),
    pl.col("game_directory").str.extract(r"csgo_v(\d+)", 1).alias("game_build"),
)
print("header matches:", H.height, "| null path-day:", H.filter(pl.col("day_path").is_null()).height,
      "| null match_date:", H.filter(pl.col("match_date").is_null()).height)
print("header build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("ASSUMPTION used: day = path day (csds/2024/02/DD/); header match_date kept as a second stratifier")

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
print("round_end rows:", RE.height)
dup = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"))
print("(match,round) groups with >1 row:", dup.filter(pl.col("n") > 1).height,
      "| of which disagree on winner:", dup.filter(pl.col("nw") > 1).height)
RE1 = (RE.sort(["match_key", "round", "tick"]).unique(subset=["match_key", "round"], keep="first",
                                                       maintain_order=True))
print("round_end unique (match,round):", RE1.height)

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code", "player_id"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").rows())
BSo = BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
outc = (BSo.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
             (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
             pl.len().alias("n_out"),
             pl.col("tick").min().alias("out_tick")))
plant = (BSo.filter(pl.col("event_type") == "bomb_planted")
         .sort(["match_key", "round", "tick"])
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_team"),
              pl.col("player_id").first().alias("planter_id"),
              pl.col("tick").min().alias("plant_tick")))
defuse = (BSo.filter(pl.col("event_type") == "bomb_defused")
          .sort(["match_key", "round", "tick"])
          .group_by(["match_key", "round"])
          .agg(pl.col("player_team_code").first().alias("defuser_team"),
               pl.col("player_id").first().alias("defuser_id")))
print("rounds with bomb outcome:", outc.height, "| both defused and exploded in one round:",
      outc.filter(pl.col("defused") & pl.col("exploded")).height)

R = (RE1.join(outc, on=["match_key", "round"], how="left")
        .join(plant, on=["match_key", "round"], how="left")
        .join(defuse, on=["match_key", "round"], how="left")
        .join(H.select(["match_key", "build_num", "day_path", "date_utc", "day_utc", "hour_utc",
                        "game_build", "rushb_version"]), on="match_key", how="left"))
R = R.with_columns(
    pl.col("defused").fill_null(False),
    pl.col("exploded").fill_null(False),
)
R = R.with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    pl.when(pl.col("defused")).then(pl.lit(3, dtype=pl.Int64))
      .when(pl.col("exploded")).then(pl.lit(2, dtype=pl.Int64))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
)
R = R.with_columns(
    pl.when(pl.col("has_out")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
      .otherwise(None).alias("mm"),
    pl.when(pl.col("has_out")).then(pl.col("bomb_side")).otherwise(pl.col("winner_team_code")).alias("w_bomb"),
)
R.write_parquet(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
                     r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_2_round.parquet"))

# ---------------------------------------------------------------- 1. reconcile
print("\n== 1. reconcile population")
pc20 = R.filter((pl.col("player_count") == 20) & pl.col("has_out"))
print("pc20 & outcome rows:", pc20.height, "| mismatches:", int(pc20["mm"].sum()),
      "| (claim: 15,436 / 499)")
r13 = pc20.filter(pl.col("round") == 13)
print("round 13 pc20 & outcome rows:", r13.height, "| mismatches:", int(r13["mm"].sum()), "| (claim: 785 / 434)")
print("all outcome rows any pc:", R.filter(pl.col("has_out")).height, "| mismatches:",
      int(R.filter(pl.col("has_out"))["mm"].sum()))
print("non-pc20 outcome rows:", R.filter(pl.col("has_out") & (pl.col("player_count") != 20)).height,
      "| mismatches:", int(R.filter(pl.col("has_out") & (pl.col("player_count") != 20))["mm"].sum()))
print("player_count values among outcome rows:",
      R.filter(pl.col("has_out")).group_by("player_count").len().sort("player_count").rows())
print("round 13 rows (any pc) with outcome, by pc:",
      R.filter(pl.col("has_out") & (pl.col("round") == 13)).group_by("player_count").len().sort("player_count").rows())
print("round 13 outcome rows, build x pc:",
      R.filter(pl.col("has_out") & (pl.col("round") == 13)).group_by(["build_num", "player_count"]).len()
       .sort(["build_num", "player_count"]).rows())

# ---------------------------------------------------------------- 2. 2x2 by build
print("\n== 2. round 13 pc20 by build_num")
g = r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("build_num")
print(g)
rows = {int(b): (int(n), int(m)) for b, n, m in g.select(["build_num", "n", "mm"]).rows()}
b9961 = rows[9961]
b9842 = rows[9842]
tab = np.array([[b9961[1], b9961[0] - b9961[1]], [b9842[1], b9842[0] - b9842[1]]])
stat, p, dof, _ = chi2_contingency(tab, correction=False)
print("2x2 [[9961 mm, 9961 ok],[9842 mm, 9842 ok]] =", tab.tolist())
print(f"crude chi-square = {stat:.2f}, df={dof}, log10 p = {log10p_chi2(stat):.1f} (claim: p = 1.1e-166)")
print(f"build 9961 mismatch rate = {b9961[1]/b9961[0]:.4f} ({b9961[1]}/{b9961[0]}); "
      f"build 9842 = {b9842[1]}/{b9842[0]}")

# ---------------------------------------------------------------- 3. by day (path and utc)
print("\n== 3. round 13 pc20 by day")
for col in ["day_path", "day_utc"]:
    t = (r13.group_by([col, "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
         .sort([col, "build_num"]))
    print(f"-- by {col} x build (n, mm):")
    print(t)

# ---------------------------------------------------------------- 4. stratified tests
def strata_table(df, col):
    out = []
    for s in sorted(df[col].unique().to_list()):
        d = df.filter(pl.col(col) == s)
        exp = d["build_num"].to_numpy() == 9961
        mm = d["mm"].to_numpy().astype(int)
        out.append((s, exp, mm))
    return out


def stratified(df, col, label, n_perm=2000):
    print(f"\n-- stratified by {label} ({col})")
    strata = strata_table(df, col)
    both = [(s, e, m) for s, e, m in strata if e.any() and (~e).any()]
    print(f"strata total {len(strata)}; strata with BOTH builds {len(both)}: {[s for s, _, _ in both]}")
    # cell counts per stratum: a=exp&mm, b=exp&ok, c=unexp&mm, d=unexp&ok
    A, B, C, D = [], [], [], []
    for s, e, m in both:
        a = int(((e) & (m == 1)).sum()); b = int(((e) & (m == 0)).sum())
        c = int(((~e) & (m == 1)).sum()); d = int(((~e) & (m == 0)).sum())
        A.append(a); B.append(b); C.append(c); D.append(d)
        print(f"   {s}: 9961 mm {a}/{a+b}; 9842 mm {c}/{c+d}")
    A, B, C, D = map(np.array, (A, B, C, D))
    N = A + B + C + D
    # Mantel-Haenszel, strata with both builds, no correction
    num = np.sum(A * D / N); den = np.sum(B * C / N)
    print(f"MH OR (no correction, both-build strata) = {num/den if den>0 else float('inf'):.3f}")
    # Haldane: add 0.5 to every cell of every stratum (both-build strata)
    a2, b2, c2, d2 = A + 0.5, B + 0.5, C + 0.5, D + 0.5
    n2 = a2 + b2 + c2 + d2
    print(f"MH OR (Haldane +0.5 all cells, both-build strata) = {np.sum(a2*d2/n2)/np.sum(b2*c2/n2):.2f}")
    # zero-cell-only Haldane
    zero = (A == 0) | (B == 0) | (C == 0) | (D == 0)
    a3 = A + 0.5 * zero; b3 = B + 0.5 * zero; c3 = C + 0.5 * zero; d3 = D + 0.5 * zero
    n3 = a3 + b3 + c3 + d3
    print(f"MH OR (Haldane zero-cell strata only; {int(zero.sum())} strata) = "
          f"{np.sum(a3*d3/n3)/np.sum(b3*c3/n3):.2f}")
    # permutation: shuffle build labels within each stratum (keep mismatch fixed), statistic = sum_s (a_s - E_s)
    stat_obs = 0.0
    e_list = []
    for s, e, m in both:
        n_s = len(m)
        e_s = int(e.sum()); m_s = int(m.sum())
        a_s = int(((e) & (m == 1)).sum())
        stat_obs += a_s - e_s * m_s / n_s
    # precompute per stratum arrays
    arrs = [(e.copy(), m.copy()) for _, e, m in both]
    cnt_ge = 0
    for _ in range(n_perm):
        st = 0.0
        for (e, m) in arrs:
            ep = RNG.permutation(e)
            n_s = len(m)
            st += int(((ep) & (m == 1)).sum()) - int(ep.sum()) * int(m.sum()) / n_s
        if st >= stat_obs - 1e-9:
            cnt_ge += 1
    print(f"stratified permutation (one-sided, {n_perm} perms): observed stat = {stat_obs:.3f}; "
          f"p = {(cnt_ge + 1) / (n_perm + 1):.5f} (floor = {1/(n_perm+1):.5f})")
    return A, B, C, D


print("\n== 4. stratified tests, round 13 pc20")
stratified(r13, "day_path", "path day")
stratified(r13, "day_utc", "header match_date day (UTC)")

# ---------------------------------------------------------------- 5. round index structure
print("\n== 5. bomb-derived mismatch by round index and build (all pc20 outcome rows)")
ri = (pc20.group_by(["round", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
          .sort(["round", "build_num"]))
flagged = ri.filter(pl.col("mm") > 0)
print("cells with mismatches (round, build, n, mm):")
print(flagged)
print("build 9842 total mismatches across all rounds:",
      int(pc20.filter(pl.col("build_num") == 9842)["mm"].sum()),
      "| build 9961 total:", int(pc20.filter(pl.col("build_num") == 9961)["mm"].sum()))
print("build 9961 bomb rows, round index 1..30 counts with n>0:",
      pc20.filter(pl.col("build_num") == 9961).group_by("round").len().sort("round").height)

# ---------------------------------------------------------------- 6. final-score test
print("\n== 6. final-score test (header scores vs round_end winner codes; regulation matches only)")
Rr = R.select(["match_key", "round", "winner_team_code", "w_bomb", "build_num", "player_count"])
per_match_rounds = Rr.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").max().alias("max_round"),
                                                pl.col("round").min().alias("min_round"))
print("matches with round_end rows:", per_match_rounds.height)
print("max_round distribution (top):",
      per_match_rounds.group_by("max_round").len().sort("max_round").tail(12).rows())
complete = per_match_rounds.filter((pl.col("max_round") <= 24) & (pl.col("min_round") == 1)
                                   & (pl.col("n_rounds") == pl.col("max_round")))
print("regulation-complete matches (rounds 1..max, max<=24):", complete.height)


def score_expr(col):
    # T-starters: T in rounds 1-12, CT in rounds 13-24; CT-starters the reverse
    t_s = (pl.when((pl.col("round") <= 12) & (pl.col(col) == 2)).then(1).otherwise(0)
           + pl.when((pl.col("round") >= 13) & (pl.col(col) == 3)).then(1).otherwise(0))
    c_s = (pl.when((pl.col("round") <= 12) & (pl.col(col) == 3)).then(1).otherwise(0)
           + pl.when((pl.col("round") >= 13) & (pl.col(col) == 2)).then(1).otherwise(0))
    return t_s, c_s


Rc = Rr.join(complete.select("match_key"), on="match_key", how="inner")
t_rec, c_rec = score_expr("winner_team_code")
t_bom, c_bom = score_expr("w_bomb")
Rc = Rc.with_columns(t_rec.alias("t_rec"), c_rec.alias("c_rec"), t_bom.alias("t_bom"), c_bom.alias("c_bom"))
SC = (Rc.group_by("match_key")
        .agg(pl.col("t_rec").sum(), pl.col("c_rec").sum(), pl.col("t_bom").sum(), pl.col("c_bom").sum(),
             pl.col("build_num").first(), pl.col("winner_team_code").is_null().sum().alias("nulls"))
        .join(H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]),
              on="match_key", how="left"))
SC = SC.with_columns(
    (pl.col("t_starters_score_final").is_null() | pl.col("ct_starters_score_final").is_null()).alias("no_final"),
    ((pl.col("t_rec") == pl.col("t_starters_score_final")) & (pl.col("c_rec") == pl.col("ct_starters_score_final")))
      .alias("ok_rec"),
    ((pl.col("t_bom") == pl.col("t_starters_score_final")) & (pl.col("c_bom") == pl.col("ct_starters_score_final")))
      .alias("ok_bom"),
    (pl.col("t_rec") - pl.col("t_starters_score_final")).alias("d_t_rec"),
)
print("matches in score test:", SC.height, "| with null header score:", int(SC["no_final"].sum()))
sc_ok = SC.filter(~pl.col("no_final"))
print("score-test matches by build (n, ok_recorded, ok_bomb):")
print(sc_ok.group_by("build_num").agg(pl.len().alias("n"), pl.col("ok_rec").sum().alias("ok_recorded"),
                                      pl.col("ok_bom").sum().alias("ok_bomb")).sort("build_num"))
print("recorded-winner score difference (t_rec - header T score) by build:")
print(sc_ok.group_by(["build_num", "d_t_rec"]).len().sort(["build_num", "d_t_rec"]))
# matches with an outcome-mismatch round vs not
mm_matches = R.filter(pl.col("mm") == True).select("match_key").unique()  # noqa: E712
sc_ok = sc_ok.with_columns(pl.col("match_key").is_in(mm_matches["match_key"]).alias("has_mm_round"))
print("score test by (build, has_mismatch_round):")
print(sc_ok.group_by(["build_num", "has_mm_round"]).agg(
    pl.len().alias("n"), pl.col("ok_rec").sum().alias("ok_recorded"), pl.col("ok_bom").sum().alias("ok_bomb"))
    .sort(["build_num", "has_mm_round"]))
print("sum check: header T+CT final vs number of rounds (should be rounds count if no draws, first 5 rows):")
chk = sc_ok.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_sum"))
print(chk.select(["match_key", "build_num", "hdr_sum", "t_rec", "c_rec", "t_bom", "c_bom"]).head(5))
SC.write_parquet(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
                     r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_2_scores.parquet"))
print("\nDONE core")
