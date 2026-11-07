"""verify_D1_5 controls.
C1  2026 tome: round_end winner vs freeze-end-derived side winner, all rounds (is the defect 2024-specific?).
C2  2024 tome: round-13 (and 9/25/28) round_end-vs-derived disagreement by header build_num and map_name,
    d12-25 only (is the defect tied to a build or map, which would matter for the map question?).
ASSUMPTIONS: A1 side codes, A5 swap rule (MR24 for 2024; for 2026 the same rule is used, which is
ASSUMPTION A9: 2026 max_rounds not checked, MR24 assumed), derived winner uses freeze(r), freeze(r+1).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
MR = 24
DAYS_HI = {f"{d:02d}" for d in range(12, 26)}


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def swap_expected(r, mr=MR):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def chan_files(base, ch, suffix=None):
    if suffix is not None:
        return sorted(glob.glob(str(base / f"basic_{ch}{suffix}" / "dataframe_*")))
    cands = sorted(glob.glob(str(base / f"basic_{ch}.*")))
    if not cands:
        raise SystemExit(f"no folder for {ch} under {base}")
    return sorted(glob.glob(str(Path(cands[0]) / "dataframe_*")))


def load(base, suffix):
    re_ = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                     for f in chan_files(base, "round_end", suffix)], how="vertical_relaxed")
    re_ = re_.unique(["match_key", "round"], keep="first")
    rs = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                    for f in chan_files(base, "round_state", suffix)], how="vertical_relaxed")
    return re_, rs


def fe_table(rs):
    fe_df = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
             .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"),
                                                  pl.col("ct_score").first().alias("c")))
    return {(m, int(r)): (int(t), int(c)) for m, r, t, c in fe_df.select(["match_key", "round", "t", "c"]).iter_rows()}


def fe_winner(fe, m, r):
    a = fe.get((m, r))
    b = fe.get((m, r + 1))
    if a is None or b is None:
        return None
    t_r, c_r = a
    t_n, c_n = b
    if swap_expected(r):
        dT, dC = c_n - t_r, t_n - c_r
    else:
        dT, dC = t_n - t_r, c_n - c_r
    if dT == 1 and dC == 0:
        return 2
    if dC == 1 and dT == 0:
        return 3
    return None


# ------------------------------------------------------------------ C1 2026 control
hdr("C1 2026 control: round_end winner vs freeze-end-derived side, all rounds")
re26, rs26 = load(T26, None)
print("2026 round_end rows", re26.height, "| round_state rows", rs26.height)
print("2026 player_count values:", re26.group_by("player_count").len().sort("len", descending=True).head(5).to_dicts())
fe26 = fe_table(rs26)
cnt, dis, amb = collections.Counter(), collections.Counter(), collections.Counter()
for m, r, w, pc in re26.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    fw = fe_winner(fe26, m, int(r))
    if fw is None:
        amb[int(r)] += 1
        continue
    cnt[int(r)] += 1
    if fw != w:
        dis[int(r)] += 1
tot_n = sum(cnt.values())
tot_d = sum(dis.values())
print(f"2026 evaluable rounds: {tot_n}, disagreements: {tot_d}, ambiguous/unavailable: {sum(amb.values())}")
print("2026 disagreements by round:", {r: dis[r] for r in sorted(dis)} if dis else "none")
print("2026 round-13 evaluable:", cnt.get(13, 0), "disagree:", dis.get(13, 0))

# ------------------------------------------------------------------ C2 2024 build and map
hdr("C2 2024 d12-25: disagreement by header build_num and map_name (rounds 13, 9, 25, 28 and all)")
re24, rs24 = load(T24, SUF24)
fe24 = fe_table(rs24)
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(T24 / "header" / "dataframe_*")))], how="vertical_relaxed")
HDm = HD0.select([pl.col("key").alias("match_key"), "build_num", "map_name", "match_date", "tick_rate"]).unique("match_key")
print("build_num distinct:", HDm.select(pl.col("build_num").n_unique()).item(), "| map_name distinct:", HDm.select(pl.col("map_name").n_unique()).item())
rows = []
for m, r, w, pc in re24.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    day = m[len("csds/2024/02/"):len("csds/2024/02/") + 2]
    if day not in DAYS_HI:
        continue
    fw = fe_winner(fe24, m, int(r))
    if fw is None:
        continue
    rows.append((m, int(r), int(w), fw))
D = pl.DataFrame(rows, schema=["match_key", "round", "w", "fw"], orient="row").with_columns(
    (pl.col("w") != pl.col("fw")).alias("dis"))
D = D.join(HDm, on="match_key", how="left")
r13 = D.filter(pl.col("round") == 13)
print("d12-25 round 13 evaluable:", r13.height, "disagree:", int(r13["dis"].sum()))
print("by build_num (round 13), top 12 by n:")
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                   pl.col("dis").mean().round(3).alias("rate")).sort("n", descending=True).head(12))
print("by map_name (round 13), top 12 by n:")
print(r13.group_by("map_name").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                  pl.col("dis").mean().round(3).alias("rate")).sort("n", descending=True).head(12))
print("by match_date (round 13), first 12 days:")
print(r13.group_by("match_date").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                    pl.col("dis").mean().round(3).alias("rate")).sort("match_date").head(12))
print("\nall d12-25 rounds: disagreement rate by round (9, 13, 25, 28 only):")
print(D.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by("round").agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("dis"), pl.col("dis").mean().round(3).alias("rate")).sort("round"))
# per-match: is the round-13 disagreement the same matches as round-9/25/28 disagreements?
per = D.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by("match_key").agg(
    pl.col("round").filter(pl.col("dis")).sort().alias("dis_rounds"), pl.col("round").sort().alias("eval_rounds"))
per = per.with_columns(pl.col("dis_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("dis_str"))
print("\nd12-25 matches by set of disagreeing rounds among {9,13,25,28}:")
print(per.group_by("dis_str").len().sort("len", descending=True).head(12))
