"""Independent verification of D1-9 (2024 round_end.player_count convention; duplicate keys).

Reads only (projected columns, one channel per scan, small):
  tomes\\2024-02-g\\tome\\csds   (round_end, player_info, header skipped)
  tomes\\2026-10-01_to_08\\tome\\csds  (round_end, player_info, header)
Writes stdout only; the log is captured to logs\\swarm2\\verify_D1_9.log.

Assumptions (marked ASSUMPTION in output where used):
  - team_code 2 = T, 3 = CT (feature_validation.py line 38).
  - round identity = (match_key, round). Match date = the csds/YYYY/MM/DD/ segment of match_key.
  - player identity: player_id_fixed (bake-off) and player_id (cross_period.py) are both reported.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(200)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
S24 = ".2023-12-28,2024-02-25"
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S26 = ".2026-09-18,2026-10-08"
T_CODE, CT_CODE = 2, 3
DATE_RE = r"^csds/(\d{4}/\d{2}/\d{2})/"


def lf(base, suf, ch):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch} under {base}")
    return pl.scan_parquet(fs)


def header_df(base):
    fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    return pl.read_parquet(fs)


def hdr(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def add_day(df):
    return df.with_columns(day=pl.col("match_key").str.extract(DATE_RE, 1))


# ---------------------------------------------------------------- 1. round_end distributions
hdr("1. round_end.player_count distribution (recomputed)")
RE24 = lf(T24, S24, "round_end").select(
    "match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count").collect()
RE26 = lf(T26, S26, "round_end").select(
    "match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count").collect()
RE24 = add_day(RE24)
RE26 = add_day(RE26)

for name, RE, target in [("2024 tome", RE24, 20), ("2026 tome", RE26, 10)]:
    n_rows = RE.height
    n_keys = RE.select(["match_key", "round"]).unique().height
    n_match = RE["match_key"].n_unique()
    n_null_day = RE["day"].null_count()
    n_t = int((RE["player_count"] == target).sum())
    print(f"{name}: rows={n_rows} distinct(match,round)={n_keys} dup_rows={n_rows - n_keys} "
          f"matches={n_match} match_key_date_null={n_null_day}")
    print(f"  player_count == {target}: {n_t} of {n_rows} rows ({100 * n_t / n_rows:.2f}%)")
    vc = (RE["player_count"].value_counts().sort("count", descending=True))
    print(vc)

print("ASSUMPTION check: the claim says 2024 pc=20 for 78,596 of 83,527 (94%); 2026 pc=10 for 32,041 of 35,992 (89%).")

# ---------------------------------------------------------------- 2. 2024 by match date
hdr("2. 2024 player_count by match date (date from match_key)")
by_day = (RE24.group_by("day").agg(
    pl.len().alias("rows"),
    (pl.col("player_count") == 20).sum().alias("pc20"),
    (pl.col("player_count") == 10).sum().alias("pc10"),
    (~pl.col("player_count").is_in([10, 20])).sum().alias("pc_other"),
    pl.col("player_count").filter(~pl.col("player_count").is_in([10, 20])).unique().sort().alias("other_values"),
).sort("day"))
print(by_day)

per_match_pc = RE24.group_by("match_key").agg(pl.col("player_count").n_unique().alias("n_pc"))
print("2024 matches with a single player_count value across all rounds:",
      per_match_pc.filter(pl.col("n_pc") == 1).height, "of", per_match_pc.height)
print("2024 matches with more than one player_count value:",
      per_match_pc.filter(pl.col("n_pc") > 1).height)
per_match_pc26 = RE26.group_by("match_key").agg(pl.col("player_count").n_unique().alias("n_pc"))
print("2026 matches with a single player_count value across all rounds:",
      per_match_pc26.filter(pl.col("n_pc") == 1).height, "of", per_match_pc26.height)

# ---------------------------------------------------------------- 3. per-round real player count from player_info
hdr("3. player_info: real players per round vs round_end.player_count")
PI24 = lf(T24, S24, "player_info").select(
    "match_key", "round", "player_id", "player_id_fixed", "team_code").collect()
PI26 = lf(T26, S26, "player_info").select(
    "match_key", "round", "player_id", "player_id_fixed", "team_code").collect()
print("2024 player_info team_code counts:", PI24["team_code"].value_counts().sort("team_code").to_dicts())
print("2026 player_info team_code counts:", PI26["team_code"].value_counts().sort("team_code").to_dicts())
print("2024 player_info null player_id_fixed:", PI24["player_id_fixed"].null_count(),
      " null player_id:", PI24["player_id"].null_count())
print("2026 player_info null player_id_fixed:", PI26["player_id_fixed"].null_count(),
      " null player_id:", PI26["player_id"].null_count())


def per_round(PI):
    return PI.group_by(["match_key", "round"]).agg(
        pl.len().alias("pi_rows"),
        pl.col("player_id_fixed").n_unique().alias("n_pid_fixed"),
        pl.col("player_id").n_unique().alias("n_pid"),
        pl.col("player_id_fixed").filter(pl.col("team_code") == T_CODE).n_unique().alias("n_t_fixed"),
        pl.col("player_id_fixed").filter(pl.col("team_code") == CT_CODE).n_unique().alias("n_ct_fixed"),
        pl.col("player_id").filter(pl.col("team_code") == T_CODE).n_unique().alias("n_t"),
        pl.col("player_id").filter(pl.col("team_code") == CT_CODE).n_unique().alias("n_ct"),
    )


PR24 = per_round(PI24)
PR26 = per_round(PI26)
RE24u = RE24.unique(["match_key", "round"], keep="first", maintain_order=True)
RE26u = RE26.unique(["match_key", "round"], keep="first", maintain_order=True)
J24 = RE24u.join(PR24, on=["match_key", "round"], how="left")
J26 = RE26u.join(PR26, on=["match_key", "round"], how="left")
print("2024 rounds (unique) without any player_info row:", J24.filter(pl.col("pi_rows").is_null()).height)
print("2026 rounds (unique) without any player_info row:", J26.filter(pl.col("pi_rows").is_null()).height)

print("\n2024 crosstab: player_count x distinct player_id_fixed per round (top 40 by count)")
print(J24.group_by(["player_count", "n_pid_fixed"]).len().sort("len", descending=True).head(40))
print("\n2026 crosstab: player_count x distinct player_id_fixed per round (top 40 by count)")
print(J26.group_by(["player_count", "n_pid_fixed"]).len().sort("len", descending=True).head(40))

for name, J in [("2024", J24), ("2026", J26)]:
    K = J.filter(pl.col("n_pid_fixed").is_not_null())
    n = K.height
    eq1 = int((K["player_count"] == K["n_pid_fixed"]).sum())
    eq2 = int((K["player_count"] == 2 * K["n_pid_fixed"]).sum())
    eq_rows = int((K["player_count"] == K["pi_rows"]).sum())
    eq_rows2 = int((K["player_count"] == 2 * K["pi_rows"]).sum())
    eq_pid = int((K["player_count"] == K["n_pid"]).sum())
    eq_pid2 = int((K["player_count"] == 2 * K["n_pid"]).sum())
    print(f"{name}: rounds with player_info={n}")
    print(f"  pc == distinct player_id_fixed        : {eq1} ({100 * eq1 / n:.2f}%)")
    print(f"  pc == 2 x distinct player_id_fixed    : {eq2} ({100 * eq2 / n:.2f}%)")
    print(f"  pc == distinct player_id              : {eq_pid} ({100 * eq_pid / n:.2f}%)")
    print(f"  pc == 2 x distinct player_id          : {eq_pid2} ({100 * eq_pid2 / n:.2f}%)")
    print(f"  pc == player_info rows                : {eq_rows} ({100 * eq_rows / n:.2f}%)")
    print(f"  pc == 2 x player_info rows            : {eq_rows2} ({100 * eq_rows2 / n:.2f}%)")
    exact5 = int(((K["n_t_fixed"] == 5) & (K["n_ct_fixed"] == 5)).sum())
    print(f"  rounds with exactly 5 T and 5 CT by player_id_fixed: {exact5} ({100 * exact5 / n:.2f}%)")

print("\n2024 rounds with pc == 20: distinct player_id_fixed distribution")
print(J24.filter(pl.col("player_count") == 20).group_by("n_pid_fixed").len().sort("n_pid_fixed"))
print("2024 rounds with pc != 20: pc x distinct player_id_fixed (all)")
print(J24.filter(pl.col("player_count") != 20).group_by(["player_count", "n_pid_fixed"]).len()
      .sort(["player_count", "n_pid_fixed"]))
print("2026 rounds with pc == 10: distinct player_id_fixed distribution")
print(J26.filter(pl.col("player_count") == 10).group_by("n_pid_fixed").len().sort("n_pid_fixed"))

# ---------------------------------------------------------------- 4. doubling hypothesis by date
hdr("4. Doubling hypothesis (pc == 2 x real players) by 2024 match date")
K24 = add_day(J24).filter(pl.col("n_pid_fixed").is_not_null())
dd = (K24.group_by("day").agg(
    pl.len().alias("rounds"),
    (pl.col("player_count") == 2 * pl.col("n_pid_fixed")).sum().alias("pc_eq_2n"),
    (pl.col("player_count") == pl.col("n_pid_fixed")).sum().alias("pc_eq_n"),
    (pl.col("n_pid_fixed") == 10).sum().alias("n10"),
    (pl.col("n_t_fixed") == 5).sum().alias("t5"),
).sort("day"))
print(dd)

# ---------------------------------------------------------------- 5. duplicate keys
hdr("5. Duplicate keys")
for name, PI, RE, base_name in [("2024", PI24, RE24, "2024"), ("2026", PI26, RE26, "2026")]:
    k = ["match_key", "round", "player_id_fixed"]
    d = PI.filter(pl.len().over(k) > 1)
    print(f"{name} player_info dup rows: {d.height}; dup keys: {d.select(k).unique().height}; "
          f"matches with dup keys: {d['match_key'].n_unique() if d.height else 0}")
    if d.height:
        conf = d.group_by(k).agg(pl.col("team_code").n_unique().alias("nt")).filter(pl.col("nt") > 1).height
        print(f"  dup keys with conflicting team_code: {conf}")
        print(f"  group size distribution: {d.group_by(k).len()['len'].value_counts().to_dicts()}")
        mm = d.group_by("match_key").agg(pl.len().alias("dup_rows"), pl.col("round").n_unique().alias("rounds")).sort("dup_rows", descending=True)
        print(mm)
        print("  dup rounds per match (pc from round_end):")
        for mk in mm["match_key"].to_list():
            rr = RE.filter(pl.col("match_key") == mk).select(["round", "player_count"]).unique().sort("round")
            print("   ", mk, "round_end rows:", RE.filter(pl.col("match_key") == mk).height,
                  "pc values:", rr["player_count"].unique().sort().to_list())
    # round_end duplicates
    dr = RE.filter(pl.len().over(["match_key", "round"]) > 1)
    exact_dup = dr.height - dr.unique().height
    print(f"{name} round_end dup rows: {dr.height}; exact duplicate rows (all columns identical): {exact_dup}")
    if dr.height:
        print(dr)
        dr_mk = set(dr["match_key"].to_list())
        dup_pi_mk = set(d["match_key"].to_list()) if d.height else set()
        print("  round_end dup matches also in player_info dup set:", sorted(dr_mk & dup_pi_mk) or "none")
        print("  round_end dup match not in the claim's 3 player_info dup matches:",
              sorted(dr_mk - dup_pi_mk))

# ---------------------------------------------------------------- 6. duplicate matches across keys (round_end signature)
hdr("6. Duplicate matches across match keys (round_end signature: round,tick,winner,pc)")
for name, RE in [("2024", RE24), ("2026", RE26)]:
    sig = (RE.with_columns(pl.format("{}:{}:{}:{}", "round", "tick", "winner_team_code", "player_count").alias("t"))
           .group_by("match_key").agg(pl.col("t").sort().str.join("|").alias("sig")))
    grp = (sig.group_by("sig").agg(pl.col("match_key").alias("mks"), pl.len().alias("n"))
           .filter(pl.col("n") > 1))
    print(f"{name}: groups of matches with identical round_end signature: {grp.height}; "
          f"matches in those groups: {int(grp['n'].sum()) if grp.height else 0}; "
          f"copies beyond the first: {int((grp['n'] - 1).sum()) if grp.height else 0}")
    if name == "2024" and grp.height:
        bad = {"csds/2024/02/12/2f5bcb89-9ef8-4788-9f3a-2fa3d633ea4d/csds",
               "csds/2024/02/02/1a02a0e2-b132-4353-bb22-7085dd810cee/csds",
               "csds/2024/02/21/1875b0fb-f5f1-4bc4-8ce8-b596f09506ee/csds",
               "csds/2024/02/21/d3ab79fa-68a1-4d7f-a383-56edb91ef8ef/csds"}
        hits = [x for mks in grp["mks"].to_list() for x in mks if x in bad]
        print("  claim-named match keys found inside duplicate-match groups:", hits or "none")

# ---------------------------------------------------------------- 7. label set: win_reason_code vs winner_team_code
hdr("7. Label set check: win_reason_code vs winner_team_code (2024 and 2026)")
for name, RE in [("2024", RE24), ("2026", RE26)]:
    print(name)
    print(RE.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))

# ---------------------------------------------------------------- 8. 2026 header: wingman flag vs player_count
hdr("8. 2026 header: is_wingman vs round_end.player_count")
H26 = header_df(T26).select(["key", "match_id", "is_wingman", "max_unique_players", "match_date", "unique_steamids"])
print("2026 header rows:", H26.height, " distinct key:", H26["key"].n_unique())
print("2026 round_end match_key in header key set:",
      RE26["match_key"].is_in(H26["key"].to_list()).all())
RE26h = RE26.join(H26.select(["key", "is_wingman", "max_unique_players"]), left_on="match_key", right_on="key", how="left")
print(RE26h.group_by(["is_wingman", "player_count"]).len().sort(["is_wingman", "player_count"]))
print("2026 pc != 10 rounds in wingman matches:",
      RE26h.filter((pl.col("player_count") != 10) & (pl.col("is_wingman") == True)).height,
      " in non-wingman matches:",
      RE26h.filter((pl.col("player_count") != 10) & (pl.col("is_wingman") == False)).height)
print("2026 header match_date sample:", H26["match_date"].head(3).to_list())
print("2026 match key years:", RE26["day"].str.slice(0, 4).value_counts().to_dicts())
print("2024 match key years:", RE24["day"].str.slice(0, 4).value_counts().to_dicts())

# ---------------------------------------------------------------- 9. cross-period relaxed rule replication (partial)
hdr("9. Cross-period relaxed rule (2024): unique-player 5v5 check, without any player_count filter")
K24all = J24.filter(pl.col("n_pid").is_not_null())
exact5_pid = int(((K24all["n_t"] == 5) & (K24all["n_ct"] == 5)).sum())
print(f"2024 rounds (unique round_end) with exactly 5 T and 5 CT by player_id: {exact5_pid} "
      f"of {K24all.height} with player_info")
print("  cross_period.json: 'kept_rounds_with_total_players_not_10': 0; "
      "'kept_rounds_with_round_end_player_count_not_10_(not filtered)': 53816")
tot_pid10 = int((K24all["n_pid"] == 10).sum())
print(f"  2024 rounds with exactly 10 distinct player_id: {tot_pid10}")
print("  pc == 20 rounds among those with exactly 5v5 by player_id:",
      int(((K24all["n_t"] == 5) & (K24all["n_ct"] == 5) & (K24all["player_count"] == 20)).sum()))
print("  pc != 20 rounds among those with exactly 5v5 by player_id:",
      int(((K24all["n_t"] == 5) & (K24all["n_ct"] == 5) & (K24all["player_count"] != 20)).sum()))

print("\nDONE")
