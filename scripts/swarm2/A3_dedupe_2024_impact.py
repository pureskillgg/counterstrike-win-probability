"""A3: 2024 tome duplicates. Characterise the near-duplicate pairs, and measure how many
round-13 bomb-outcome mismatches (scripts/swarm/e2_side_check_r13.py definition) sit in copies.

Read-only. Same definitions as scripts/swarm/e2_side_check_r13.py:
  rounds with outcome = round_end.player_count == 20 joined to a bomb_defused or bomb_exploded row;
  bomb_side = 3 (CT) if defused, else 2 (T); mismatch = bomb_side != winner_team_code.
Duplicate copies (2024 tome, all matches, no channel or round filter):
  exact: same round_end, player_death, player_info content (bakeoff rule), lowest key kept;
  near:  the non-grouped pairs found by A3_dedupe_2024.py (>= 8 rounds with same winner and
         round_end within 64 ticks), lowest key kept.
ASSUMPTION: mismatch status is compared per (match, round) row. Copies are assumed to have the
same winner (checked below).
"""
import collections
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def main():
    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    BS = read("bomb_state", ["match_key", "round", "event_type"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    H = {r["key"]: r for r in read("header", ["key", "match_date"]).iter_rows(named=True)}
    mkeys = sorted(H)

    # ---------- exact groups (bakeoff rule, all 2024 matches)
    parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in parts.items()}
    groups = collections.defaultdict(list)
    for m in mkeys:
        groups[sigs.get(m, ())].append(m)
    exact_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    exact_drop = {m for g in exact_groups for m in g[1:]}
    print(f"exact groups {len(exact_groups)}; copies {len(exact_drop)}")

    # ---------- near pairs (tick-tolerant, same winners, >= 8 rounds, not one signature)
    per_re = collections.defaultdict(dict)
    for x in RE.iter_rows(named=True):
        per_re[x["match_key"]][int(x["round"])] = (int(x["tick"]), int(x["winner_team_code"]))
    idx = {m: i for i, m in enumerate(mkeys)}
    n = len(mkeys)
    max_r = int(RE["round"].max())
    cnt = np.zeros((n, n), dtype=np.int16)
    for r in range(1, max_r + 1):
        tk = np.full(n, np.nan, dtype=np.float32)
        wn = np.full(n, -1, dtype=np.int64)
        for m, d in per_re.items():
            if r in d:
                tk[idx[m]], wn[idx[m]] = d[r][0], d[r][1]
        have = ~np.isnan(tk)
        if have.sum() < 2:
            continue
        dd = np.abs(tk[:, None] - tk[None, :])
        sw = (wn[:, None] == wn[None, :]) & have[:, None] & have[None, :]
        cnt += ((dd <= 64) & sw).astype(np.int16)
        del dd, sw
    near = []
    for i, j in zip(*np.where(np.triu(cnt >= 8, k=1))):
        a, b = mkeys[i], mkeys[j]
        if sigs.get(a, ()) != sigs.get(b, ()):
            near.append((a, b, int(cnt[i, j])))
    print(f"non-grouped near pairs (>=8 tolerant rounds): {len(near)}")

    # characterise each near pair
    print("\n== near-duplicate pairs ==")
    for a, b, c in near:
        ra, rb = per_re[a], per_re[b]
        common = sorted(set(ra) & set(rb))
        diffs = [rb[r][0] - ra[r][0] for r in common]
        wins_eq = sum(1 for r in common if ra[r][1] == rb[r][1])
        da = collections.Counter()
        db = collections.Counter()
        def ev(m):
            return set(map(tuple, DE.filter(pl.col("match_key") == m).select(["round", "tick", "player_id"]).rows()))
        ea, eb = ev(a), ev(b)
        pa = set(map(tuple, PI.filter(pl.col("match_key") == a).select(["round", "player_id", "team_code"]).rows()))
        pb = set(map(tuple, PI.filter(pl.col("match_key") == b).select(["round", "player_id", "team_code"]).rows()))
        print(f"  {a.split('/')[3]}/{a.split('/')[4][:8]} vs {b.split('/')[3]}/{b.split('/')[4][:8]}")
        print(f"    header dates {H[a]['match_date'][:19]} | {H[b]['match_date'][:19]}; "
              f"rounds {len(ra)} vs {len(rb)}; common rounds {len(common)}; same winner on {wins_eq}")
        print(f"    round_end tick diffs (b-a) over common rounds: min {min(diffs)}, max {max(diffs)}, "
              f"sorted sample {sorted(diffs)[:6]}...{sorted(diffs)[-4:]}")
        print(f"    deaths {len(ea)} vs {len(eb)}; shared (round,tick,pid) {len(ea & eb)}; "
              f"player_info rows {len(pa)} vs {len(pb)}; shared {len(pa & pb)}")

    near_drop = {b for a, b, c in near}
    # ---------- round-13 impact
    print("\n== round-13 mismatch counts (e2 definition) ==")
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (RE.filter(pl.col("player_count") == 20)
         .join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
             pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
    print(f"rounds with outcome (all 2024): {j.height}; mismatches: {int(j['mismatch'].sum())}")
    j = j.with_columns(
        pl.col("match_key").is_in(list(exact_drop)).alias("exact_copy"),
        pl.col("match_key").is_in(list(near_drop)).alias("near_copy"),
    )
    j = j.with_columns((pl.col("exact_copy") | pl.col("near_copy")).alias("any_copy"))
    tot = j.height
    for label, flag in (("exact copies", "exact_copy"), ("near copies", "near_copy"), ("any copy", "any_copy")):
        sub = j.filter(pl.col(flag))
        print(f"  {label}: rows {sub.height}; mismatches {int(sub['mismatch'].sum())}")
    keep = j.filter(~pl.col("any_copy"))
    print(f"after removing copies (exact + near): rows {keep.height}; mismatches {int(keep['mismatch'].sum())}")
    r13 = j.filter(pl.col("round") == 13)
    r13k = keep.filter(pl.col("round") == 13)
    print(f"round 13: rows {r13.height}, mismatches {int(r13['mismatch'].sum())}; after copies removed: "
          f"rows {r13k.height}, mismatches {int(r13k['mismatch'].sum())}")
    d = (r13.group_by("day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
         .sort("day"))
    dk = (r13k.group_by("day").agg(pl.len().alias("n_keep"), pl.col("mismatch").sum().alias("mm_keep"))
          .sort("day"))
    print("round 13 by day (all rows vs copies removed):")
    print(d.join(dk, on="day", how="left").with_columns(pl.col("n_keep").fill_null(0), pl.col("mm_keep").fill_null(0)))
    dates_12_25 = j.filter(pl.col("round") == 13).filter(pl.col("day").cast(pl.Int32).is_between(12, 25))
    print(f"round 13 in days 12-25: rows {dates_12_25.height}, mismatches {int(dates_12_25['mismatch'].sum())}")
    dk1 = keep.filter(pl.col("round") == 13).filter(pl.col("day").cast(pl.Int32).is_between(12, 25))
    print(f"  after copies removed: rows {dk1.height}, mismatches {int(dk1['mismatch'].sum())}")
    other = j.filter(pl.col("round") != 13)
    print(f"other rounds: rows {other.height}, mismatches {int(other['mismatch'].sum())}; after copies removed: "
          f"rows {keep.filter(pl.col('round') != 13).height}, mismatches "
          f"{int(keep.filter(pl.col('round') != 13)['mismatch'].sum())}")

    # copies agree with their lowest key on mismatch status?
    lk = {}
    for g in exact_groups:
        for m in g[1:]:
            lk[m] = g[0]
    for a, b, c in near:
        lk[b] = a
    rows_by = {(r["match_key"], r["round"]): r["mismatch"] for r in j.iter_rows(named=True)}
    disagree = 0
    checked = 0
    for (m, rnd), mm in rows_by.items():
        if m in lk and (lk[m], rnd) in rows_by:
            checked += 1
            if rows_by[(lk[m], rnd)] != mm:
                disagree += 1
    print(f"copy vs lowest-key rows compared: {checked}; mismatch status disagrees: {disagree}")


if __name__ == "__main__":
    main()
