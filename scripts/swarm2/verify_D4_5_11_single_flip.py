"""verify D4-5, step 11: which single round would have to flip for round_end to reproduce the header?

Complete 2024 matches, days 12-25, labels from 1, per-match swap convention (mr from the halftime event).
For each match with round_end-based X total != header hT: list every round r whose single winner flip makes
X == hT. Then tabulate the set of such rounds by match kind and disputed pattern. A round that appears
for most mismatched matches points at the wrong round label.
Also reports how often the score-side fix for round 9 / 25 / 28 is the flip that is needed.
Writes stdout only.
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, T, CT  # noqa: E402

base = TOMES["2024"]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_label"))
M = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label")).join(
    H, on="match_key").join(half, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("half_label") == 7).then(pl.lit(16)).otherwise(pl.lit(24)).alias("mr"),
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("day"))
M = M.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25))
keep = set(M["match_key"].to_list())
rows = re_.filter(pl.col("match_key").is_in(list(keep))).sort(["match_key", "round"]).select(
    ["match_key", "round", "winner_team_code"]).rows()
mr_of = {r["match_key"]: r["mr"] for r in M.iter_rows(named=True)}
hT_of = {r["match_key"]: r["hT"] for r in M.iter_rows(named=True)}
kind_of = {r["match_key"]: ("MR8" if r["mr"] == 16 else "MR12") for r in M.iter_rows(named=True)}


def swap_after(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


by_match = defaultdict(list)
for mk, rnd, w in rows:
    by_match[mk].append((rnd, w))


def xtotal(seq, mr):
    s = 0
    x = 0
    for rnd, w in seq:
        side_x_on_T = (s % 2 == 0)
        if (w == T and side_x_on_T) or (w == CT and not side_x_on_T):
            x += 1
        if swap_after(rnd, mr):
            s += 1
    return x


mismatch = 0
flip_counter = Counter()
flip_by_kind = defaultdict(Counter)
unfixable = 0
n_by_kind = Counter()
for mk, seq in by_match.items():
    mr = mr_of[mk]
    hT = hT_of[mk]
    kind = kind_of[mk]
    n_by_kind[kind] += 1
    if xtotal(seq, mr) == hT:
        continue
    mismatch += 1
    fixers = []
    for i, (rnd, w) in enumerate(seq):
        seq2 = list(seq)
        seq2[i] = (rnd, CT if w == T else T)
        if xtotal(seq2, mr) == hT:
            fixers.append(rnd)
    if len(fixers) == 0:
        unfixable += 1
        continue
    for rnd in fixers:
        flip_counter[rnd] += 1
        flip_by_kind[kind][rnd] += 1
    if len(fixers) == 1:
        flip_by_kind[kind]["unique_" + str(fixers[0])] += 1

print(f"complete days 12-25 matches: {len(by_match)}; round_end-based X total != header: {mismatch}; "
      f"no single flip reproduces header: {unfixable}")
print("matches by kind:", dict(n_by_kind))
print("\nrounds whose single flip reproduces the header (counts across mismatched matches, all kinds):")
print(sorted(flip_counter.items(), key=lambda kv: -kv[1])[:12])
for kind in ["MR12", "MR8"]:
    c = flip_by_kind[kind]
    top = sorted([(k, v) for k, v in c.items() if not str(k).startswith("unique")], key=lambda kv: -kv[1])[:10]
    print(f"\n{kind} mismatched matches, rounds whose flip fixes the header:", top)
    uniq = sorted([(k, v) for k, v in c.items() if str(k).startswith("unique")], key=lambda kv: -kv[1])[:10]
    print(f"{kind} mismatched matches with a UNIQUE fixing round:", uniq)
