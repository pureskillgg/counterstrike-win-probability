# verify_A1_1_recompute.py
# Independent recomputation for finding A1-1 (header final scores vs round_end rows, 2026 tome).
# Reads only the 2026 tome (header + basic_round_end + basic_score_update). No E: reads. Lazy/projection, small.
# ASSUMPTION (same as bakeoff.py swap_expected): sides swap after round r if r == max_rounds//2,
#   or r > max_rounds and (r - max_rounds - 3) % 6 == 0.  winner_team_code 2 = T side, 3 = CT side.
# ASSUMPTION: win threshold = max_rounds//2 + 1 (first-to-13 for 24-round, first-to-9 for 16-round).
import os, glob, collections
import polars as pl

print("== verify_A1_1_recompute ==")
print("polars", pl.__version__)
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")

hp = sorted(glob.glob(os.path.join(T, "header", "dataframe_*")))
print("header pages:", [os.path.basename(p) for p in hp])
h = pl.concat([pl.read_parquet(p, columns=["key", "match_id", "match_date", "max_rounds",
                                           "t_starters_score_final", "ct_starters_score_final", "map_name"])
               for p in hp], how="diagonal_relaxed")
print("header rows", h.height, "| unique key", h.get_column("key").n_unique(),
      "| unique match_id", h.get_column("match_id").n_unique())
print("header max_rounds dist:", h.group_by("max_rounds").len().sort("max_rounds").rows())
print("header null final scores:", h.filter(pl.col("t_starters_score_final").is_null() | pl.col("ct_starters_score_final").is_null()).height)

rp = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
print("round_end pages:", len(rp))
re_ = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "winner_team_code", "win_reason_code", "tick"])
                 for p in rp], how="diagonal_relaxed")
print("round_end rows", re_.height, "| distinct match_key", re_.get_column("match_key").n_unique())
print("winner_team_code dist:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("null winner rows:", re_.filter(pl.col("winner_team_code").is_null()).height)
dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match_key, round) groups:", dup.height)

hk = set(h.get_column("key").to_list())
rk = set(re_.get_column("match_key").unique().to_list())
print("round_end keys not in header:", len(rk - hk), "| header keys with no round_end rows:", len(hk - rk))

# Per-match round lists (winner per round), kept in python dicts. Small: 35,992 rows.
rounds = collections.defaultdict(dict)
for k, r, w in re_.select(["match_key", "round", "winner_team_code"]).iter_rows():
    rounds[k][int(r)] = w

def swap_after(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)

def split_for(k, mr, rounds_map, swap_fn=swap_after):
    """Round-derived starter totals (team that started on T side, team that started on CT side)."""
    rt = 0; rc = 0; swaps = 0
    for r in sorted(rounds_map):
        w = rounds_map[r]
        starter_t_on_T = (swaps % 2 == 0)
        if (w == 2) == starter_t_on_T:
            rt += 1
        else:
            rc += 1
        if swap_fn(r, mr):
            swaps += 1
    return rt, rc

rows = []
for k, mid, mr, tS, cS, mdate, mname in h.select(["key", "match_id", "max_rounds", "t_starters_score_final",
                                                    "ct_starters_score_final", "match_date", "map_name"]).iter_rows():
    rm = rounds.get(k, {})
    n = len(rm)
    rmax = max(rm) if rm else 0
    rmin = min(rm) if rm else 0
    contiguous = (n > 0 and rmin == 1 and rmax == n)
    rt, rc = split_for(k, mr, rm)
    thr = mr // 2 + 1
    # first round at which a team's round-derived total reaches the threshold
    cum_t = 0; cum_c = 0; first_reach = None
    swaps = 0
    for r in sorted(rm):
        w = rm[r]
        starter_t_on_T = (swaps % 2 == 0)
        if (w == 2) == starter_t_on_T:
            cum_t += 1
        else:
            cum_c += 1
        if first_reach is None and max(cum_t, cum_c) >= thr:
            first_reach = r
        if swap_after(r, mr):
            swaps += 1
    rows.append(dict(key=k, day=k.split("/")[3], match_id=mid, mr=mr, thr=thr, S=tS + cS, hdr_t=tS, hdr_c=cS,
                     n=n, rmin=rmin, rmax=rmax, contiguous=contiguous, rnd_t=rt, rnd_c=rc,
                     first_reach=first_reach if first_reach is not None else -1,
                     match_date=str(mdate), map_name=mname))

# Category definitions
def cat(r):
    if r["n"] < r["S"]:
        return "C_sum_short"
    if r["n"] > r["S"]:
        return "D_sum_long"
    if r["rnd_t"] == r["hdr_t"] and r["rnd_c"] == r["hdr_c"]:
        return "A_agree"
    return "B_sum_eq_split_diff"

for r in rows:
    r["cat"] = cat(r)
    r["split_ok"] = (r["rnd_t"] == r["hdr_t"] and r["rnd_c"] == r["hdr_c"])
    r["split_swapped"] = (r["rnd_t"] == r["hdr_c"] and r["rnd_c"] == r["hdr_t"])

N = len(rows)
print("\n== 1. Sum-level and split-level agreement (1752 matches) ==")
print("matches", N)
print("total header S", sum(r["S"] for r in rows), "| total round_end rows", sum(r["n"] for r in rows),
      "| diff S - n", sum(r["S"] for r in rows) - sum(r["n"] for r in rows))
cats = collections.Counter(r["cat"] for r in rows)
print("categories:", dict(cats))
print("sum agree (n == S):", sum(1 for r in rows if r["n"] == r["S"]))
print("split agree (both teams):", sum(1 for r in rows if r["split_ok"]))
print("split agree sum-only subset (n==S and split ok):", sum(1 for r in rows if r["n"] == r["S"] and r["split_ok"]))
print("\nby day, category counts:")
bd = collections.defaultdict(collections.Counter)
for r in rows:
    bd[r["day"]][r["cat"]] += 1
for d in sorted(bd):
    print("  day", d, dict(bd[d]), "total", sum(bd[d].values()))
keyday = collections.Counter(r["day"] for r in rows)
print("key-day sizes:", dict(sorted(keyday.items())))

C = [r for r in rows if r["cat"] == "C_sum_short"]
B = [r for r in rows if r["cat"] == "B_sum_eq_split_diff"]
print("\n== 2. Shortfall set (C: n < S) ==")
print("C count", len(C), "| gap min/max", min(r["S"] - r["n"] for r in C), max(r["S"] - r["n"] for r in C),
      "| total gap", sum(r["S"] - r["n"] for r in C))
print("C rounds in these matches (round_end rows):", sum(r["n"] for r in C))
print("C contiguous rounds 1..n:", sum(1 for r in C if r["contiguous"]), "of", len(C))
print("C by day:", dict(collections.Counter(r["day"] for r in C)))
print("C by max_rounds:", dict(collections.Counter(r["mr"] for r in C)))
print("C header S distribution:", sorted(collections.Counter(r["S"] for r in C).items()))
print("C round-derived (rnd_t,rnd_c) vs header (hdr_t,hdr_c) examples:")
for r in C[:6]:
    print("   ", r["key"][-40:], "mr", r["mr"], "n", r["n"], "rnd", (r["rnd_t"], r["rnd_c"]), "hdr", (r["hdr_t"], r["hdr_c"]))
print("C: header diff per team (hdr_t - rnd_t, hdr_c - rnd_c) counts:")
dd = collections.Counter((r["hdr_t"] - r["rnd_t"], r["hdr_c"] - r["rnd_c"]) for r in C)
print("   ", sorted(dd.items(), key=lambda x: (-x[1], x[0]))[:20])
print("C: exactly one team agrees with header:", sum(1 for r in C if (r["hdr_t"] == r["rnd_t"]) != (r["hdr_c"] == r["rnd_c"])),
      "| both teams differ:", sum(1 for r in C if r["hdr_t"] != r["rnd_t"] and r["hdr_c"] != r["rnd_c"]),
      "| both agree (impossible if n<S):", sum(1 for r in C if r["hdr_t"] == r["rnd_t"] and r["hdr_c"] == r["rnd_c"]))
print("C: the team with threshold-level total per round-derived split is the same team as header threshold team:")
n_thr_round = sum(1 for r in C if max(r["rnd_t"], r["rnd_c"]) >= r["thr"])
n_thr_hdr = sum(1 for r in C if max(r["hdr_t"], r["hdr_c"]) >= r["thr"])
print("   round-derived max team >= thr at last recorded round:", n_thr_round, "of", len(C))
print("   header max team >= thr:", n_thr_hdr, "of", len(C))
same_team = sum(1 for r in C if (max(r["hdr_t"], r["hdr_c"]) == r["hdr_t"]) == (max(r["rnd_t"], r["rnd_c"]) == r["rnd_t"]))
print("   header winner-side equals round-derived winner-side:", same_team, "of", len(C))
print("C: first_reach round == n (clinch at last recorded round):", sum(1 for r in C if r["first_reach"] == r["n"]),
      "| first_reach < n:", sum(1 for r in C if 0 < r["first_reach"] < r["n"]),
      "| never reached thr:", sum(1 for r in C if r["first_reach"] == -1), "of", len(C))
print("C: clinch-at-last-round cases by day:", dict(collections.Counter(r["day"] for r in C if r["first_reach"] == r["n"])))
nonclinch = [r for r in C if r["first_reach"] != r["n"]]
print("C not clinched at last round (candidates for truncation):")
for r in nonclinch:
    print("   ", r["key"], "mr", r["mr"], "thr", r["thr"], "n", r["n"], "S", r["S"], "rnd", (r["rnd_t"], r["rnd_c"]), "hdr", (r["hdr_t"], r["hdr_c"]), "first_reach", r["first_reach"])

print("\n== 3. Split-only disagreement set (B: n == S but split differs) ==")
print("B count", len(B), "by day", dict(collections.Counter(r["day"] for r in B)))
print("B split swapped (header t/c swapped vs round-derived):", sum(1 for r in B if r["split_swapped"]), "of", len(B))
for r in B:
    print("   ", r["key"][-40:], "mr", r["mr"], "n", r["n"], "S", r["S"], "rnd", (r["rnd_t"], r["rnd_c"]),
          "hdr", (r["hdr_t"], r["hdr_c"]), "first_reach", r["first_reach"], "clinch_at_n", r["first_reach"] == r["n"])

print("\n== 4. Clinch-stop validation on agreeing matches (A) ==")
A = [r for r in rows if r["cat"] == "A_agree"]
A_clinch_last = sum(1 for r in A if r["first_reach"] == r["n"])
A_clinch_before = sum(1 for r in A if 0 < r["first_reach"] < r["n"])
A_noclinch = [r for r in A if r["first_reach"] == -1]
print("A count", len(A), "| clinched exactly at last round:", A_clinch_last,
      "| clinched before last round (game continued past clinch):", A_clinch_before,
      "| never clinched:", len(A_noclinch))
print("A never-clinched by n vs mr: n == mr:", sum(1 for r in A_noclinch if r["n"] == r["mr"]),
      "| n > mr (OT):", sum(1 for r in A_noclinch if r["n"] > r["mr"]),
      "| n < mr (suspicious):", sum(1 for r in A_noclinch if r["n"] < r["mr"]))
print("A never-clinched cases with n < mr:", [(r["key"][-40:], r["mr"], r["n"], r["S"]) for r in A_noclinch if r["n"] < r["mr"]][:20])
print("A clinched-at-last-round by day:", dict(collections.Counter(r["day"] for r in A if r["first_reach"] == r["n"])))
print("A clinch-before-last by day:", dict(collections.Counter(r["day"] for r in A if 0 < r["first_reach"] < r["n"])))
Ad48 = [r for r in rows if r["day"] in ("04", "05", "06", "07", "08")]
print("days 04-08 (all categories):", dict(collections.Counter(r["cat"] for r in Ad48)),
      "| clinched at last round:", sum(1 for r in Ad48 if r["first_reach"] == r["n"]),
      "| clinched before last:", sum(1 for r in Ad48 if 0 < r["first_reach"] < r["n"]),
      "| never clinched:", sum(1 for r in Ad48 if r["first_reach"] == -1))

print("\n== 5. Round-count arithmetic for the claim ==")
allD = [r for r in rows if r["cat"] == "D_sum_long"]
print("D (n > S) count:", len(allD))
print("rows in 90 split-mismatch matches (B + C):", sum(r["n"] for r in rows if r["cat"] in ("B_sum_eq_split_diff", "C_sum_short")),
      "| share of 35,992:", round(100 * sum(r["n"] for r in rows if r["cat"] in ("B_sum_eq_split_diff", "C_sum_short")) / sum(r["n"] for r in rows), 3))
print("rows in 80 sum-short matches (C):", sum(r["n"] for r in C), "| header S sum over C:", sum(r["S"] for r in C))
print("totals: sum over all of S =", sum(r["S"] for r in rows), "| sum of n =", sum(r["n"] for r in rows))

# Dump per-match table for the other scripts
import csv
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
out_csv = os.path.join(SCRATCH, "verify_A1_1_permatch.csv")
cols = ["key", "day", "match_id", "mr", "thr", "S", "hdr_t", "hdr_c", "n", "rmin", "rmax", "contiguous",
        "rnd_t", "rnd_c", "first_reach", "cat", "match_date", "map_name"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
print("\nper-match table written to scratchpad:", out_csv)
