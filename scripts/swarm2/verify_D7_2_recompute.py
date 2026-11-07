"""verify D7-2 (part 1): recompute the first-after-swap test from the raw 2024 tome.

Inputs (raw tome only, no D7 intermediates as inputs):
  header: key, match_date, build_num
  basic_round_end: match_key, round, winner_team_code, player_count
  basic_round_state: round_freeze_end rows (t_score, ct_score)
Own implementation of the side ledger: minimum-orientation-change DP per run of consecutive rounds.
Compared afterwards with D7_ledger_dp.parquet only as a reproduction check.

ASSUMPTION A1: the minimum number of orientation changes equals the true number of side swaps (same as D7-8).
ASSUMPTION A2: round winner = team whose score rose by exactly 1 between consecutive freeze_end rows (same as D7-8).
ASSUMPTION A3: first-after-swap (fas) round = swap round + 1, rounds 9 and 13 only, player_count == 20 (same as D7-12).
Threshold tests use only valid cut points (between DISTINCT key values), in both directions.
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
INF = 10**9


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


# ---------- header ----------
H = (pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num"]) for f in files(BASE / "header")],
               how="vertical_relaxed")
     .rename({"key": "match_key"}))
H = H.with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"),
    pl.col("match_date").str.contains(r"\.\d{9}Z$").alias("ns_timestamp"),
)
print("header rows:", H.height, "| mdt nulls:", H.filter(pl.col("mdt").is_null()).height,
      "| path_day nulls:", H.filter(pl.col("path_day").is_null()).height,
      "| nanosecond match_date rows:", H.filter(pl.col("ns_timestamp")).height)
print("nanosecond match_date rows (key, raw, build):")
print(H.filter(pl.col("ns_timestamp")).select(["match_key", "match_date", "build_num"]).with_columns(
    pl.col("match_key").str.slice(0, 45)).rows())

# ---------- round_end ----------
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                for f in files(BASE / f"basic_round_end{SUF}")], how="vertical_relaxed")
RE20 = RE.filter(pl.col("player_count") == 20)
dup_re = RE20.height - RE20.unique(["match_key", "round"]).height
print("\nround_end rows:", RE.height, "| player_count==20 rows:", RE20.height, "| duplicate (match,round) in 20p:", dup_re)

# ---------- freeze_end scores ----------
RSraw = (pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")])
         .filter(pl.col("event_type") == "round_freeze_end")
         .select(["match_key", "round", "t_score", "ct_score"]).collect())
multi = (RSraw.group_by(["match_key", "round"]).agg(pl.col("t_score").n_unique().alias("nt"),
                                                  pl.col("ct_score").n_unique().alias("nc"))
         .filter((pl.col("nt") > 1) | (pl.col("nc") > 1)))
print("\nfreeze_end rows:", RSraw.height, "| (match,round) keys:", RSraw.unique(["match_key", "round"]).height,
      "| keys with conflicting scores:", multi.height)
RS = RSraw.unique(subset=["match_key", "round"], keep="first")

# ---------- own ledger DP ----------
def ledger_for_match(rows):
    """rows: list of (round, t, c) sorted by round. Returns list of (round, side, swap, status)."""
    out, stat = [], collections.Counter()
    runs, cur = [], [rows[0]]
    for row in rows[1:]:
        if row[0] == cur[-1][0] + 1:
            cur.append(row)
        else:
            runs.append(cur); cur = [row]
    runs.append(cur)
    for run in runs:
        L = len(run)
        if L < 2:
            stat["single_round_run"] += 1
            continue
        trans = []
        for i in range(L - 1):
            _, t0, c0 = run[i]
            _, t1, c1 = run[i + 1]
            m = {}
            for a in (0, 1):
                A0, B0 = (t0, c0) if a == 0 else (c0, t0)
                for b in (0, 1):
                    A1, B1 = (t1, c1) if b == 0 else (c1, t1)
                    dA, dB = A1 - A0, B1 - B0
                    if (dA, dB) == (1, 0):          # team A (T at orientation a=0) won round i
                        m[(a, b)] = (True, 2 if a == 0 else 3)
                    elif (dA, dB) == (0, 1):        # team B won round i; B is T iff a == 1
                        m[(a, b)] = (True, 2 if a == 1 else 3)
                    else:
                        m[(a, b)] = (False, None)
            trans.append(m)
        F = [[0, 0]]
        for i in range(L - 1):
            nf = [INF, INF]
            for a in (0, 1):
                if F[i][a] >= INF:
                    continue
                for b in (0, 1):
                    if trans[i][(a, b)][0]:
                        nf[b] = min(nf[b], F[i][a] + (1 if a != b else 0))
            F.append(nf)
        total = min(F[L - 1])
        if total >= INF:
            stat["infeasible_run"] += 1
            for i in range(L - 1):
                out.append((run[i][0], None, None, "infeasible"))
            continue
        G = [None] * L
        G[L - 1] = [0, 0]
        for i in range(L - 2, -1, -1):
            gi = [INF, INF]
            for a in (0, 1):
                for b in (0, 1):
                    if trans[i][(a, b)][0] and G[i + 1][b] < INF:
                        gi[a] = min(gi[a], (1 if a != b else 0) + G[i + 1][b])
            G[i] = gi
        stat["runs_feasible"] += 1
        for i in range(L - 1):
            sides, swaps = set(), set()
            for a in (0, 1):
                for b in (0, 1):
                    ok, side = trans[i][(a, b)]
                    if not ok or F[i][a] >= INF or G[i + 1][b] >= INF:
                        continue
                    if F[i][a] + (1 if a != b else 0) + G[i + 1][b] == total:
                        sides.add(side); swaps.add(a != b)
            r = run[i][0]
            if len(sides) == 1 and len(swaps) == 1:
                out.append((r, sides.pop(), swaps.pop(), "unique")); stat["pairs_unique"] += 1
            elif len(sides) == 1:
                out.append((r, sides.pop(), None, "unique_side_swap_ambiguous")); stat["pairs_swap_ambiguous"] += 1
            else:
                out.append((r, None, None, "ambiguous")); stat["pairs_ambiguous"] += 1
    return out, stat


by_match = collections.defaultdict(list)
for k, r, t, c in RS.iter_rows():
    by_match[k].append((r, t, c))
ledger_rows, total_stat = [], collections.Counter()
for k in by_match:
    rows = sorted(by_match[k])
    out, st = ledger_for_match(rows)
    total_stat.update(st)
    for r, side, swap, status in out:
        ledger_rows.append((k, r, side, swap, status))
print("\nown ledger stats:", dict(total_stat))
LED = pl.DataFrame(ledger_rows, schema={"match_key": pl.Utf8, "round": pl.Int64, "ledger_side": pl.Int64,
                                        "swap": pl.Boolean, "status": pl.Utf8}, orient="row")
print("own ledger rows by status:", LED.group_by("status").len().sort("status").rows())

# reproduction check against the D7-8 output (reference only, not an input)
ref = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))
cmp = LED.join(ref, on=["match_key", "round"], how="full", suffix="_ref", coalesce=True)
mism = cmp.filter(
    (pl.col("ledger_side") != pl.col("ledger_side_ref")) |
    (pl.col("swap") != pl.col("swap_ref")) |
    (pl.col("status") != pl.col("status_ref")) |
    pl.col("ledger_side").is_null() | pl.col("ledger_side_ref").is_null())
print("reproduction vs D7_ledger_dp: own rows", LED.height, "| ref rows", ref.height,
      "| rows differing (side/swap/status/missing):", mism.height)

# ---------- rounds joined with ledger and winner ----------
X = (RE20.join(LED, on=["match_key", "round"], how="left")
     .join(H.select(["match_key", "build_num", "mdt", "path_day", "ns_timestamp"]), on="match_key", how="left")
     .with_columns(
         pl.when(pl.col("ledger_side").is_null() | pl.col("winner_team_code").is_null()).then(None)
           .otherwise(pl.col("ledger_side") == pl.col("winner_team_code")).alias("w_ok")))

fas_keys = (LED.filter(pl.col("swap") == True)
            .with_columns((pl.col("round") + 1).alias("round"))
            .select(["match_key", "round"]).with_columns(pl.lit(True).alias("fas")))
X = X.join(fas_keys, on=["match_key", "round"], how="left").with_columns(pl.col("fas").fill_null(False))
X = X.with_columns(pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"))

print("\nfas rows (all rounds):", X.filter(pl.col("fas")).height,
      "| fas rows with known w_ok:", X.filter(pl.col("fas") & pl.col("w_ok").is_not_null()).height)
print("fas by round x period x w_ok (count):")
print(X.filter(pl.col("fas") & pl.col("w_ok").is_not_null()).group_by(["round", "period", "w_ok"]).len()
      .sort(["round", "period", "w_ok"]).rows())

F = X.filter(pl.col("fas") & pl.col("round").is_in([9, 13]) & pl.col("w_ok").is_not_null())
print("\nCLAIM SET (fas, rounds 9/13, known w_ok, player_count 20): n =", F.height, "(claimed 3287)")
print("flips (w_ok False) =", F.filter(~pl.col("w_ok")).height, "| agree =", F.filter(pl.col("w_ok")).height)
print("distinct mdt values:", F["mdt"].n_unique(), "| distinct path_day values:", F["path_day"].n_unique(),
      "| distinct build values:", F["build_num"].n_unique())
print("build x w_ok (fas 9/13):", F.group_by(["build_num", "w_ok"]).len().sort(["build_num", "w_ok"]).rows())
print("ns-timestamp rows inside claim set:", F.filter(pl.col("ns_timestamp")).select(["match_key", "round", "mdt", "w_ok", "build_num"]).rows())

F = F.with_columns(
    (pl.col("path_day").cast(pl.Int64)).alias("key_day_num"),
    pl.col("w_ok").cast(pl.Int8).alias("y_agree"),
)


# ================= threshold tests on the claim set =================
def cut_errors(df, key_col, direction, valid_only=True):
    """direction 'agree_early': rows with key < cut predicted agree, >= cut predicted flip.
    direction 'flip_early' : rows with key < cut predicted flip, >= cut predicted agree.
    valid_only=True: cut only between DISTINCT key values (a real threshold on that variable)."""
    d = df.select([key_col, "w_ok"]).sort(key_col)
    keys = d[key_col].to_list(); y = d["w_ok"].to_list(); n = len(y)
    pa = [0] * (n + 1); pf = [0] * (n + 1)
    for i, v in enumerate(y):
        pa[i + 1] = pa[i] + (1 if v else 0)
        pf[i + 1] = pf[i] + (0 if v else 1)
    best = None
    for c in range(n + 1):
        if valid_only and 0 < c < n and keys[c - 1] == keys[c]:
            continue
        if direction == "agree_early":
            e = pf[c] + (pa[n] - pa[c])
        else:
            e = pa[c] + (pf[n] - pf[c])
        if best is None or e < best[0]:
            best = (e, c, keys[c - 1] if c > 0 else None, keys[c] if c < n else None)
    return n, best


print("\n================ THRESHOLD TESTS (claim set, n=%d) ================" % F.height)
Fm = F.with_columns(pl.col("mdt").alias("mdt_key"), pl.col("path_day").cast(pl.Int64).alias("day_key"))
for key_col, label in [("mdt_key", "match_date (mdt)"), ("day_key", "path day")]:
    for direction in ["agree_early", "flip_early"]:
        for valid in [True, False]:
            n, best = cut_errors(Fm, key_col, direction, valid_only=valid)
            tag = "valid cuts only" if valid else "ties split (claim-style)"
            print(f"{label:<18} {direction:<12} {tag:<26} errors={best[0]:<5} cut between {best[2]} and {best[3]}")

# misclassified rows at the best valid mdt cut (agree_early)
n, best = cut_errors(Fm, "mdt_key", "agree_early", True)
cut_lo, cut_hi = best[2], best[3]
bad = Fm.filter(((pl.col("mdt_key") <= cut_lo) & (~pl.col("w_ok"))) | ((pl.col("mdt_key") >= cut_hi) & (pl.col("w_ok"))))
print("\nrows misclassified at best valid mdt cut:")
print(bad.select(["mdt_key", "round", "w_ok", "build_num", "path_day", "ns_timestamp"]).with_columns(
    pl.col("mdt_key").cast(pl.Utf8)).rows())

# build as a categorical marker (no threshold needed)
ba = F.filter((pl.col("build_num") == 9842) & (~pl.col("w_ok"))).height + F.filter((pl.col("build_num") == 9961) & pl.col("w_ok")).height
bb = F.filter((pl.col("build_num") == 9961) & (~pl.col("w_ok"))).height + F.filter((pl.col("build_num") == 9842) & pl.col("w_ok")).height
print("\nbuild_num categorical: errors (9842->agree, 9961->flip) =", ba, "| reverse mapping errors =", bb)

# mdt test with the 3 nanosecond-timestamp matches removed (sensitivity)
Fns = Fm.filter(~pl.col("ns_timestamp"))
n2, best2 = cut_errors(Fns, "mdt_key", "agree_early", True)
print("mdt valid-cut errors excluding the nanosecond-timestamp rows: n=%d errors=%d cut %s .. %s" % (n2, best2[0], best2[2], best2[3]))
n3, best3 = cut_errors(Fns, "day_key", "agree_early", True)
print("path-day valid-cut errors excluding the nanosecond-timestamp rows: n=%d errors=%d" % (n3, best3[0]))

# ================= full-population crosstab (all 4,435 matches) =================
print("\n================ FULL POPULATION: build vs match_date (all 4,435 matches) ================")
cut_all = pl.datetime(2024, 2, 7, 1, 1, 18)
pop = H.with_columns(
    pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7, 0, 24, 22)).then(pl.lit("<00:24:22 (before switch)"))
      .when(pl.col("mdt") < cut_all).then(pl.lit("gap 00:24:22..01:01:18"))
      .otherwise(pl.lit(">=01:01:18 (after switch)")).alias("time_zone"))
print("build x time zone (matches):")
print(pop.group_by(["time_zone", "build_num"]).len().sort(["time_zone", "build_num"]).rows())
print("9961 matches with mdt before 01:01:18 Feb 7:", pop.filter((pl.col("build_num") == 9961) & (pl.col("mdt") < cut_all)).height)
late9842 = pop.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= cut_all))
print("9842 matches with mdt at or after 01:01:18 Feb 7:", late9842.height)
print(late9842.select(["mdt", "match_date", "path_day", "match_key"]).with_columns(pl.col("match_key").str.slice(0, 45)).sort("mdt").head(40).rows())
print("\nmonth of 9961 matches:", pop.filter(pl.col("build_num") == 9961).with_columns(pl.col("mdt").dt.strftime("%Y-%m").alias("m")).group_by("m").len().sort("m").rows())
print("month of 9842 matches:", pop.filter(pl.col("build_num") == 9842).with_columns(pl.col("mdt").dt.strftime("%Y-%m").alias("m")).group_by("m").len().sort("m").rows())
print("\nbuild x path_day (full population):")
print(pop.group_by(["path_day", "build_num"]).len().sort(["path_day", "build_num"]).rows())


# ================= flips by build and round, all rounds (ledger label), outside the claim set =================
print("\n================ LEDGER FLIPS BY BUILD AND ROUND (all rounds with known w_ok, player_count 20) ================")
Wk = X.filter(pl.col("w_ok").is_not_null()).with_columns(pl.col("w_ok").not_().alias("flip"))
fl = (Wk.group_by(["round", "build_num", "fas"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"))
      .sort(["round", "build_num", "fas"]))
print("round, build_num, fas -> (n, flips), rounds with any flip:")
print(fl.filter(pl.col("flips") > 0).rows())
print("flips by build, all rounds, excluding fas:", Wk.filter(~pl.col("fas") & pl.col("flip")).group_by("build_num").len().rows())
print("flips by build, fas rounds other than 9/13:", Wk.filter(pl.col("fas") & ~pl.col("round").is_in([9, 13]) & pl.col("flip")).group_by("build_num").len().rows())
r27 = Wk.filter((pl.col("round") == 27) & pl.col("flip"))
print("round-27 flips: n =", r27.height, "| build x fas:", r27.group_by(["build_num", "fas"]).len().sort(["build_num", "fas"]).rows(),
      "| mdt range:", r27["mdt"].min(), "to", r27["mdt"].max())


# ================= map confound check (header map_name) =================
print("\n================ MAP CONFOUND CHECK ================")
HM = (pl.concat([pl.read_parquet(f, columns=["key", "map_name"]) for f in files(BASE / "header")], how="vertical_relaxed")
      .rename({"key": "match_key"}))
Fmap = F.join(HM, on="match_key", how="left")
print("claim set by map x build (n, flips):")
print(Fmap.group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("flips"))
      .sort(["map_name", "build_num"]).rows())
Wmap = X.filter(pl.col("w_ok").is_not_null() & (pl.col("round") == 13)).join(HM, on="match_key", how="left")
print("round-13 ledger flips by map x build (n, flips):")
print(Wmap.group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), (~pl.col("w_ok")).sum().alias("flips"))
      .sort(["map_name", "build_num"]).rows())


# ================= identity check: first flipped first-after-swap row (9/13) by match_date =================
ff = F.filter(~pl.col("w_ok")).sort("mdt").head(3).select(["match_key", "mdt", "round", "build_num", "path_day"])
print("\nIDENTITY first flipped fas 9/13 rows by mdt:", ff.with_columns(pl.col("match_key").str.slice(0, 45)).rows())
