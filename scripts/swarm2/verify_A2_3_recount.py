# verify_A2_3_recount.py
# Independent recount for finding A2-3 (verifier). Reads tomes\2024-02-g only, projected columns,
# one parquet page per channel (memory well under 3 GB).
#
# Part 1: direct count, A2_rst_filter logic: round_end (match, round) pairs with no round_start row
#         with the same (match, round), over ALL 2024 header matches.
# Part 2: channel filter, mirroring cross_period.py / bakeoff.py (keyset membership; zero-row dense
#         channels), giving the kept match set.
# Part 3: sequential replication of cross_period.py _run_2024 round filters in code order:
#         (a) not exactly 5 T and 5 CT in player_info, (b) disconnect/bot takeover in round,
#         (c) no round_start row, (d) no live row, (e) null live score.
#         The "no round_start row" count is the stage-(c) residual, so it is conditional on (a) and (b).
# Part 4: classify the direct-count misses by the stage at which cross_period would have dropped them.
# Output: logs\swarm2\verify_A2_3.log (written by this script; overwrites).

import collections
import glob
import os
import sys

import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2024-02-g", "tome", "csds")
SUFFIX = ".2023-12-28,2024-02-25"  # same suffix cross_period.py uses for the 2024 tome
LOG_PATH = os.path.join(ROOT, "logs", "swarm2", "verify_A2_3.log")
T, CT, START = 2, 3, 5  # same constants as cross_period.py / bakeoff.py
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE = ["player_disconnect", "bot_takeover"]

LOG = open(LOG_PATH, "a", encoding="utf-8")  # append: the log is shared with other verify_A2_3 writers


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def folder(ch):
    return os.path.join(TOME, "header" if ch == "header" else f"basic_{ch}{SUFFIX}")


def pages(ch):
    files = sorted(glob.glob(os.path.join(folder(ch), "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder(ch)}")
    return files


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in pages(ch)], how="vertical_relaxed")


def keyset(ch):
    files = sorted(glob.glob(os.path.join(folder(ch), "keyset_*")))
    if not files:
        raise SystemExit(f"no keyset in {folder(ch)}")
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


say("== verify_A2_3_recount.py (tome 2024-02-g) ==")
say("tome folder:", TOME)
say("page files per channel (should be 1 each):",
    {ch: len(pages(ch)) for ch in DENSE + SPARSE + ["header"]})

# ---------- header and channel filter
hdr = read("header", ["key"])
all_m = set(hdr["key"].to_list())
say(f"header matches: {len(all_m)}")

chan_drop = set()
for ch in DENSE + SPARSE:
    ks = keyset(ch)
    if ch == "bot_takeover":
        df_m = set()
        n_rows = sum(pl.read_parquet(f).height for f in pages(ch))
    else:
        df = read(ch, ["match_key"])
        df_m = set(df["match_key"].unique().to_list())
        n_rows = df.height
    zero = all_m - df_m
    no_ks = all_m - ks
    dense = ch in DENSE
    miss = no_ks | (zero if dense else set())
    chan_drop |= miss
    say(f"  {ch:<20} {'dense' if dense else 'sparse':<6} rows={n_rows:>8} matches_with_rows={len(df_m):>5} "
        f"keyset={len(ks):>5} zero_row={len(zero):>5} not_in_keyset={len(no_ks):>5} counted_missing={len(miss):>5}")
kept = all_m - chan_drop
say(f"channel filter: dropped matches {len(chan_drop)}; kept matches {len(kept)}")

# ---------- Part 1: direct count (A2 logic, all matches)
RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
RST = read("round_start", ["match_key", "round", "tick"])
E_pairs = RE.select(["match_key", "round"]).unique()
S_pairs = RST.select(["match_key", "round"]).unique()
say("")
say("== Part 1: direct count (all 2024 matches, A2_rst_filter logic) ==")
say(f"round_end rows: {RE.height}; unique (match,round): {E_pairs.height}")
say(f"round_start rows: {RST.height}; unique (match,round): {S_pairs.height}")
miss_all = E_pairs.join(S_pairs, on=["match_key", "round"], how="anti")
say(f"round_end (match,round) with NO round_start row with same key: {miss_all.height}")

mx = RE.group_by("match_key").agg(pl.col("round").max().alias("mx"))
miss_all = (miss_all.join(mx, on="match_key", how="left")
            .with_columns((pl.col("round") == pl.col("mx")).alias("is_final"),
                          pl.col("match_key").is_in(sorted(kept)).alias("in_kept_match")))
say("crosstab of direct misses: in_kept_match x is_final")
say(miss_all.group_by(["in_kept_match", "is_final"]).len().sort(["in_kept_match", "is_final"]).rows())
say(f"direct misses in channel-kept matches: {miss_all.filter(pl.col('in_kept_match')).height}")

# ---------- Part 2/3: sequential replication of cross_period.py _run_2024 round filters
say("")
say("== Part 3: sequential replication of cross_period.py round filters (kept matches only) ==")
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
DC = read("player_disconnect", ["match_key", "round", "tick"])
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])

rd = {}  # last-wins, as in cross_period.py
for x in RE.iter_rows(named=True):
    if x["match_key"] in kept:
        rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                 "pc": x["player_count"]}
rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True) if x["match_key"] in kept}
live = {}  # first freeze_end row per key, as in cross_period.py
for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
    m = x["match_key"]
    if m not in kept:
        continue
    k = (m, int(x["round"]))
    if k not in live:
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
players = collections.defaultdict(dict)
for x in PI.iter_rows(named=True):
    if x["match_key"] in kept:
        players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
disc = set()
for x in DC.iter_rows(named=True):
    if x["match_key"] in kept:
        disc.add((x["match_key"], int(x["round"])))

say(f"rounds in kept matches (round_end, last-wins): {len(rd)}  (notes: 80236)")
reasons = collections.Counter()
survivors = []
stage_of = {}  # stage at which cross_period drops a round, for Part 4
for k in sorted(rd):
    pl_ = players.get(k, {})
    n_t = sum(1 for v in pl_.values() if v == T)
    n_ct = sum(1 for v in pl_.values() if v == CT)
    if not (n_t == START and n_ct == START):
        reasons["not exactly 5 T and 5 CT at start"] += 1
        stage_of[k] = "a_not5v5"
        continue
    if k in disc:
        reasons["disconnect or bot takeover event in round"] += 1
        stage_of[k] = "b_disc"
        continue
    if k not in rst_keys:
        reasons["no round_start row"] += 1
        stage_of[k] = "c_no_rst"
        continue
    if k not in live:
        reasons["no round_freeze_end (live start) row"] += 1
        stage_of[k] = "d_no_live"
        continue
    if live[k]["t"] is None or live[k]["ct"] is None:
        reasons["null score at live start"] += 1
        stage_of[k] = "e_null_score"
        continue
    survivors.append(k)
    stage_of[k] = "pass"
say("drop reasons in code order (cross_period.py):")
for name, n in reasons.most_common():
    say(f"   {name}: {n}")
say(f"survivors before duplicate and death stages: {len(survivors)}  (notes-table implied: 60351)")

# ---------- Part 4: where do the direct misses fall in the sequential pipeline
say("")
say("== Part 4: direct misses classified by cross_period stage ==")
miss_rows = miss_all.select(["match_key", "round", "is_final", "in_kept_match"]).iter_rows(named=True)
cls = collections.Counter()
for x in miss_rows:
    k = (x["match_key"], int(x["round"]))
    if not x["in_kept_match"]:
        st = "not_in_kept_match"
    else:
        st = stage_of.get(k, "not_in_rd")
    cls[(st, "final" if x["is_final"] else "nonfinal")] += 1
for key, n in sorted(cls.items()):
    say(f"   stage={key[0]:<20} round={key[1]:<9} count={n}")
say(f"stage c (no_rst) total from direct misses: {sum(v for (s, _), v in cls.items() if s == 'c_no_rst')}")
say(f"stage c count via cross_period-style residual: {reasons['no round_start row']}")

LOG.close()
