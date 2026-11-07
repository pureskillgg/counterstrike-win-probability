"""Independent recount for finding A2-3 (verifier). Prints to stdout only; writes no files.

Claim under test (A2-3): the cross-period 'no round_start row' drop count of 141
(notes/model/sufficiency/cross_period.md:61 and cross_period.json) cannot be reproduced on
tomes/2024-02-g with the same key logic; the direct count is 4,463 (scripts/swarm2/A2_rst_filter.py).

What this script does:
  1. Recomputes the direct (unconditional) count: every round_end (match, round) pair in every
     2024 header match with no round_start row of the same (match, round).
  2. Replays the cross_period.py _run_2024 channel filter and round filters in code order
     (lines 189-265), on the channel-kept matches only, and counts the drop reasons.
  3. Splits the direct count by the first cross_period.py stage that would drop each pair.
  4. Checks alternative explanations: reader choice (glob vs tome metadata), round_start label
     offsets, whether final rounds reach the kept set, and the same unconditional count on 2026.

ASSUMPTIONS (marked ASSUMPTION in the output):
  A1  Reader: every dataframe_* page in the channel folder (same as feature_validation.read).
      Cross-check: pages listed in the channel's tome metadata file.
  A2  Key = (match_key, int(round)); the round_end label defines the round (as in cross_period.py).
  A3  2024 folder suffix .2023-12-28,2024-02-25 (cross_period.SUFFIX_2024). 2026 suffix
      .2026-09-18,2026-10-08 (feature_validation.SUFFIX), used only for the 2026 check.
  A4  Channel filter and round filter order are copied from cross_period.py, not imported.
  A5  Stages after the round_start check (duplicate copies, death side) cannot change the 141
      count, so they are not recomputed here.
"""

import collections
import glob
import json
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
TOME26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF26 = ".2026-09-18,2026-10-08"
T, CT, START = 2, 3, 5
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE = ["player_disconnect", "bot_takeover"]

# stage labels
S_CHAN = "A channel-dropped match (round filters never see it)"
S_5V5 = "B fails 5 T and 5 CT at start (player_info)"
S_DISC = "C disconnect or bot takeover event in round"
S_RST = "D reaches the round_start check and has no row (counted as 'no round_start row')"
S_KEPT = "E survives the round_start check"


def folder(tome, suf, ch):
    return tome / ("header" if ch == "header" else f"basic_{ch}{suf}")


def pages(tome, suf, ch):
    return sorted(glob.glob(str(folder(tome, suf, ch) / "dataframe_*")))


def read(tome, suf, ch, cols):
    files = pages(tome, suf, ch)
    assert files, f"no pages for {ch} in {tome}"
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def keyset(tome, suf, ch):
    files = sorted(glob.glob(str(folder(tome, suf, ch) / "keyset_*")))
    assert files, f"no keyset for {ch}"
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def pairs(df):
    return df.select([pl.col("match_key"), pl.col("round").cast(pl.Int64)]).unique()


def pair_set(df):
    return set(df.iter_rows())


def read_by_metadata(tome, suf, ch, cols):
    """Cross-check reader (A1): take the page list from the tome metadata file."""
    fd = folder(tome, suf, ch)
    meta = json.loads((fd / "tome").read_text(encoding="utf8"))
    parts = [pl.read_parquet(fd / p["dataframe"]["dataframe"], columns=cols) for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


def main():
    print("== verify_A2_3_indep.py: independent recount, tome 2024-02-g ==")
    print(f"polars {pl.__version__}")
    print(f"TOME24 = {TOME24}")
    print(f"ASSUMPTION A1: reader = all dataframe_* pages (cross-check: tome metadata pages)")
    print(f"ASSUMPTION A2: key = (match_key, int(round)); round_end label defines the round")
    print(f"ASSUMPTION A3: 2024 suffix = {SUF24}; 2026 suffix = {SUF26}")
    print("ASSUMPTION A4: channel filter and round filter order copied from cross_period.py")
    print("ASSUMPTION A5: stages after the round_start check are not recomputed (cannot change 141)")

    print("\n== 0. inventory ==")
    for ch in ["header"] + DENSE + SPARSE:
        print(f"  pages {ch:<20} {len(pages(TOME24, SUF24, ch))}")
    hdr = read(TOME24, SUF24, "header", ["key"])
    all_m = set(hdr["key"].to_list())
    print(f"header matches: {len(all_m)}; unique keys: {hdr['key'].n_unique()}")

    RE = read(TOME24, SUF24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RST = read(TOME24, SUF24, "round_start", ["match_key", "round"])
    print(f"dtype round_end.round = {RE.schema['round']}; round_start.round = {RST.schema['round']}")
    re_p = pairs(RE)
    rst_p = pairs(RST)
    print(f"round_end rows = {RE.height}; unique pairs = {re_p.height}")
    print(f"round_start rows = {RST.height}; unique pairs = {rst_p.height}")
    outside = re_p.filter(~pl.col("match_key").is_in(list(all_m))).height
    print(f"round_end pairs with match_key outside the header: {outside}")

    # cross-check reader (A1)
    re_meta = pairs(read_by_metadata(TOME24, SUF24, "round_end", ["match_key", "round"]))
    rst_meta = pairs(read_by_metadata(TOME24, SUF24, "round_start", ["match_key", "round"]))
    print(f"reader cross-check: round_end pair set equal (glob vs metadata): "
          f"{pair_set(re_meta) == pair_set(re_p)}; round_start equal: {pair_set(rst_meta) == pair_set(rst_p)}")

    print("\n== 1. direct (unconditional) count: all header matches ==")
    direct = re_p.join(rst_p, on=["match_key", "round"], how="anti")
    print(f"round_end pairs with no round_start row of the same key: {direct.height}")

    print("\n== 2. channel filter (cross_period.py lines 189-205) ==")
    dropped = set()
    for ch in DENSE + SPARSE:
        ks = keyset(TOME24, SUF24, ch)
        if ch == "bot_takeover":
            n = sum(pl.read_parquet(f).height for f in pages(TOME24, SUF24, ch))
            assert n == 0, "bot_takeover has rows; this replica does not read them"
            have = set()
        else:
            df = read(TOME24, SUF24, ch, ["match_key"])
            n = df.height
            have = set(df["match_key"].unique().to_list())
        dense = ch in DENSE
        zero = all_m - have
        no_ks = all_m - ks
        miss = no_ks | (zero if dense else set())
        dropped |= miss
        print(f"  {ch:<20} {'dense ' if dense else 'sparse'} rows={n:>8} matches_with_rows={len(have):>5} "
              f"keyset={len(ks):>5} zero_row={len(zero):>5} not_in_keyset={len(no_ks):>5} "
              f"counted_missing={len(miss):>5}")
    kept = all_m - dropped
    print(f"kept matches after channel filter: {len(kept)}; dropped: {len(dropped)}")

    print("\n== 3. sequential replica of cross_period.py round filters (kept matches) ==")
    kept_list = list(kept)
    rd_keys = pair_set(re_p.filter(pl.col("match_key").is_in(kept_list)))
    rst_keys = pair_set(rst_p.filter(pl.col("match_key").is_in(kept_list)))
    PI = read(TOME24, SUF24, "player_info", ["match_key", "round", "player_id", "team_code"])
    print(f"dtype player_info.team_code = {PI.schema['team_code']}")
    players = {}
    for m, r, pid, tc in PI.filter(pl.col("match_key").is_in(kept_list)).iter_rows():
        players.setdefault((m, int(r)), {})[pid] = tc
    DC = read(TOME24, SUF24, "player_disconnect", ["match_key", "round"])
    disc = set()
    for m, r in DC.filter(pl.col("match_key").is_in(kept_list)).iter_rows():
        disc.add((m, int(r)))
    RS = read(TOME24, SUF24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    RSf = RS.filter((pl.col("event_type") == "round_freeze_end") & pl.col("match_key").is_in(kept_list))
    live = {}
    for m, r, tk, _ev, ts, cs in RSf.iter_rows():
        live.setdefault((m, int(r)), (tk, ts, cs))

    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd_keys):
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if not (n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start"] += 1
            continue
        if k in disc:
            reasons["disconnect or bot takeover event in round"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no round_freeze_end (live start) row"] += 1
            continue
        if live[k][1] is None or live[k][2] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)
    print(f"round_end pairs in kept matches: {len(rd_keys)}")
    for name in ["not exactly 5 T and 5 CT at start", "disconnect or bot takeover event in round",
                 "no round_start row", "no round_freeze_end (live start) row", "null score at live start"]:
        print(f"  dropped: {name}: {reasons.get(name, 0)}")
    print(f"survivors before duplicate and death stages: {len(survivors)}")
    print("  saved values (cross_period.md:58-62 and :17): 14960 / 4784 / 141 / 0 / 0; "
          "survivors 60351; rounds in kept matches 80236")
    surv_final = 0
    last = {m: mx for m, mx in re_p.group_by("match_key").agg(pl.col("round").max().alias("mx")).iter_rows()}
    for m, r in survivors:
        if last.get(m) == r:
            surv_final += 1
    print(f"surviving rounds that are a match's final round: {surv_final}")

    print("\n== 4. direct count split by the first cross_period.py stage that drops each pair ==")
    re_sets0 = {}
    for m, r in re_p.iter_rows():
        re_sets0.setdefault(m, set()).add(int(r))
    rst_sets0 = {}
    for m, r in rst_p.iter_rows():
        rst_sets0.setdefault(m, set()).add(int(r))

    def label_group(m):
        a = re_sets0.get(m, set())
        b = rst_sets0.get(m, set())
        if not b:
            return "no round_start rows"
        return "shifted (RE min = RST min + 1)" if min(a) == min(b) + 1 else "normal (RE min = RST min)"

    cnt = collections.Counter()
    cnt_g = collections.Counter()
    no_pi_final = 0
    lowcount = collections.Counter()
    for m, r in direct.iter_rows():
        r = int(r)
        k = (m, r)
        is_final = last.get(m) == r
        if m not in kept:
            stage = S_CHAN
        else:
            pl_ = players.get(k, {})
            n_t = sum(1 for v in pl_.values() if v == T)
            n_ct = sum(1 for v in pl_.values() if v == CT)
            if not (n_t == START and n_ct == START):
                stage = S_5V5
                if is_final:
                    if len(pl_) == 0:
                        no_pi_final += 1
                    lowcount[(n_t, n_ct)] += 1
            elif k in disc:
                stage = S_DISC
            else:
                stage = S_RST
        cnt[(stage, is_final)] += 1
        cnt_g[(stage, is_final, label_group(m))] += 1
    for stage in [S_CHAN, S_5V5, S_DISC, S_RST]:
        for fin in (True, False):
            print(f"  {stage:<90} final_round={str(fin):<5} count={cnt.get((stage, fin), 0)}")
    print("  split of the direct misses by label group (verify_A2_3_labels.py patterns):")
    for (stage, fin, g), n in sorted(cnt_g.items()):
        print(f"    {stage[:1]} final_round={str(fin):<5} {g:<32} count={n}")
    total = sum(cnt.values())
    print(f"  total = {total}; direct count = {direct.height}; sums equal: {total == direct.height}")
    print(f"  final-round pairs in kept matches that fail 5v5 and have zero player_info rows: {no_pi_final}")
    print(f"  final-round 5v5 failures by (n_T, n_CT) at start, top 8: {lowcount.most_common(8)}")
    kept_final_direct = sum(v for (s, f), v in cnt.items() if f and s != S_CHAN)
    print(f"  final-round direct misses in kept matches: {kept_final_direct} "
          f"(kept matches: {len(kept)}; each kept match has one final round in round_end: "
          f"{len(kept) == sum(1 for m in kept if m in last)})")

    reproduced_141 = cnt.get((S_RST, True), 0) + cnt.get((S_RST, False), 0)
    print("\n== 5. alternative explanations ==")
    # 5a: label offsets: where do round_start rows sit relative to the match's last round_end label?
    re_sets = {}
    for m, r in re_p.iter_rows():
        re_sets.setdefault(m, set()).add(int(r))
    rst_sets = {}
    for m, r in rst_p.iter_rows():
        rst_sets.setdefault(m, set()).add(int(r))
    miss_off = collections.Counter()
    extra_off = collections.Counter()
    extra_no_re = 0
    for m in set(re_sets) | set(rst_sets):
        a = re_sets.get(m, set())
        b = rst_sets.get(m, set())
        if not a:
            extra_no_re += len(b)
            continue
        mx = max(a)
        for r in a - b:
            miss_off[r - mx] += 1
        for r in b - a:
            extra_off[r - mx] += 1
    print(f"  round_end pairs missing a round_start row, offset from match's last round_end label: "
          f"{miss_off.most_common(6)}; distinct offsets: {len(miss_off)}")
    print(f"  round_start pairs with no round_end row, offset from match's last round_end label: "
          f"{extra_off.most_common(6)}; distinct offsets: {len(extra_off)}; "
          f"round_start pairs in matches with no round_end rows: {extra_no_re}")
    n_match_rst_short = sum(1 for m in re_sets if m in rst_sets and max(rst_sets[m]) < max(re_sets[m]))
    no_rst_matches = [m for m in re_sets if m not in rst_sets]
    print(f"  matches whose round_start max label is below their round_end max: {n_match_rst_short}")
    print(f"  matches with no round_start rows at all: {len(no_rst_matches)}")
    for m in no_rst_matches:
        print(f"    {m}: round_end labels {min(re_sets[m])}..{max(re_sets[m])} ({len(re_sets[m])} rounds), "
              f"in kept matches: {m in kept}")

    # 5b: unconditional count on 2026 (same key logic)
    re26 = pairs(read(TOME26, SUF26, "round_end", ["match_key", "round"]))
    rst26 = pairs(read(TOME26, SUF26, "round_start", ["match_key", "round"]))
    anti26 = re26.join(rst26, on=["match_key", "round"], how="anti")
    print(f"  2026 tome, unconditional: round_end pairs with no round_start row: {anti26.height} "
          f"(round_end pairs {re26.height}, round_start pairs {rst26.height})")
    print("  bake-off log (logs/bakeoff.log) 2026 drop list has no 'no round_start row' line: "
          "the 2026 round_start filter drops 0 rounds")

    print("\n== 6. verdict inputs ==")
    print(f"  saved 141 reproduced by the sequential replica: {reproduced_141 == 141} (replica = {reproduced_141})")
    print(f"  direct 4,463 reproduced: {direct.height == 4463} (direct = {direct.height})")
    print(f"  split of the direct count sums to it: {total == direct.height}")
    print(f"  direct misses that the cross_period filters remove before the round_start check: "
          f"{direct.height - reproduced_141}")
    print(f"  kept set final rounds: {surv_final} (expected 0 if every final round lacks a round_start row)")


if __name__ == "__main__":
    main()
