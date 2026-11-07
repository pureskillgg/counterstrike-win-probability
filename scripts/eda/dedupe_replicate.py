"""Task 3: replicate the duplicate-match counts of the CURRENT scripts/model/bakeoff.py (read-only).

Re-derives, from the tome, the same quantities bakeoff.py prints:
  - channel-kept matches (missing-channel filter, DENSE + SPARSE channels)
  - signature dedupe over channel-kept matches (round_end, player_death, player_info events)
  - duplicate groups, matches in groups, matches dropped (keep lowest key), dropped copies with kept rounds
  - groups straddling the primary / chronological split (before removal)
  - round-fingerprint check (>=5 rounds, all matches), the diagnostic used in the report
Also compares the signature groups over all matches (no channel filter) and the fingerprint groups.
make_splits and the channel lists come from bakeoff.py (imported by path, no side effects).

Run: pixi run python -I scripts\\eda\\dedupe_replicate.py
"""
import collections
import glob
import importlib.util
import sys
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
T, CT, START = 2, 3, 5


def load_bakeoff():
    spec = importlib.util.spec_from_file_location("bakeoff_mod", ROOT / "scripts" / "model" / "bakeoff.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def keyset(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def main():
    bo = load_bakeoff()
    DENSE, SPARSE = bo.DENSE, bo.SPARSE
    H = {}
    hdr = read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                          "t_starters_score_final", "ct_starters_score_final"])
    for r in hdr.iter_rows(named=True):
        H[r["key"]] = r
    all_matches = set(H)
    print(f"header matches: {len(all_matches)}")

    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    RST = read("round_start", ["match_key", "round", "tick"])
    RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                               "weapon_name", "attacker_id"])
    BS = read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = read("player_disconnect", ["match_key", "round", "tick"])
    BT = read("bot_takeover", ["match_key", "round", "tick"])
    TC = read("team_change", ["match_key", "round", "tick"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}

    # 1. channel-kept matches (same rule as bakeoff.py section 1)
    missing_by_match = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        df = frames[ch]
        ks = keyset(ch)
        have = set(df["match_key"].unique().to_list())
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        dense = ch in DENSE
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
    kept_matches = all_matches - set(missing_by_match)
    print(f"channel-kept matches: {len(kept_matches)} (dropped {len(missing_by_match)})")

    # 2. signatures (same repr scheme as bakeoff.py)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]),
                                               int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]),
                                               str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]),
                                               str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}

    # 3. round filters (same rules as bakeoff.py section 2), to get survivors before dedupe
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept_matches}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        players[(m, int(x["round"]))][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc_rounds.add((x["match_key"], int(x["round"])))
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            continue
        if k in disc_rounds:
            continue
        if k not in rst_keys:
            continue
        if k not in live:
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            continue
        survivors.append(k)
    pre_final = {k[0] for k in survivors}
    print(f"matches with >=1 surviving round (before dedupe): {len(pre_final)}")

    # 4. dedupe exactly as bakeoff.py does it
    pre_sets = bo.make_splits(pre_final, H)
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    members_total = sum(len(g) for g in dedup_groups)
    dup_matches_with_rounds = len({k[0] for k in survivors if k[0] in dup_drop})
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    print("\n== Current dedupe step (channel-kept matches, signature = round_end + player_death + player_info) ==")
    print(f"duplicate groups: {len(dedup_groups)}")
    print(f"matches in groups: {members_total}")
    print(f"matches dropped by dedupe (group size minus 1, lowest key kept): {len(dup_drop)}")
    print(f"dropped matches that had surviving rounds: {dup_matches_with_rounds}; "
          f"rounds removed: {dup_rounds}")
    print(f"group sizes: {dict(collections.Counter(len(g) for g in dedup_groups))}")
    print(f"dropped matches with no surviving round: {len(dup_drop) - dup_matches_with_rounds}")
    print("kept key rule: first key in sorted(kept_matches) within each group (lowest match_key string).")

    # 5. straddle before removal
    straddle = {"primary": 0, "chronological": 0}
    for g in dedup_groups:
        members = [m for m in g if m in pre_final]
        if len(members) < 2:
            continue
        for name in ("primary", "chronological"):
            parts = {part for m in members for part in ("train", "val", "test") if m in pre_sets[name][part]}
            if len(parts) > 1:
                straddle[name] += 1
    print(f"groups with members on more than one side (before removal): primary {straddle['primary']}, "
          f"chronological {straddle['chronological']}")
    # 6. signature groups over ALL matches (no channel filter)
    sig_all = collections.defaultdict(list)
    for m in sorted(all_matches):
        sig_all[sigs.get(m, ())].append(m)
    all_groups = [g for g in sig_all.values() if len(g) > 1]
    print(f"\n== Signature groups over ALL {len(all_matches)} matches (no channel filter) ==")
    print(f"groups: {len(all_groups)}; matches in groups: {sum(len(g) for g in all_groups)}; "
          f"matches above group minimum: {sum(len(g) - 1 for g in all_groups)}")
    for g in all_groups:
        if any(m not in kept_matches for m in g):
            print(f"  group with channel-dropped members: "
                  f"{[(m.split('/')[-1][:8], m in kept_matches, sorted(missing_by_match.get(m, []))) for m in g]}")

    # 7. round-fingerprint diagnostic (all matches, >=5 rounds): what bakeoff.py prints as "47 groups"
    per_match_rounds = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        per_match_rounds[x["match_key"]].append((int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
    fps = collections.defaultdict(list)
    for m, lst in per_match_rounds.items():
        if len(lst) >= 5:
            fps[tuple(sorted(lst))].append(m)
    fp_groups = [v for v in fps.values() if len(v) > 1]
    print(f"\n== Round-fingerprint diagnostic (>=5 rounds, all matches) ==")
    print(f"groups: {len(fp_groups)}; matches in groups: {sum(len(v) for v in fp_groups)}; "
          f"matches above group minimum: {sum(len(v) - 1 for v in fp_groups)}")
    # reconcile: fingerprint groups vs signature groups over all matches
    sig_member_sets = {frozenset(g) for g in all_groups}
    fp_member_sets = {frozenset(v) for v in fp_groups}
    only_fp = [v for v in fp_groups if frozenset(v) not in sig_member_sets]
    only_sig = [g for g in all_groups if frozenset(g) not in fp_member_sets]
    print(f"fingerprint groups that are not a signature group: {len(only_fp)}; "
          f"signature groups that are not a fingerprint group: {len(only_sig)}")
    for v in only_fp[:5]:
        print(f"  fp-only group: {[m.split('/')[-1][:8] + ' in_kept=' + str(m in kept_matches) for m in v]}")
    for g in only_sig[:5]:
        print(f"  sig-only group: {[m.split('/')[-1][:8] + ' in_kept=' + str(m in kept_matches) for m in g]}")


if __name__ == "__main__":
    main()
