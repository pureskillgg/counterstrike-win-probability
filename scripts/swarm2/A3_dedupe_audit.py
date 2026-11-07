"""A3 dedupe audit for scripts/model/bakeoff.py (read-only).

Recomputes, from the 2026-10-01..08 tome, the bake-off match filter, the event-signature
duplicate rule, the round filters and both splits. Checks each against the counts in
logs/bakeoff.log and logs/bakeoff_run1_no_dedupe.log. Then probes what the rule can miss:
  (a) groups where the kept (lowest-key) copy has no surviving round, so the content is lost;
  (b) copies that differ in some channel;
  (c) near-duplicates (partial overlap) that the exact-signature rule does not group;
  (d) straddling of duplicates across the train/val/test boundary, for run1 and the final run.

Nothing is written except this log (stdout). Constants and split code are copied from
bakeoff.py and feature_validation.py (T=2, CT=3, START=5, SEED=0, fractions as in bakeoff).
ASSUMPTION: the replicated pipeline matches bakeoff.py; every replicated count is checked
against the logged value before the probes run.
"""
import collections
import datetime as dt
import glob
import hashlib
import random
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
T, CT, START, TICK_RATE = 2, 3, 5, 64
SEED = 0
TEST_FRAC, VAL_FRAC, CHRONO_TRAIN_FRAC = 0.20, 0.10, 0.80
PRIMARY, CHRONO = "primary", "chronological"

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  CHECK {label}: got {got}, logged {want} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        FAILS.append((label, got, want))


def files(channel):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no pages in {folder}")
    return fs


def read(channel, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(channel)],
                     how="vertical_relaxed")


def keyset_keys(ch):
    folder = TOME / f"basic_{ch}{SUFFIX}"
    fs = sorted(glob.glob(str(folder / "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in fs])["_"].to_list())


def parse_date(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def make_splits(matches, H):
    """Copied from bakeoff.py make_splits (same shuffles, same rounding)."""
    keys = sorted(matches)
    shuffled = list(keys)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(TEST_FRAC * len(keys))
    test, pool = shuffled[:n_test], shuffled[n_test:]
    n_val = round(VAL_FRAC * len(pool))
    val, train = pool[:n_val], pool[n_val:]
    prim = {"train": set(train), "val": set(val), "test": set(test)}
    ordered = sorted(keys, key=lambda k: (parse_date(H[k]["match_date"]), k))
    n_pool = round(CHRONO_TRAIN_FRAC * len(ordered))
    pool_c, test_c = ordered[:n_pool], ordered[n_pool:]
    n_val_c = round(VAL_FRAC * len(pool_c))
    train_c, val_c = pool_c[:len(pool_c) - n_val_c], pool_c[len(pool_c) - n_val_c:]
    chrono = {"train": set(train_c), "val": set(val_c), "test": set(test_c)}
    return {PRIMARY: prim, CHRONO: chrono}


def side_of(sp, m):
    for part in ("train", "val", "test"):
        if m in sp[part]:
            return part
    return None


def md5_of(rows):
    return hashlib.md5("\n".join(sorted(repr(r) for r in rows)).encode("utf-8")).hexdigest()[:12]


def main():
    print(f"polars {pl.__version__}; tome {TOME}")
    # ---------- header
    hdr = read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                          "t_starters_score_final", "ct_starters_score_final",
                          "demo_id", "sharecode"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    print(f"header matches: {len(all_matches)}")
    print(f"  distinct demo_id: {hdr['demo_id'].n_unique()}; distinct sharecode: "
          f"{hdr['sharecode'].n_unique()}; sample demo_id: {hdr['demo_id'].head(2).to_list()}; "
          f"sample sharecode: {hdr['sharecode'].head(2).to_list()}")
    uuid_match = sum(1 for k, h in H.items() if k.split("/")[4] == h["match_id"])
    print(f"  key uuid equals header match_id: {uuid_match} of {len(H)}")
    check("header matches", len(all_matches), 1752)

    # ---------- channels (only the columns the bake-off uses)
    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RST = read("round_start", ["match_key", "round", "tick"])
    RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                               "weapon_name", "attacker_id"])
    PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
    BS = read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = read("player_disconnect", ["match_key", "round", "tick"])
    BT = read("bot_takeover", ["match_key", "round", "tick"])
    TC = read("team_change", ["match_key", "round", "tick"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}
    DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info",
             "bomb_state", "team_change"]
    SPARSE = ["player_disconnect", "bot_takeover"]

    # ---------- 1. match filter (same rule as bakeoff section 1)
    print("\n== 1. Match filter ==")
    missing_by_match = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        df = frames[ch]
        ks = keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        miss = no_keyset | (zero_rows if ch in DENSE else set())
        for m in miss:
            missing_by_match[m].append(ch)
    dropped_matches = set(missing_by_match)
    kept_matches = all_matches - dropped_matches
    print(f"dropped for a missing channel: {len(dropped_matches)}; kept: {len(kept_matches)}")
    check("matches dropped for missing channel", len(dropped_matches), 43)
    check("kept after channel filter", len(kept_matches), 1709)

    # ---------- 2. round filters (bakeoff section 2 order); survivors per round
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
        live.setdefault((m, int(x["round"])), {"tick": int(x["tick"]), "t": x["t_score"],
                                               "ct": x["ct_score"]})
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            players[(m, int(x["round"]))][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc_rounds.add((x["match_key"], int(x["round"])))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                 "team": x["player_team_code"]})
    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            reasons["player_count"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["5v5"] += 1
            continue
        if k in disc_rounds:
            reasons["disc"] += 1
            continue
        if k not in rst_keys:
            reasons["no_round_start"] += 1
            continue
        if k not in live:
            reasons["no_freeze_end"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null_score"] += 1
            continue
        # unresolved victim side inside the live window (bakeoff build_states returns None)
        frz, end = live[k]["tick"], rd[k]["end"]
        unresolved = False
        for d in deaths.get(k, []):
            if frz <= d["tick"] <= end:
                side = d["team"] if d["team"] is not None else players.get(k, {}).get(d["pid"])
                if side not in (T, CT):
                    unresolved = True
                    break
        if unresolved:
            reasons["unresolved_victim"] += 1
            continue
        survivors.append(k)
    surv_by_match = collections.Counter(k[0] for k in survivors)
    print(f"rounds in kept matches: {len(rd)}; round filter reasons: {dict(reasons)}")
    print(f"rounds surviving the filters (before dedupe): {len(survivors)} in "
          f"{len(surv_by_match)} matches")
    check("rounds in kept matches", len(rd), 35502)
    check("rounds surviving before dedupe", len(survivors), 29596)
    check("matches with a surviving round (pre-dedupe)", len(surv_by_match), 1568)

    # ---------- 3. event signature and dedupe groups (bakeoff rule)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(("D", int(x["round"]), int(x["tick"]), str(x["player_id"])))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"])))
    sigs = {m: tuple(sorted(repr(p) for p in v)) for m, v in sig_parts.items()}
    sig_rows = {m: len(v) for m, v in sig_parts.items()}
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    n_copies = sum(len(g) - 1 for g in dedup_groups)
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    print(f"\n== 3. Signature groups among channel-kept matches ==")
    print(f"groups: {len(dedup_groups)}; matches in groups: {sum(len(g) for g in dedup_groups)}; "
          f"redundant copies (members minus one): {n_copies}; dropped copies: {len(dup_drop)}")
    size_dist = collections.Counter(len(g) for g in dedup_groups)
    print(f"group size distribution: {dict(sorted(size_dist.items()))}")
    check("sig groups among kept", len(dedup_groups), 46)
    check("matches in sig groups", sum(len(g) for g in dedup_groups), 97)
    dropped_with_rounds = {m for m in dup_drop if surv_by_match.get(m, 0) > 0}
    print(f"dropped copies that had surviving rounds: {len(dropped_with_rounds)}; "
          f"rounds they held: {sum(surv_by_match[m] for m in dropped_with_rounds)}")
    check("dropped copies with rounds", len(dropped_with_rounds), 47)
    check("rounds removed by dedupe", sum(surv_by_match[m] for m in dropped_with_rounds), 913)

    # ---------- 3a. does the lowest key carry the content?
    print("\n== 3a. Lowest-key rule vs surviving content ==")
    lost_groups = []
    for g in dedup_groups:
        kept_m, dropped = g[0], g[1:]
        if surv_by_match.get(kept_m, 0) == 0 and any(surv_by_match.get(m, 0) > 0 for m in dropped):
            lost_groups.append(g)
    lost_rounds = sum(surv_by_match.get(m, 0) for g in lost_groups for m in g[1:])
    print(f"groups where the kept (lowest key) copy has 0 surviving rounds but a dropped copy has "
          f"rounds: {len(lost_groups)}; rounds removed in those groups: {lost_rounds}")
    for g in lost_groups:
        print("   group:", [(m.split('/')[4][:8] + '..', surv_by_match.get(m, 0), H[m]["match_date"][:19])
                            for m in g])
    groups_with_any_rounds = [g for g in dedup_groups if any(surv_by_match.get(m, 0) > 0 for m in g)]
    print(f"groups with rounds in any member: {len(groups_with_any_rounds)}; groups where the kept copy "
          f"holds the rounds: {sum(1 for g in groups_with_any_rounds if surv_by_match.get(g[0], 0) > 0)}")
    # rounds per member within groups where members differ
    n_diff_surv = 0
    for g in dedup_groups:
        vals = {surv_by_match.get(m, 0) for m in g}
        if len(vals) > 1:
            n_diff_surv += 1
    print(f"groups whose members keep different numbers of rounds after the filters: {n_diff_surv}")

    # ---------- 3b. what differs among copies
    print("\n== 3b. Channel-level agreement among copies (md5 of sorted rows per match) ==")
    chan_specs = {
        "round_end": (RE, ["round", "tick", "winner_team_code", "player_count"]),
        "round_start": (RST, ["round", "tick"]),
        "round_state": (RS, ["round", "tick", "event_type", "t_score", "ct_score"]),
        "player_death": (DE, ["round", "tick", "player_id", "player_team_code", "weapon_name", "attacker_id"]),
        "player_info": (PI, ["round", "player_id", "team_code"]),
        "bomb_state": (BS, ["round", "tick", "event_type"]),
        "team_change": (TC, ["round", "tick"]),
        "player_disconnect": (DC, ["round", "tick"]),
        "bot_takeover": (BT, ["round", "tick"]),
    }
    chan_hash = {}
    for name, (df, cols) in chan_specs.items():
        per = collections.defaultdict(list)
        for row in df.select(["match_key"] + cols).iter_rows():
            per[row[0]].append(tuple(row[1:]))
        chan_hash[name] = {m: md5_of(v) for m, v in per.items()}
    for name in chan_specs:
        differ = 0
        for g in dedup_groups:
            hs = {chan_hash[name].get(m, "none") for m in g}
            if len(hs) > 1:
                differ += 1
        print(f"  {name:<18} groups where copies differ: {differ} of {len(dedup_groups)}")
    hdr_diff = collections.Counter()
    date_gaps = []
    for g in dedup_groups:
        mds = {H[m]["max_rounds"] for m in g}
        fs = {(H[m]["t_starters_score_final"], H[m]["ct_starters_score_final"]) for m in g}
        if len(mds) > 1:
            hdr_diff["max_rounds differs"] += 1
        if len(fs) > 1:
            hdr_diff["final score differs"] += 1
        ds = [parse_date(H[m]["match_date"]) for m in g]
        date_gaps.append((max(ds) - min(ds)).total_seconds())
    print(f"  header: {dict(hdr_diff) if hdr_diff else 'max_rounds and final score agree in all groups'}")
    date_gaps.sort()
    print(f"  match_date spread within groups (s): min {date_gaps[0]:.0f}, median "
          f"{date_gaps[len(date_gaps)//2]:.0f}, max {date_gaps[-1]:.0f}; groups with spread 0: "
          f"{sum(1 for x in date_gaps if x == 0)}")
    small = sorted(((sig_rows.get(g[0], 0), len(g), g[0]) for g in dedup_groups))[:5]
    print(f"  smallest signatures in groups (rows, members): {[(a, b) for a, b, _ in small]}")
    big_small = [x for x in small if x[0] < 50]
    print(f"  groups with fewer than 50 event rows in the signature: {len(big_small)}")

    # ---------- 4. fingerprint (round_end only) check, all matches with >= 5 rounds
    print("\n== 4. Round-end fingerprint groups (all matches, >=5 rounds) ==")
    per_m = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        per_m[x["match_key"]].append((int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
    fps = collections.defaultdict(list)
    for m, lst in per_m.items():
        if len(lst) >= 5:
            fps[tuple(sorted(lst))].append(m)
    fp_groups = [v for v in fps.values() if len(v) > 1]
    print(f"fingerprint groups: {len(fp_groups)}; matches: {sum(len(v) for v in fp_groups)}")
    check("fingerprint groups", len(fp_groups), 47)
    check("fingerprint matches", sum(len(v) for v in fp_groups), 99)
    n_fp_not_sig = 0
    for v in fp_groups:
        in_kept = [m for m in v if m in kept_matches]
        excluded = [m for m in v if m not in kept_matches]
        sig_ids = {sigs.get(m, ()) for m in v}
        if len(sig_ids) == 1 and not excluded:
            continue
        n_fp_not_sig += 1
        print(f"  fp group {len(v)} members; kept {len(in_kept)}; excluded by channel filter "
              f"{len(excluded)}; distinct signatures {len(sig_ids)}")
        for m in v:
            print(f"     {m.split('/')[4][:8]}.. kept={m in kept_matches} "
                  f"sig_rows={sig_rows.get(m, 0)} surviving_rounds={surv_by_match.get(m, 0)} "
                  f"missing={missing_by_match.get(m, [])} date={H[m]['match_date'][:19]} "
                  f"sig_hash={hashlib.md5(repr(sigs.get(m, ())).encode()).hexdigest()[:8]}")
    print(f"fingerprint groups that are not one identical signature among kept matches: {n_fp_not_sig}")

    # ---------- 5. partial-overlap (near-duplicate) search across all matches
    print("\n== 5. Near-duplicate search (shared exact events) ==")
    re_df = RE.select(["match_key", pl.format("{}|{}|{}", "round", "tick", "winner_team_code").alias("k")]).unique()
    de_df = DE.select(["match_key", pl.format("{}|{}|{}", "round", "tick", "player_id").alias("k")]).unique()
    re_pairs = (re_df.join(re_df, on="k", suffix="_b")
                .filter(pl.col("match_key") < pl.col("match_key_b"))
                .group_by(["match_key", "match_key_b"]).agg(pl.len().alias("shared_re")))
    de_pairs = (de_df.join(de_df, on="k", suffix="_b")
                .filter(pl.col("match_key") < pl.col("match_key_b"))
                .group_by(["match_key", "match_key_b"]).agg(pl.len().alias("shared_de")))
    pairs = re_pairs.join(de_pairs, on=["match_key", "match_key_b"], how="full", coalesce=True)
    pairs = pairs.with_columns(pl.col("shared_re").fill_null(0), pl.col("shared_de").fill_null(0))
    print(f"pairs sharing at least one round_end or death event: {pairs.height}")
    n_re = {m: len(v) for m, v in per_m.items()}
    n_de = collections.Counter(x["match_key"] for x in DE.iter_rows(named=True))
    cand = pairs.filter((pl.col("shared_re") >= 3) | (pl.col("shared_de") >= 5)).to_dicts()
    print(f"pairs with shared_re>=3 or shared_de>=5: {len(cand)}")
    sig_pair = set()
    for g in dedup_groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                sig_pair.add((g[i], g[j]))
    near = []
    for r in cand:
        a, b = r["match_key"], r["match_key_b"]
        same_sig = (a, b) in sig_pair
        if not same_sig:
            near.append(r)
    print(f"  of which are exact-signature pairs: {len(cand) - len(near)}; not grouped by the rule: {len(near)}")
    near_sorted = sorted(near, key=lambda r: -(r["shared_de"] + r["shared_re"]))
    for r in near_sorted[:25]:
        a, b = r["match_key"], r["match_key_b"]
        print(f"   shared_re={r['shared_re']:>3} shared_de={r['shared_de']:>3} "
              f"rounds=({n_re.get(a, 0)},{n_re.get(b, 0)}) deaths=({n_de.get(a, 0)},{n_de.get(b, 0)}) "
              f"kept=({a in kept_matches},{b in kept_matches}) "
              f"surv=({surv_by_match.get(a, 0)},{surv_by_match.get(b, 0)}) "
              f"dates=({H[a]['match_date'][:10]},{H[b]['match_date'][:10]}) "
              f"a={a.split('/')[4][:8]} b={b.split('/')[4][:8]}")

    # ---------- 5b. same-game check that does not use exact ticks
    print("\n== 5b. Same-game check without exact ticks (max_rounds, header final scores, winner sequence) ==")
    per_full = collections.defaultdict(list)
    for m, lst in per_m.items():
        per_full[m] = sorted(lst)
    fz_key = {}
    for m in all_matches:
        h = H[m]
        fz_key[m] = (h["max_rounds"], h["t_starters_score_final"], h["ct_starters_score_final"],
                     tuple(w for (_, _, w) in per_full.get(m, [])))
    fz_groups = collections.defaultdict(list)
    for m in all_matches:
        fz_groups[fz_key[m]].append(m)
    fz_multi = [g for g in fz_groups.values() if len(g) > 1]
    print(f"groups with equal max_rounds, final scores and winner sequence: {len(fz_multi)}; "
          f"members {sum(len(g) for g in fz_multi)}")
    sig_not_in_fz = sum(1 for g in dedup_groups if len({fz_key[m] for m in g}) > 1)
    print(f"  signature groups (kept) whose members do not share the same fuzzy key: {sig_not_in_fz}")
    fz_not_sig = [g for g in fz_multi if len({sigs.get(m, ()) for m in g}) > 1]
    print(f"  fuzzy groups that are not one identical signature: {len(fz_not_sig)}")
    for g in fz_not_sig:
        diffs = []
        for m in g[1:]:
            a, b = per_full[g[0]], per_full[m]
            diffs.append(sorted(abs(x[1] - y[1]) for x, y in zip(a, b)))
        med = [d[len(d) // 2] for d in diffs]
        mx = [d[-1] for d in diffs]
        print(f"   group {len(g)} members {[m.split('/')[4][:8] for m in g]} kept "
              f"{[m in kept_matches for m in g]} rounds {[len(per_full[m]) for m in g]} "
              f"deaths {[n_de.get(m, 0) for m in g]} median tick diff vs first {med} max {mx} "
              f"sig_rows {[sig_rows.get(m, 0) for m in g]}")
    n_fz_multi_kept = sum(1 for g in fz_multi if sum(1 for m in g if m in kept_matches) > 1)
    print(f"  fuzzy groups with more than one kept member: {n_fz_multi_kept}")

    # ---------- 5c. tick-tolerant pairwise round-end match (all matches, numpy)
    print("\n== 5c. Tick-tolerant pair check: same round, same winner, round_end within 64 ticks ==")
    import numpy as np
    mlist = sorted(all_matches)
    midx = {m: i for i, m in enumerate(mlist)}
    max_r = int(RE["round"].max())
    n_m = len(mlist)
    cnt = np.zeros((n_m, n_m), dtype=np.int16)
    exact = np.zeros((n_m, n_m), dtype=np.int16)
    for r in range(1, max_r + 1):
        tk = np.full(n_m, np.nan)
        wn = np.full(n_m, -1, dtype=np.int64)
        sub = RE.filter(pl.col("round") == r)
        for x in sub.iter_rows(named=True):
            i = midx[x["match_key"]]
            tk[i] = x["tick"]
            wn[i] = x["winner_team_code"]
        have = ~np.isnan(tk)
        if have.sum() < 2:
            continue
        d = np.abs(tk[:, None] - tk[None, :])
        same_w = (wn[:, None] == wn[None, :]) & have[:, None] & have[None, :]
        cnt += ((d <= 64) & same_w).astype(np.int16)
        exact += ((d == 0) & same_w).astype(np.int16)
    iu = np.triu_indices(n_m, k=1)
    c_up = cnt[iu]
    hist = collections.Counter(int(v) for v in c_up if v >= 3)
    print(f"pairs with >=3 rounds matched within 64 ticks: {int((c_up >= 3).sum())}")
    print(f"  histogram of matched rounds (>=3): {dict(sorted(hist.items()))}")
    tol_pairs = [(mlist[i], mlist[j], int(cnt[i, j]), int(exact[i, j]))
                 for i, j in zip(*np.where(np.triu(cnt >= 3, k=1)))]
    not_sig_tol = []
    for a, b, c, e in tol_pairs:
        if sigs.get(a, ()) != sigs.get(b, ()):
            not_sig_tol.append((a, b, c, e))
    print(f"  pairs with >=3 tolerant rounds that do not share one signature: {len(not_sig_tol)}; "
          f"max tolerant rounds among them: {max((c for _, _, c, _ in not_sig_tol), default=0)}")
    n_sig_pairs_tol = sum(1 for a, b, c, e in tol_pairs if sigs.get(a, ()) == sigs.get(b, ()))
    print(f"  pairs with >=3 tolerant rounds that share one signature: {n_sig_pairs_tol}; "
          f"min tolerant rounds among them: {min((c for a, b, c, e in tol_pairs if sigs.get(a, ()) == sigs.get(b, ())), default=0)}")
    big_missed = [t for t in not_sig_tol if t[2] >= 8]
    print(f"  NON-GROUPED pairs with >=8 tolerant rounds (real near-duplicate signal): {len(big_missed)}")
    check("non-grouped pairs with >=8 tolerant rounds (all matches)", len(big_missed), 0)
    kept_tol_missed = [(a, b) for a, b, c, e in not_sig_tol if a in kept_matches and b in kept_matches]
    print(f"  kept-kept pairs with 3-4 tolerant rounds and no shared signature (coincidence level): "
          f"{len(kept_tol_missed)}")

    # constant-offset copies: same winners and header totals, round_end ticks differ by one fixed amount
    print("\n== 5d. Constant-offset copies (same winners and header totals) ==")
    n_const = 0
    for g in fz_multi:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if sigs.get(a, ()) == sigs.get(b, ()):
                    continue
                pa, pb = per_full.get(a, []), per_full.get(b, [])
                if len(pa) != len(pb) or not pa:
                    continue
                dd = [pb_[1] - pa_[1] for pa_, pb_ in zip(pa, pb)]
                if max(dd) - min(dd) <= 128:
                    n_const += 1
                    print(f"   constant offset {dd[0]} ticks across {len(dd)} rounds: "
                          f"{a.split('/')[4][:8]} {b.split('/')[4][:8]}")
    print(f"constant-offset non-identical pairs: {n_const}")
    check("constant-offset non-identical pairs", n_const, 0)

    # ---------- 6. splits and straddling
    print("\n== 6. Splits and straddling ==")
    pre_final = set(surv_by_match)
    sp_pre = make_splits(pre_final, H)
    sp_final = make_splits({k[0] for k in survivors if k[0] not in dup_drop}, H)
    for name in (PRIMARY, CHRONO):
        print(f"  run1-equivalent {name}: " + ", ".join(
            f"{p} {len(sp_pre[name][p])}" for p in ("train", "val", "test")))
    check("run1 primary train", len(sp_pre[PRIMARY]["train"]), 1129)
    check("run1 primary test", len(sp_pre[PRIMARY]["test"]), 314)
    for name in (PRIMARY, CHRONO):
        print(f"  final {name}: " + ", ".join(
            f"{p} {len(sp_final[name][p])}" for p in ("train", "val", "test")))
    check("final primary test", len(sp_final[PRIMARY]["test"]), 304)
    check("final primary train", len(sp_final[PRIMARY]["train"]), 1095)
    final_set = {k[0] for k in survivors if k[0] not in dup_drop}
    # signature groups straddling sides, counting only members each split covers
    for label, sp, covered in (("run1", sp_pre, pre_final), ("final", sp_final, final_set)):
        for name in (PRIMARY, CHRONO):
            n_str = 0
            for g in dedup_groups:
                mem = [m for m in g if m in covered]
                if len(mem) < 2:
                    continue
                parts = {side_of(sp[name], m) for m in mem}
                if len(parts) > 1:
                    n_str += 1
            print(f"  {label} split {name}: signature groups straddling sides: {n_str}")
            if label == "run1" and name == PRIMARY:
                check("run1 primary straddling groups", n_str, 17)
            if label == "run1" and name == CHRONO:
                check("run1 chronological straddling groups", n_str, 0)
    # non-grouped near pairs straddling sides, only pairs with both members covered
    for label, sp, covered in (("run1", sp_pre, pre_final), ("final", sp_final, final_set)):
        for name in (PRIMARY, CHRONO):
            n_str = 0
            for r in near:
                a, b = r["match_key"], r["match_key_b"]
                if not (a in covered and b in covered):
                    continue
                sa, sb = side_of(sp[name], a), side_of(sp[name], b)
                if sa != sb:
                    n_str += 1
            print(f"  {label} split {name}: non-grouped near pairs straddling sides: {n_str}")
    # exposure: primary test matches with an exact or near twin in train or val
    twin = collections.defaultdict(set)
    for g in dedup_groups:
        for m in g:
            twin[m].update(x for x in g if x != m)
    for r in near:
        twin[r["match_key"]].add(r["match_key_b"])
        twin[r["match_key_b"]].add(r["match_key"])
    for label, sp, covered in (("run1", sp_pre, pre_final), ("final", sp_final, final_set)):
        tr_set = sp[PRIMARY]["train"] | sp[PRIMARY]["val"]
        test_ms = sp[PRIMARY]["test"]
        exposed = [m for m in test_ms if any(t in tr_set for t in twin.get(m, ()))]
        test_rounds = sum(surv_by_match.get(m, 0) for m in test_ms)
        exp_rounds = sum(surv_by_match.get(m, 0) for m in exposed)
        print(f"  {label} primary test matches with a twin in train or val: {len(exposed)} of {len(test_ms)}; "
              f"rounds {exp_rounds} of {test_rounds}")
        if label == "run1":
            check("run1 exposed test matches", len(exposed), 12)
            for m in exposed:
                tw = sorted(twin[m] & tr_set) if (twin[m] & tr_set) else []
                print(f"     exposed test {m.split('/')[4][:8]} rounds={surv_by_match.get(m, 0)} "
                      f"twins in train/val: {[t.split('/')[4][:8] + ('(train)' if t in sp[PRIMARY]['train'] else '(val)') for t in tw]}")
    print("\nFAILED CHECKS: " + str(len(FAILS)))
    for f in FAILS:
        print("  ", f)


if __name__ == "__main__":
    main()
