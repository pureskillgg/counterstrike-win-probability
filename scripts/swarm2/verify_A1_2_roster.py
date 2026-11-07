# verify A1-2: roster-based starter totals with NO swap-rule assumption.
# Starter-T group A = players whose player_info team_code is 2 (T) in the first round; B = team_code 3 in that round.
# Each round: side of A and B from observed player_info team_code of their members in that round.
# A_won(r) = round winner (winner_team_code, verified against win_reason_message) equals the side of group A in round r.
# Compare: roster totals vs header t/ct, vs swap-model totals (ASSUMPTION: swap after max_rounds//2, bakeoff.py line ~129).
# Also: observed halftime side flip round for A, and last-kill attacker side vs winner for elimination rounds.
import glob, os, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
OUT_CSV = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A1_2_roster.csv"


def pages(ch):
    return sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))


def load(ch, cols):
    cols = list(dict.fromkeys(cols + ["match_key"]))
    return pl.concat([pl.scan_parquet(p).select(cols) for p in pages(ch)], how="diagonal_relaxed").collect()


h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final", "match_date"])
re_ = load("round_end", ["round", "tick", "winner_team_code", "win_reason_code"]).rename({"match_key": "key"})
pi = load("player_info", ["round", "player_id", "team_code"]).rename({"match_key": "key"})
pd_ = load("player_death", ["round", "tick", "player_id", "player_team_code", "attacker_id", "attacker_team_code"]).rename({"match_key": "key"})
print("loaded: header", h.height, "| round_end", re_.height, "| player_info", pi.height, "| player_death", pd_.height)
print("player_info duplicate (key,round,player) rows:", pi.height - pi.unique(["key", "round", "player_id"]).height)
per_match_np = pi.group_by("key").agg(pl.col("player_id").n_unique().alias("np")).get_column("np")
print("distinct players per match in player_info (np: matches):", sorted(collections.Counter(per_match_np.to_list()).items()))

pi_idx = collections.defaultdict(lambda: collections.defaultdict(dict))  # key -> round -> pid -> team
for k, r, p, tc in pi.select(["key", "round", "player_id", "team_code"]).iter_rows():
    pi_idx[k][r][p] = tc
pd_idx = collections.defaultdict(list)
for k, r, tk, vt, att, at in pd_.select(["key", "round", "tick", "player_team_code", "attacker_id", "attacker_team_code"]).iter_rows():
    pd_idx[(k, r)].append((tk, vt, at))
re_idx = collections.defaultdict(dict)
for k, r, tk, w, rc in re_.select(["key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows():
    re_idx[k][r] = (tk, w, rc)


def swap_expected(r, max_rounds):
    # ASSUMPTION copied from scripts/model/bakeoff.py swap_expected (line ~129)
    return r == max_rounds // 2 or (r > max_rounds and (r - max_rounds - 3) % 6 == 0)


rows = []
for k, mr, tS, cS, md in h.select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final", "match_date"]).iter_rows():
    rounds = re_idx.get(k, {})
    n = len(rounds)
    prounds = pi_idx.get(k, {})
    if not rounds or not prounds:
        rows.append(dict(key=k, max_rounds=mr, n=n, S=tS + cS, hdr_t=tS, hdr_ct=cS, A_roster=None, B_roster=None,
                         A_swap=None, B_swap=None, unresolved=n, A_flip_round=None, A_players_r1=0,
                         last_kill_ok=None, last_kill_n=0, match_date=md, note="no_data"))
        continue
    r1 = min(prounds.keys())
    first = prounds[r1]
    A = {p for p, tc in first.items() if tc == 2}
    B = {p for p, tc in first.items() if tc == 3}
    A_roster = B_roster = unresolved = 0
    side_A_by_round = {}
    for r in sorted(rounds):
        tk, w, rc = rounds[r]
        pr = prounds.get(r, {})
        sa = [pr[p] for p in A if p in pr]
        sb = [pr[p] for p in B if p in pr]
        if not sa or not sb:
            unresolved += 1
            continue
        side_A = collections.Counter(sa).most_common(1)[0][0]
        side_B = collections.Counter(sb).most_common(1)[0][0]
        side_A_by_round[r] = side_A
        A_roster += int(side_A == w)
        B_roster += int(side_B == w)
    # swap model (ASSUMPTION)
    A_swap = B_swap = 0
    s = 0
    for r in sorted(rounds):
        tk, w, rc = rounds[r]
        starter_t_on_T = (s % 2 == 0)
        if (w == 2) == starter_t_on_T:
            A_swap += 1
        else:
            B_swap += 1
        if swap_expected(r, mr):
            s += 1
    flip = None
    rs = sorted(side_A_by_round)
    for a, b in zip(rs, rs[1:]):
        if side_A_by_round[a] != side_A_by_round[b]:
            flip = (a, b)
            break
    # last-kill check on elimination rounds (reason 8 or 9): attacker side of last death at or before round end == winner
    last_ok = 0
    last_n = 0
    for r in sorted(rounds):
        tk, w, rc = rounds[r]
        if rc not in (8, 9):
            continue
        ev = sorted((e for e in pd_idx.get((k, r), []) if e[0] is not None and tk is not None and e[0] <= tk),
                    key=lambda e: e[0])
        if not ev:
            continue
        last = ev[-1]
        last_n += 1
        if last[2] == w:
            last_ok += 1
    rows.append(dict(key=k, max_rounds=mr, n=n, S=tS + cS, hdr_t=tS, hdr_ct=cS, A_roster=A_roster, B_roster=B_roster,
                     A_swap=A_swap, B_swap=B_swap, unresolved=unresolved,
                     A_flip_round=(f"{flip[0]}->{flip[1]}" if flip else None),
                     A_players_r1=len(A), last_kill_ok=last_ok, last_kill_n=last_n, match_date=md, note=""))

d = pl.DataFrame(rows, infer_schema_length=None)
d = d.with_columns(
    pl.when((pl.col("A_roster") == pl.col("hdr_t")) & (pl.col("B_roster") == pl.col("hdr_ct"))).then(pl.lit("hdr_matches_roster"))
    .when((pl.col("A_roster") == pl.col("hdr_ct")) & (pl.col("B_roster") == pl.col("hdr_t"))).then(pl.lit("hdr_labels_swapped_vs_roster"))
    .otherwise(pl.lit("other")).alias("cat"))
d = d.with_columns(
    (pl.col("A_roster") == pl.col("A_swap")).alias("roster_eq_swapmodel_A"),
    (pl.col("n") == pl.col("S")).alias("n_eq_S"))
d.write_csv(OUT_CSV)
print("matches", d.height)
print("category x n_eq_S:", d.group_by(["cat", "n_eq_S"]).len().sort(["cat", "n_eq_S"]).rows())
print("roster A == swap-model A (all matches):", d.filter(pl.col("roster_eq_swapmodel_A")).height, "of", d.height)
print("category x roster==swapmodel:", d.group_by(["cat", "roster_eq_swapmodel_A"]).len().sort(["cat", "roster_eq_swapmodel_A"]).rows())
sw = d.filter(pl.col("cat") == "hdr_labels_swapped_vs_roster")
print("SWAPPED category keys:", sw.get_column("key").to_list())
print("SWAPPED day counts:", sw.with_columns(pl.col("key").str.split("/").list.get(3).alias("day")).group_by("day").len().sort("day").rows())
print("SWAPPED max_rounds:", sw.group_by("max_rounds").len().rows(), "| n==S:", sw.filter(pl.col("n_eq_S")).height,
      "| max unresolved:", sw.get_column("unresolved").max())
print("SWAPPED key,A_roster,B_roster,hdr_t,hdr_ct,n,S,flip,last_kill_ok/n:",
      sw.select(["key", "A_roster", "B_roster", "hdr_t", "hdr_ct", "n", "S", "A_flip_round", "last_kill_ok", "last_kill_n"]).rows())
print("SWAPPED observed A halftime flip rounds:", sw.group_by("A_flip_round").len().rows())
print("ALL cat counts:", d.group_by("cat").len().rows())
print("ALL A_flip_round by max_rounds:", d.group_by(["max_rounds", "A_flip_round"]).len().sort(["max_rounds", "A_flip_round"]).rows())
print("last-kill attacker side == winner (elimination rounds) SWAPPED (ok, n):", sw.select(pl.col("last_kill_ok").sum(), pl.col("last_kill_n").sum()).rows())
print("last-kill attacker side == winner (elimination rounds) ALL (ok, n):", d.select(pl.col("last_kill_ok").sum(), pl.col("last_kill_n").sum()).rows())
print("A_players_r1 distribution:", sorted(collections.Counter(d.get_column("A_players_r1").to_list()).items()))
print("no_data matches:", d.filter(pl.col("note") == "no_data").height)
