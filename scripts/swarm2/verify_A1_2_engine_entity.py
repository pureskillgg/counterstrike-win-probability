# verify A1-2: is score_update.new_score a per-TEAM (entity) counter or a per-SIDE counter?
# Entity A = starter-T group (players on T in the first round, player_info), entity B = starter-CT group.
# Per round r: winner side W_r (round_end, verified against win_reason_message), entity E_r = A if A's side in round r == W_r else B.
# Per-entity prediction for the score_update row of round r: (wins of E_r before r) + 1.
# Per-side prediction for that row: (wins of side W_r before r) + 1.
# Discriminator: at the first post-halftime rounds, A1 (A wins in first half) and B1 differ in most matches.
# Also: m_930de9b0 unresolved round, from player_death sides and team_change.
import glob, os, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
OUT = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A1_2_engine_entity.csv"


def pages(ch):
    return sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))


def load(ch, cols):
    cols = list(dict.fromkeys(cols + ["match_key"]))
    return pl.concat([pl.scan_parquet(p).select(cols) for p in pages(ch)], how="diagonal_relaxed").collect()


h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
re_ = load("round_end", ["round", "tick", "winner_team_code"]).rename({"match_key": "key"})
su = load("score_update", ["round", "tick", "team_code", "old_score", "new_score"]).rename({"match_key": "key"})
pi = load("player_info", ["round", "player_id", "team_code"]).rename({"match_key": "key"})
pi_idx = collections.defaultdict(lambda: collections.defaultdict(dict))
for k, r, p, tc in pi.select(["key", "round", "player_id", "team_code"]).iter_rows():
    pi_idx[k][r][p] = tc
re_idx = collections.defaultdict(dict)
for k, r, tk, w in re_.select(["key", "round", "tick", "winner_team_code"]).iter_rows():
    re_idx[k][r] = (tk, w)
su_idx = collections.defaultdict(dict)  # key -> round -> (team_code, new_score, old_score)
for k, r, tc, old, new in su.select(["key", "round", "team_code", "old_score", "new_score"]).iter_rows():
    su_idx[k][r] = (tc, new, old)

rows = []
agg = collections.Counter()
for k, mr, tS, cS in h.select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"]).iter_rows():
    rounds = re_idx.get(k, {})
    prounds = pi_idx.get(k, {})
    srows = su_idx.get(k, {})
    if not rounds or not prounds:
        continue
    r1 = min(prounds.keys())
    first = prounds[r1]
    A = {p for p, tc in first.items() if tc == 2}
    B = {p for p, tc in first.items() if tc == 3}
    ent_wins = collections.Counter()
    side_wins = collections.Counter()
    pe_ok = ps_ok = cmp_n = 0
    halftime_rows = 0
    halftime_pe_ok = halftime_ps_ok = 0
    sideA_last = None
    A_cnt_end = B_cnt_end = None
    for r in sorted(rounds):
        tk, W = rounds[r]
        pr = prounds.get(r, {})
        sa = [pr[p] for p in A if p in pr]
        if not sa:
            ent = None
        else:
            sideA = collections.Counter(sa).most_common(1)[0][0]
            sideA_last = sideA
            ent = "A" if sideA == W else "B"
        # score row for this round
        if r in srows:
            tc, new, old = srows[r]
            cmp_n += 1
            if tc != W:
                agg["score_team_code_ne_winner"] += 1
            if ent is not None:
                pe_pred = ent_wins[ent] + 1
                if new == pe_pred:
                    pe_ok += 1
                if old == ent_wins[ent]:
                    agg["per_entity_old_match"] += 1
            ps_pred = side_wins[W] + 1
            if new == ps_pred:
                ps_ok += 1
            if r > mr // 2 and r <= mr // 2 + 3 and ent is not None:
                halftime_rows += 1
                halftime_pe_ok += int(new == ent_wins[ent] + 1)
                halftime_ps_ok += int(new == side_wins[W] + 1)
        # update counters after the round
        side_wins[W] += 1
        if ent is not None:
            ent_wins[ent] += 1
    rows.append(dict(key=k, max_rounds=mr, n_rounds=len(rounds), score_rows=cmp_n, per_entity_ok=pe_ok, per_side_ok=ps_ok,
                     ht_rows=halftime_rows, ht_per_entity_ok=halftime_pe_ok, ht_per_side_ok=halftime_ps_ok,
                     A_total_roster=ent_wins["A"], B_total_roster=ent_wins["B"], hdr_t=tS, hdr_ct=cS))

d = pl.DataFrame(rows)
d.write_csv(OUT)
print("matches with score rows and roster:", d.height)
print("score rows total:", d.get_column("score_rows").sum(), "| per_entity_ok:", d.get_column("per_entity_ok").sum(), "| per_side_ok:", d.get_column("per_side_ok").sum())
print("halftime-window rows total:", d.get_column("ht_rows").sum(), "| ht per_entity_ok:", d.get_column("ht_per_entity_ok").sum(), "| ht per_side_ok:", d.get_column("ht_per_side_ok").sum())
print("per-match: all rows per_entity_ok:", d.filter(pl.col("per_entity_ok") == pl.col("score_rows")).height, "of", d.height)
print("per-match: all rows per_side_ok:", d.filter(pl.col("per_side_ok") == pl.col("score_rows")).height, "of", d.height)
print("counter flags:", dict(agg))
print("entity totals equal header t/ct (A==hdr_t and B==hdr_ct):", d.filter((pl.col("A_total_roster") == pl.col("hdr_t")) & (pl.col("B_total_roster") == pl.col("hdr_ct"))).height,
      "| A==hdr_ct and B==hdr_t:", d.filter((pl.col("A_total_roster") == pl.col("hdr_ct")) & (pl.col("B_total_roster") == pl.col("hdr_t"))).height)
sw = d.filter((pl.col("A_total_roster") == pl.col("hdr_ct")) & (pl.col("B_total_roster") == pl.col("hdr_t")))
print("swapped-vs-header (per entity) ht info:", sw.select(["key", "ht_rows", "ht_per_entity_ok", "ht_per_side_ok", "per_entity_ok", "per_side_ok", "score_rows"]).rows())
