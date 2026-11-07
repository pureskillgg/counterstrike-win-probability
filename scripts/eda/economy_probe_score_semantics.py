# Probe: are round_state t_score / ct_score attached to the side or to the team?
# Team identity is tracked across halves with player_id_fixed sets from basic_player_info.
# Compares the freeze-end score fields with (a) team-level prior wins of the team on each side
# and (b) side-level cumulative prior wins. G-side tomes only; no E: reads.
import csv, glob, os
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
keys = sorted({r["match_key"] for r in csv.DictReader(open(os.path.join(ROOT, "logs", "eda", "economy_snapshot_matches.csv"), encoding="utf-8"))})

def files(ch):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % ch)) if os.path.isdir(x)][0]
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

rend = pq.read_table(files("round_end")[0], columns=["match_key", "round", "winner_team_code"],
                     filters=[("match_key", "in", keys)]).to_pandas()
rs = pd.concat([pq.read_table(f, columns=["match_key", "round", "t_score", "ct_score"],
                              filters=[("event_type", "==", "round_freeze_end"), ("match_key", "in", keys)]).to_pandas()
                for f in files("round_state")], ignore_index=True)
pi = pq.read_table(files("player_info")[0], columns=["match_key", "round", "player_id_fixed", "team_code"],
                   filters=[("match_key", "in", keys)]).to_pandas()
pi = pi[pi["team_code"].isin([2, 3]) & pi["player_id_fixed"].notna()]

counts = Counter()
rounds_total = 0
skipped = Counter()
for mk in keys:
    w = rend[rend["match_key"] == mk].set_index("round")["winner_team_code"].to_dict()
    f = rs[rs["match_key"] == mk].drop_duplicates("round").set_index("round")
    p = pi[pi["match_key"] == mk]
    rounds = sorted(set(w) & set(f.index))
    team_wins = Counter()      # keyed 'A'/'B'
    side_wins = Counter()      # keyed 'CT'/'T'
    A = None
    for r in rounds:
        pr = p[p["round"] == r]
        T_set = set(pr.loc[pr["team_code"] == 2, "player_id_fixed"].astype("int64"))
        C_set = set(pr.loc[pr["team_code"] == 3, "player_id_fixed"].astype("int64"))
        if A is None:
            if not T_set or not C_set:
                skipped["no_players_round1"] += 1
                continue
            A = frozenset(T_set)  # team A = the side that was T in the first round seen
            B = frozenset(C_set)
        ov_A_T = len(T_set & A); ov_A_C = len(C_set & A)
        if ov_A_C >= ov_A_T:
            ct_team, t_team = "A", "B"
        else:
            ct_team, t_team = "B", "A"
        s_ct_team = team_wins[ct_team]; s_t_team = team_wins[t_team]
        s_ct_side = side_wins["CT"]; s_t_side = side_wins["T"]
        ct_score, t_score = int(f.loc[r, "ct_score"]), int(f.loc[r, "t_score"])
        rounds_total += 1
        counts["ct_field==team_on_CT_wins"] += int(ct_score == s_ct_team)
        counts["t_field==team_on_T_wins"] += int(t_score == s_t_team)
        counts["ct_field==CT_side_cum_wins"] += int(ct_score == s_ct_side)
        counts["t_field==T_side_cum_wins"] += int(t_score == s_t_side)
        counts["ct_field==team_on_T_wins (flipped)"] += int(ct_score == s_t_team)
        counts["both_fields_match_team_attached"] += int(ct_score == s_ct_team and t_score == s_t_team)
        counts["both_fields_match_side_cumulative"] += int(ct_score == s_ct_side and t_score == s_t_side)
        # update after the round, using the recorded winner
        win = w[r]
        if win == 3:
            team_wins[ct_team] += 1; side_wins["CT"] += 1
        elif win == 2:
            team_wins[t_team] += 1; side_wins["T"] += 1
        else:
            skipped["winner_not_2_or_3"] += 1

print("matches", len(keys), "rounds compared", rounds_total, "skipped", dict(skipped))
for k in sorted(counts):
    print("%-45s %6d  (%.1f%%)" % (k, counts[k], 100.0 * counts[k] / max(rounds_total, 1)))
