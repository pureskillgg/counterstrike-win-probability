import csv, os, json, collections
import pyarrow.parquet as pq

E = "E:/csds_final/csds"
W = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
rows = list(csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8")))

players_per_match = collections.Counter()
rounds_per_match = collections.Counter()
codes_per_player_hist = collections.Counter()
zero_rows = []
bad_pairs = 0
sample_matches = []
for r in rows:
    mk = r["match_key"]
    p = mk.split("/")
    folder = os.path.join(E, p[1], p[2], p[3], p[4])
    t = pq.read_table(os.path.join(folder, "player_info"), columns=["round", "player_id", "team_code"])
    rd = t.column("round").to_pylist()
    pid = t.column("player_id").to_pylist()
    tc = t.column("team_code").to_pylist()
    players = set(pid)
    rounds = set(rd)
    players_per_match[len(players)] += 1
    rounds_per_match[len(rounds)] += 1
    per_player = collections.defaultdict(set)
    for a, b, c in zip(rd, pid, tc):
        if c is None:
            continue
        per_player[b].add(c)
        if c == 0:
            zero_rows.append((mk, a, b))
    for b, s in per_player.items():
        codes_per_player_hist[len(s)] += 1
    if len(sample_matches) < 3:
        sample_matches.append(mk)

out = {
    "matches": len(rows),
    "distinct_players_per_match_hist": dict(sorted(players_per_match.items())),
    "rounds_per_match_hist_min_max": [min(rounds_per_match), max(rounds_per_match)] if rounds_per_match else None,
    "distinct_team_codes_per_player_hist": dict(codes_per_player_hist),
    "zero_team_code_rows": len(zero_rows),
    "zero_team_code_examples": zero_rows[:9],
}
with open(W + "skeptic_e_plausibility.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
