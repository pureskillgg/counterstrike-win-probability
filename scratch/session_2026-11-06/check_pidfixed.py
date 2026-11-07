import csv, collections, os, json
import polars as pl
W = "/home/sagemaker-user/adam/builder/fixes/fix332/"
FIX = "/home/sagemaker-user/adam/builder/fixes/"

# 1. sanity of per-match alive counts (recovered laptop rounds)
pm = pl.read_parquet(W + "per_match_round_flags.parquet")
print("per-match rounds", pm.height, "| alive_n hist", sorted(collections.Counter(pm["alive_n"].to_list()).items()))
print("per-match rounds with kind", pm.filter(pl.col("kind") != "").height, "| roster pids mean", round(pm["n_info_pids"].mean(), 2))

# 2. player_id vs player_id_fixed in candidate rounds, from local per-match raw (fixes/raw)
cand = list(csv.DictReader(open(W + "candidates.csv")))
by_match = collections.defaultdict(list)
for r in cand:
    by_match[r["match_key"]].append(r)
rows = collections.Counter()
examples = []
for mk, rr in by_match.items():
    folder = mk.rsplit("/", 1)[0]
    d = FIX + "raw/" + folder.replace("/", "_")
    pi = d + "/player_info"
    if not os.path.exists(pi):
        rows["no_local_raw"] += 1
        continue
    info = pl.read_parquet(pi, columns=["round", "player_id", "player_id_fixed", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("player_id_fixed").cast(pl.Int64), pl.col("team_code").cast(pl.Int64))
    for r in rr:
        rd = int(r["round"])
        sub = info.filter(pl.col("round") == rd)
        diff = sub.filter(pl.col("player_id") != pl.col("player_id_fixed"))
        rows["candidate_rounds_checked"] += 1
        if diff.height:
            rows["candidate_rounds_with_pid_diff"] += 1
        if len(examples) < 6 and diff.height:
            examples.append({"match": mk.split("/")[4][:8], "round": rd, "kind": r["kind"], "dup": r["dup_ids"], "unro": r["unro_ids"],
                             "diff_rows": diff.select(["player_id", "player_id_fixed", "team_code"]).rows()[:6]})
    # non-candidate rounds in same matches for comparison
    cand_rounds = {int(r["round"]) for r in rr}
    other = info.filter(~pl.col("round").is_in(list(cand_rounds)))
    rows["other_rounds_in_same_matches"] += other["round"].n_unique()
    rows["other_rounds_with_pid_diff"] += other.filter(pl.col("player_id") != pl.col("player_id_fixed"))["round"].n_unique()
print("pid vs pid_fixed:", dict(rows))
print("examples:", json.dumps(examples, default=str)[:2500])
