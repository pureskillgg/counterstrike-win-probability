# A1: final engine score sum per match from the score_update channel (last new_score per team_code), vs round count and header sum.
# Engine pair: the last row's new_score (team L), and the most recent new_score of the other team_code.
import os, glob, collections
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
pages = sorted(glob.glob(os.path.join(T, "basic_score_update.2026-09-18,2026-10-08", "dataframe_*")))
su = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").select(["match_key", "round", "tick", "team_code", "new_score"]).collect()
print("score_update rows", su.height, "| team_code values:", su.get_column("team_code").unique().sort().to_list())
su = su.sort(["match_key", "round", "tick"])
last = su.group_by("match_key", maintain_order=True).last()
last_rows = dict((r[0], (r[3], r[4], r[1])) for r in last.select(["match_key", "round", "tick", "team_code", "new_score"]).iter_rows())
# most recent value of each team_code
latest = {}
for mk, tc, ns in su.select(["match_key", "team_code", "new_score"]).iter_rows():
    latest[(mk, tc)] = ns
# count of score_update rows per match
cnt = su.group_by("match_key").len()
cnt = dict(cnt.iter_rows())
rounds_re = pl.concat([pl.scan_parquet(p) for p in sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))], how="diagonal_relaxed").select("match_key").collect()
n_by = dict(rounds_re.group_by("match_key").len().iter_rows())
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key", "t_starters_score_final", "ct_starters_score_final"])
rows = []
for k, tS, cS in h.iter_rows():
    if k not in last_rows:
        rows.append((k, None, None, None, tS + cS, n_by.get(k, 0), cnt.get(k, 0), None)); continue
    tc_last, ns_last, _ = last_rows[k]
    other = 3 if tc_last == 2 else 2
    ns_other = latest.get((k, other))
    eng = (ns_last or 0) + (ns_other or 0)
    rows.append((k, ns_last, ns_other, tc_last, tS + cS, n_by.get(k, 0), cnt.get(k, 0), eng))
d = pl.DataFrame(rows, schema=["key", "last_ns", "other_ns", "last_team", "S", "n", "su_rows", "eng_sum"], orient="row")
d = d.with_columns((pl.col("n") == pl.col("S")).alias("n_eq_S"), (pl.col("eng_sum") == pl.col("n")).alias("eng_eq_n"), (pl.col("eng_sum") == pl.col("S")).alias("eng_eq_S"))
d = d.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"))
print("matches", d.height)
print("n == S:", d.filter(pl.col("n_eq_S")).height, "| eng_sum == n:", d.filter(pl.col("eng_eq_n")).height, "| eng_sum == S:", d.filter(pl.col("eng_eq_S")).height)
print("among n<S (80): eng_sum == n:", d.filter((pl.col("n") < pl.col("S")) & pl.col("eng_eq_n")).height, "| eng_sum == S:", d.filter((pl.col("n") < pl.col("S")) & pl.col("eng_eq_S")).height)
print("among n==S (1672): eng_sum == n:", d.filter(pl.col("n_eq_S") & pl.col("eng_eq_n")).height, "of", d.filter(pl.col("n_eq_S")).height)
print("by day, n!=S count / eng!=n count:", d.with_columns((~pl.col("n_eq_S")).alias("nbad"), (~pl.col("eng_eq_n")).alias("ebad")).group_by("day").agg(pl.col("nbad").sum(), pl.col("ebad").sum(), pl.len()).sort("day").rows())
d.write_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_engine_score_check.csv")
print("engine-vs-round disagreements:", d.filter(~pl.col("eng_eq_n")).select(["key", "last_ns", "other_ns", "S", "n", "eng_sum"]).head(12).rows())
