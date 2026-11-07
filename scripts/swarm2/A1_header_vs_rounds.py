# A1: rebuild starter-team final scores from round_end winners (side swap rule) and compare with header.
# ASSUMPTION: sides swap after round max_rounds//2 and after every 3 OT rounds (r = max+3, max+9, ...),
#             the same swap rule as bakeoff.py swap_expected (scripts/model/bakeoff.py line ~129).
# winner_team_code 2 = T side, 3 = CT side (round_end). Starters_T = team on T side in round 1.
import os, glob, collections
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final", "sharecode", "demo_id", "match_id", "match_date"])
print("sharecode values (first 3):", h.get_column("sharecode").head(3).to_list(), "| demo_id:", h.get_column("demo_id").head(3).to_list())
print("sharecode distinct:", h.get_column("sharecode").n_unique(), "| demo_id distinct:", h.get_column("demo_id").n_unique())
pages = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
re_ = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").select(["match_key", "round", "winner_team_code"]).collect()
rows = collections.defaultdict(list)
for k, r, w in re_.iter_rows():
    rows[k].append((r, w))
def swaps_before(r, mr):
    # number of swaps that happened after a round < r
    s = 0
    half = mr // 2
    if half < r: s += 1          # swap after round `half`
    # OT swaps after rounds mr+3, mr+9, ...
    x = mr + 3
    while x < r:
        s += 1; x += 6
    return s
out = []
for k, mr, tS, cS in h.select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"]).iter_rows():
    t_st = 0; c_st = 0; last_r = 0
    for r, w in sorted(rows.get(k, [])):
        s = swaps_before(r, mr)
        starter_t_on_T = (s % 2 == 0)
        if (w == 2) == starter_t_on_T:
            t_st += 1
        else:
            c_st += 1
        last_r = max(last_r, r)
    n = len(rows.get(k, []))
    thr = mr // 2 + 1
    out.append(dict(key=k, max_rounds=mr, n=n, S=tS + cS, rnd_t=t_st, rnd_ct=c_st, hdr_t=tS, hdr_ct=cS,
                    round_match=(t_st == tS and c_st == cS), max_team_round=max(t_st, c_st), threshold=thr))
d = pl.DataFrame(out)
d.write_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_header_vs_rounds.csv")
print("matches:", d.height, "| round-derived starter totals == header (both teams):", d.filter(pl.col("round_match")).height)
print("by (round_match):", d.group_by("round_match").len().rows())
sh = d.filter(~pl.col("round_match"))
print("mismatch: n rounds == header S:", sh.filter(pl.col("n") == pl.col("S")).height, "| n < S:", sh.filter(pl.col("n") < pl.col("S")).height, "| n > S:", sh.filter(pl.col("n") > pl.col("S")).height)
print("mismatch: round-derived t==hdr_t:", sh.filter(pl.col("rnd_t") == pl.col("hdr_t")).height, " ct==hdr_ct:", sh.filter(pl.col("rnd_ct") == pl.col("hdr_ct")).height)
print("mismatch: rounds-derived sum == n:", sh.filter((pl.col("rnd_t") + pl.col("rnd_ct")) == pl.col("n")).height, "of", sh.height)
print("mismatch: winner reached threshold on the last recorded round (max_team_round >= threshold):", sh.filter(pl.col("max_team_round") >= pl.col("threshold")).height, "of", sh.height)
print("all matches: max_team_round >= threshold:", d.filter(pl.col("max_team_round") >= pl.col("threshold")).height, "of", d.height)
print("mismatch sample:", sh.select(["key", "max_rounds", "n", "S", "rnd_t", "rnd_ct", "hdr_t", "hdr_ct"]).head(12).rows())
