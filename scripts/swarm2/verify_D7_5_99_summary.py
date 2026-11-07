"""verify_D7_5_99_summary: remaining numbers for the D7-5 verdict (player_count==20 unless stated).

(1) which rounds hold the 'other' bomb-side disagreements of the DP ledger and of within-round labels (claim: 23);
(2) final-round coverage: rounds with no DP ledger winner (the last round of each match has no next freeze_end);
(3) post round-13 bomb rows: code vs bomb by day (claim: day 12 51% -> day 17 100%);
(4) header-consistent post subset: within-available matches, code vs DP vs within (lenient rule, rounds 1..N-1).
"""
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
R20 = R.filter(pl.col("player_count") == 20)

print("== (1) bomb rows where DP ledger != bomb, 'other' rounds (player_count==20), by period and round ==")
oth = R20.filter((pl.col("rbin") == "other") & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0) &
                 pl.col("W_dp").is_not_null() & (pl.col("W_dp") != pl.col("W_bomb")))
print("count:", oth.height, "| by period/round:", oth.group_by(["period", "round"]).len().sort(["period", "round"]).rows())
oth_w = R20.filter((pl.col("rbin") == "other") & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0) &
                   pl.col("W_within").is_not_null() & (pl.col("W_within") != pl.col("W_bomb")))
print("within != bomb, other rounds:", oth_w.height, "| by period/round:", oth_w.group_by(["period", "round"]).len().sort(["period", "round"]).rows())

print("\n== (2) final-round coverage: rounds with no DP ledger winner (player_count==20) ==")
print(R20.group_by("period").agg(pl.len().alias("rows"), pl.col("W_dp").is_null().sum().alias("no_dp"),
                                 pl.col("W_within").is_null().sum().alias("no_within"),
                                 pl.col("W_code").is_null().sum().alias("no_code")).sort("period").rows())
last = R20.sort(["match_key", "round"]).group_by("match_key").agg(pl.col("round").max().alias("N"))
Rl = R20.join(last, on="match_key").filter(pl.col("round") == pl.col("N"))
print("last-round rows:", Rl.height, "| last-round rows with no DP winner:", Rl.filter(pl.col("W_dp").is_null()).height,
      "| with no within label:", Rl.filter(pl.col("W_within").is_null()).height)

print("\n== (3) post round-13 bomb rows: code == bomb by day (player_count==20) ==")
b13 = R20.filter((pl.col("period") == "post") & (pl.col("round") == 13) & pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
b13 = b13.with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("day"))
g = b13.group_by("day").agg(pl.len().alias("n"), (pl.col("W_code") == pl.col("W_bomb")).sum().alias("code_eq")).sort("day")
print([(d, n, ce, round(100 * (1 - ce / n), 1)) for d, n, ce in g.select(["day", "n", "code_eq"]).rows()])

print("\n== (4) header-consistent post subset (tF+cF==N, rounds contiguous), within-available matches only ==")
import pickle
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "rb") as fh:
    hdr = pickle.load(fh)["hdr"]
from collections import Counter
res = Counter()
for (mk,), g2 in R.sort(["match_key", "round"]).group_by(["match_key"], maintain_order=True):
    rounds = g2["round"].to_list()
    N = max(rounds)
    if g2["period"][0] != "post" or rounds != list(range(1, N + 1)):
        continue
    h = hdr[mk]
    tF, cF = h["t_starters_score_final"], h["ct_starters_score_final"]
    if tF + cF != N:
        continue
    o = g2["orient"].to_list()
    if any(x is None for x in o):
        continue
    wl = g2["W_within"].to_list()[:-1]
    if any(x is None for x in wl):
        continue
    out = {}
    for src in ["W_code", "W_within", "W_dp"]:
        s = g2[src].to_list()[:-1]
        if any(x is None for x in s):
            out[src] = "missing"
            continue
        A1 = sum(1 for x, oo in zip(s, o[:-1]) if (x == 2 and oo == 0) or (x == 3 and oo == 1))
        B1 = (N - 1) - A1
        out[src] = "pass" if (tF - A1 in (0, 1) or tF - B1 in (0, 1)) else "fail"
    res[tuple(sorted(out.items()))] += 1
print("matches with within available on rounds 1..N-1 (post, header-consistent):", sum(res.values()))
for k, v in res.most_common():
    print(v, k)
