"""verify_D7_5_40_parity: label-free header parity test and round-count check.

Uses only round_freeze_end scores (team slots) and the DP orientation at the LAST round of each match
(from verify_D7_5_10_main.py), plus header final scores. No winner label (code, within-round, bomb) is used.
For each match with a single contiguous run covering the match's last round N:
  A_before = score of team A (the team on T at round 1) at freeze_end(N), B_before likewise.
  mapping 'direct': tF - A_before in {0,1} and cF - B_before in {0,1} and their sum == 1.
  mapping 'swapped': tF - B_before in {0,1} and cF - A_before in {0,1} and their sum == 1.
  parity test uses o_N as computed; the flipped-orientation control uses 1 - o_N.
Also reports N vs tF+cF (round count vs header totals).
"""
import sys
import pickle
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "rb") as fh:
    ST = pickle.load(fh)
hdr = ST["hdr"]

last = R.sort(["match_key", "round"]).group_by("match_key", maintain_order=True).agg(
    pl.col("round").max().alias("N"),
    pl.col("round").count().alias("n_rows"),
    pl.col("period").first().alias("period"),
    pl.col("build").first().alias("build"),
)
R_last = R.join(last.select(["match_key", "N"]), on="match_key").filter(pl.col("round") == pl.col("N"))
rows = []
for d in R_last.to_dicts():
    mk = d["match_key"]
    h = hdr[mk]
    tF, cF = h["t_starters_score_final"], h["ct_starters_score_final"]
    o = d["orient"]
    if o is None or d["fe_t"] is None:
        rows.append({"match_key": mk, "period": d["period"], "verdict": "no_orient_or_fe", "N": d["N"]})
        continue
    tb, cb = d["fe_t"], d["fe_c"]
    res = {}
    for label, oo in [("o", o), ("flip", 1 - o)]:
        A_b = tb if oo == 0 else cb
        B_b = cb if oo == 0 else tb
        direct = (tF - A_b in (0, 1)) and (cF - B_b in (0, 1)) and ((tF - A_b) + (cF - B_b) == 1)
        swapped = (tF - B_b in (0, 1)) and (cF - A_b in (0, 1)) and ((tF - B_b) + (cF - A_b) == 1)
        res[label] = "direct" if direct else ("swapped" if swapped else "neither")
    rows.append({"match_key": mk, "period": d["period"], "N": d["N"], "tF": tF, "cF": cF, "tb": tb, "cb": cb,
                 "o": o, "verdict_o": res["o"], "verdict_flip": res["flip"],
                 "rounds_minus_totals": d["N"] - (tF + cF)})
P = pl.DataFrame(rows, infer_schema_length=None)
print("matches evaluated:", P.height)
print("verdict with orientation o_N, by period:", P.group_by(["period", "verdict_o"]).len().sort(["period", "verdict_o"]).rows())
print("verdict with flipped orientation (control), by period:", P.group_by(["period", "verdict_flip"]).len().sort(["period", "verdict_flip"]).rows())
print("N - (tF+cF) distribution by period:", P.group_by(["period", "rounds_minus_totals"]).len().sort(["period", "rounds_minus_totals"]).rows())
print("\nfirst 12 'neither' examples (post):")
print(P.filter((pl.col("verdict_o") == "neither") & (pl.col("period") == "post")).select(
    ["match_key", "N", "tF", "cF", "tb", "cb", "o", "rounds_minus_totals"]).head(12))
print("\nfirst 6 'neither' examples (pre):")
print(P.filter((pl.col("verdict_o") == "neither") & (pl.col("period") == "pre")).select(
    ["match_key", "N", "tF", "cF", "tb", "cb", "o", "rounds_minus_totals"]).head(6))
print("\nswapped-mapping matches by period:", P.filter(pl.col("verdict_o") == "swapped").group_by("period").len().rows())
P.write_parquet(str(SCRATCH / "verify_D7_5_parity.parquet"))
