# verify_A1_1_early_end.py
# Compare the two non-clinch regulation flagged matches with agreeing early-ended matches (n < max_rounds, no team at
# threshold). Same raw test as verify_A1_1_tails.py: tail after last round_end, and whether a round n+1 start falls
# inside the tick stream (it would, at ~448 ticks after round_end, if the match had continued).
import os, csv, collections
import polars as pl
E = r"E:\csds_final\csds\2026\10"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    for c in ["mr", "thr", "S", "hdr_t", "hdr_c", "n", "rnd_t", "rnd_c", "first_reach"]:
        r[c] = int(r[c])
early = [r for r in rows if r["cat"] == "A_agree" and r["first_reach"] == -1 and r["n"] < r["mr"]]
flag_nc = [r for r in rows if r["cat"] == "C_sum_short" and r["first_reach"] == -1]
print("== verify_A1_1_early_end ==")
print("agreeing early-ended (n<mr, no team at threshold):", len(early), "| flagged no-clinch regulation:", len(flag_nc))

def probe(r):
    dd, mid = r["key"].split("/")[3], r["key"].split("/")[4]
    base = os.path.join(E, dd, mid)
    rend = pl.read_parquet(os.path.join(base, "round_end"), columns=["round", "tick"])
    rs = pl.read_parquet(os.path.join(base, "round_start"), columns=["round", "tick"])
    tk = pl.read_parquet(os.path.join(base, "tick"), columns=["round", "tick"])
    last_end = int(rend.get_column("tick").max())
    last_round = int(rend.get_column("round").max())
    tail = int(tk.get_column("tick").max()) - last_end
    max_tick_round = int(tk.get_column("round").max())
    # expected next-round start: last round_end + median mid-game gap 448 ticks
    expected_next = last_end + 448
    stream_covers = int(tk.get_column("tick").max()) >= expected_next
    return tail, max_tick_round, last_round, stream_covers

for label, grp in [("agreeing early-ended", early), ("flagged no-clinch regulation", flag_nc)]:
    tails = collections.Counter(); covers = 0; more = 0
    for r in grp:
        tail, mtr, lr, cov = probe(r)
        tails[tail] += 1
        covers += int(cov)
        more += int(mtr > r["n"])
        print("  ", label, r["key"][-40:], "mr", r["mr"], "n", r["n"], "S", r["S"], "tail", tail, "tick_max_round", mtr,
              "stream covers expected next-round start", cov)
    print(label, "tails:", tails.most_common(), "| stream covers expected next start:", covers, "of", len(grp),
          "| tick stream beyond n:", more)
