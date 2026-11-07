# verify_A1_1_tails.py
# Population test of the tick-stream tail: for each match, tail = max tick in the raw 'tick' channel minus the tick of the
# last round_end; max_tick_round = highest round number seen in the tick stream. Also the mid-game round gap
# (round_start(r+1) - round_end(r)) from agreeing matches, to see whether a next-round start would fall inside the tail.
# Raw E: read-only, extensionless parquet per channel. Groups: B+C (90), A overtime (156), A regulation sample (150).
import os, csv, collections, random, statistics
import polars as pl
E = r"E:\csds_final\csds\2026\10"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    r["n"] = int(r["n"]); r["mr"] = int(r["mr"]); r["S"] = int(r["S"]); r["first_reach"] = int(r["first_reach"])
    r["ot"] = r["n"] > r["mr"]
flag = [r for r in rows if r["cat"] in ("B_sum_eq_split_diff", "C_sum_short")]
A_ot = [r for r in rows if r["cat"] == "A_agree" and r["ot"]]
A_reg = [r for r in rows if r["cat"] == "A_agree" and not r["ot"]]
rng = random.Random(7)
A_reg_s = rng.sample(A_reg, 150)
print("== verify_A1_1_tails ==")
print("groups: flagged", len(flag), "| A overtime", len(A_ot), "| A regulation sample", len(A_reg_s))

def tail_info(r):
    dd, mid = r["key"].split("/")[3], r["key"].split("/")[4]
    base = os.path.join(E, dd, mid)
    rend = pl.read_parquet(os.path.join(base, "round_end"), columns=["round", "tick"])
    last_end = int(rend.get_column("tick").max())
    tk = pl.read_parquet(os.path.join(base, "tick"), columns=["round", "tick"])
    max_tick = int(tk.get_column("tick").max())
    max_round_tick = int(tk.get_column("round").max())
    rs = pl.read_parquet(os.path.join(base, "round_start"), columns=["round", "tick"]).sort("round")
    gaps = []
    rend_s = rend.sort("round")
    re_map = {int(a): int(b) for a, b in rend_s.select(["round", "tick"]).iter_rows()}
    rs_map = {int(a): int(b) for a, b in rs.select(["round", "tick"]).iter_rows()}
    for rr in re_map:
        if rr + 1 in rs_map:
            gaps.append(rs_map[rr + 1] - re_map[rr])
    return dict(tail=max_tick - last_end, max_round_tick=max_round_tick, n=r["n"], gaps=gaps,
                round_start_rounds=max(rs_map) if rs_map else None)

def summarize(name, group):
    tails = collections.Counter()
    more_rounds = 0
    gaps_all = []
    rs_gt_n = 0
    for r in group:
        try:
            info = tail_info(r)
        except Exception as ex:
            tails[f"err:{type(ex).__name__}"] += 1
            continue
        tails[info["tail"]] += 1
        if info["max_round_tick"] > r["n"]:
            more_rounds += 1
        if info["round_start_rounds"] is not None and info["round_start_rounds"] > r["n"]:
            rs_gt_n += 1
        gaps_all += info["gaps"]
        r["_info"] = info
    vals = [k for k in tails if isinstance(k, int)]
    print(f"\n-- {name}: n={len(group)}")
    print("   tail (max tick - last round_end tick) top values:", tails.most_common(6))
    if vals:
        print("   tail median", statistics.median(vals), "| min", min(vals), "| max", max(vals))
    print("   tick stream reaches a round number above n (more rounds in ticks than round_end):", more_rounds)
    print("   round_start rounds above n:", rs_gt_n)
    if gaps_all:
        print("   mid-game gap round_end(r)->round_start(r+1) ticks: median", statistics.median(gaps_all), "| min", min(gaps_all), "| max", max(gaps_all), "| count", len(gaps_all))
    return group

summarize("flagged B+C", flag)
summarize("A overtime (agreeing)", A_ot)
summarize("A regulation sample (agreeing)", A_reg_s)
print("\nflagged regulation vs OT tails:")
for r in flag:
    info = r.get("_info")
    if info is None:
        continue
    if r["n"] > r["mr"] or info["tail"] != 1279:
        print("   ", r["cat"], r["day"], "n", r["n"], "mr", r["mr"], "S", r["S"], "tail", info["tail"], "tick_max_round", info["max_round_tick"], "first_reach", r["first_reach"])
