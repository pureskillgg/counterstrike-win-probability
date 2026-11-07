# verify_A2_1 lead test: replicate score_lead_fix.derive_leads (bake-off swap rule, recorded winners) and compare
# with round_state ct_score - t_score at round_freeze_end, by build and round bucket. Variants:
#   V0: recorded winners, bake-off swap rule (swap after r = max_rounds//2 and after r = max_rounds+3, +6, ...)
#   V1: V0 plus an extra swap after round max_rounds (overtime start), i.e. the swap rule the data suggest
#   V2: V0 with winners inverted on 9961 switch rounds (first round after each swap: 9, 13, 25, 28)
# Agreement is reported per build and per bucket (<=12, 13..24, >=25; MR8 uses max_rounds=16).
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)

SP = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3

R = pl.read_parquet(os.path.join(SP, "verify_A2_1_round_table.parquet")).select(
    ["match_key", "round", "winner_team_code", "build_num"])
hdr = pl.scan_parquet(os.path.join(BASE, "header", "dataframe_00000")).select(["key", "build_num"]).collect().rename(
    {"key": "match_key"})
fes = (pl.scan_parquet(sorted(glob.glob(os.path.join(BASE, f"basic_round_state{SUF}", "dataframe_*")))[0])
       .select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
       .filter(pl.col("event_type") == "round_freeze_end").collect()
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fsec = fes.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("first_second"))
fmt = fes.select("match_key").unique().join(fsec, on="match_key", how="left")


def max_rounds_for(first_second):
    if first_second == 9:
        return 16
    if first_second == 13:
        return 24
    return None


mr_map = {r[0]: max_rounds_for(r[1]) for r in fmt.rows()}
bld = {r[0]: r[1] for r in hdr.rows()}
live = {(r[0], r[1]): (r[2], r[3]) for r in fes.select(["match_key", "round", "t_score", "ct_score"]).rows()}

rec = {}
for mk, rd, w in R.select(["match_key", "round", "winner_team_code"]).rows():
    rec[(mk, rd)] = w


def swap_base(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def swap_alt(r, mr):
    return swap_base(r, mr) or r == mr


def run(variant):
    out = {}  # (build, bucket) -> [n, agree]
    by_match = {}
    for (mk, rd), w in rec.items():
        by_match.setdefault(mk, {})[rd] = w
    for mk, rounds in by_match.items():
        mr = mr_map.get(mk)
        if mr is None:
            continue
        bn = bld[mk]
        side_a = T
        a = b = 0
        for r in sorted(rounds):
            lead = (b - a) if side_a == T else (a - b)
            if (mk, r) in live:
                t_s, ct_s = live[(mk, r)]
                bucket = "r<=12" if r <= 12 else ("r13-24" if r <= 24 else "r>=25")
                key = (bn, bucket)
                o = out.setdefault(key, [0, 0])
                o[0] += 1
                o[1] += int(lead == (ct_s - t_s))
            w = rounds[r]
            if variant == "V2" and bn == 9961 and r in ([9] if mr == 16 else [13] if mr == 24 else []) + ([25, 28] if mr == 24 else []):
                w = 5 - w
            if w == side_a:
                a += 1
            elif w == (CT if side_a == T else T):
                b += 1
            sw = swap_alt(r, mr) if variant == "V1" else swap_base(r, mr)
            if sw:
                side_a = CT if side_a == T else T
    return out


for variant in ["V0", "V1", "V2"]:
    res = run(variant)
    print(f"== variant {variant}: (build, bucket) -> (n, agree, pct)")
    for key in sorted(res):
        n, ag = res[key]
        print(f"   {key}: n={n} agree={ag} pct={100.0 * ag / n:.2f}", flush=True)
print("DONE lead", flush=True)
