"""verify D5-9 checks: worked examples, the 9->10 'swap' tie-artifact caveat, and the regime-marker agreement.

Uses scratchpad tables from verify_D5_9_build.py (2024) and the 2026 tome (projected reads, C: scratchpad only).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_checks.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")
FE = pl.read_parquet(SCR / "fe.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
REr = pl.read_parquet(SCR / "end_raw.parquet")
BS = pl.read_parquet(SCR / "bomb.parquet")
DE = pl.read_parquet(SCR / "deaths.parquet")
H = pl.read_parquet(SCR / "header.parquet")

regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())
RE = REr.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("winner_team_code").first().alias("w"), pl.col("win_reason_code").first().alias("reason"),
    pl.col("tick").first().alias("end_tick"))
rounds = defaultdict(dict)
R = FE.select(["match_key", "round", "fe_t", "fe_ct", "fe_tick"]).join(RE, on=["match_key", "round"], how="full",
                                                                       coalesce=True)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row
BO = {}
for r_ in BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).iter_rows(named=True):
    BO.setdefault((r_["match_key"], r_["round"]), set()).add(r_["event_type"])
DEs = DE.sort(["match_key", "round", "tick"])
lv = {}
for (mk, r_), grp in DEs.group_by(["match_key", "round"], maintain_order=True):
    last = grp.tail(1)
    lv[(mk, r_)] = int(last["player_team_code"][0]) if last["player_team_code"][0] is not None else None
hmap = {r["match_key"]: r for r in H.iter_rows(named=True)}


def g(mk, r):
    return rounds.get(mk, {}).get(r)


print("=== 1. worked examples (values from the tome)")
for mk in ["csds/2024/02/12/1d1017f6-2e4b-4735-be2e-4d0fd3a5a225/csds",
           "csds/2024/02/01/053996fe-a1e1-44ea-8cde-3ca99eeb57e1/csds"]:
    if mk not in rounds:
        print("  missing key:", mk)
        continue
    print("  ", mk, "map", hmap[mk]["map_name"], "regime", "A" if mk in regA else "B")
    for r in range(7, 11):
        row = g(mk, r) or {}
        print(f"     r{r:2d}: fe(t,ct)=({row.get('fe_t')},{row.get('fe_ct')})  end winner={row.get('w')} "
              f"reason={row.get('reason')}  bomb={sorted(BO.get((mk, r), []))}  last victim team={lv.get((mk, r))}")

print("\n=== 2. worked example arithmetic (regime A) : fe8=(5,2) fe9=(2,6) fe10=(2,7)")
print("   8->9 under side-swap mapping: t9 - ct8 = 2-2 = 0, ct9 - t8 = 6-5 = 1 -> T won round 8 (w8 must be 2)")
print("   9->10 under no-swap mapping: ct10 - ct9 = 7-6 = 1 -> CT won round 9 (truth 3); round_end(9) reports 2")

print("\n=== 3. tie-artifact caveat: boundary 9->10 side-swap classification with round_end winner w9")
def classify(t, ct, t2, ct2, w):
    if None in (t, ct, t2, ct2, w) or w not in (2, 3):
        return "na"
    a = 1 if w == 2 else 0
    b = 1 if w == 3 else 0
    ns = (t2 == t + a) and (ct2 == ct + b)
    sw = (t2 == ct + b) and (ct2 == t + a)
    if ns and sw:
        return "ambig"
    if ns:
        return "ns"
    if sw:
        return "sw"
    return "other"

c = Counter()
examples = []
for mk, rr in rounds.items():
    a_, b_ = rr.get(9), rr.get(10)
    if not a_ or not b_:
        continue
    cl = classify(a_.get("fe_t"), a_.get("fe_ct"), b_.get("fe_t"), b_.get("fe_ct"), a_.get("w"))
    if cl != "sw":
        continue
    reg = "A" if mk in regA else "B"
    tie = a_.get("fe_t") == a_.get("fe_ct")
    c[(reg, "tie_at_fe9" if tie else "no_tie")] += 1
    if len(examples) < 3:
        examples.append((mk, reg, a_.get("fe_t"), a_.get("fe_ct"), b_.get("fe_t"), b_.get("fe_ct"), a_.get("w")))
print("  boundary 9->10 classified sw (round_end w9 used):", dict(c))
for e in examples:
    print("   e.g.", e)

print("\n=== 4. regime marker vs round_end label 1 (2024)")
lab1 = set(REr.filter(pl.col("round") == 1)["match_key"].to_list())
allk = set(rounds.keys())
print("  marker present:", len(regA), " label-1 round_end present:", len(lab1),
      " marker&label1:", len(regA & lab1), " marker only:", len(regA - lab1), " label1 only:", len(lab1 - regA))

print("\n=== 5. 2026 tome: marker and label-1 presence (is the 2024 regime signature present there?)")
B = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S = ".2026-09-18,2026-10-08"
def scan(ch):
    fs = sorted(glob.glob(str(B / f"basic_{ch}{S}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")
ms26 = set(scan("round_state").filter(pl.col("event_type") == "round_announce_match_start").select("match_key").unique().collect()["match_key"].to_list())
re26 = scan("round_end").filter(pl.col("round") == 1).select("match_key").unique().collect()
lab1_26 = set(re26["match_key"].to_list())
all26 = set(scan("round_end").select("match_key").unique().collect()["match_key"].to_list())
print("  2026 matches with round_end rows:", len(all26), " marker:", len(ms26), " label-1 round_end:", len(lab1_26),
      " marker&label1:", len(ms26 & lab1_26))
