"""verify D5-9 mode: is regime (A/B) confounded with game mode (player count) for the 8-round halftime group?

Per match: regime (round_announce_match_start marker), mode (modal round_end player_count and
player_death roster size), halftime format (last-round-half announcement label 7 or 11), and the
round-level inversion test at the first post-halftime label (9 for label 7, 13 for label 11)
using bomb and last-victim truths (independent of score fields and of the round_end winner).

Uses scratchpad tables from verify_D5_9_build.py. Writes nothing to the project or to E:.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_mode.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")
H = pl.read_parquet(SCR / "header.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
MK = pl.read_parquet(SCR / "marks.parquet")
REr = pl.read_parquet(SCR / "end_raw.parquet")
BS = pl.read_parquet(SCR / "bomb.parquet")
DE = pl.read_parquet(SCR / "deaths.parquet")

regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())
ann_lab = {r["match_key"]: r["min_round"] for r in
           MK.filter(pl.col("event_type") == "round_announce_last_round_half").iter_rows(named=True)}
RE = (REr.sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"], maintain_order=True).agg(
          pl.col("winner_team_code").first().alias("w"),
          pl.col("tick").first().alias("end_tick"),
          pl.col("player_count").first().alias("pc")))
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
BO = BO.with_columns(
    bo_side=pl.when(pl.col("evs").list.contains("bomb_exploded") & ~pl.col("evs").list.contains("bomb_defused")).then(2)
    .when(pl.col("evs").list.contains("bomb_defused") & ~pl.col("evs").list.contains("bomb_exploded")).then(3)
    .otherwise(None)).select(["match_key", "round", "bo_side"])
PL = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.len().alias("n_pl"))
DEs = DE.sort(["match_key", "round", "tick"])
LVt = DEs.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lv_tick"))
LV = (DEs.join(LVt, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lv_tick"))
      .group_by(["match_key", "round"]).agg(pl.col("player_team_code").n_unique().alias("lv_nteam"),
                                            pl.col("player_team_code").max().alias("lv_victim")))
R = (FE.select(["match_key", "round", "fe_t", "fe_ct", "fe_tick"])
     .join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(LV, on=["match_key", "round"], how="left"))
rounds = defaultdict(dict)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row

# roster size from player_death: distinct victim team codes per team is not a roster; use modal pc instead
pc_mode = {}
for mk, rr in rounds.items():
    c = Counter(v["pc"] for v in rr.values() if v.get("pc") is not None)
    pc_mode[mk] = c.most_common(1)[0][0] if c else None

# halftime label expected: 7 -> first half of 8 rounds (first post-halftime round = 9); 11 -> 12 rounds (13)
def first_post(mk):
    lab = ann_lab.get(mk)
    if lab == 7:
        return 9
    if lab == 11:
        return 13
    return None


def inv_stats(mks, r):
    bn = bd = ln = ld = 0
    for mk in mks:
        row = rounds.get(mk, {}).get(r)
        if not row or row.get("w") not in (2, 3):
            continue
        if row.get("bo_side") is not None:
            bn += 1
            bd += int(row["bo_side"] != row["w"])
        elif row.get("n_pl") is None and row.get("lv_nteam") == 1 and row.get("lv_victim") in (2, 3):
            ln += 1
            ld += int((5 - row["lv_victim"]) != row["w"])
    return bn, bd, ln, ld


def fmt(x):
    bn, bd, ln, ld = x
    p = lambda a, b: f"{100.0 * a / b:5.1f}%" if b else "  -   "
    return f"bomb {bd}/{bn} ({p(bd, bn)})  LV {ld}/{ln} ({p(ld, ln)})"


all_mk = list(rounds.keys())
print("=== 1. mode (modal round_end player_count) by regime and halftime label")
tab = Counter()
for mk in all_mk:
    tab[("A" if mk in regA else "B", ann_lab.get(mk, "none"), pc_mode.get(mk))] += 1
for k, v in sorted(tab.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), -kv[1])):
    if v >= 3:
        print("  regime", k[0], " ann_label", k[1], " pc_mode", k[2], " matches", v)

print("\n=== 2. first post-halftime inversion test by (regime, pc_mode, halftime label)")
groups = defaultdict(list)
for mk in all_mk:
    fp = first_post(mk)
    if fp is None:
        continue
    groups[("A" if mk in regA else "B", ann_lab[mk], "5v5" if pc_mode.get(mk) == 20 else
            ("2v2-like(pc8)" if pc_mode.get(mk) == 8 else f"pc{pc_mode.get(mk)}"))].append(mk)
for key in sorted(groups, key=lambda k: (k[1], k[0], k[2])):
    mks = groups[key]
    fp = 9 if key[1] == 7 else 13
    print(f"  regime {key[0]}, ann {key[1]} (first post-halftime label {fp}), mode {key[2]}, matches {len(mks)}:")
    print(f"      label {fp}: {fmt(inv_stats(mks, fp))}")
    prev = fp - 1
    print(f"      label {prev} (last pre-halftime): {fmt(inv_stats(mks, prev))}")
    nxt = fp + 1
    print(f"      label {nxt} (next): {fmt(inv_stats(mks, nxt))}")

print("\n=== 3. label-7 matches outside the claim group, detail")
sw_cls = {}
for mk in all_mk:
    a_, b_ = rounds[mk].get(8), rounds[mk].get(9)
    if not a_ or not b_:
        sw_cls[mk] = "absent"
        continue
    sw_cls[mk] = "have"
for mk in [m for m in all_mk if ann_lab.get(m) == 7]:
    fp = 9
    row8 = rounds[mk].get(8, {})
    row9 = rounds[mk].get(9, {})
    if row8 and row9 and None not in (row8.get("fe_t"), row8.get("fe_ct"), row9.get("fe_t"), row9.get("fe_ct"), row8.get("w")):
        pass
c = Counter()
for mk in [m for m in all_mk if ann_lab.get(m) == 7]:
    c[("A" if mk in regA else "B", "pc20" if pc_mode.get(mk) == 20 else f"pc{pc_mode.get(mk)}")] += 1
print("  ann-7 matches by (regime, mode):", dict(c))

print("\n=== 4. all regime-A 5v5 (pc20) matches: first post-halftime inversion by announced format")
for lab in [7, 11, None]:
    mks = [m for m in all_mk if (m in regA) and pc_mode.get(m) == 20 and ann_lab.get(m) == lab]
    if lab is None:
        print("  ann None:", len(mks), "matches")
        continue
    fp = 9 if lab == 7 else 13
    print(f"  ann {lab}, n={len(mks)}: label {fp} {fmt(inv_stats(mks, fp))}")

print("\n=== 5. all regime-B 5v5 (pc20) matches, same test")
for lab in [7, 11, None]:
    mks = [m for m in all_mk if (m not in regA) and pc_mode.get(m) == 20 and ann_lab.get(m) == lab]
    if lab is None:
        print("  ann None:", len(mks), "matches")
        continue
    fp = 9 if lab == 7 else 13
    print(f"  ann {lab}, n={len(mks)}: label {fp} {fmt(inv_stats(mks, fp))}")

print("\n=== 6. regime-B matches with pc 8 (2v2-like), by ann label")
for lab in [7, 11, None]:
    mks = [m for m in all_mk if (m not in regA) and pc_mode.get(m) == 8 and ann_lab.get(m) == lab]
    if lab is None:
        print("  ann None:", len(mks), "matches")
        continue
    fp = 9 if lab == 7 else 13
    print(f"  ann {lab}, n={len(mks)}: label {fp} {fmt(inv_stats(mks, fp))}")
